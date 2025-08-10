"""
Source adapters for different data formats.

These adapters implement the SourceAdapter protocol and handle
the specifics of reading from various data sources.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import pyarrow.dataset as ds
import pyarrow as pa
from glob import glob
from loguru import logger

from .protocols import SourceAdapter


class ParquetCaptionAdapter:
    """
    Adapter for multi-part Parquet datasets containing images and captions.
    
    This adapter efficiently handles large datasets split across multiple
    Parquet files using PyArrow's lazy loading capabilities.
    """
    
    def __init__(self, file_glob: str, base_path: Optional[str] = None):
        """
        Initialize the adapter with a glob pattern for Parquet files.
        
        Args:
            file_glob: Glob pattern for Parquet files (e.g., "data-captions/part-*.parquet")
            base_path: Optional base path to prepend to file_glob
        """
        if base_path:
            full_pattern = str(Path(base_path) / file_glob)
        else:
            full_pattern = file_glob
            
        # Find all matching files
        self.files = sorted(glob(full_pattern))
        if not self.files:
            raise ValueError(f"No files found matching pattern: {full_pattern}")
            
        logger.info(f"Found {len(self.files)} Parquet files matching {full_pattern}")
        
        # Create Arrow dataset - this is lazy and doesn't load data yet
        self.arrow_dataset = ds.dataset(self.files, format="parquet")
        
        # Cache the total row count
        self._real_length = self.arrow_dataset.count_rows()
        logger.info(f"Total rows in dataset: {self._real_length:,}")
        
        # Get schema information
        self.schema = self.arrow_dataset.schema
        logger.debug(f"Dataset schema: {self.schema}")
    
    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return self._real_length
    
    def get_canonical_sample(self, index: int) -> Dict[str, Any]:
        """
        Fetch a single sample by index and convert to canonical format.
        
        Args:
            index: Index of the sample to fetch
            
        Returns:
            Dictionary with canonical format:
            - image: bytes (raw image data)
            - prompts: List[str] (all available captions)
            - image_id: str (unique identifier)
        """
        if index < 0 or index >= self._real_length:
            raise IndexError(f"Index {index} out of range [0, {self._real_length})")
        
        # Use scanner to efficiently fetch just one row
        scanner = self.arrow_dataset.scanner()
        batch = scanner.take([index])
        
        # Convert to Python dictionary
        row_dict = batch.to_pydict()
        
        # Extract values (each field is a list with one item)
        sample = {}
        
        # Handle image field - could be named 'image', 'img', or 'image_bytes'
        image_field = None
        for field_name in ['image', 'img', 'image_bytes']:
            if field_name in row_dict:
                image_field = field_name
                break
        
        if image_field:
            sample['image'] = row_dict[image_field][0]
        else:
            raise KeyError(f"No image field found in Parquet. Available fields: {list(row_dict.keys())}")
        
        # Handle caption/prompt fields - could be single or multiple
        prompts = []
        
        # Check for various caption field names
        caption_fields = ['caption', 'captions', 'prompt', 'prompts', 'text', 'texts']
        for field_name in caption_fields:
            if field_name in row_dict:
                value = row_dict[field_name][0]
                if isinstance(value, list):
                    prompts.extend(value)
                elif isinstance(value, str):
                    prompts.append(value)
                break
        
        if not prompts:
            # Fallback: look for any field containing "caption" or "prompt"
            for field_name, values in row_dict.items():
                if 'caption' in field_name.lower() or 'prompt' in field_name.lower():
                    value = values[0]
                    if isinstance(value, list):
                        prompts.extend(value)
                    elif isinstance(value, str):
                        prompts.append(value)
        
        sample['prompts'] = prompts if prompts else ["No caption available"]
        
        # Generate image_id from file index and row index
        file_idx = index // 10000  # Assuming roughly even distribution
        sample['image_id'] = f"parquet_{file_idx:03d}_row_{index:08d}"
        
        # Include any other metadata fields
        excluded_fields = {image_field} | set(caption_fields)
        for field_name, values in row_dict.items():
            if field_name not in excluded_fields:
                sample[f"meta_{field_name}"] = values[0]
        
        return sample


class HuggingFaceDatasetAdapter:
    """
    Adapter for HuggingFace datasets.
    
    This adapter wraps HuggingFace datasets to provide the same
    interface as other adapters.
    """
    
    def __init__(self, dataset_name: str, split: str = "train", **kwargs):
        """
        Initialize adapter with a HuggingFace dataset.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Which split to use (train/validation/test)
            **kwargs: Additional arguments for load_dataset
        """
        from datasets import load_dataset
        
        self.dataset = load_dataset(dataset_name, split=split, **kwargs)
        self._real_length = len(self.dataset)
        logger.info(f"Loaded {dataset_name} ({split}) with {self._real_length:,} samples")
    
    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return self._real_length
    
    def get_canonical_sample(self, index: int) -> Dict[str, Any]:
        """
        Fetch a single sample and convert to canonical format.
        """
        if index < 0 or index >= self._real_length:
            raise IndexError(f"Index {index} out of range [0, {self._real_length})")
        
        row = self.dataset[index]
        
        # Convert to canonical format
        sample = {}
        
        # Handle image - HF datasets often have PIL images
        if 'image' in row:
            from io import BytesIO
            img = row['image']
            if hasattr(img, 'save'):  # PIL Image
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                sample['image'] = buffer.getvalue()
            else:
                sample['image'] = row['image']  # Already bytes
        
        # Handle captions
        prompts = []
        for field in ['caption', 'text', 'prompt']:
            if field in row:
                value = row[field]
                if isinstance(value, list):
                    prompts.extend(value)
                else:
                    prompts.append(str(value))
        
        sample['prompts'] = prompts if prompts else ["No caption available"]
        sample['image_id'] = f"hf_{index:08d}"
        
        return sample