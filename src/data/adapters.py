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
    
    def __init__(self, file_glob: str, base_path: Optional[str] = None, preload_all: bool = True):
        """
        Initialize the adapter with a glob pattern for Parquet files.
        
        Args:
            file_glob: Glob pattern for Parquet files (e.g., "data-captions/part-*.parquet")
            base_path: Optional base path to prepend to file_glob
            preload_all: If True, load all files into memory at startup (default: True)
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
        
        # Initialize file cache for loaded Parquet tables
        self._file_cache = {}  # file_idx -> pyarrow.Table
        self.preload_all = preload_all
        
        # Preload all files if requested
        if self.preload_all:
            self._preload_all_files()
    
    def _preload_all_files(self):
        """Preload all Parquet files into memory."""
        import pyarrow.parquet as pq
        import time
        
        logger.info(f"Preloading {len(self.files)} Parquet files into memory...")
        total_size = 0
        start_time = time.time()
        
        for i, file_path in enumerate(self.files):
            file_start = time.time()
            table = pq.read_table(file_path)
            self._file_cache[i] = table
            
            # Estimate memory usage
            size_mb = table.nbytes / (1024 * 1024)
            total_size += size_mb
            
            logger.info(f"  Loaded file {i+1}/{len(self.files)}: {Path(file_path).name} ({size_mb:.1f} MB) in {time.time() - file_start:.1f}s")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ All files preloaded: {total_size:.1f} MB total in {elapsed:.1f}s")
    
    def _build_index_mapping(self):
        """Build mapping from global index to (file_index, local_index)."""
        import pyarrow.parquet as pq
        self._file_indices = []
        cumulative = 0
        
        if self.preload_all:
            # Use cached tables for row counts
            for i in range(len(self.files)):
                num_rows = len(self._file_cache[i])
                self._file_indices.append((cumulative, cumulative + num_rows))
                cumulative += num_rows
        else:
            # Read metadata for row counts
            for i, file_path in enumerate(self.files):
                metadata = pq.read_metadata(file_path)
                num_rows = metadata.num_rows
                self._file_indices.append((cumulative, cumulative + num_rows))
                cumulative += num_rows
        
        logger.debug(f"Built index mapping for {len(self.files)} files")
    
    def _get_file_and_local_index(self, global_idx: int) -> tuple[int, int]:
        """Convert global index to (file_index, local_index within file)."""
        for file_idx, (start, end) in enumerate(self._file_indices):
            if start <= global_idx < end:
                return file_idx, global_idx - start
        raise IndexError(f"Global index {global_idx} out of range")
    
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
        
        # For efficient random access, we need to map indices to files
        if not hasattr(self, '_file_indices'):
            # Build index mapping on first access
            self._build_index_mapping()
        
        # Find which file contains this index
        file_idx, local_idx = self._get_file_and_local_index(index)
        
        # Get the table (either preloaded or load on demand)
        if self.preload_all:
            # Use preloaded table
            table = self._file_cache[file_idx]
        else:
            # Load on demand (fallback for low memory systems)
            import pyarrow.parquet as pq
            if file_idx not in self._file_cache:
                logger.debug(f"Loading Parquet file {file_idx} on demand...")
                self._file_cache[file_idx] = pq.read_table(self.files[file_idx])
            table = self._file_cache[file_idx]
        
        # Get the specific row
        row = table.slice(local_idx, 1)
        
        # Convert to Python dictionary
        row_dict = row.to_pydict()
        
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