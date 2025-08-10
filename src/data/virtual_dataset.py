"""
VirtualDataset implementation for on-the-fly data generation.

This module provides the VirtualDataset class which orchestrates
adapters and transforms to create training samples dynamically.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Union
import numpy as np
from loguru import logger
import importlib
from torch.utils.data import IterableDataset

from .protocols import SourceAdapter, Transform
from .transforms import ComposeTransforms


class VirtualDataset(IterableDataset):
    """
    Virtual dataset that generates samples on-the-fly.
    
    This dataset doesn't store pre-computed samples but instead
    generates them dynamically by:
    1. Loading data from adapters
    2. Applying transforms
    3. Yielding augmented samples
    """
    
    def __init__(
        self,
        manifest_path: Optional[str] = None,
        adapter: Optional[SourceAdapter] = None,
        transforms: Optional[List[Transform]] = None,
        virtual_scale_factor: int = 1,
        seed: Optional[int] = None
    ):
        """
        Initialize VirtualDataset.
        
        Args:
            manifest_path: Path to YAML manifest file (if using manifest)
            adapter: Source adapter instance (if not using manifest)
            transforms: List of transforms (if not using manifest)
            virtual_scale_factor: How many virtual epochs per real epoch
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        if manifest_path:
            self._load_from_manifest(manifest_path)
        else:
            if adapter is None:
                raise ValueError("Either manifest_path or adapter must be provided")
            self.adapter = adapter
            self.transforms = transforms or []
            self.virtual_scale_factor = virtual_scale_factor
        
        # Calculate virtual length
        self.real_length = len(self.adapter)
        self.virtual_length = self.real_length * self.virtual_scale_factor
        
        logger.info(
            f"VirtualDataset initialized: "
            f"real_length={self.real_length:,}, "
            f"virtual_length={self.virtual_length:,}, "
            f"scale_factor={self.virtual_scale_factor}"
        )
        
        # Create transform pipeline
        if self.transforms:
            self.transform_pipeline = ComposeTransforms(self.transforms)
        else:
            self.transform_pipeline = None
    
    def _load_from_manifest(self, manifest_path: str):
        """
        Load configuration from YAML manifest.
        
        The manifest should contain:
        - adapter_class: Full module path to adapter class
        - adapter_args: Arguments for adapter initialization
        - transforms: List of transform configurations
        - virtual_scale_factor: Scale factor for virtual dataset
        """
        with open(manifest_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loading VirtualDataset from manifest: {manifest_path}")
        
        # Load adapter
        adapter_config = config['adapter']
        adapter_class_path = adapter_config['class']
        adapter_args = adapter_config.get('args', {})
        
        # Import adapter class
        module_path, class_name = adapter_class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        adapter_class = getattr(module, class_name)
        
        # Initialize adapter
        self.adapter = adapter_class(**adapter_args)
        logger.info(f"Loaded adapter: {adapter_class_name} with {len(self.adapter)} samples")
        
        # Load transforms
        self.transforms = []
        for transform_config in config.get('transforms', []):
            transform_class_path = transform_config['class']
            transform_args = transform_config.get('args', {})
            
            # Import transform class
            module_path, class_name = transform_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            transform_class = getattr(module, class_name)
            
            # Initialize transform
            transform = transform_class(**transform_args)
            self.transforms.append(transform)
            logger.debug(f"Loaded transform: {class_name}")
        
        # Get virtual scale factor
        self.virtual_scale_factor = config.get('virtual_scale_factor', 1)
    
    def __len__(self) -> int:
        """
        Return virtual length of dataset.
        
        This is what the trainer uses to determine epoch size.
        """
        return self.virtual_length
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through virtual dataset.
        
        Yields:
            Augmented samples ready for collator
        """
        # Create a new RNG for this iteration
        # This ensures reproducibility across epochs
        worker_info = None
        try:
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
        except ImportError:
            pass
        
        if worker_info is not None:
            # In multi-worker setup, use worker id to seed
            worker_seed = self.seed + worker_info.id if self.seed else worker_info.id
            iter_rng = np.random.default_rng(worker_seed)
        else:
            # Single worker
            iter_rng = np.random.default_rng(self.seed)
        
        # Generate virtual samples
        for virtual_idx in range(self.virtual_length):
            # Pick a random real sample
            real_idx = iter_rng.integers(0, self.real_length)
            
            # Get canonical sample from adapter
            sample = self.adapter.get_canonical_sample(real_idx)
            
            # Apply transforms
            if self.transform_pipeline:
                # Create a new RNG for this sample (for reproducibility)
                sample_seed = virtual_idx + (self.seed or 0)
                sample_rng = np.random.default_rng(sample_seed)
                sample = self.transform_pipeline(sample, sample_rng)
            
            yield sample
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        Get a specific sample by index.
        
        This is useful for debugging and inspection.
        
        Args:
            index: Virtual index of sample
            
        Returns:
            Processed sample
        """
        if index < 0 or index >= self.virtual_length:
            raise IndexError(f"Index {index} out of range [0, {self.virtual_length})")
        
        # Use the index as seed for reproducibility
        sample_rng = np.random.default_rng(index + (self.seed or 0))
        
        # Map virtual index to real index
        real_idx = sample_rng.integers(0, self.real_length)
        
        # Get canonical sample
        sample = self.adapter.get_canonical_sample(real_idx)
        
        # Apply transforms
        if self.transform_pipeline:
            sample = self.transform_pipeline(sample, sample_rng)
        
        return sample


def create_virtual_dataset_from_config(config_path: str) -> VirtualDataset:
    """
    Convenience function to create VirtualDataset from config file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configured VirtualDataset instance
    """
    return VirtualDataset(manifest_path=config_path)