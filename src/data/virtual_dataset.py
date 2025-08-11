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
from torch.utils.data import Dataset
import time
import os
import logfire
import random

# Initialize Logfire for performance tracking
logfire_token = os.getenv('LOGFIRE_TOKEN')
if logfire_token:
    # Disable inspect_arguments to avoid introspection warnings during eval
    logfire.configure(token=logfire_token, service_name='virtual-dataset', inspect_arguments=False)
    logger.info("Logfire instrumentation enabled for VirtualDataset")

from .protocols import SourceAdapter, Transform
from .transforms import ComposeTransforms


class VirtualDataset(Dataset):
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
        virtual_scale_factor: float = 1,
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
        self.virtual_length = int(self.real_length * self.virtual_scale_factor)

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
        
        # Pre-generate orders for deterministic indexing
        self._generate_orders()

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

    def _generate_orders(self):
        """Pre-generate shuffled orders for deterministic indexing."""
        # Phase A: Shuffled order for all real samples
        self.order_a = np.arange(self.real_length)
        self.rng.shuffle(self.order_a)
        
        # Phase B: Random indices with replacement
        phase_b_count = int(self.virtual_length - self.real_length)
        if phase_b_count > 0:
            self.order_b = self.rng.integers(0, self.real_length, size=phase_b_count)
        else:
            self.order_b = np.array([])
        
        logger.debug(f"Generated orders: Phase A={len(self.order_a)}, Phase B={len(self.order_b)}")

    def __len__(self) -> int:
        """
        Return virtual length of dataset.

        This is what the trainer uses to determine epoch size.
        """
        return self.virtual_length
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample by index (supports resuming from checkpoint).
        
        MAINTAINS LAZY LOADING: Only the indices are pre-generated,
        actual data is loaded on-demand from Parquet files.
        
        Args:
            index: Index of the sample to get (0 to virtual_length-1)
            
        Returns:
            Sample ready for collator
        """
        start_time = time.time()
        
        with logfire.span(
            'virtual_dataset.get_sample',
            index=index,
            phase='A' if index < self.real_length else 'B'
        ):
            if index < 0 or index >= self.virtual_length:
                raise IndexError(f"Index {index} out of range [0, {self.virtual_length})")
            
            # Determine which phase this index belongs to
            if index < self.real_length:
                # Phase A: Identity only
                real_idx = self.order_a[index]
                force_identity = True
                phase = "Phase A (identity)"
            else:
                # Phase B: 30/70 split handled by IdentityOrAug
                phase_b_idx = index - self.real_length
                real_idx = self.order_b[phase_b_idx]
                force_identity = False
                phase = "Phase B (mixed)"
            
            # Get canonical sample from adapter (LAZY LOADING HERE)
            with logfire.span('adapter.get_sample', real_idx=int(real_idx)):
                adapter_start = time.time()
                sample = self.adapter.get_canonical_sample(int(real_idx))
                adapter_time = time.time() - adapter_start
                # logfire.debug(f"Loaded sample from Parquet in {adapter_time:.3f}s", adapter_time_ms=adapter_time*1000)
            
            # Apply transforms
            if self.transform_pipeline:
                with logfire.span('apply_transforms', force_identity=force_identity):
                    transform_start = time.time()
                    # Create deterministic RNG for this specific index
                    sample_seed = index + (self.seed or 0)
                    sample_rng = np.random.default_rng(sample_seed)
                    
                    # Pass force_identity flag for Phase A
                    if force_identity:
                        sample['__force_identity__'] = True
                    
                    sample = self.transform_pipeline(sample, sample_rng)
                    
                    # Clean up the flag
                    sample.pop('__force_identity__', None)
                    
                    transform_time = time.time() - transform_start
                    # logfire.debug(f"Applied transforms in {transform_time:.3f}s", transform_time_ms=transform_time*1000)
            
            # Format output for collator
            with logfire.span('format_output'):
                # For caption datasets: put caption in assistant slot, minimal prompt in user
                # Use varied prompts for training diversity (caption-appropriate)
                caption_prompts = [
                    "Please generate a caption for this image.",
                    "Write a caption for this image.",
                    "What's shown in this image?",
                    "Describe what you see.",
                    "Caption this image.",
                    "What does this image show?",
                    "Provide a caption."
                ]
                
                # Randomly select a prompt for training diversity
                # (eval uses the base prompt for fair comparison)
                prompt = random.choice(caption_prompts) if self.training else "Please generate a caption for this image."
                
                output = {
                    "images": [sample["image"]],  # PIL image in a list
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": sample["prompts"][0]}  # Caption goes here for training
                            ]
                        }
                    ]
                }
            
            total_time = time.time() - start_time
            # logfire.debug(
            #     f"{phase}: Generated sample {index} in {total_time:.3f}s",
            #     total_time_ms=total_time*1000,
            #     phase=phase,
            #     index=index
            # )
            
            return output

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through virtual dataset.
        
        Simply iterates through indices using __getitem__.
        This maintains compatibility while supporting indexing.
        
        Yields:
            Samples ready for collator
        """
        for i in range(len(self)):
            yield self[i]

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
