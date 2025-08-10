"""
Protocol definitions for the virtual dataset system.

These protocols define the contracts that different components must implement,
enabling a flexible and testable data loading pipeline.
"""
from typing import Protocol, Dict, Any, runtime_checkable
import numpy as np


@runtime_checkable
class SourceAdapter(Protocol):
    """
    Protocol for data source adapters.
    
    Adapters know how to count and fetch from specific raw data sources
    (e.g., Parquet files, HuggingFace datasets, local directories).
    """
    
    def __len__(self) -> int:
        """
        Returns the total number of items in the source.
        This should be fast and preferably cached.
        """
        ...
    
    def get_canonical_sample(self, index: int) -> Dict[str, Any]:
        """
        Returns one item by index, translated to canonical format.
        
        The canonical format should include:
        - "image": bytes (raw PNG/JPEG bytes)
        - "prompts": List[str] (all available captions/prompts)
        - "image_id": str (unique identifier)
        - Any other metadata fields
        
        Args:
            index: Index of the item to fetch (0 <= index < len(self))
            
        Returns:
            Dictionary in canonical format
        """
        ...


@runtime_checkable
class Transform(Protocol):
    """
    Protocol for data transformation steps.
    
    Transforms are callable objects that modify a sample dictionary
    in place or return a modified copy. They should be composable
    in a pipeline.
    """
    
    def __call__(self, sample: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Apply transformation to a sample.
        
        Args:
            sample: Dictionary containing the data to transform
            rng: NumPy random generator for reproducible randomness
            
        Returns:
            Transformed sample dictionary
        """
        ...