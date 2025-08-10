"""
Tests for protocol definitions and compliance.
"""
import pytest
from typing import Dict, Any
import numpy as np

from src.data.protocols import SourceAdapter, Transform


def test_source_adapter_protocol():
    """Test that SourceAdapter protocol is properly defined."""
    # This just verifies the protocol exists and has the right methods
    assert hasattr(SourceAdapter, '__len__')
    assert hasattr(SourceAdapter, 'get_canonical_sample')


def test_transform_protocol():
    """Test that Transform protocol is properly defined."""
    assert hasattr(Transform, '__call__')


class MockAdapter:
    """A mock adapter that implements the SourceAdapter protocol."""
    
    def __init__(self, data: list):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_canonical_sample(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of range")
        return self.data[index]


class MockTransform:
    """A mock transform that implements the Transform protocol."""
    
    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value
    
    def __call__(self, sample: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        sample[self.key] = self.value
        return sample


def test_mock_adapter_implements_protocol():
    """Test that our mock adapter properly implements SourceAdapter."""
    adapter = MockAdapter([
        {"image": b"fake_image_1", "prompts": ["caption 1"]},
        {"image": b"fake_image_2", "prompts": ["caption 2"]},
    ])
    
    # Check it implements the protocol
    assert isinstance(adapter, SourceAdapter)
    
    # Test the methods work
    assert len(adapter) == 2
    
    sample = adapter.get_canonical_sample(0)
    assert sample["image"] == b"fake_image_1"
    assert sample["prompts"] == ["caption 1"]
    
    # Test out of bounds
    with pytest.raises(IndexError):
        adapter.get_canonical_sample(10)


def test_mock_transform_implements_protocol():
    """Test that our mock transform properly implements Transform."""
    transform = MockTransform("test_key", "test_value")
    
    # Check it implements the protocol
    assert isinstance(transform, Transform)
    
    # Test it works
    rng = np.random.default_rng(42)
    sample = {"original": "data"}
    result = transform(sample, rng)
    
    assert result["original"] == "data"
    assert result["test_key"] == "test_value"


def test_protocol_runtime_checkable():
    """Test that protocols are runtime checkable."""
    from src.data.protocols import SourceAdapter, Transform
    
    # A class that doesn't implement the protocol
    class NotAnAdapter:
        pass
    
    # Should not be instance of protocol
    not_adapter = NotAnAdapter()
    assert not isinstance(not_adapter, SourceAdapter)
    assert not isinstance(not_adapter, Transform)
    
    # Our mocks should be instances
    adapter = MockAdapter([])
    transform = MockTransform("key", "value")
    
    assert isinstance(adapter, SourceAdapter)
    assert isinstance(transform, Transform)