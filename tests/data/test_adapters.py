"""
Tests for data source adapters.
"""
import pytest
from pathlib import Path
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
from io import BytesIO

from src.data.adapters import ParquetCaptionAdapter, HuggingFaceDatasetAdapter
from src.data.protocols import SourceAdapter


def create_test_parquet_files(temp_dir: Path, num_files: int = 2, rows_per_file: int = 5):
    """Create test Parquet files with image data and captions."""
    files = []
    
    for file_idx in range(num_files):
        # Create fake image bytes (small 2x2 RGB images)
        images = []
        captions = []
        
        for row_idx in range(rows_per_file):
            # Create a small test image
            img = Image.new('RGB', (2, 2), color=(file_idx * 50, row_idx * 50, 128))
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            images.append(buffer.getvalue())
            
            # Create captions
            captions.append([
                f"Caption 1 for file {file_idx} row {row_idx}",
                f"Caption 2 for file {file_idx} row {row_idx}",
            ])
        
        # Create table
        table = pa.table({
            'image': images,
            'captions': captions,
            'metadata': [f"meta_{file_idx}_{i}" for i in range(rows_per_file)]
        })
        
        # Write to file
        file_path = temp_dir / f"part-{file_idx:03d}.parquet"
        pq.write_table(table, file_path)
        files.append(file_path)
    
    return files


class TestParquetCaptionAdapter:
    """Tests for ParquetCaptionAdapter."""
    
    def test_implements_protocol(self):
        """Test that ParquetCaptionAdapter implements SourceAdapter protocol."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            create_test_parquet_files(temp_path, num_files=1, rows_per_file=2)
            
            adapter = ParquetCaptionAdapter(f"{temp_dir}/part-*.parquet")
            assert isinstance(adapter, SourceAdapter)
    
    def test_initialization(self):
        """Test adapter initialization with multiple Parquet files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            files = create_test_parquet_files(temp_path, num_files=3, rows_per_file=5)
            
            adapter = ParquetCaptionAdapter(f"{temp_dir}/part-*.parquet")
            
            # Check that files were found
            assert len(adapter.files) == 3
            
            # Check total length
            assert len(adapter) == 15  # 3 files * 5 rows each
            
            # Check schema is loaded
            assert adapter.schema is not None
            assert 'image' in adapter.schema.names
            assert 'captions' in adapter.schema.names
    
    def test_get_canonical_sample(self):
        """Test fetching samples in canonical format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            create_test_parquet_files(temp_path, num_files=2, rows_per_file=3)
            
            adapter = ParquetCaptionAdapter(f"{temp_dir}/part-*.parquet")
            
            # Test first sample
            sample = adapter.get_canonical_sample(0)
            
            # Check canonical format
            assert 'image' in sample
            assert 'prompts' in sample
            assert 'image_id' in sample
            
            # Check types
            assert isinstance(sample['image'], bytes)
            assert isinstance(sample['prompts'], list)
            assert len(sample['prompts']) == 2  # We created 2 captions per row
            assert isinstance(sample['image_id'], str)
            
            # Check that image is valid PNG
            img = Image.open(BytesIO(sample['image']))
            assert img.format == 'PNG'
            assert img.size == (2, 2)
            
            # Check metadata is included
            assert 'meta_metadata' in sample
    
    def test_get_sample_bounds_checking(self):
        """Test that index bounds are properly checked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            create_test_parquet_files(temp_path, num_files=1, rows_per_file=5)
            
            adapter = ParquetCaptionAdapter(f"{temp_dir}/part-*.parquet")
            
            # Valid indices
            sample = adapter.get_canonical_sample(0)
            assert sample is not None
            
            sample = adapter.get_canonical_sample(4)
            assert sample is not None
            
            # Invalid indices
            with pytest.raises(IndexError):
                adapter.get_canonical_sample(-1)
            
            with pytest.raises(IndexError):
                adapter.get_canonical_sample(5)
            
            with pytest.raises(IndexError):
                adapter.get_canonical_sample(100)
    
    def test_different_caption_field_names(self):
        """Test adapter handles different caption field names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Parquet with 'caption' instead of 'captions'
            table = pa.table({
                'image': [b"fake_image_1", b"fake_image_2"],
                'caption': ["Single caption 1", "Single caption 2"],
            })
            
            file_path = Path(temp_dir) / "test.parquet"
            pq.write_table(table, file_path)
            
            adapter = ParquetCaptionAdapter(str(file_path))
            sample = adapter.get_canonical_sample(0)
            
            assert sample['prompts'] == ["Single caption 1"]
    
    def test_missing_caption_fallback(self):
        """Test adapter handles missing captions gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Parquet without captions
            table = pa.table({
                'image': [b"fake_image_1", b"fake_image_2"],
                'other_field': ["data1", "data2"],
            })
            
            file_path = Path(temp_dir) / "test.parquet"
            pq.write_table(table, file_path)
            
            adapter = ParquetCaptionAdapter(str(file_path))
            sample = adapter.get_canonical_sample(0)
            
            # Should have fallback caption
            assert sample['prompts'] == ["No caption available"]
    
    def test_no_files_found_error(self):
        """Test that adapter raises error when no files match pattern."""
        with pytest.raises(ValueError, match="No files found"):
            ParquetCaptionAdapter("/nonexistent/path/*.parquet")
    
    def test_with_base_path(self):
        """Test adapter with base_path parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            subdir = temp_path / "data"
            subdir.mkdir()
            
            create_test_parquet_files(subdir, num_files=1, rows_per_file=2)
            
            # Use base_path
            adapter = ParquetCaptionAdapter("part-*.parquet", base_path=str(subdir))
            assert len(adapter) == 2


def test_adapter_protocol_compliance():
    """Test that all adapters properly implement the SourceAdapter protocol."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        create_test_parquet_files(temp_path, num_files=1, rows_per_file=2)
        
        # Test ParquetCaptionAdapter
        parquet_adapter = ParquetCaptionAdapter(f"{temp_dir}/part-*.parquet")
        
        # Should be instance of protocol
        assert isinstance(parquet_adapter, SourceAdapter)
        
        # Should have required methods
        assert hasattr(parquet_adapter, '__len__')
        assert hasattr(parquet_adapter, 'get_canonical_sample')
        
        # Methods should work
        assert len(parquet_adapter) > 0
        sample = parquet_adapter.get_canonical_sample(0)
        assert 'image' in sample
        assert 'prompts' in sample
        assert 'image_id' in sample