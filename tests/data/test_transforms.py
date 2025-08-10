"""
Tests for data transforms.
"""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Any

from src.data.transforms import (
    DecodeImage,
    RandomResizedCrop,
    MildColorJitter,
    SelectRandomPrompt,
    PrepareForModel,
    ComposeTransforms
)
from src.data.protocols import Transform


def create_test_image(size=(100, 100), color=(255, 0, 0)):
    """Create a test PIL Image."""
    return Image.new('RGB', size, color)


def image_to_bytes(img: Image.Image, format='PNG') -> bytes:
    """Convert PIL Image to bytes."""
    buffer = BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


class TestDecodeImage:
    """Tests for DecodeImage transform."""
    
    def test_implements_protocol(self):
        """Test that DecodeImage implements Transform protocol."""
        transform = DecodeImage()
        assert isinstance(transform, Transform)
    
    def test_decode_bytes_to_pil(self):
        """Test decoding image bytes to PIL Image."""
        # Create test image and convert to bytes
        img = create_test_image()
        img_bytes = image_to_bytes(img)
        
        # Create sample
        sample = {'image': img_bytes}
        
        # Apply transform
        transform = DecodeImage()
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        # Check result
        assert 'image' in result
        assert isinstance(result['image'], Image.Image)
        assert result['image'].size == (100, 100)
        assert result['image'].mode == 'RGB'
    
    def test_already_pil_image(self):
        """Test that PIL images are passed through."""
        img = create_test_image()
        sample = {'image': img}
        
        transform = DecodeImage()
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        assert result['image'] is img
    
    def test_mode_conversion(self):
        """Test conversion to different modes."""
        # Create RGBA image
        img = Image.new('RGBA', (10, 10), (255, 0, 0, 128))
        img_bytes = image_to_bytes(img, 'PNG')
        
        sample = {'image': img_bytes}
        
        # Convert to RGB
        transform = DecodeImage(mode='RGB')
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        assert result['image'].mode == 'RGB'
    
    def test_missing_image_key(self):
        """Test error when image key is missing."""
        sample = {'other': 'data'}
        
        transform = DecodeImage()
        rng = np.random.default_rng(42)
        
        with pytest.raises(KeyError, match="Sample must have 'image' key"):
            transform(sample, rng)


class TestRandomResizedCrop:
    """Tests for RandomResizedCrop transform."""
    
    def test_implements_protocol(self):
        """Test that RandomResizedCrop implements Transform protocol."""
        transform = RandomResizedCrop(size=(224, 224))
        assert isinstance(transform, Transform)
    
    def test_crop_and_resize(self):
        """Test that image is cropped and resized to target size."""
        img = create_test_image(size=(500, 500))
        sample = {'image': img}
        
        target_size = (224, 224)
        transform = RandomResizedCrop(size=target_size)
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        assert result['image'].size == target_size
    
    def test_different_scales(self):
        """Test different scale ranges."""
        img = create_test_image(size=(500, 500))
        
        # Small scale - should crop small portion
        transform_small = RandomResizedCrop(size=(224, 224), scale=(0.08, 0.2))
        # Large scale - should crop large portion
        transform_large = RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
        
        rng = np.random.default_rng(42)
        
        result_small = transform_small({'image': img}, rng)
        result_large = transform_large({'image': img}, rng)
        
        # Both should produce correct size
        assert result_small['image'].size == (224, 224)
        assert result_large['image'].size == (224, 224)
    
    def test_aspect_ratio_range(self):
        """Test different aspect ratio ranges."""
        img = create_test_image(size=(500, 500))
        sample = {'image': img}
        
        # Different aspect ratios
        transform = RandomResizedCrop(
            size=(224, 224),
            ratio=(0.5, 2.0)  # Wide range of aspect ratios
        )
        
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        assert result['image'].size == (224, 224)
    
    def test_error_on_non_pil_image(self):
        """Test error when image is not PIL Image."""
        sample = {'image': b'not_an_image'}
        
        transform = RandomResizedCrop(size=(224, 224))
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="Image must be PIL Image"):
            transform(sample, rng)


class TestMildColorJitter:
    """Tests for MildColorJitter transform."""
    
    def test_implements_protocol(self):
        """Test that MildColorJitter implements Transform protocol."""
        transform = MildColorJitter()
        assert isinstance(transform, Transform)
    
    def test_color_jitter_applied(self):
        """Test that color jitter modifies the image."""
        # Create a mid-gray image (easier to see changes)
        img = Image.new('RGB', (100, 100), (128, 128, 128))
        sample = {'image': img}
        
        transform = MildColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        # Image should be modified
        assert result['image'] != img
        # But size should be same
        assert result['image'].size == img.size
        assert result['image'].mode == 'RGB'
    
    def test_mild_perturbations(self):
        """Test that mild settings produce small changes."""
        img = Image.new('RGB', (10, 10), (128, 128, 128))
        
        # Very mild jitter
        transform = MildColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        )
        
        # Apply multiple times with different seeds
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        
        result1 = transform({'image': img}, rng1)
        result2 = transform({'image': img}, rng2)
        
        # Results should be different
        img1_array = np.array(result1['image'])
        img2_array = np.array(result2['image'])
        assert not np.array_equal(img1_array, img2_array)
        
        # But changes should be mild (within expected range)
        original_array = np.array(img)
        diff1 = np.abs(img1_array.astype(float) - original_array.astype(float))
        diff2 = np.abs(img2_array.astype(float) - original_array.astype(float))
        
        # Max difference should be reasonable (roughly brightness + contrast effect)
        assert diff1.max() < 100  # Out of 255
        assert diff2.max() < 100
    
    def test_no_jitter_with_zero_params(self):
        """Test that zero parameters produce no change."""
        img = create_test_image()
        sample = {'image': img}
        
        transform = MildColorJitter(
            brightness=0,
            contrast=0,
            saturation=0,
            hue=0
        )
        
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        # Image should be unchanged
        assert np.array_equal(np.array(result['image']), np.array(img))


class TestSelectRandomPrompt:
    """Tests for SelectRandomPrompt transform."""
    
    def test_implements_protocol(self):
        """Test that SelectRandomPrompt implements Transform protocol."""
        transform = SelectRandomPrompt()
        assert isinstance(transform, Transform)
    
    def test_select_single_prompt(self):
        """Test selecting from single prompt."""
        sample = {
            'prompts': ["Single caption"],
            'other': 'data'
        }
        
        transform = SelectRandomPrompt()
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        assert 'prompt' in result
        assert result['prompt'] == "Single caption"
        assert 'prompts' in result  # Original list should remain
    
    def test_select_from_multiple_prompts(self):
        """Test random selection from multiple prompts."""
        prompts = [f"Caption {i}" for i in range(10)]
        sample = {'prompts': prompts}
        
        transform = SelectRandomPrompt()
        
        # Test multiple selections
        selected = set()
        for seed in range(100):
            rng = np.random.default_rng(seed)
            result = transform(sample, rng)
            selected.add(result['prompt'])
        
        # Should have selected multiple different prompts
        assert len(selected) > 1
        # All selected should be from original list
        assert all(p in prompts for p in selected)
    
    def test_custom_prompt_key(self):
        """Test using custom key for selected prompt."""
        sample = {'prompts': ["Caption 1", "Caption 2"]}
        
        transform = SelectRandomPrompt(prompt_key="selected_caption")
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        assert 'selected_caption' in result
        assert result['selected_caption'] in sample['prompts']
    
    def test_missing_prompts_key(self):
        """Test error when prompts key is missing."""
        sample = {'other': 'data'}
        
        transform = SelectRandomPrompt()
        rng = np.random.default_rng(42)
        
        with pytest.raises(KeyError, match="Sample must have 'prompts' key"):
            transform(sample, rng)
    
    def test_empty_prompts_list(self):
        """Test error with empty prompts list."""
        sample = {'prompts': []}
        
        transform = SelectRandomPrompt()
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="Prompts list is empty"):
            transform(sample, rng)


class TestPrepareForModel:
    """Tests for PrepareForModel transform."""
    
    def test_implements_protocol(self):
        """Test that PrepareForModel implements Transform protocol."""
        transform = PrepareForModel()
        assert isinstance(transform, Transform)
    
    def test_prepare_basic_sample(self):
        """Test preparing a basic sample for model."""
        img = create_test_image()
        sample = {
            'image': img,
            'prompt': "Test caption",
            'image_id': 'test_001'
        }
        
        transform = PrepareForModel()
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        # Check structure
        assert 'images' in result
        assert 'messages' in result
        
        # Check images
        assert isinstance(result['images'], list)
        assert len(result['images']) == 1
        assert result['images'][0] is img
        
        # Check messages
        assert isinstance(result['messages'], list)
        assert len(result['messages']) == 2
        
        # Check user message
        user_msg = result['messages'][0]
        assert user_msg['role'] == 'user'
        assert len(user_msg['content']) == 2
        assert user_msg['content'][0]['type'] == 'image'
        assert user_msg['content'][1]['type'] == 'text'
        assert user_msg['content'][1]['text'] == "Test caption"
        
        # Check assistant message
        assistant_msg = result['messages'][1]
        assert assistant_msg['role'] == 'assistant'
        assert assistant_msg['content'][0]['type'] == 'text'
        assert assistant_msg['content'][0]['text'] == ""
    
    def test_include_image_id(self):
        """Test including image_id in output."""
        img = create_test_image()
        sample = {
            'image': img,
            'prompt': "Test",
            'image_id': 'test_123'
        }
        
        transform = PrepareForModel(include_image_id=True)
        rng = np.random.default_rng(42)
        result = transform(sample, rng)
        
        assert 'image_id' in result
        assert result['image_id'] == 'test_123'
    
    def test_missing_required_keys(self):
        """Test error when required keys are missing."""
        transform = PrepareForModel()
        rng = np.random.default_rng(42)
        
        # Missing image
        with pytest.raises(KeyError, match="Sample must have 'image' and 'prompt' keys"):
            transform({'prompt': "Test"}, rng)
        
        # Missing prompt
        with pytest.raises(KeyError, match="Sample must have 'image' and 'prompt' keys"):
            transform({'image': create_test_image()}, rng)


class TestComposeTransforms:
    """Tests for ComposeTransforms."""
    
    def test_implements_protocol(self):
        """Test that ComposeTransforms implements Transform protocol."""
        transform = ComposeTransforms([])
        assert isinstance(transform, Transform)
    
    def test_compose_pipeline(self):
        """Test composing multiple transforms."""
        # Create a full pipeline
        pipeline = ComposeTransforms([
            DecodeImage(),
            RandomResizedCrop(size=(224, 224)),
            MildColorJitter(brightness=0.1),
            SelectRandomPrompt(),
            PrepareForModel()
        ])
        
        # Create input sample
        img = create_test_image(size=(500, 500))
        img_bytes = image_to_bytes(img)
        sample = {
            'image': img_bytes,
            'prompts': ["Caption 1", "Caption 2", "Caption 3"]
        }
        
        # Apply pipeline
        rng = np.random.default_rng(42)
        result = pipeline(sample, rng)
        
        # Check final output
        assert 'images' in result
        assert 'messages' in result
        assert isinstance(result['images'][0], Image.Image)
        assert result['images'][0].size == (224, 224)
        assert result['messages'][0]['content'][1]['text'] in sample['prompts']
    
    def test_empty_pipeline(self):
        """Test empty pipeline returns input unchanged."""
        pipeline = ComposeTransforms([])
        sample = {'test': 'data'}
        
        rng = np.random.default_rng(42)
        result = pipeline(sample, rng)
        
        assert result == sample
    
    def test_transform_order_matters(self):
        """Test that transform order is preserved."""
        # Create transforms that add keys in order
        class AddKey1:
            def __call__(self, sample, rng):
                sample['key1'] = 'first'
                return sample
        
        class AddKey2:
            def __call__(self, sample, rng):
                sample['key2'] = 'second'
                sample['key1'] = 'modified'  # Modify first key
                return sample
        
        pipeline = ComposeTransforms([AddKey1(), AddKey2()])
        
        rng = np.random.default_rng(42)
        result = pipeline({}, rng)
        
        assert result['key1'] == 'modified'  # Should be modified by second transform
        assert result['key2'] == 'second'