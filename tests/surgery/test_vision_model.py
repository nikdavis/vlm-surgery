#!/usr/bin/env python3
"""
Test the Qwen2.5-VL vision model in isolation to debug NaN issues.
Run with: CUDA_VISIBLE_DEVICES="2" CUDA_DEVICE_ORDER=PCI_BUS_ID uv run pytest tests/surgery/test_vision_model.py -v
"""

import torch
import pytest
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
from pathlib import Path
import json

class TestQwenVisionModel:
    """Test suite for Qwen2.5-VL vision model"""
    
    @pytest.fixture(scope="class")
    def model_and_processor(self):
        """Load model and processor once for all tests"""
        print("\nLoading Qwen2.5-VL model...")
        # Load on CPU first, then move to GPU
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
        )
        # Move to the available cuda device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
        # Put in eval mode as we do in training
        model.eval()
        
        print(f"Model loaded on device: {device}")
        
        return model, processor
    
    @pytest.fixture
    def sample_image(self):
        """Create a simple test image"""
        img = Image.new('RGB', (336, 336), color='red')
        return img
    
    @pytest.fixture
    def real_training_image(self):
        """Load a real image from our training dataset"""
        dataset_path = Path("./datasetv2/combined_dataset.json")
        if dataset_path.exists():
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                if data and len(data) > 0:
                    # Get first example's image path
                    img_path = data[0].get('images', [None])[0]
                    if img_path and Path(img_path).exists():
                        return Image.open(img_path).convert('RGB')
        # Fallback to generated image
        return Image.new('RGB', (512, 512), color='blue')
    
    def test_vision_model_basic(self, model_and_processor, sample_image):
        """Test basic vision model forward pass"""
        model, processor = model_and_processor
        
        # Process image as in training
        inputs = processor(
            text="What is in this image?",
            images=sample_image,
            return_tensors="pt"
        ).to(model.device)
        
        print(f"\nInputs shapes:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}, dtype={v.dtype}")
        
        # Get vision features directly
        with torch.no_grad():
            vision_model = model.visual
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs.get("image_grid_thw")
            
            # Call vision model
            vision_features = vision_model(pixel_values, grid_thw=image_grid_thw)
            
            print(f"\nVision features:")
            print(f"  Shape: {vision_features.shape}")
            print(f"  Min: {vision_features.min().item():.4f}")
            print(f"  Max: {vision_features.max().item():.4f}")
            print(f"  Mean: {vision_features.mean().item():.4f}")
            print(f"  Std: {vision_features.std().item():.4f}")
            
            # Check for NaN/Inf
            assert not torch.isnan(vision_features).any(), "Vision features contain NaN!"
            assert not torch.isinf(vision_features).any(), "Vision features contain Inf!"
            assert vision_features.abs().max() < 100, "Vision features have extreme values!"
    
    def test_vision_model_multiple_passes(self, model_and_processor, sample_image):
        """Test that multiple forward passes don't degrade"""
        model, processor = model_and_processor
        
        inputs = processor(
            text="Test prompt",
            images=sample_image,
            return_tensors="pt"
        ).to(model.device)
        
        vision_model = model.visual
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs.get("image_grid_thw")
        
        outputs = []
        with torch.no_grad():
            for i in range(5):
                vision_features = vision_model(pixel_values, grid_thw=image_grid_thw)
                
                # Store stats
                stats = {
                    'iteration': i,
                    'min': vision_features.min().item(),
                    'max': vision_features.max().item(),
                    'mean': vision_features.mean().item(),
                    'has_nan': torch.isnan(vision_features).any().item(),
                }
                outputs.append(stats)
                print(f"  Pass {i}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}")
                
                assert not stats['has_nan'], f"NaN detected on pass {i}!"
        
        # Check consistency across passes
        means = [o['mean'] for o in outputs]
        assert np.std(means) < 0.01, f"Outputs vary too much across passes: {means}"
    
    def test_vision_model_with_training_setup(self, model_and_processor):
        """Test vision model with exact training setup (dtype, no autocast)"""
        model, processor = model_and_processor
        
        # Create pixel values like in training
        batch_size = 1
        num_patches = 256  # Typical for one image
        channels = 1176  # Qwen2.5-VL channels
        
        # Random pixel values in typical range
        pixel_values = torch.randn(num_patches, channels, dtype=torch.float32).to(model.device)
        pixel_values = pixel_values * 2.0  # Scale to typical range seen in training
        
        # Create grid_thw
        image_grid_thw = torch.tensor([[16, 16, 1]], dtype=torch.long).to(model.device)  # 16x16 patches, 1 frame
        
        print(f"\nTest input:")
        print(f"  pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}")
        print(f"  pixel min/max: {pixel_values.min().item():.4f} / {pixel_values.max().item():.4f}")
        print(f"  image_grid_thw: {image_grid_thw}")
        
        vision_model = model.visual
        
        # Test 1: With float32 input converted to float16
        pixel_values_f16 = pixel_values.to(dtype=torch.float16)
        with torch.no_grad():
            vision_features = vision_model(pixel_values_f16, grid_thw=image_grid_thw)
            
            print(f"\nFloat16 output:")
            print(f"  Shape: {vision_features.shape}")
            print(f"  Has NaN: {torch.isnan(vision_features).any().item()}")
            print(f"  Min: {vision_features.min().item():.4f}")
            print(f"  Max: {vision_features.max().item():.4f}")
            
            assert not torch.isnan(vision_features).any(), "NaN in float16 forward!"
    
    def test_vision_model_with_gradient_flow(self, model_and_processor, sample_image):
        """Test vision model with gradient flow (like in training)"""
        model, processor = model_and_processor
        
        inputs = processor(
            text="Test",
            images=sample_image,
            return_tensors="pt"
        ).to(model.device)
        
        vision_model = model.visual
        pixel_values = inputs["pixel_values"].requires_grad_(True)  # Enable gradients
        image_grid_thw = inputs.get("image_grid_thw")
        
        # Forward pass WITHOUT no_grad (like in training)
        vision_features = vision_model(pixel_values, grid_thw=image_grid_thw)
        
        print(f"\nWith gradient flow:")
        print(f"  Output shape: {vision_features.shape}")
        print(f"  Has NaN: {torch.isnan(vision_features).any().item()}")
        print(f"  Requires grad: {vision_features.requires_grad}")
        
        # Simulate a loss and backward
        fake_loss = vision_features.mean()
        fake_loss.backward()
        
        print(f"  After backward:")
        print(f"  Pixel gradients exist: {pixel_values.grad is not None}")
        if pixel_values.grad is not None:
            print(f"  Grad has NaN: {torch.isnan(pixel_values.grad).any().item()}")
            print(f"  Grad max: {pixel_values.grad.abs().max().item():.4f}")
        
        # Check model still works after backward
        with torch.no_grad():
            vision_features_2 = vision_model(inputs["pixel_values"], grid_thw=image_grid_thw)
            print(f"  Second forward has NaN: {torch.isnan(vision_features_2).any().item()}")
            assert not torch.isnan(vision_features_2).any(), "NaN after backward pass!"
    
    def test_merger_layer_directly(self, model_and_processor):
        """Test the merger MLP layers directly"""
        model, processor = model_and_processor
        vision_model = model.visual
        
        # Create dummy input matching merger input size
        # Merger expects output from vision encoder
        dummy_input = torch.randn(256, 5120, dtype=torch.float16).to(model.device)
        
        print("\nTesting merger layers:")
        with torch.no_grad():
            # Test each layer
            x = dummy_input
            for i, layer in enumerate(vision_model.merger.mlp):
                x = layer(x)
                print(f"  After layer {i} ({layer.__class__.__name__}): shape={x.shape}, has_nan={torch.isnan(x).any().item()}")
                assert not torch.isnan(x).any(), f"NaN after merger layer {i}!"
        
        # Final output should be [256, 3584]
        assert x.shape[-1] == 3584, f"Unexpected output dim: {x.shape}"
    
    def test_with_actual_collator_output(self, model_and_processor):
        """Test with pixel values matching what collator produces"""
        model, processor = model_and_processor
        
        # These are the typical shapes we see from collator
        test_cases = [
            (11972, 1176),  # From debug output
            (16352, 1176),
            (3996, 1176),
            (256, 1176),    # Minimal case
        ]
        
        vision_model = model.visual
        
        for num_patches, channels in test_cases:
            print(f"\nTesting shape ({num_patches}, {channels}):")
            
            # Create pixel values in typical range
            pixel_values = torch.randn(num_patches, channels, dtype=torch.float32).to(model.device)
            pixel_values = pixel_values * 1.5  # Typical std
            
            # Grid for this many patches (assuming square-ish)
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                grid_size = 16  # Default
                while grid_size * grid_size < num_patches:
                    grid_size += 1
            
            image_grid_thw = torch.tensor([[grid_size, grid_size, 1]], dtype=torch.long).to(model.device)
            
            # Convert to float16 and test
            pixel_values_f16 = pixel_values.to(dtype=torch.float16)
            
            with torch.no_grad():
                try:
                    vision_features = vision_model(pixel_values_f16, grid_thw=image_grid_thw)
                    has_nan = torch.isnan(vision_features).any().item()
                    print(f"  Output: shape={vision_features.shape}, has_nan={has_nan}")
                    assert not has_nan, f"NaN for shape {num_patches}x{channels}!"
                except Exception as e:
                    print(f"  ERROR: {e}")
                    pytest.fail(f"Failed for shape {num_patches}x{channels}: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))