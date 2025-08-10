#!/usr/bin/env python3
"""
Test to understand and fix grid_thw calculation for Qwen2.5-VL
"""

import torch
import pytest
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import numpy as np

def test_grid_thw_requirements():
    """Figure out the correct grid_thw calculation"""
    print("\nLoading model to check spatial_merge_size...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    vision_model = model.visual
    spatial_merge_size = vision_model.spatial_merge_size
    print(f"spatial_merge_size: {spatial_merge_size}")
    
    # Test cases from actual training
    num_patches_list = [11972, 16352, 3996, 256, 576]  # 576 is from successful test
    
    for num_patches in num_patches_list:
        print(f"\nTesting num_patches={num_patches}:")
        
        # The number of patches after merging should be divisible by spatial_merge_size^2
        patches_per_image = num_patches // (spatial_merge_size * spatial_merge_size)
        print(f"  Patches after merge: {patches_per_image}")
        
        # Find valid grid size
        # The grid h,w must be divisible by spatial_merge_size
        grid_size = int(np.sqrt(num_patches))
        print(f"  Initial grid_size guess: {grid_size}")
        
        # Adjust to be divisible by spatial_merge_size
        if grid_size % spatial_merge_size != 0:
            # Round up to nearest multiple
            grid_size = ((grid_size + spatial_merge_size - 1) // spatial_merge_size) * spatial_merge_size
            print(f"  Adjusted grid_size: {grid_size}")
        
        # Check if this works
        if grid_size * grid_size != num_patches:
            print(f"  WARNING: {grid_size}x{grid_size}={grid_size*grid_size} != {num_patches}")
            # Try to find the actual h,w from the processor output
            # The processor likely knows the right dimensions
            
        # Create test tensor
        pixel_values = torch.randn(num_patches, 1176, dtype=torch.float16).to(device)
        image_grid_thw = torch.tensor([[grid_size, grid_size, 1]], dtype=torch.long).to(device)
        
        print(f"  Testing grid_thw: {image_grid_thw.tolist()}")
        
        try:
            with torch.no_grad():
                vision_features = vision_model(pixel_values, grid_thw=image_grid_thw)
                print(f"  ✓ Success! Output shape: {vision_features.shape}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

def test_processor_grid_calculation():
    """Check how the processor calculates grid_thw"""
    from PIL import Image
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Test with various image sizes
    test_sizes = [(336, 336), (512, 512), (1024, 768), (224, 224)]
    
    for w, h in test_sizes:
        img = Image.new('RGB', (w, h), color='blue')
        inputs = processor(
            text="Test",
            images=img,
            return_tensors="pt"
        )
        
        print(f"\nImage size {w}x{h}:")
        print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"  image_grid_thw: {inputs['image_grid_thw'].tolist()}")
        
        # Calculate what we would expect
        num_patches = inputs['pixel_values'].shape[0]
        grid_thw = inputs['image_grid_thw'][0].tolist()
        expected_patches = grid_thw[0] * grid_thw[1] * grid_thw[2]
        print(f"  Grid h={grid_thw[0]}, w={grid_thw[1]}, t={grid_thw[2]}")
        print(f"  Expected patches: {expected_patches}, actual: {num_patches}")

if __name__ == "__main__":
    test_grid_thw_requirements()
    print("\n" + "="*60)
    test_processor_grid_calculation()