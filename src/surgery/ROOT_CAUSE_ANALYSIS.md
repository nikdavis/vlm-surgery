# Root Cause Analysis: Vision Model NaN Issue

## Summary
The Qwen2.5-VL vision model works perfectly in isolation but produces NaN when integrated with our surgery approach and training loop.

## Test Results

### ✅ What Works
1. Vision model forward pass with proper inputs
2. Multiple forward passes (no degradation)
3. Forward pass with gradient flow
4. Backward pass through vision model
5. Vision model remains stable after backward

### ❌ What Fails
In our training setup, the vision model produces NaN after the first training batch.

## Discovered Issues

### 1. Grid Dimension Format (CRITICAL)
The `image_grid_thw` parameter has format **[t, h, w]** where:
- t = temporal dimension (1 for images)
- h = height in patches
- w = width in patches

Example: For a 336x336 image → `[1, 24, 24]` NOT `[24, 24, 1]`

### 2. Spatial Merge Size Constraint
- The vision model has `spatial_merge_size = 2`
- Grid dimensions (h, w) must be divisible by 2
- Invalid grid dimensions cause reshape errors

### 3. Model Surgery Issue
When we replace `vision_model.merger.mlp[2]` with a new Linear layer:
- First forward pass works (produces valid loss ~1.3)
- First backward pass computes but `grad_norm` is NaN
- Second forward pass: vision model outputs NaN

This suggests the backward pass through our replaced layer is corrupting the vision model's internal state, even though the model parameters are frozen.

## Root Cause Hypothesis

The issue is NOT with the vision model itself, but with our model surgery approach:

1. **Gradient Flow Through Frozen Model**: When we replace an internal layer and allow gradients to flow through the frozen vision model (necessary for gradients to reach our trainable layer), the backward pass corrupts internal buffers or running statistics.

2. **BatchNorm/LayerNorm Statistics**: Even though we put the model in eval mode, some internal state is being modified during the backward pass.

3. **Incompatible Gradient Magnitudes**: The replaced layer may produce gradients of vastly different magnitude than the original, causing numerical instability.

## Solutions to Try

### Option 1: Clean Separation (Recommended)
Instead of replacing internal layers, use a separate adapter module:
```python
# Don't replace vision_model.merger.mlp[2]
# Instead, add adapter after the complete vision model
vision_features = vision_model(pixel_values, grid_thw)
vision_features = vision_features.detach()  # Stop gradients here
adapted_features = trainable_adapter(vision_features)
```

### Option 2: Gradient Checkpointing Fix
Ensure gradient checkpointing is properly configured for partially frozen models.

### Option 3: LoRA Instead of Replacement
Use LoRA adapters on the merger layer instead of replacing it entirely.

## Next Steps
1. Implement clean separation approach
2. Test with detached vision features
3. Verify training stability