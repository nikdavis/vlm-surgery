# Gradient Flow Issue Analysis

## Problem
The Qwen2.5-VL vision model produces NaN outputs after the first training batch, even though:
1. The vision model parameters are frozen (requires_grad=False)
2. Only the replaced merger projection layer is trainable
3. First forward pass works correctly (produces valid loss)

## Pattern
- Sample 0: Works, produces loss ~1.3-1.6
- Sample 1+: Vision model outputs NaN
- grad_norm is NaN from the first backward pass

## Root Cause Hypothesis
When we replace `vision_model.merger.mlp[2]` with a new Linear layer and allow gradients to flow through the frozen vision model (necessary for the gradients to reach our trainable layer), the backward pass seems to corrupt the vision model's internal state.

This could be due to:
1. Gradient checkpointing interacting poorly with partially frozen models
2. The replaced layer having incompatible gradient magnitudes
3. BatchNorm or other stateful layers in the vision model getting updated despite being frozen

## Things We've Tried
1. ✅ Removed all torch.no_grad() blocks (except initialization)
2. ✅ Put vision model in eval mode
3. ✅ Override train() to keep vision model in eval
4. ✅ Very small initialization (std=0.0001)
5. ✅ Lower learning rates (down to 1e-7)
6. ✅ Disabled gradient checkpointing for vision model
7. ❌ All still produce NaN after first batch

## Potential Solutions
1. Use a separate adapter module after the vision model instead of replacing internal layers
2. Use gradient accumulation with detach() between batches
3. Clone the vision features before passing through adapter
4. Use a different surgery approach (e.g., LoRA on the merger)