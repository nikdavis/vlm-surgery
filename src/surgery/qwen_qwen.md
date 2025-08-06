# Qwen3-4B + Qwen2.5-VL Vision Surgery Plan

## Overview
Combine Qwen2.5-VL's vision encoder with Qwen3-4B language model for a more stable training setup.

## Key Advantages Over R1 Approach
1. **Same model family**: Both are Qwen models, likely similar internal representations
2. **Native thinking support**: Qwen3 has built-in thinking mode via `enable_thinking` parameter
3. **No cross-family token issues**: Same tokenizer family, same special tokens
4. **Simpler training**: Can train without thinking first, then test thinking mode

## Architecture Plan

### Models
- **Vision**: `Qwen/Qwen2.5-VL-7B-Instruct` (vision encoder only)
- **Language**: `Qwen/Qwen3-4B` or `Qwen/Qwen3-4B-Instruct`
- **Adapter**: Simple projection from vision (3584) to Qwen3 dims (2560 for 4B model)

### Training Strategy
1. **Phase 1**: Train WITHOUT thinking tokens
   - Remove `<think>` sections from training data
   - Train only the vision→language adapter
   - Simpler loss masking (just mask user prompt)

2. **Phase 2**: Test with thinking enabled
   - Use `enable_thinking=True` in tokenizer
   - Model should naturally use its thinking capability

## Implementation Steps

### 1. Model Loading
```python
# Load Qwen3-4B
from transformers import AutoModelForCausalLM, AutoTokenizer
language_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct")

# Load Qwen2.5-VL vision encoder
from transformers import Qwen2_5_VLForConditionalGeneration
vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```

### 2. Dimension Matching
- Qwen2.5-VL vision output: 3584
- Qwen3-4B hidden size: 2560
- Adapter: Linear(3584 → 2560)

### 3. Key Simplifications
- **No thinking token masking**: Train on clean input→output pairs
- **Same tokenizer family**: Vision tokens should be compatible
- **Smaller adapter**: 3584→2560 instead of 3584→4096

### 4. Dataset Modifications
- Strip `<think>...</think>` sections from responses during training
- Keep clean input→output pairs
- Can still use CoT dataset, just extract final answers

### 5. Training Configuration
- Lower learning rate: 1e-5 or 5e-6
- No warmup (simpler)
- Gradient clipping: 1.0
- Batch size: 1 (due to vision model size)

## Expected Improvements
1. **More stable training**: Same model family = compatible representations
2. **No NaN issues**: Simpler architecture, no cross-family bridging
3. **Natural thinking**: Can enable thinking mode after training
4. **Smaller memory footprint**: 4B vs 8B language model

## Testing Plan
1. Train adapter without thinking tokens
2. Test generation quality on visual tasks
3. Enable thinking mode: `enable_thinking=True`
4. Test if model naturally uses thinking for complex tasks

## Files to Create/Modify
- `qwen_qwen_vision.py`: New hybrid model class
- `train_qwen_qwen.py`: Modified training script
- Dataset loader: Option to strip thinking tokens

## Potential Issues
- Qwen3-4B might have different special tokens than expected
- Vision token compatibility needs verification
- May need to check if Qwen3-4B has vision placeholders

## Next Steps
1. Verify Qwen3-4B model structure and hidden dimensions
2. Check tokenizer compatibility between models
3. Implement simplified adapter without complex stabilization
4. Test with small subset first