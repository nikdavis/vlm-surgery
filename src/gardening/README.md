# Model Gardening

Post-training optimization scripts for model deployment.

## Stage 1: Merge LoRA

**Input**: LoRA adapter + base model  
**Output**: Merged standalone model

```bash
uv run python -m src.gardening.merge_lora \
  --adapter-path outputs/final_model \
  --output-path outputs/merged_model \
  --base-model unsloth/Pixtral-12B-2409
```

**Output format**: HuggingFace model directory with:
- `config.json`
- `model.safetensors` (full weights)
- `tokenizer.json`, `tokenizer_config.json`
- `preprocessor_config.json`

## Stage 2: FP8 Quantization

**Input**: Merged model  
**Output**: FP8 quantized model

```bash
uv run python -m src.gardening.quantize_fp8 \
  --model-path outputs/merged_model \
  --output-path outputs/merged_model_fp8
```

**Output format**: HuggingFace model directory with:
- Same files as merged model but with FP8 weights
- `quantization_config.json` (quantization metadata)