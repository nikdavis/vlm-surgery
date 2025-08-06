#!/usr/bin/env python3
"""Verify Qwen3-4B dimensions and tokenizer compatibility"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading Qwen3-4B to check dimensions...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
print(f"\nTokenizer loaded: {type(tokenizer)}")
print(f"Vocab size: {len(tokenizer)}")

# Check for vision tokens
vision_tokens = [
    "<|vision_start|>",
    "<|vision_end|>", 
    "<|image_pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|>",
    "<think>",
    "</think>"
]

print("\nChecking for special tokens:")
for token in vision_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    exists = token_id != tokenizer.unk_token_id
    print(f"  {token}: {'EXISTS' if exists else 'MISSING'} (ID: {token_id if exists else 'N/A'})")

# Load model config to check dimensions
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen3-4B")
print(f"\nModel config:")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Intermediate size: {config.intermediate_size}")
print(f"  Num layers: {config.num_hidden_layers}")
print(f"  Num attention heads: {config.num_attention_heads}")

# Check if model has vision config
if hasattr(config, 'vision_config'):
    print(f"  Has vision config: Yes")
else:
    print(f"  Has vision config: No (will need to add vision tokens)")

# Compare with Qwen2.5-VL dimensions
print(f"\nFor reference, Qwen2.5-VL vision output: 3584")
print(f"Adapter needed: Linear(3584 -> {config.hidden_size})")
print(f"Parameter count: {3584 * config.hidden_size:,} + {config.hidden_size} bias = {3584 * config.hidden_size + config.hidden_size:,}")