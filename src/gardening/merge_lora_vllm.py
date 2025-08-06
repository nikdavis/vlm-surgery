#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for vLLM serving.
This version ensures vLLM compatibility by properly handling configs.
"""

import argparse
from pathlib import Path
import torch
import json
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model for vLLM")
    parser.add_argument("--adapter-path", type=Path, default=Path("outputs_qwen/final_model"),
                       help="Path to LoRA adapter")
    parser.add_argument("--output-path", type=Path, default=Path("outputs_qwen/merged_model_vllm"),
                       help="Path to save merged model")
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-VL-7B-Instruct",
                       help="Base model name (non-quantized version)")

    args = parser.parse_args()

    print(f"Loading adapter config from: {args.adapter_path}")
    
    # Read adapter config to get the actual base model used
    adapter_config_path = args.adapter_path / "adapter_config.json"
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    
    # The base model in adapter_config might be the 4bit version
    trained_base_model = adapter_config.get("base_model_name_or_path", args.base_model)
    print(f"Adapter was trained on: {trained_base_model}")
    
    # For merging, we need the non-quantized version
    # Convert 4bit model names to their standard versions
    if "bnb-4bit" in trained_base_model or "unsloth" in trained_base_model.lower():
        # Use the provided base model which should be the non-quantized version
        merge_base_model = args.base_model
        print(f"Using non-quantized base model for merge: {merge_base_model}")
    else:
        merge_base_model = trained_base_model

    print(f"\nLoading base model: {merge_base_model}")
    
    # Load the base model in full precision
    model = AutoModelForVision2Seq.from_pretrained(
        merge_base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(merge_base_model, trust_remote_code=True)
    
    print(f"\nLoading LoRA adapter from: {args.adapter_path}")
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    print("\nMerging LoRA weights into base model...")
    
    # Merge LoRA weights
    model = model.merge_and_unload()
    
    print(f"\nSaving merged model to: {args.output_path}")
    args.output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the merged model
    model.save_pretrained(str(args.output_path))
    tokenizer.save_pretrained(str(args.output_path))
    
    # Also save the processor
    try:
        processor = AutoProcessor.from_pretrained(merge_base_model, trust_remote_code=True)
        processor.save_pretrained(str(args.output_path))
        print("Processor saved successfully")
    except Exception as e:
        print(f"Note: Could not save processor: {e}")
    
    # Clean up the config for vLLM
    config_path = args.output_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Remove quantization config if present
    if 'quantization_config' in config:
        del config['quantization_config']
        print("Removed quantization_config from config.json")
    
    # Remove unsloth-specific fields
    for key in ['unsloth_fixed', 'unsloth_version']:
        if key in config:
            del config[key]
        if 'text_config' in config and key in config['text_config']:
            del config['text_config'][key]
    
    # Save cleaned config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nMerge complete! The merged model is ready for vLLM.")
    print(f"Model saved to: {args.output_path}")
    print("\nTo test with vLLM, update your docker-compose.yml to point to this path.")


if __name__ == "__main__":
    main()