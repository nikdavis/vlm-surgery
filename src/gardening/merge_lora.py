#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for vLLM serving.
"""

import argparse
from pathlib import Path
import torch
import json
from transformers import AutoProcessor
from unsloth import FastVisionModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter-path", type=Path, default=Path("outputs/final_model"),
                       help="Path to LoRA adapter")
    parser.add_argument("--output-path", type=Path, default=Path("outputs_qwen/merged_model"),
                       help="Path to save merged model")
    parser.add_argument("--base-model", default="unsloth/Pixtral-12B-2409",
                       help="Base model name")

    args = parser.parse_args()

    print(f"Loading model and adapter from: {args.adapter_path}")

    # Load model with adapter using unsloth (since that's how we trained)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(args.adapter_path),  # Convert Path to string
        load_in_4bit=False,  # Load in full precision for merging
    )

    print("Saving merged model in 16-bit format for vLLM...")
    args.output_path.mkdir(parents=True, exist_ok=True)
    
    # Use Unsloth's save_pretrained_merged for vLLM compatibility
    model.save_pretrained_merged(
        str(args.output_path),
        tokenizer,
        save_method="merged_16bit",
    )
    
    # Also save the processor if available
    try:
        processor = AutoProcessor.from_pretrained(args.base_model)
        processor.save_pretrained(str(args.output_path))
        print("Processor saved successfully")
    except Exception as e:
        print(f"Note: Could not save processor: {e}")

    # Clean up config for vLLM compatibility
    print("\nCleaning up config for vLLM...")
    config_path = args.output_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Remove quantization config if present
        if 'quantization_config' in config:
            del config['quantization_config']
            print("Removed quantization_config")
        
        # Remove unsloth-specific fields
        for key in ['unsloth_fixed', 'unsloth_version']:
            if key in config:
                del config[key]
            if 'text_config' in config and key in config['text_config']:
                del config['text_config'][key]
        
        # For Qwen2.5-VL, remove text_config if present (should use top-level params)
        if 'text_config' in config and config.get('model_type') == 'qwen2_5_vl':
            del config['text_config']
            print("Removed text_config for vLLM compatibility")
        
        # Fix rope_scaling format for Qwen2.5-VL
        if config.get('model_type') == 'qwen2_5_vl' and 'rope_scaling' in config:
            if 'rope_type' in config['rope_scaling']:
                del config['rope_scaling']['rope_type']
            config['rope_scaling']['type'] = 'mrope'
            print("Fixed rope_scaling format")
        
        # Add bos_token_id if missing (required by some tokenizers)
        if config.get('model_type') == 'qwen2_5_vl' and 'bos_token_id' not in config:
            config['bos_token_id'] = 151643
            print("Added bos_token_id")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("Config cleaned for vLLM compatibility")
    
    print("\nMerge complete! The merged model can now be loaded directly with vLLM.")
    print(f"Model saved to: {args.output_path}")


if __name__ == "__main__":
    main()
