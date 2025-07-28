#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for vLLM serving.
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoProcessor
from unsloth import FastVisionModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter-path", type=Path, default=Path("outputs/final_model"),
                       help="Path to LoRA adapter")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/merged_model"),
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

    print("\nMerge complete! The merged model can now be loaded directly with vLLM.")
    print(f"Model saved to: {args.output_path}")


if __name__ == "__main__":
    main()
