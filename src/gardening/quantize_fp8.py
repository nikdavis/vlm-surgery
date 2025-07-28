#!/usr/bin/env python3
"""
Quantize merged model to FP8 for efficient inference.
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from transformers.utils.quantization_config import QuantizationMethod


def quantize_model_fp8(model_path: Path, output_path: Path):
    """Quantize model to FP8 format."""
    print(f"Loading model from: {model_path}")
    
    # Load the merged model
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Start with FP16
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("Quantizing model to FP8...")
    
    # For FP8 quantization, we'll use the model's built-in quantization if available
    # Otherwise, we'll need to use a quantization library like quanto or awq
    try:
        # Try using the quanto library for FP8 quantization
        from optimum.quanto import quantize, freeze, qfloat8
        
        # Quantize the model to FP8
        quantize(model, weights=qfloat8)
        freeze(model)
        
        print(f"Saving quantized model to: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save quantized model
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        processor.save_pretrained(output_path)
        
        # Save quantization config
        config = {
            "quantization_method": "fp8",
            "quantization_library": "quanto",
        }
        import json
        with open(output_path / "quantization_config.json", "w") as f:
            json.dump(config, f, indent=2)
            
    except ImportError:
        print("\nWarning: quanto library not found. Trying alternative FP8 quantization...")
        
        # Alternative: Use torch.float8 if available (PyTorch 2.1+)
        if hasattr(torch, 'float8_e4m3fn'):
            print("Using PyTorch native FP8 quantization...")
            
            # Convert model weights to FP8
            for name, param in model.named_parameters():
                if param.dtype in [torch.float16, torch.float32]:
                    # Convert to FP8 E4M3
                    param.data = param.data.to(torch.float8_e4m3fn)
            
            print(f"Saving FP8 quantized model to: {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            # Save quantization config
            config = {
                "quantization_method": "fp8_e4m3fn",
                "quantization_library": "pytorch_native",
            }
            import json
            with open(output_path / "quantization_config.json", "w") as f:
                json.dump(config, f, indent=2)
        else:
            print("\nError: FP8 quantization not available. Please install quanto:")
            print("  pip install optimum-quanto")
            print("Or upgrade PyTorch to 2.1+ for native FP8 support")
            return
    
    print("\nQuantization complete!")
    print(f"Quantized model saved to: {output_path}")
    
    # Calculate size reduction
    import os
    def get_dir_size(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total += os.path.getsize(filepath)
        return total
    
    if model_path.exists():
        original_size = get_dir_size(model_path)
        quantized_size = get_dir_size(output_path)
        reduction = (1 - quantized_size / original_size) * 100
        print(f"\nSize reduction: {reduction:.1f}%")
        print(f"Original: {original_size / 1e9:.1f} GB")
        print(f"Quantized: {quantized_size / 1e9:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Quantize model to FP8")
    parser.add_argument("--model-path", type=Path, default=Path("outputs/merged_model"),
                       help="Path to merged model")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/merged_model_fp8"),
                       help="Path to save FP8 quantized model")
    
    args = parser.parse_args()
    
    quantize_model_fp8(args.model_path, args.output_path)


if __name__ == "__main__":
    main()