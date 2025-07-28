#!/usr/bin/env python3
"""
Inference script for fine-tuned Pixtral OCR model.
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
from unsloth import FastVisionModel
from transformers import TextStreamer


def load_model(model_path: str, load_in_4bit: bool = True):
    """Load fine-tuned model with LoRA adapter."""
    print(f"Loading LoRA model from: {model_path}")
    
    # Load model and tokenizer with LoRA weights
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path,  # Path to LoRA adapter
        load_in_4bit=load_in_4bit,
    )
    
    # Enable inference mode
    FastVisionModel.for_inference(model)
    
    return model, tokenizer


def run_inference(model, tokenizer, image_path: Path, ocr_text: str, prompt: str = None):
    """Run inference on a single example."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Default prompt if not provided
    if prompt is None:
        prompt = "Extract this encyclopedia page as structured XML. Schema: header (page info), article (contains: title, body, bibliography), footer. Articles may have continuation=\"true\" attribute."
    
    # Format input - image must come FIRST (matching training format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{prompt}\n\nOCR Text:\n{ocr_text}"}
            ]
        }
    ]
    
    # Tokenize with images
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        images=[image],  # Pass as list
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate
    print("\nGenerating structured output...\n")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=1024,
        use_cache=True,
        temperature=0.1,  # Low temp for consistent structure
        min_p=0.1,
    )
    
    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part
    generated = output_text.split(input_text)[-1].strip()
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Pixtral")
    parser.add_argument("--model", default="outputs/final_model", help="Path to fine-tuned model")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--ocr", type=Path, help="Path to OCR text file")
    parser.add_argument("--ocr-text", type=str, help="OCR text directly")
    parser.add_argument("--prompt", type=str, help="Custom prompt")
    parser.add_argument("--output", type=Path, help="Save output to file")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4bit quantization")
    
    args = parser.parse_args()
    
    # Get OCR text
    if args.ocr:
        with open(args.ocr, 'r', encoding='utf-8') as f:
            ocr_text = f.read()
    elif args.ocr_text:
        ocr_text = args.ocr_text
    else:
        print("Error: Must provide either --ocr or --ocr-text")
        return
    
    # Load model
    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)
    
    # Run inference
    output = run_inference(model, tokenizer, args.image, ocr_text, args.prompt)
    
    # Save if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()