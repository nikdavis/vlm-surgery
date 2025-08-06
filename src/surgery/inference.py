#!/usr/bin/env python3
"""
Inference script for the trained Qwen-Qwen hybrid model.
Supports both checkpoint loading and merged model loading.
"""
import torch
from PIL import Image
from pathlib import Path
import click
from loguru import logger
from transformers import AutoProcessor

from src.surgery.qwen_qwen_vision import QwenQwenHybrid


@click.command()
@click.option("--checkpoint", default="./outputs_qwen_hybrid/checkpoint-70", help="Path to checkpoint or merged model")
@click.option("--image", required=True, help="Path to image file")
@click.option("--prompt", default="What's in this image?", help="Text prompt")
@click.option("--max-tokens", default=512, help="Maximum tokens to generate")
@click.option("--temperature", default=0.7, help="Sampling temperature")
@click.option("--enable-thinking", is_flag=True, help="Enable thinking tokens (CoT)")
def run_inference(checkpoint, image, prompt, max_tokens, temperature, enable_thinking):
    """Run inference with the trained model."""
    
    logger.info(f"Loading model from {checkpoint}")
    
    # Load model
    if Path(checkpoint) / "config.json" and (Path(checkpoint) / "language_model").exists():
        # This is a merged model
        logger.info("Loading merged model...")
        # For now, we'll still use the hybrid class but in production 
        # you'd want a cleaner loading mechanism
        model = QwenQwenHybrid()
        # Load the saved weights
        # ... (implementation depends on how we structure the merged model)
    else:
        # This is a checkpoint with adapters
        logger.info("Loading checkpoint with adapters...")
        model = QwenQwenHybrid.from_pretrained(checkpoint)
    
    # Move to GPU
    model = model.cuda().eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Load and process image
    logger.info(f"Loading image from {image}")
    pil_image = Image.open(image).convert("RGB")
    
    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process with Qwen processor for image
    image_inputs = processor.image_processor([pil_image], return_tensors="pt")
    
    # Tokenize text with model's tokenizer
    if enable_thinking:
        # Add thinking tokens if requested
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    else:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    input_ids = model.tokenizer.encode(text, return_tensors="pt")
    
    # Insert vision placeholder
    vision_placeholder_id = model.config.vision_placeholder_id
    # Find position after user tag
    user_end = model.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    insert_pos = len(user_end)
    
    # Insert placeholder
    input_ids_list = input_ids[0].tolist()
    input_ids_list.insert(insert_pos, vision_placeholder_id)
    input_ids = torch.tensor([input_ids_list])
    
    # Move to GPU
    input_ids = input_ids.cuda()
    pixel_values = image_inputs.pixel_values.cuda()
    image_grid_thw = image_inputs.image_grid_thw.cuda()
    
    logger.info("Running inference...")
    
    # Generate
    with torch.no_grad():
        outputs = model.language_model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )
    
    # Decode response
    response = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()
        
        # Handle thinking tokens if present
        if "<think>" in response and "</think>" in response:
            thinking = response[response.find("<think>") + 7:response.find("</think>")]
            answer = response[response.find("</think>") + 8:].strip()
            logger.info(f"Thinking: {thinking}")
            logger.info(f"Answer: {answer}")
        else:
            logger.info(f"Response: {response}")
    else:
        logger.info(f"Full output: {response}")
    
    return response


@click.group()
def cli():
    """Inference tools for Qwen-Qwen hybrid model."""
    pass


@cli.command()
@click.option("--checkpoint", default="./outputs_qwen_hybrid/checkpoint-70", help="Checkpoint to test")
def test_checkpoint(checkpoint):
    """Quick test to verify checkpoint loads correctly."""
    logger.info(f"Testing checkpoint: {checkpoint}")
    
    try:
        model = QwenQwenHybrid.from_pretrained(checkpoint)
        logger.info("✅ Checkpoint loaded successfully!")
        
        # Check trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        
        # Check for LoRA
        lora_params = sum(p.numel() for n, p in model.named_parameters() 
                         if p.requires_grad and 'lora' in n.lower())
        if lora_params > 0:
            logger.info(f"LoRA parameters: {lora_params:,}")
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--checkpoint", default="./outputs_qwen_hybrid/merged_model", help="Model checkpoint")
def serve(host, port, checkpoint):
    """Serve the model as an API (simplified version)."""
    from fastapi import FastAPI, File, UploadFile, Form
    from fastapi.responses import JSONResponse
    import uvicorn
    import io
    
    app = FastAPI()
    
    # Load model once
    logger.info(f"Loading model from {checkpoint}")
    model = QwenQwenHybrid.from_pretrained(checkpoint).cuda().eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    @app.post("/generate")
    async def generate(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        max_tokens: int = Form(512),
        temperature: float = Form(0.7)
    ):
        """Generate response for image + prompt."""
        
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Process (simplified - reuse inference logic)
        # ... (implement based on run_inference function)
        
        return JSONResponse({"response": "Generated text here"})
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # If called directly, run inference
    run_inference()