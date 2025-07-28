#!/usr/bin/env python3
"""
FastAPI server for Pixtral OCR inference.
"""

import os
from typing import Optional, List, Dict
from pathlib import Path
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch

# Disable Triton to avoid compilation errors
os.environ["TRITON_PTXAS_PATH"] = ""
os.environ["TORCH_COMPILE_BACKEND"] = "eager"

from unsloth import FastVisionModel


# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")

app = FastAPI(title="Pixtral OCR Inference Server")

# Global model storage
model = None
tokenizer = None


class InferenceRequest(BaseModel):
    """Request model for text + base64 image."""
    image_base64: str
    ocr_text: str
    prompt: Optional[str] = None
    output_format: Optional[str] = "xml"  # xml or json
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = TEMPERATURE


class InferenceResponse(BaseModel):
    """Response model."""
    output: str
    format: str
    prompt_used: str


def load_model():
    """Load the fine-tuned Pixtral model."""
    global model, tokenizer
    
    print(f"Loading model from: {MODEL_PATH}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_PATH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastVisionModel.for_inference(model)
    print("Model loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


def get_prompt(output_format: str, prompt: Optional[str] = None) -> str:
    """Get appropriate prompt based on output format."""
    if prompt:
        return prompt
    
    if output_format == "json":
        return "Convert to JSON: {header: \"...\", articles: [{title?: \"...\", body: \"text with [N] for footnotes\", footnotes?: [{id: N, text: \"...\"}], bibliography?: \"...\", continuation?: true}]}"
    else:  # xml
        return "Extract as XML: <header>, <article continuation=\"true\"?> containing <title>?, <body> with <ref id=\"N\"/> marking footnote references, <footnotes><footnote id=\"N\"> for corresponding notes, <bibliography>?"


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Run inference on image + OCR text."""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Get prompt
        prompt = get_prompt(request.output_format, request.prompt)
        
        # Format messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{prompt}\n\nOCR Text:\n{request.ocr_text}"}
                ]
            }
        ]
        
        # Tokenize
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            images=[image],
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                use_cache=True,
                min_p=0.1,
            )
        
        # Decode output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = output_text.split(input_text)[-1].strip()
        
        return InferenceResponse(
            output=generated,
            format=request.output_format,
            prompt_used=prompt
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/file", response_model=InferenceResponse)
async def inference_file(
    image: UploadFile = File(...),
    ocr_text: str = Form(...),
    prompt: Optional[str] = Form(None),
    output_format: str = Form("xml"),
    max_tokens: int = Form(MAX_NEW_TOKENS),
    temperature: float = Form(TEMPERATURE)
):
    """Run inference with file upload."""
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Convert to base64 for consistency
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Create request and process
        request = InferenceRequest(
            image_base64=image_base64,
            ocr_text=ocr_text,
            prompt=prompt,
            output_format=output_format,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return await inference(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device": DEVICE
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Pixtral OCR Inference Server",
        "endpoints": {
            "/inference": "POST - Run inference with base64 image",
            "/inference/file": "POST - Run inference with file upload",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)