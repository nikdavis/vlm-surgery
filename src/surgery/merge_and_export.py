#!/usr/bin/env python3
"""
Merge the trained adapters into a full model for inference.
This creates a standalone model that can be loaded normally.
"""
import torch
from pathlib import Path
import click
from loguru import logger
import json
import shutil
from transformers import AutoProcessor

# Import our hybrid model
from src.surgery.qwen_qwen_vision import QwenQwenHybrid


@click.command()
@click.option("--checkpoint-dir", default="./outputs_qwen_hybrid/checkpoint-70", help="Path to checkpoint with adapters")
@click.option("--output-dir", default="./outputs_qwen_hybrid/merged_model", help="Output directory for merged model")
@click.option("--merge-lora", is_flag=True, default=True, help="Merge LoRA weights into base model")
def merge_model(checkpoint_dir, output_dir, merge_lora):
    """Merge trained adapters into a full model ready for serving."""
    
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint directory {checkpoint_path} does not exist!")
        return
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the model with adapters
    model = QwenQwenHybrid.from_pretrained(str(checkpoint_path))
    
    # Move to GPU for merging
    model = model.cuda()
    
    if merge_lora:
        logger.info("Merging LoRA weights into base model...")
        # Merge LoRA weights into the base model
        if hasattr(model.language_model, 'merge_and_unload'):
            model.language_model = model.language_model.merge_and_unload()
            logger.info("LoRA weights merged successfully")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the full model (not just adapters)
    logger.info(f"Saving merged model to {output_path}")
    
    # Save vision model
    logger.info("Saving vision model components...")
    vision_path = output_path / "vision_model"
    vision_path.mkdir(exist_ok=True)
    
    # Save the entire vision model with modified merger
    torch.save(model.vision_model.state_dict(), vision_path / "vision_model.pt")
    
    # Save vision config
    vision_config = {
        "spatial_merge_size": model.spatial_merge_size,
        "vision_placeholder_id": model.config.vision_placeholder_id,
        "vision_start_id": model.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "vision_end_id": model.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }
    with open(vision_path / "vision_config.json", "w") as f:
        json.dump(vision_config, f, indent=2)
    
    # Save language model (with merged LoRA if applicable)
    logger.info("Saving language model...")
    model.language_model.save_pretrained(output_path / "language_model")
    
    # Save tokenizer with special tokens
    logger.info("Saving tokenizer...")
    model.tokenizer.save_pretrained(output_path / "tokenizer")
    
    # Save vision embeddings
    embeddings = {
        "vision_start_embedding": model.vision_start_embedding.cpu(),
        "vision_end_embedding": model.vision_end_embedding.cpu(),
    }
    torch.save(embeddings, output_path / "vision_embeddings.pt")
    
    # Create a config for loading
    config = {
        "model_type": "qwen_qwen_hybrid",
        "vision_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "language_model": "Qwen/Qwen3-4B",
        "vision_placeholder_id": model.config.vision_placeholder_id,
        "merged_checkpoint": str(checkpoint_path),
        "lora_merged": merge_lora,
    }
    
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Also save the processor for convenience
    logger.info("Saving processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    processor.save_pretrained(output_path / "processor")
    
    # Create a simple loading script
    load_script = '''#!/usr/bin/env python3
"""Simple script to load and test the merged model."""
import torch
from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image

def load_merged_model(model_dir):
    """Load the merged model for inference."""
    model_path = Path(model_dir)
    
    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path / "processor")
    
    # Load language model
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path / "language_model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load vision components
    vision_state = torch.load(model_path / "vision_model" / "vision_model.pt")
    vision_embeddings = torch.load(model_path / "vision_embeddings.pt")
    
    print(f"Model loaded from {model_dir}")
    print(f"LoRA merged: {config.get('lora_merged', False)}")
    
    return {
        "tokenizer": tokenizer,
        "processor": processor,
        "language_model": language_model,
        "vision_state": vision_state,
        "vision_embeddings": vision_embeddings,
        "config": config
    }

if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "./outputs_qwen_hybrid/merged_model"
    components = load_merged_model(model_dir)
    print("Model components loaded successfully!")
'''
    
    with open(output_path / "load_model.py", "w") as f:
        f.write(load_script)
    
    # Make it executable
    (output_path / "load_model.py").chmod(0o755)
    
    logger.info(f"""
    ✅ Model successfully merged and saved to {output_path}
    
    The merged model contains:
    - Vision model with trained merger projection
    - Language model with {"merged" if merge_lora else "separate"} LoRA weights  
    - Tokenizer with vision tokens
    - Processor for image handling
    
    To load the model:
    python {output_path}/load_model.py
    
    To serve with vLLM:
    - Update docker-compose.yml to point to {output_path}
    - The model uses Qwen3 chat template with vision support
    """)

if __name__ == "__main__":
    merge_model()