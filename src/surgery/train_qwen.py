#!/usr/bin/env python3
"""
Train the Qwen2.5-VL + DeepSeekR1 hybrid model on a single GPU.
This script uses gradient checkpointing to save memory.
"""
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoProcessor
from typing import List, Dict, Any
import click
from datetime import datetime
import mlflow

# Import the new Qwen-based hybrid model and the existing dataset loader
from src.surgery.qwen_vision import FinishedQwenR1Hybrid
from src.unified_dataset_loader import UnifiedOCRDataset

class CoTDataCollator:
    """
    Handles batching and creates masked labels for CoT training.
    Manual handling for cross-model compatibility (Qwen2.5-VL vision + Qwen3 R1 language).
    """
    def __init__(self, tokenizer: AutoTokenizer, processor: AutoProcessor, placeholder_id: int):
        self.tokenizer = tokenizer  # Qwen3 R1 tokenizer
        self.image_processor = processor.image_processor  # Only use image processor from Qwen2.5-VL
        self.placeholder_id = placeholder_id  # The unused token we're repurposing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-fetch token IDs for special tokens used in CoT
        # R1 uses <think> and </think> tokens
        self.think_start_id = tokenizer.convert_tokens_to_ids("<think>")
        self.think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        print(f"Think tokens: start={self.think_start_id}, end={self.think_end_id}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Now that we know the tokens are compatible, we can use the processor!
        images = []
        messages_for_processor = []
        
        for feature in features:
            # Get images directly from the feature
            feature_images = feature["images"]
            images.extend(feature_images)
            
            # Extract prompt and response from messages
            user_content = feature["messages"][0]["content"]
            prompt = None
            for item in user_content:
                if item["type"] == "text":
                    prompt = item["text"]
                    break
            
            # Get response text
            response = feature["messages"][1]["content"][0]["text"]
            
            # Build text-only messages for chat template
            messages = [
                {
                    "role": "user",
                    "content": prompt  # Just the text
                },
                {"role": "assistant", "content": response}
            ]
            
            # Apply chat template with text only
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Now we need to insert vision tokens manually
            # Replace the start of user content with vision tokens
            # The format should be: <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>prompt...
            
            # For now, store the text and we'll insert vision tokens after tokenization
            messages_for_processor.append(text)
        
        # Tokenize text first
        inputs = self.tokenizer(
            messages_for_processor,
            padding=False,  # We'll pad after inserting vision tokens
            truncation=True,
            max_length=1024,  # Reduced from 2048 to help with OOM
            return_tensors=None  # Get list format first
        )
        
        # Now insert vision tokens after "<|im_start|>user\n"
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
        modified_input_ids = []
        image_idx = 0
        
        for i, (input_ids, feature) in enumerate(zip(inputs.input_ids, features)):
            # Find position after user tag to insert vision tokens
            # Qwen3 R1 uses <｜User｜> format
            # Try multiple patterns
            insert_pos = None
            
            # Pattern 1: Look for <｜User｜> token (151669)
            user_token_id = 151669  # <｜User｜>
            for j in range(len(input_ids) - 1):
                if input_ids[j] == user_token_id:
                    # Insert right after the user token
                    insert_pos = j + 1
                    break
            
            # Pattern 2: Try the old format as fallback
            if insert_pos is None:
                user_start_tokens = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
                for j in range(len(input_ids) - len(user_start_tokens) + 1):
                    if all(input_ids[j + k] == user_start_tokens[k] for k in range(len(user_start_tokens))):
                        insert_pos = j + len(user_start_tokens)
                        break
            
            if insert_pos is None:
                # Fallback: just use as-is
                print(f"WARNING: Could not find user prompt position in sample {i}")
                modified_input_ids.append(torch.tensor(input_ids))
                continue
            
            # Insert ONLY image_pad tokens - the model will add vision_start/end
            num_images = len(feature["images"])
            image_pad_tokens = [self.placeholder_id] * num_images
            
            # Debug: Show where we're inserting (only for first batch)
            if i < 2 and not hasattr(self, '_insertion_logged'):  # First 2 samples, first batch only
                tokens_before = self.tokenizer.decode(input_ids[:insert_pos])
                tokens_after = self.tokenizer.decode(input_ids[insert_pos:insert_pos+50])  # First 50 chars after
                print(f"\nDEBUG Sample {i} insertion:")
                print(f"  Inserting {num_images} image_pad tokens at position {insert_pos}")
                print(f"  Text before: ...{tokens_before[-50:]}")
                print(f"  Text after: {tokens_after}...")
            
            # Insert only the image_pad tokens
            new_ids = input_ids[:insert_pos] + image_pad_tokens + input_ids[insert_pos:]
            modified_input_ids.append(torch.tensor(new_ids))
            
            image_idx += num_images
        
        # Now pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(modified_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(ids) for ids in modified_input_ids],
            batch_first=True,
            padding_value=0
        )
        
        # Debug: Check if vision tokens were inserted
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"\nDEBUG Collator Summary:")
            print(f"  Placeholder id (image_pad): {self.placeholder_id}")
            print(f"  Vision start id: {vision_start_id}")
            print(f"  Vision end id: {vision_end_id}")
            print(f"  Total images processed: {image_idx}")
            for i in range(min(2, input_ids.shape[0])):  # Check first 2 samples
                num_placeholders = (input_ids[i] == self.placeholder_id).sum().item()
                num_vision_start = (input_ids[i] == vision_start_id).sum().item()
                num_vision_end = (input_ids[i] == vision_end_id).sum().item()
                print(f"  Sample {i}: {num_placeholders} image_pad, {num_vision_start} vision_start, {num_vision_end} vision_end")
                # Show actual token sequence around placeholders
                if num_placeholders > 0:
                    placeholder_pos = (input_ids[i] == self.placeholder_id).nonzero(as_tuple=True)[0][0].item()
                    start = max(0, placeholder_pos - 5)
                    end = min(len(input_ids[i]), placeholder_pos + 5)
                    token_window = input_ids[i][start:end].tolist()
                    print(f"    Token window around placeholder: {token_window}")
            self._insertion_logged = True  # Mark that we've logged insertion
        
        # Process images
        image_outputs = {}
        if images:
            image_inputs = self.image_processor(images, return_tensors="pt")
            image_outputs = image_inputs
        
        labels = input_ids.clone()

        # Create the masked labels for CoT
        for i in range(len(features)):
            # R1 uses <｜Assistant｜> token
            assistant_token_id = 151670  # <｜Assistant｜>
            input_id_list = input_ids[i].tolist()

            try:
                # Find where assistant response starts
                assistant_pos = None
                for j, token_id in enumerate(input_id_list):
                    if token_id == assistant_token_id:
                        assistant_pos = j + 1  # Position after assistant token
                        break
                
                if assistant_pos is not None:
                    # Mask everything before assistant response (including the assistant token)
                    labels[i, :assistant_pos] = -100
                    if i < 2 and not hasattr(self, '_mask_debug'):
                        print(f"    Masked up to position {assistant_pos} (after <｜Assistant｜>)")
                else:
                    print(f"Warning: Could not find assistant token in sample {i}")
            except Exception as e:
                print(f"Error masking labels for sample {i}: {e}")
                continue

            # Mask CoT thinking if present
            if self.think_start_id and self.think_end_id:
                think_starts = (labels[i] == self.think_start_id).nonzero(as_tuple=True)[0]
                think_ends = (labels[i] == self.think_end_id).nonzero(as_tuple=True)[0]
                if len(think_starts) > 0 and len(think_ends) > 0:
                    labels[i, think_starts[0] : think_ends[0] + 1] = -100

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Debug: Check label masking
        if not hasattr(self, '_label_debug_printed'):
            self._label_debug_printed = True
            print(f"\nDEBUG Label masking:")
            for i in range(min(2, labels.shape[0])):
                non_masked = (labels[i] != -100).sum().item()
                total = labels[i].shape[0]
                print(f"  Sample {i}: {non_masked}/{total} non-masked tokens ({non_masked/total*100:.1f}%)")
                # Show first few non-masked tokens
                non_masked_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(non_masked_indices) > 0:
                    first_tokens = labels[i][non_masked_indices[:10]].tolist()
                    decoded = [self.tokenizer.decode([t]) for t in first_tokens]
                    print(f"    First non-masked tokens: {decoded}")
                    # Also show what comes after thinking
                    think_end_pos = (input_ids[i] == self.think_end_id).nonzero(as_tuple=True)[0]
                    if len(think_end_pos) > 0:
                        after_think = input_ids[i][think_end_pos[0]+1:think_end_pos[0]+11].tolist()
                        decoded_after = self.tokenizer.decode(after_think)
                        print(f"    After </think>: '{decoded_after}...'")

        # Combine everything
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        # Add image outputs if we have them
        if image_outputs:
            result.update(image_outputs)
            
        return result

@click.command()
@click.option("--output-dir", default="./outputs_qwen_hybrid", help="Output directory for checkpoints.")
@click.option("--data-path", default="./datasetv2/combined_dataset.json", help="Path to the unified dataset JSON file.")
@click.option("--batch-size", type=int, default=1, help="Training batch size per device.")
@click.option("--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps.")
@click.option("--learning-rate", type=float, default=5e-5, help="Learning rate for the adapter.")
@click.option("--num-epochs", type=int, default=1, help="Number of training epochs.")
def main(output_dir, data_path, batch_size, gradient_accumulation, learning_rate, num_epochs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("qwen-r1-surgery-20250805")

    # 1. Instantiate the new Qwen-based hybrid model
    model = FinishedQwenR1Hybrid()
    tokenizer = model.tokenizer

    # Load Qwen processor for image processing
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # R1 already has <think> and </think> tokens, no need to add them
    # Just verify they exist
    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    print(f"Using R1's native thinking tokens: <think> ({think_start_id}), </think> ({think_end_id})")
    
    # No need to resize embeddings since tokens already exist

    # 2. Load the dataset
    full_dataset = UnifiedOCRDataset(data_path, enable_cot=True)
    train_dataset, val_dataset = full_dataset.get_train_val_split(val_ratio=0.02)

    # Create run name
    run_name = f"Qwen-R1-Surgery-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 3. Define TrainingArguments (simple, single-GPU version)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=True, # Use bfloat16 for memory efficiency
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        do_train=True,
        do_eval=True,
        remove_unused_columns=False,
        gradient_checkpointing=True, # ESSENTIAL: Creates gradient path through frozen model
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="mlflow",  # Enable MLflow logging
        run_name=run_name,
        max_grad_norm=1.0,  # Tighter gradient clipping for stability
        warmup_steps=10,  # Fixed small warmup - just 10 steps instead of ratio
    )

    # The Trainer will handle gradient checkpointing based on training_args
    # Disable cache when using gradient checkpointing
    model.language_model.config.use_cache = False

    # 4. Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CoTDataCollator(tokenizer, processor, placeholder_id=model.config.vision_placeholder_id),
    )

    # 5. Start training with MLflow tracking
    print("\n--- Starting Training ---")
    with mlflow.start_run(run_name=run_name) as run:
        # Log initial parameters
        mlflow.log_params({
            "vision_model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "language_model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
            "vision_placeholder_id": model.config.vision_placeholder_id,
            "dataset_size": len(full_dataset),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "enable_cot": True,
        })
        
        trainer_stats = trainer.train()
        
        # Log final training stats
        mlflow.log_metrics({
            "final_train_loss": trainer_stats.metrics.get('train_loss', 0),
            "total_train_time": trainer_stats.metrics.get('train_runtime', 0),
            "train_samples_per_second": trainer_stats.metrics.get('train_samples_per_second', 0),
        })
        
        print("\n--- Training Complete ---")
    
        # 6. Save the final trained adapter
        model.save_pretrained(str(output_dir / "final_model"))
        print(f"Final model adapter saved to {output_dir / 'final_model'}")
        
        # Log the final model artifacts
        mlflow.log_artifacts(str(output_dir / "final_model"), "model")
    
    print(f"MLflow tracking at: http://localhost:5000")

if __name__ == "__main__":
    main()
