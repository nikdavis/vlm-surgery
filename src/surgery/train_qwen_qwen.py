#!/usr/bin/env python3
"""
Train the Qwen2.5-VL + DeepSeekR1 hybrid model on a single GPU.
This script uses gradient checkpointing to save memory.
"""
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoProcessor, TrainerCallback
from transformers.optimization import Adafactor, AdafactorSchedule
from typing import List, Dict, Any, Optional
import click
from datetime import datetime
import mlflow
from loguru import logger
import os
import torch.utils.data
from math import ceil, cos, pi

# Configure loguru from environment
log_level = os.getenv('LOGURU_LEVEL', 'INFO')
logger.remove()
logger.add(lambda msg: print(msg, end=""), level=log_level, format="{time:HH:mm:ss} | {level: <8} | {message}")

# Import the Qwen-Qwen hybrid model and the existing dataset loader
from src.surgery.qwen_qwen_vision import QwenQwenHybrid
from src.unified_dataset_loader import UnifiedOCRDataset


class AdapterOnlyTrainer(Trainer):
    """Custom trainer that only saves the adapter weights, not the full model."""
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override save to only save merger MLP weights using the model's save_pretrained method."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving merger MLP weights to {output_dir}")
        
        # Use the model's custom save_pretrained method which only saves merger weights
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            logger.error("Model doesn't have save_pretrained method!")
            
        # Save training state (optimizer, scheduler, etc.) separately
        # This is important for resuming training
        torch.save({
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'best_metric': self.state.best_metric,
            'best_model_checkpoint': self.state.best_model_checkpoint,
        }, os.path.join(output_dir, 'trainer_state.pt'))
        
        # Save training args for reference
        import json
        with open(os.path.join(output_dir, 'training_args.json'), 'w') as f:
            json.dump(self.args.to_dict(), f, indent=2)
        
        logger.info(f"Merger MLP checkpoint saved successfully")


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

        # No thinking tokens for this training
        self.think_start_id = None
        self.think_end_id = None

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
            max_length=1024,  # Increased to ensure we capture assistant response
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
                logger.info(f"Could not find user prompt position in sample {i}")
                modified_input_ids.append(torch.tensor(input_ids))
                continue
            
            # Insert ONLY image_pad tokens - the model will add vision_start/end
            num_images = len(feature["images"])
            image_pad_tokens = [self.placeholder_id] * num_images
            
            # Debug: Show where we're inserting (only for first batch)
            if i < 2 and not hasattr(self, '_insertion_logged'):  # First 2 samples, first batch only
                tokens_before = self.tokenizer.decode(input_ids[:insert_pos])
                tokens_after = self.tokenizer.decode(input_ids[insert_pos:insert_pos+50])  # First 50 chars after
                logger.info(f"Sample {i} insertion:")
                logger.info(f"  Inserting {num_images} image_pad tokens at position {insert_pos}")
                logger.info(f"  Text before: ...{tokens_before[-50:]}")
                logger.info(f"  Text after: {tokens_after}...")
            
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
            logger.info("Collator Summary:")
            logger.info(f"  Placeholder id (image_pad): {self.placeholder_id}")
            logger.info(f"  Vision start id: {vision_start_id}")
            logger.info(f"  Vision end id: {vision_end_id}")
            logger.info(f"  Total images processed: {image_idx}")
            for i in range(min(2, input_ids.shape[0])):  # Check first 2 samples
                num_placeholders = (input_ids[i] == self.placeholder_id).sum().item()
                num_vision_start = (input_ids[i] == vision_start_id).sum().item()
                num_vision_end = (input_ids[i] == vision_end_id).sum().item()
                logger.info(f"  Sample {i}: {num_placeholders} image_pad, {num_vision_start} vision_start, {num_vision_end} vision_end")
                # Show actual token sequence around placeholders
                if num_placeholders > 0:
                    placeholder_pos = (input_ids[i] == self.placeholder_id).nonzero(as_tuple=True)[0][0].item()
                    start = max(0, placeholder_pos - 5)
                    end = min(len(input_ids[i]), placeholder_pos + 5)
                    token_window = input_ids[i][start:end].tolist()
                    logger.info(f"    Token window around placeholder: {token_window}")
            self._insertion_logged = True  # Mark that we've logged insertion
        
        # Process images
        image_outputs = {}
        if images:
            image_inputs = self.image_processor(images, return_tensors="pt")
            image_outputs = image_inputs
        
        labels = input_ids.clone()

        # Create the masked labels - mask everything before assistant response
        for i in range(len(features)):
            input_id_list = input_ids[i].tolist()
            
            # Qwen3 uses <|im_start|>assistant format
            assistant_prompt = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            assistant_pos = None
            
            # Debug: Log what we're looking for vs what we have (only for first few samples)
            if i < 2 and not hasattr(self, '_assistant_debug_logged'):
                logger.info(f"Sample {i} assistant token search:")
                logger.info(f"  Looking for tokens: {assistant_prompt}")
                logger.info(f"  Total tokens in sample: {len(input_id_list)}")
                logger.info(f"  First 20 tokens: {input_id_list[:20]}")
                # Check if assistant tokens exist anywhere
                assistant_found = False
                for j in range(len(input_id_list) - len(assistant_prompt) + 1):
                    if all(input_id_list[j + k] == assistant_prompt[k] for k in range(len(assistant_prompt))):
                        assistant_found = True
                        logger.info(f"  ✓ Found assistant tokens at position {j}")
                        # Show context around it
                        context_start = max(0, j - 10)
                        context_end = min(len(input_id_list), j + 20)
                        context_text = self.tokenizer.decode(input_id_list[context_start:context_end])
                        logger.info(f"  Context: ...{context_text}...")
                        break
                if not assistant_found:
                    logger.warning(f"  ✗ Assistant tokens NOT found in sample!")
                    # Show the full decoded text to debug
                    full_text = self.tokenizer.decode(input_id_list)
                    logger.info(f"  Full text (first 500 chars): {full_text[:500]}")
                self._assistant_debug_logged = True
            
            if len(assistant_prompt) > 0:
                # Find this sequence in the input
                for j in range(len(input_id_list) - len(assistant_prompt) + 1):
                    if all(input_id_list[j + k] == assistant_prompt[k] for k in range(len(assistant_prompt))):
                        assistant_pos = j + len(assistant_prompt)
                        break
            
            # Try alternative patterns if standard pattern not found
            if assistant_pos is None:
                # Try pattern 2: <｜Assistant｜> (DeepSeek R1 style)
                assistant_token_id = 151670  # <｜Assistant｜> token
                for j in range(len(input_id_list)):
                    if input_id_list[j] == assistant_token_id:
                        assistant_pos = j + 1  # Position after the assistant token
                        if i < 2:
                            logger.info(f"  Found assistant token 151670 at position {j}")
                        break
            
            # Try pattern 3: Look for "assistant" in any form
            if assistant_pos is None:
                # Encode just "assistant" and look for it
                assistant_variants = [
                    self.tokenizer.encode("assistant", add_special_tokens=False),
                    self.tokenizer.encode("Assistant", add_special_tokens=False),
                    self.tokenizer.encode("\nassistant", add_special_tokens=False),
                ]
                for variant in assistant_variants:
                    if len(variant) > 0:
                        for j in range(len(input_id_list) - len(variant) + 1):
                            if all(input_id_list[j + k] == variant[k] for k in range(len(variant))):
                                assistant_pos = j + len(variant)
                                if i < 2:
                                    logger.info(f"  Found assistant variant at position {j}")
                                break
                        if assistant_pos is not None:
                            break
            
            if assistant_pos is not None:
                # Mask everything before assistant response (user prompt and assistant marker)
                labels[i, :assistant_pos] = -100
                # Only log for first batch to reduce spam
                if i < 2 and not hasattr(self, '_mask_logged'):
                    masked_count = assistant_pos
                    total_count = (labels[i] != self.tokenizer.pad_token_id).sum().item()
                    logger.info(f"  Sample {i}: Masked {masked_count} tokens before assistant response, {total_count - masked_count} tokens remain for training")
                    self._mask_logged = True
            else:
                # Only log occasionally to reduce spam
                if not hasattr(self, '_assistant_not_found_count'):
                    self._assistant_not_found_count = 0
                self._assistant_not_found_count += 1
                if self._assistant_not_found_count <= 3:  # Only log first 3 times
                    logger.warning(f"Could not find assistant token in sample {i} - will train on full sequence")
                    # Show what we have
                    if i == 0:
                        decoded = self.tokenizer.decode(input_id_list[:200])
                        logger.info(f"  First 200 chars: {decoded[:200]}")
                # As fallback, mask just the user prompt portion
                # Look for where the image_pad token is and mask up to there plus a bit
                if self.placeholder_id in input_id_list:
                    pad_idx = input_id_list.index(self.placeholder_id)
                    # Mask up to and including the placeholder + some context
                    labels[i, :pad_idx+10] = -100  # Mask user prompt + a bit of context

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Debug: Check label masking
        if not hasattr(self, '_label_debug_printed'):
            self._label_debug_printed = True
            logger.info("Label masking:")
            for i in range(min(2, labels.shape[0])):
                non_masked = (labels[i] != -100).sum().item()
                total = (labels[i] != self.tokenizer.pad_token_id).sum().item()  # Don't count padding
                logger.info(f"  Sample {i}: {non_masked}/{total} non-padded tokens are unmasked ({non_masked/total*100:.1f}% if total > 0 else 0)")
                # Show first few non-masked tokens
                non_masked_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(non_masked_indices) > 0:
                    first_tokens = labels[i][non_masked_indices[:10]].tolist()
                    decoded = [self.tokenizer.decode([t]) for t in first_tokens]
                    logger.info(f"    First non-masked tokens: {decoded}")

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

class PercentCurriculum(TrainerCallback):
    """Curriculum callback that adjusts K and gate based on training progress percentage."""
    def __init__(self, start_k=16, end_k=64, gate_max=1.5, stage1_pct=0.15, ramp_end_pct=0.50):
        self.start_k, self.end_k = start_k, end_k
        self.gate_max = gate_max
        self.stage1_pct = stage1_pct
        self.ramp_end_pct = ramp_end_pct
        self.total_updates = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Model is passed directly, not in kwargs
        if model is None:
            return control
            
        # Initialize total updates based on state/args
        if self.total_updates is None:
            if state.max_steps and state.max_steps > 0:
                self.total_updates = state.max_steps
            else:
                # Estimate from epochs and dataset size
                # This is approximate since we don't have direct access to dataloader
                self.total_updates = int(args.num_train_epochs * 100)  # Rough estimate
        
        t = state.global_step
        T = max(1, self.total_updates)
        p = min(1.0, t / T)  # progress 0..1

        # Stage 1: keep extras off
        if p <= self.stage1_pct:
            model.base_k = self.start_k
            if hasattr(model, "use_soft_gate") and model.use_soft_gate:
                with torch.no_grad():
                    model.extra_gate.data.fill_(0.0)
            return control

        # Stage 2: ramp K and gate (cosine ramp)
        ramp_p = (p - self.stage1_pct) / max(1e-6, (self.ramp_end_pct - self.stage1_pct))
        ramp_p = max(0.0, min(1.0, ramp_p))
        # cosine 0→1: smooth
        smooth = 0.5 * (1 - cos(pi * ramp_p))
        K = int(round(self.start_k + smooth * (self.end_k - self.start_k)))
        model.base_k = K
        if hasattr(model, "use_soft_gate") and model.use_soft_gate:
            g = smooth * self.gate_max
            with torch.no_grad():
                model.extra_gate.data.fill_(g)
        return control

@click.command()
@click.option("--output-dir", default="./outputs_qwen_hybrid", help="Output directory for checkpoints.")
@click.option("--data-path", default="./datasetv2/combined_dataset.json", help="Path to the unified dataset JSON file.")
@click.option("--batch-size", type=int, default=1, help="Training batch size per device.")
@click.option("--gradient-accumulation", type=int, default=32, help="Gradient accumulation steps.")
@click.option("--learning-rate", type=float, default=2e-5, help="Learning rate for the adapter.")
@click.option("--num-epochs", type=int, default=3, help="Number of training epochs.")
@click.option("--use-8bit-adam", is_flag=True, help="Use 8-bit Adam optimizer instead of AdaFactor")
@click.option("--exclude-v1-data/--include-v1-data", default=True, help="Exclude/include v1 data (test_data_200 examples)")
@click.option("--exclude-captions/--include-captions", default=False, help="Exclude/include captions dataset")
@click.option("--quick-test", is_flag=True, help="Quick test mode: 10 examples, no accumulation, eval at step 1")
def main(output_dir, data_path, batch_size, gradient_accumulation, learning_rate, num_epochs, use_8bit_adam, exclude_v1_data, exclude_captions, quick_test):
    # Quick test mode overrides
    if quick_test:
        logger.info("🚀 QUICK TEST MODE - Minimal training for validation")
        batch_size = 1
        gradient_accumulation = 1
        num_epochs = 1
        exclude_captions = True  # Skip large caption dataset
        logger.info(f"  Settings: batch={batch_size}, accum={gradient_accumulation}, epochs={num_epochs}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("qwen-qwen-hybrid-20250805")

    # 1. Instantiate the Qwen-Qwen hybrid model
    model = QwenQwenHybrid()
    tokenizer = model.tokenizer

    # Load Qwen processor for image processing
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # For Qwen3, we'll train without thinking tokens for stability
    # The model has native thinking support that can be enabled at inference
    logger.info("Training WITHOUT thinking tokens for stability")
    logger.info("Thinking can be enabled at inference with enable_thinking=True")

    # 2. Load the dataset without CoT (we'll strip thinking tokens)
    # Always include captions dataset by default
    data_paths = [data_path]
    
    # Check for captions in Docker or local path - use normalized filename
    captions_paths = [
        Path("/app/data-captions/pixelprose_captions.parquet"),  # Docker
        Path("./data-captions/pixelprose_captions.parquet"),  # Local
    ]
    
    if not exclude_captions:
        for captions_path in captions_paths:
            if captions_path.exists():
                data_paths.append(str(captions_path))
                logger.info(f"Including captions dataset: {captions_path}")
                break
        else:
            logger.warning("Captions dataset not found in expected locations")
    
    full_dataset = UnifiedOCRDataset(data_paths, enable_cot=False)
    
    # Filter out v1 data if requested
    if exclude_v1_data:
        logger.info("Filtering out v1 data (test_data_200 examples)...")
        original_len = len(full_dataset)
        
        # Create filtered indices - exclude items with test_data_200 in image path
        filtered_indices = []
        for idx in range(len(full_dataset)):
            sample = full_dataset[idx]
            # Check if any image path contains test_data_200
            has_v1_data = False
            if "images" in sample:
                for img in sample["images"]:
                    if hasattr(img, '_path'):
                        # PIL Image with _path attribute
                        if "test_data_200" in str(img._path):
                            has_v1_data = True
                            break
            if not has_v1_data:
                filtered_indices.append(idx)
        
        # Create subset with filtered indices
        import torch.utils.data
        full_dataset = torch.utils.data.Subset(full_dataset, filtered_indices)
        logger.info(f"Filtered dataset: {original_len} -> {len(full_dataset)} samples (removed {original_len - len(full_dataset)} v1 samples)")
    
    
    # Quick test mode: use tiny dataset
    if quick_test:
        logger.info("Quick test: Using only 10 training examples, 2 validation")
        # Take first 12 examples total
        if hasattr(full_dataset, 'examples'):
            # Direct dataset
            full_dataset.examples = full_dataset.examples[:12]
        else:
            # It's a Subset
            full_dataset = torch.utils.data.Subset(full_dataset, range(min(12, len(full_dataset))))
        
        # Split 10 train, 2 val
        train_dataset = torch.utils.data.Subset(full_dataset, range(10))
        val_dataset = torch.utils.data.Subset(full_dataset, range(10, min(12, len(full_dataset))))
    else:
        # Normal split
        if hasattr(full_dataset, 'get_train_val_split'):
            train_dataset, val_dataset = full_dataset.get_train_val_split(val_ratio=0.02)
        else:
            # full_dataset is a Subset, do manual split
            dataset_len = len(full_dataset)
            val_size = int(dataset_len * 0.02)
            train_size = dataset_len - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
    
    logger.info(f"Training set size: {len(train_dataset)} samples")
    logger.info(f"Validation set size: {len(val_dataset)} samples")

    # Create run name
    run_name = f"Qwen-Qwen-QLoRA8bit-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Choose optimizer based on flag
    if use_8bit_adam:
        optimizer = "paged_adamw_8bit"  # Paged 8-bit Adam from bitsandbytes
        logger.info("Using paged 8-bit Adam optimizer")
    else:
        optimizer = "adafactor"
        logger.info("Using AdaFactor optimizer")
    
    # 3. Define TrainingArguments (simple, single-GPU version)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,  # Keep eval batch size at 1 to avoid OOM
        gradient_accumulation_steps=gradient_accumulation,
        eval_accumulation_steps=4,  # Process eval data in chunks of 4 to avoid OOM
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=True, # Use bfloat16 for memory efficiency
        fp16=False, # BF16 is better than FP16 for stability
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,  # Save every 20 steps
        save_safetensors=False,  # Disable safetensors to avoid shared tensor issues
        eval_strategy="steps",
        eval_steps=20,  # Evaluate every 20 steps
        do_train=True,
        do_eval=True,
        remove_unused_columns=False,
        gradient_checkpointing=True, # Enable for memory but we'll customize it
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optimizer,  # Use chosen optimizer
        report_to="mlflow",  # Enable MLflow logging
        run_name=run_name,
        max_grad_norm=1.0,  # Standard gradient clipping
        warmup_ratio=0.1,  # 10% warmup is more stable than fixed steps
        dataloader_num_workers=0,  # Disable multiprocessing to save memory
    )

    # The Trainer will handle gradient checkpointing based on training_args
    # Disable cache when using gradient checkpointing
    model.language_model.config.use_cache = False

    # 4. Instantiate the custom AdapterOnlyTrainer (Adafactor will be used automatically)
    # This custom trainer only saves adapter weights to avoid shared tensor issues
    trainer = AdapterOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CoTDataCollator(tokenizer, processor, placeholder_id=model.config.vision_placeholder_id),
    )
    
    # Add curriculum callback for token progression
    trainer.add_callback(PercentCurriculum(
        start_k=16, end_k=64, gate_max=1.5, stage1_pct=0.15, ramp_end_pct=0.50
    ))
    logger.info("Added curriculum callback: Stage 1 (0-15%), Stage 2 ramp (15-50%), Stage 3 (50-100%)")

    # 5. Start training with MLflow tracking
    logger.info("Starting Training")
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
        
        logger.info("Training Complete")
    
        # 6. Save the final trained adapter
        model.save_pretrained(str(output_dir / "final_model"))
        logger.info(f"Final model adapter saved to {output_dir / 'final_model'}")
        
        # Log the final model artifacts
        mlflow.log_artifacts(str(output_dir / "final_model"), "model")
    
    logger.info(f"MLflow tracking at: http://localhost:5000")

if __name__ == "__main__":
    main()
