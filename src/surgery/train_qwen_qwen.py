#!/usr/bin/env python3
"""
Train the Qwen2.5-VL + DeepSeekR1 hybrid model on single or multi-GPU.
This script uses gradient checkpointing to save memory and supports DDP.
"""
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoProcessor, TrainerCallback, SchedulerType
from transformers.optimization import Adafactor, AdafactorSchedule
from typing import List, Dict, Any, Optional
import click
from datetime import datetime
import mlflow
from loguru import logger
import os
import torch.utils.data
from math import ceil, cos, pi
import torch.distributed as dist

# Configure loguru from environment
log_level = os.getenv('LOGURU_LEVEL', 'INFO')
logger.remove()
logger.add(lambda msg: print(msg, end=""), level=log_level, format="{time:HH:mm:ss} | {level: <8} | {message}")

# Import the Qwen-Qwen hybrid model and dataset loaders
from src.surgery.qwen_qwen_vision import QwenQwenHybrid
from src.unified_dataset_loader import UnifiedOCRDataset
# Import virtual dataset components
from src.data.adapters import ParquetCaptionAdapter
from src.data.virtual_dataset import VirtualDataset
from src.data.transforms import (
    DecodeImage, RandomResize, RandomCrop, MildColorJitter,
    IdentityOrAug, RandomChoice, GaussianNoise, JPEGCompression
)


class AdapterOnlyTrainer(Trainer):
    """Custom trainer that only saves the adapter weights, not the full model."""

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override save to only save adapter weights (merger MLP, post_proj_ln, extra_gate) using the model's save_pretrained method."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving adapter weights (merger MLP, post_proj_ln, extra_gate) to {output_dir}")

        # Use the model's custom save_pretrained method which saves all adapter components
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            logger.error("Model doesn't have save_pretrained method!")

        # Save training state (optimizer, scheduler, etc.) properly
        # This is important for resuming training
        self.save_state()  # This saves optimizer, scheduler, training state

        # Save training args for reference
        import json
        with open(os.path.join(output_dir, 'training_args.json'), 'w') as f:
            json.dump(self.args.to_dict(), f, indent=2)

        logger.info(f"Adapter checkpoint saved successfully")
    
    def _load_from_checkpoint(self, checkpoint: str):
        """Override to properly load our adapter-only checkpoints."""
        import torch
        checkpoint = os.path.abspath(checkpoint.rstrip("/"))
        adapter_path = os.path.join(checkpoint, "vision_adapter.pt")
        
        if os.path.exists(adapter_path):
            logger.info(f"Loading adapter-only checkpoint from {checkpoint}")
            
            # Load the adapter weights
            state = torch.load(adapter_path, map_location="cpu")
            
            # Restore merger and marker embeddings
            self.model.vision_model.merger.load_state_dict(state["vision_merger_state"])
            self.model.vision_start_embedding.data.copy_(state["vision_start_embedding"].data)
            self.model.vision_end_embedding.data.copy_(state["vision_end_embedding"].data)
            
            # Restore post_proj_ln if present (backward compatibility)
            if "post_proj_ln_state" in state:
                self.model.post_proj_ln.load_state_dict(state["post_proj_ln_state"])
                logger.info("✓ post_proj_ln restored")
            else:
                logger.warning("post_proj_ln_state missing in checkpoint - using fresh initialization (expect loss spike)")
            
            # Restore extra_gate if present (backward compatibility)
            if "extra_gate" in state:
                self.model.extra_gate.data.copy_(state["extra_gate"].data)
                logger.info(f"✓ extra_gate restored (value={state['extra_gate'].item():.3f})")
            else:
                logger.warning("extra_gate missing in checkpoint - using fresh initialization")
            
            # Log sanity check values after loading
            logger.info(f"Checkpoint loaded: post_proj_ln gamma mean={self.model.post_proj_ln.weight.mean().item():.4f}, "
                       f"beta mean={self.model.post_proj_ln.bias.mean().item():.4f}, "
                       f"extra_gate={self.model.extra_gate.item():.3f}")
            
            logger.info("✓ Adapter weights restored")
            
            # Restore optimizer if present (with error handling for mismatches)
            opt_path = os.path.join(checkpoint, "optimizer.pt")
            if self.optimizer and os.path.exists(opt_path):
                try:
                    self.optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                    logger.info("✓ Optimizer state restored")
                except Exception as e:
                    logger.warning(f"Could not restore optimizer state ({type(e).__name__}: {e})")
                    logger.warning("Initializing fresh optimizer (momentum will reset)")
                    # Optimizer stays freshly initialized
            
            # Restore scheduler if present (with error handling)
            sched_path = os.path.join(checkpoint, "scheduler.pt")
            if self.lr_scheduler and os.path.exists(sched_path):
                try:
                    self.lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
                    logger.info("✓ Scheduler state restored")
                except Exception as e:
                    logger.warning(f"Could not restore scheduler state ({type(e).__name__}: {e})")
                    logger.warning("Using fresh scheduler (learning rate schedule will restart)")
                    # Scheduler stays freshly initialized
            
            # Restore RNG state if present
            rng_path = os.path.join(checkpoint, "rng_state.pth")
            if os.path.exists(rng_path):
                self._load_rng_state(checkpoint)
                logger.info("✓ RNG state restored")
            
            # Restore trainer state but RESET history to avoid polluted running averages
            trainer_state_path = os.path.join(checkpoint, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                from transformers.trainer_callback import TrainerState
                self.state = TrainerState.load_from_json(trainer_state_path)
                
                # CRITICAL: Reset log history to clear stale running averages from before fixes
                # This prevents misleading high train_loss values from old spikes
                old_global_step = self.state.global_step
                self.state.log_history = []  # Clear all history
                self.state.best_metric = None  # Reset best metric tracking
                # Keep important state like global_step and max_steps
                
                self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
                logger.info(f"✓ Trainer state partially restored (global_step={self.state.global_step}, history cleared)")
            
            # Stop memory tracker and return (skip default weight loading)
            if hasattr(self, '_memory_tracker') and self._memory_tracker is not None:
                self._memory_tracker.stop_and_update_metrics(self.state)
            
            logger.info(f"Successfully resumed from checkpoint at step {self.state.global_step}")
            return
        
        # Fall back to standard HF checkpoint loading if not adapter-only
        return super()._load_from_checkpoint(checkpoint)


class CoTDataCollator:
    """
    Handles batching and creates masked labels for CoT training.
    Manual handling for cross-model compatibility (Qwen2.5-VL vision + Qwen3 R1 language).
    """
    def __init__(self, tokenizer: AutoTokenizer, processor: AutoProcessor, placeholder_id: int, model=None):
        self.tokenizer = tokenizer  # Qwen3 R1 tokenizer
        self.image_processor = processor.image_processor  # Only use image processor from Qwen2.5-VL
        self.placeholder_id = placeholder_id  # The unused token we're repurposing
        self.model = model  # Optional reference to track vision token settings
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
            max_length=2048,  # Increased to 2k tokens
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

            # Debug: Show where we're inserting (only for first batch) - commented out for cleaner output
            # if i < 2 and not hasattr(self, '_insertion_logged'):  # First 2 samples, first batch only
            #     tokens_before = self.tokenizer.decode(input_ids[:insert_pos])
            #     tokens_after = self.tokenizer.decode(input_ids[insert_pos:insert_pos+50])  # First 50 chars after
            #     logger.debug(f"Sample {i} insertion:")
            #     logger.debug(f"  Inserting {num_images} image_pad tokens at position {insert_pos}")
            #     logger.debug(f"  Text before: ...{tokens_before[-50:]}")
            #     logger.debug(f"  Text after: {tokens_after}...")

            # Insert only the image_pad tokens
            new_ids = input_ids[:insert_pos] + image_pad_tokens + input_ids[insert_pos:]
            modified_input_ids.append(torch.tensor(new_ids))

            image_idx += num_images

        # --- REPLACED: Deterministic sequence build so labels are correct ---
        
        # Token IDs we need
        user_role_id = 151669  # <｜User｜>
        assistant_role_id = 151670  # <｜Assistant｜>
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        image_pad_id = self.placeholder_id
        
        max_len = 2048  # Using the 2k limit
        
        seq_ids_list = []
        labels_list = []
        
        for feature in features:
            num_images = len(feature["images"])
            
            # Extract prompt and response text
            user_content = feature["messages"][0]["content"]
            prompt_txt = next((item["text"] for item in user_content if item["type"] == "text"), "")
            resp_txt = feature["messages"][1]["content"][0]["text"] if feature["messages"][1]["content"] else ""
            
            # Safety check: if assistant is still empty for some reason, use a fallback
            if not resp_txt or not resp_txt.strip():
                logger.warning(f"Empty assistant response found, using fallback")
                resp_txt = "This image shows various visual elements."  # Fallback response
            
            # Plain tokenization (no chat template)
            prompt_ids = self.tokenizer(prompt_txt, add_special_tokens=False).input_ids
            resp_ids = self.tokenizer(resp_txt, add_special_tokens=False).input_ids
            
            # Log response length once
            if not hasattr(self, '_resp_len_logged'):
                logger.info(f"[collator] response_token_len={len(resp_ids)}")
                self._resp_len_logged = True
            
            # Build: [User] [<image_pad> * num_images] prompt [Assistant] response [<im_end>]
            seq = [user_role_id] + [image_pad_id] * num_images + prompt_ids + [assistant_role_id] + resp_ids + [im_end_id]
            
            # Fit to max_len: *always* keep the full assistant; trim prompt from the left if needed
            overflow = max(0, len(seq) - max_len)
            if overflow > 0:
                # How many can we cut from the prompt slice (between pads and assistant token)?
                cut_zone_start = 1 + num_images
                cut_zone_end = cut_zone_start + len(prompt_ids)
                cut = min(overflow, len(prompt_ids))
                if cut > 0:
                    # Drop from the left of the prompt window
                    seq = seq[:cut_zone_start] + seq[cut_zone_start + cut:]
                    logger.debug(f"Trimmed {cut} tokens from prompt to fit max_len")
            
            # Create labels: only the assistant response tokens should be targets
            # Find the assistant_role position -> response starts right after it
            try:
                a_pos = seq.index(assistant_role_id)
            except ValueError:
                a_pos = len(seq)  # fallback (shouldn't happen)
            r_start = a_pos + 1
            
            # Response ends before <|im_end|>
            try:
                r_end = seq.index(im_end_id, r_start)
            except ValueError:
                r_end = len(seq)
            
            labels = torch.full((len(seq),), -100, dtype=torch.long)
            
            # Copy the actual target ids for response span (mask specials just in case)
            for j in range(r_start, r_end):
                tok = seq[j]
                if tok not in (user_role_id, assistant_role_id, image_pad_id, im_end_id, self.tokenizer.pad_token_id):
                    labels[j] = tok
            
            # Debug first few samples
            if len(seq_ids_list) < 2 and not hasattr(self, '_seq_debug_logged'):
                logger.info(f"Sample {len(seq_ids_list)} sequence build:")
                logger.info(f"  Total sequence length: {len(seq)}")
                logger.info(f"  Assistant response position: {r_start}-{r_end}")
                logger.info(f"  Response length: {r_end - r_start} tokens")
                logger.info(f"  Unmasked tokens: {(labels != -100).sum().item()}")
                if r_end > r_start:
                    # Show what we're training on
                    resp_preview = self.tokenizer.decode(seq[r_start:min(r_end, r_start+20)])
                    logger.info(f"  Training on (first 20 tokens): '{resp_preview}...'")
            
            seq_ids_list.append(torch.tensor(seq, dtype=torch.long))
            labels_list.append(labels)
        
        if len(seq_ids_list) > 0 and not hasattr(self, '_seq_debug_logged'):
            self._seq_debug_logged = True
        
        # Now pad everything
        input_ids = torch.nn.utils.rnn.pad_sequence(seq_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(x) for x in seq_ids_list], 
            batch_first=True, 
            padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        # Track batch statistics for comprehensive logging
        if not hasattr(self, '_batch_count'):
            self._batch_count = 0
            self._total_seq_length = 0
            self._total_valid_tokens = 0
            self._batches_with_images = 0
            self._log_interval = 10  # Log every 10 batches
        
        self._batch_count += 1
        
        # Sanity check
        if not hasattr(self, '_labels_final_logged'):
            valid_before = (labels != -100).sum().item()
            valid_after = (labels[:, 1:] != -100).sum().item() if labels.shape[1] > 1 else 0
            logger.info(f"[labels] valid(before shift)={valid_before}  valid(after shift)={valid_after}")
            if valid_after < 10:
                logger.warning("⚠️ Still very few tokens for training!")
            else:
                logger.info(f"✓ Good: {valid_after} tokens for training")
            self._labels_final_logged = True

        # Process images
        image_outputs = {}
        if images:
            image_inputs = self.image_processor(images, return_tensors="pt")
            image_outputs = image_inputs
            self._batches_with_images += 1

        # Collect statistics
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        valid_tokens = (labels != -100).sum().item()
        
        self._total_seq_length += seq_length * batch_size
        self._total_valid_tokens += valid_tokens
        
        # Log statistics periodically
        if self._batch_count % self._log_interval == 0:
            avg_seq_length = self._total_seq_length / (self._batch_count * batch_size)
            avg_valid_tokens = self._total_valid_tokens / self._batch_count
            pct_with_images = (self._batches_with_images / self._batch_count) * 100
            
            logger.info(f"[BATCH STATS {self._batch_count}] "
                       f"avg_seq_len={avg_seq_length:.1f}, "
                       f"avg_valid_tokens={avg_valid_tokens:.1f}, "
                       f"batches_with_images={pct_with_images:.1f}%")

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
    """Curriculum callback that adjusts K, gate, and vision token cap based on training progress percentage."""
    def __init__(self, start_k=16, end_k=64, gate_max=1.5, 
                 start_cap=64, mid_cap=256, end_cap=512,
                 stage1_pct=0.15, ramp_end_pct=0.50):
        self.start_k, self.end_k = start_k, end_k
        self.gate_max = gate_max
        self.start_cap, self.mid_cap, self.end_cap = start_cap, mid_cap, end_cap
        self.stage1_pct = stage1_pct
        self.ramp_end_pct = ramp_end_pct
        self.total_updates = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Compute true total updates once at the start."""
        # Prefer explicit max_steps
        if state.max_steps and state.max_steps > 0:
            self.total_updates = state.max_steps
        else:
            trainer = kwargs.get("trainer")
            if trainer:
                ga = args.gradient_accumulation_steps
                updates_per_epoch = ceil(len(trainer.get_train_dataloader()) / ga)
                epochs = int(args.num_train_epochs)
                self.total_updates = max(1, updates_per_epoch * epochs)

        if self.total_updates:
            logger.info(f"Curriculum: total_updates={self.total_updates}, stages at {int(self.stage1_pct*100)}% and {int(self.ramp_end_pct*100)}%")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Adjust curriculum parameters based on progress."""
        if model is None or self.total_updates is None:
            return control
        
        # Skip curriculum if pooling is disabled (all tokens pass through)
        if getattr(model, 'disable_pooling', False):
            return control

        p = min(1.0, state.global_step / self.total_updates)

        # Stage 1: keep extras off, minimal vision tokens
        if p <= self.stage1_pct:
            model.base_k = self.start_k
            model.max_vision_tokens = self.start_cap
            model.allow_extra_tokens = False
            if getattr(model, "use_soft_gate", False):
                with torch.no_grad():
                    model.extra_gate.data.fill_(0.0)
            return control

        # Stage 2: ramp K, gate, and cap (cosine ramp)
        ramp = max(0.0, min(1.0, (p - self.stage1_pct) / (self.ramp_end_pct - self.stage1_pct + 1e-6)))
        smooth = 0.5 * (1 - cos(pi * ramp))
        model.base_k = int(round(self.start_k + smooth * (self.end_k - self.start_k)))
        model.max_vision_tokens = int(round(self.start_cap + smooth * (self.mid_cap - self.start_cap)))
        model.allow_extra_tokens = True  # Enable extras once ramping starts

        if getattr(model, "use_soft_gate", False):
            with torch.no_grad():
                model.extra_gate.data.fill_(smooth * self.gate_max)
        
        # Stage 3: After ramp_end_pct, continue increasing cap to end_cap
        if p >= self.ramp_end_pct:
            tail = (p - self.ramp_end_pct) / max(1e-6, (1.0 - self.ramp_end_pct))
            tail_smooth = 0.5 * (1 - cos(pi * tail))
            model.max_vision_tokens = int(round(self.mid_cap + tail_smooth * (self.end_cap - self.mid_cap)))

        return control

@click.command()
@click.option("--output-dir", default="./outputs_qwen_hybrid", help="Output directory for checkpoints.")
@click.option("--data-path", default="./datasetv2/combined_dataset.json", help="Path to the unified dataset JSON file.")
@click.option("--batch-size", type=int, default=1, help="Training batch size per device.")
@click.option("--gradient-accumulation", type=int, default=32, help="Gradient accumulation steps.")
@click.option("--learning-rate", type=float, default=5e-4, help="Learning rate for the adapter (default 5e-4 for stability).")
@click.option("--num-epochs", type=int, default=3, help="Number of training epochs.")
@click.option("--max-steps", type=int, default=None, help="Max training steps (overrides num-epochs if set).")
@click.option("--use-8bit-adam", is_flag=True, help="Use 8-bit Adam optimizer instead of AdaFactor")
@click.option("--exclude-v1-data/--include-v1-data", default=True, help="Exclude/include v1 data (test_data_200 examples)")
@click.option("--exclude-captions/--include-captions", default=False, help="Exclude/include captions dataset")
@click.option("--quick-test", is_flag=True, help="Quick test mode: 10 examples, no accumulation, eval at step 1")
@click.option("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
@click.option("--disable-curriculum", is_flag=True, help="Disable curriculum learning (auto-disabled when resuming)")
@click.option("--enable-vision-pooling", is_flag=True, help="Enable vision token pooling/sampling (default is to pass all tokens)")
def main(output_dir, data_path, batch_size, gradient_accumulation, learning_rate, num_epochs, max_steps, use_8bit_adam, exclude_v1_data, exclude_captions, quick_test, resume_from_checkpoint, disable_curriculum, enable_vision_pooling):
    # DDP setup (safe for single GPU - just no-ops)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Debug GPU visibility
    logger.info(f"[rank {local_rank}] NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"[rank {local_rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"[rank {local_rank}] cuda_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"[rank {local_rank}] device {i}: {torch.cuda.get_device_name(i)}")
    
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        # Docker remaps GPUs 0,3 to 0,1 inside container
        # So LOCAL_RANK maps directly to the remapped device index
        torch.cuda.set_device(local_rank)
        logger.info(f"[DDP] Rank {local_rank}/{world_size}, current_device={torch.cuda.current_device()}")
    else:
        logger.info("[Single GPU] Running without DDP")
    
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
    if enable_vision_pooling:
        logger.info("Vision pooling ENABLED - using token sampling/pooling")
        model = QwenQwenHybrid(disable_pooling=False)
    else:
        logger.info("Vision pooling DISABLED (default) - passing all vision tokens directly")
        model = QwenQwenHybrid(disable_pooling=True)
    tokenizer = model.tokenizer

    # Load Qwen processor for image processing
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # For Qwen3, we'll train without thinking tokens for stability
    # The model has native thinking support that can be enabled at inference
    logger.info("Training WITHOUT thinking tokens for stability")
    logger.info("Thinking can be enabled at inference with enable_thinking=True")

    # 2. Load the dataset
    if not exclude_captions:
        # Use virtual dataset for captions
        # Look for dataset in subdirectory first, then fall back to root
        captions_patterns = [
            Path("/app/data-captions/dataset_*/*.parquet"),  # Docker with subdirectory
            Path("./data-captions/dataset_*/*.parquet"),  # Local with subdirectory
            Path("/app/data-captions/*.parquet"),  # Docker root (fallback)
            Path("./data-captions/*.parquet"),  # Local root (fallback)
        ]

        captions_glob = None
        for pattern in captions_patterns:
            # Check if any files match this pattern
            import glob
            if glob.glob(str(pattern)):
                captions_glob = str(pattern)
                break

        if captions_glob:
            logger.info(f"Loading captions from {captions_glob} using VirtualDataset")
            adapter = ParquetCaptionAdapter(captions_glob)
            
            # Two-phase transform pipeline:
            # Phase A: Identity only (DecodeImage)
            # Phase B: 30% identity, 70% augmented with max 2 transforms
            transforms = [
                DecodeImage(),
                IdentityOrAug(
                    p_identity=0.3,  # 30% identity in Phase B
                    aug_transforms=[
                        RandomChoice(
                            n=2,  # Apply max 2 augmentations
                            transforms=[
                                RandomResize(scale=(0.7, 1.3)),  # Slightly stronger than before
                                RandomCrop(min_crop_ratio=0.6),  # As requested
                                MildColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                GaussianNoise(sigma=0.015),
                                JPEGCompression(quality_range=(70, 95))
                            ]
                        )
                    ]
                )
            ]
            
            full_dataset = VirtualDataset(
                adapter=adapter,
                transforms=transforms,
                virtual_scale_factor=1.5  # 1.5x for coverage + augmentation
            )
            logger.info(f"Virtual dataset created with {len(full_dataset)} virtual samples")
        else:
            logger.error("Captions dataset not found in expected locations!")
            return
    else:
        # Use original UnifiedOCRDataset for non-caption data
        logger.info("Using UnifiedOCRDataset for non-caption data")
        full_dataset = UnifiedOCRDataset([data_path], enable_cot=False)

    # Skip v1 filtering - we're only using caption data with VirtualDataset
    if exclude_v1_data and not exclude_captions:
        logger.info("V1 data filtering not needed for VirtualDataset (captions only)")

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
        # Normal split - improved for eval hygiene
        if hasattr(full_dataset, 'real_length'):
            # VirtualDataset: Use only Phase A (identity transforms) for eval
            phase_a_indices = list(range(full_dataset.real_length))
            val_size = min(int(full_dataset.real_length * 0.02), 500)  # Cap at 500 for speed
            
            # Fixed seed for deterministic eval split
            eval_generator = torch.Generator().manual_seed(42)
            val_indices = torch.randperm(full_dataset.real_length, generator=eval_generator)[:val_size].tolist()
            train_indices = [i for i in phase_a_indices if i not in val_indices]
            
            # Train gets Phase A (minus val) + all Phase B; Val gets only Phase A subset
            train_indices_full = train_indices + list(range(full_dataset.real_length, len(full_dataset)))
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices_full)
            val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
            
            logger.info(f"Eval using Phase A only (identity transforms) with fixed seed")
        elif hasattr(full_dataset, 'get_train_val_split'):
            train_dataset, val_dataset = full_dataset.get_train_val_split(val_ratio=0.02)
        else:
            # full_dataset is a Subset, do manual split with fixed seed
            dataset_len = len(full_dataset)
            val_size = int(dataset_len * 0.02)
            train_size = dataset_len - val_size
            
            # Use fixed generator for deterministic split
            split_generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator=split_generator
            )

    # Set eval dataset to use fixed prompt (not training mode)
    if hasattr(full_dataset, 'training'):
        # For VirtualDataset wrapped in Subset, need to access the underlying dataset
        if hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'training'):
            val_dataset.dataset.training = False
            logger.info("Set eval dataset to use fixed prompt (training=False)")
        elif hasattr(val_dataset, 'training'):
            val_dataset.training = False
            logger.info("Set eval dataset to use fixed prompt (training=False)")

    logger.info(f"Training set size: {len(train_dataset)} samples")
    logger.info(f"Validation set size: {len(val_dataset)} samples")

    # Create run name
    run_name = f"Qwen-Qwen-QLoRA8bit-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Make output dir unique to avoid overwriting
    output_dir = Path(output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose optimizer based on flag and DDP mode
    if world_size > 1:
        # DDP mode: Use standard torch AdamW to avoid bitsandbytes multi-GPU issues
        optimizer = "adamw_torch"  # Standard PyTorch AdamW
        logger.info(f"[DDP] Using torch AdamW optimizer (avoids bitsandbytes multi-GPU issues)")
    elif use_8bit_adam:
        optimizer = "paged_adamw_8bit"  # Paged 8-bit Adam from bitsandbytes
        logger.info("Using paged 8-bit Adam optimizer")
    else:
        optimizer = "adafactor"
        logger.info("Using AdaFactor optimizer")

    # 3. Define TrainingArguments with better defaults
    # Use max_steps if provided, otherwise epochs
    if max_steps and not quick_test:
        training_steps = max_steps
        training_epochs = 1  # Ignored when max_steps > 0
        logger.info(f"Training for {max_steps} steps")
    else:
        training_steps = -1 if not quick_test else 5  # Quick test: 5 steps
        training_epochs = num_epochs
        if not quick_test:
            logger.info(f"Training for {num_epochs} epochs")

    # DDP-aware training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,  # Keep eval batch size at 1 to avoid OOM
        gradient_accumulation_steps=gradient_accumulation,
        eval_accumulation_steps=4,  # Process eval data in chunks of 4 to avoid OOM
        # Duration control
        max_steps=training_steps,
        num_train_epochs=training_epochs,
        # LR & schedule
        learning_rate=learning_rate,
        lr_scheduler_type=SchedulerType.COSINE,  # Cosine schedule
        warmup_ratio=0.1 if not quick_test else 0.0,  # 10% warmup (skip in quick test)
        # Precision
        bf16=True,
        fp16=False,
        # Logging & checkpointing
        logging_steps=10 if not quick_test else 1,  # Log every 10 steps
        eval_strategy="steps",
        eval_steps=25 if not quick_test else 1,  # Eval every 25 steps
        save_strategy="steps",
        save_steps=25 if not quick_test else 1,  # Save every 25 steps
        save_safetensors=False,  # Disable safetensors to avoid shared tensor issues
        # Training config
        do_train=True,
        do_eval=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        dataloader_num_workers=2,  # Use 2 workers for better IO
        # DDP settings
        ddp_find_unused_parameters=False,  # Better performance, we handle all params properly
        ddp_backend="nccl" if world_size > 1 else None,  # Use NCCL for multi-GPU
        # Optimizer
        optim=optimizer,
        # Reporting
        report_to="mlflow" if not quick_test else "none",
        run_name=run_name,
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
        data_collator=CoTDataCollator(tokenizer, processor, placeholder_id=model.config.vision_placeholder_id, model=model),
    )

    # Add curriculum callback for token progression (unless disabled or resuming)
    if not enable_vision_pooling:
        logger.info("Curriculum disabled (vision pooling is disabled - all tokens pass through)")
    elif disable_curriculum or resume_from_checkpoint:
        if resume_from_checkpoint:
            logger.info("Curriculum disabled (resuming from checkpoint - gate should already be trained)")
        else:
            logger.info("Curriculum disabled by flag")
        
        # Set model to final curriculum state (fully open gate, max K, max cap)
        if hasattr(model, 'base_k'):
            model.base_k = 64  # Final K value
            logger.info(f"Set model.base_k to final value: 64")
        
        if hasattr(model, 'max_vision_tokens'):
            model.max_vision_tokens = 193  # 1 global + 64 pooled + 128 random = 193 total
            model.allow_extra_tokens = True  # Enable extras
            logger.info(f"Set model.max_vision_tokens to final value: 193 (1 global + 64 pooled + 128 random)")
        
        if hasattr(model, 'use_soft_gate') and model.use_soft_gate:
            if hasattr(model, 'extra_gate'):
                with torch.no_grad():
                    model.extra_gate.data.fill_(1.5)  # Final gate value
                logger.info(f"Set model.extra_gate to final value: 1.5")
    elif enable_vision_pooling:  # Only add curriculum if pooling is enabled
        trainer.add_callback(PercentCurriculum(
            start_k=16, end_k=64, gate_max=1.5,
            start_cap=64, mid_cap=128, end_cap=193,  # End at 193 for 128 random tokens
            stage1_pct=0.15, ramp_end_pct=0.50
        ))
        logger.info("Added curriculum callback: Stage 1 (0-15%), Stage 2 ramp (15-50%), Stage 3 (50-100%)")
        logger.info("Vision token cap will ramp: 64 → 128 → 193 (final: 1 global + 64 pooled + 128 random)")

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

        # Train - the AdapterOnlyTrainer will handle checkpoint loading properly
        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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
