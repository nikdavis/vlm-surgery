#!/usr/bin/env python3
"""
Fine-tune Pixtral for OCR structured output extraction.
Following best practices from CLAUDE.md.
"""

import os
import argparse
from pathlib import Path
import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import sys
from datetime import datetime
import mlflow

from src.expanded_dataset_loader import ExpandedOCRDataset
from src.cot_dataset_loader import OCRCoTDataset


def setup_model(model_name: str = "unsloth/Pixtral-12B-2409", load_in_4bit: bool = True):
    """Load and prepare Pixtral model with LoRA."""
    print(f"Loading model: {model_name}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",  # For memory efficiency
    )

    # Add LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        # Fine-tune both vision and language layers for OCR task
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=False,  # Save memory
        finetune_mlp_modules=True,

        r=16,  # Rank - higher for more complex tasks
        lora_alpha=16,  # Usually same as r
        lora_dropout=0.05,  # Small dropout for regularization
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Pixtral for OCR")
    parser.add_argument("--model", default="unsloth/Pixtral-12B-2409", help="Model to fine-tune")
    # parser.add_argument("--ready-dir", type=Path, default=Path("ready"), help="Directory with training examples")  # Not used anymore
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--cot", action="store_true", help="Use CoT dataset for training")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps (overrides epochs)")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--logging-steps", type=int, default=5, help="Logging frequency")
    parser.add_argument("--save-steps", type=int, default=10, help="Save checkpoint frequency")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--eval-steps", type=int, default=10, help="Eval frequency (set same as save-steps for eval at checkpoints)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (number of evals without improvement)")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4bit quantization")

    args = parser.parse_args()

    # Set up paths
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("pixtral-ocr-20250724")

    # Load dataset
    if args.cot:
        print("\nLoading CoT dataset from data-cot/")
        full_dataset = OCRCoTDataset("data-cot")
    else:
        print("\nLoading expanded dataset from outputs-synth/expanded_dataset.json")
        full_dataset = ExpandedOCRDataset()

    if len(full_dataset) == 0:
        print("Error: No training examples found!")
        return

    # Split into train/val
    train_dataset, val_dataset = full_dataset.get_train_val_split(args.val_ratio)
    print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")

    # Load model
    model, tokenizer = setup_model(args.model, load_in_4bit=not args.no_4bit)

    # Enable for training
    FastVisionModel.for_training(model)

    # Show memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    # Training arguments
    # Set up training config based on steps or epochs
    if args.max_steps is not None:
        training_config = {
            "max_steps": args.max_steps,
            # Don't set num_train_epochs when using max_steps
        }
    else:
        training_config = {
            "num_train_epochs": args.num_epochs,
        }

    # Create run name
    run_name = f"Pixtral-OCR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Changed from linear to cosine
        seed=args.seed,
        output_dir=str(args.output_dir),
        report_to="mlflow",  # Enable MLflow logging
        run_name=run_name,

        # Validation and early stopping
        do_eval=True,
        eval_strategy="steps",  # Note: eval_strategy not evaluation_strategy
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only best 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Vision model specific (MUST have these)
        max_seq_length=8192,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        **training_config,  # Add steps or epochs config
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add validation dataset
        data_collator=UnslothVisionDataCollator(model, tokenizer, max_seq_length=8192, resize="max"),
        args=training_args,
    )

    # Train with MLflow tracking
    print("\nStarting training...")
    with mlflow.start_run(run_name=run_name) as run:
        # Log initial parameters
        mlflow.log_params({
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "finetune_vision_layers": True,
            "finetune_language_layers": True,
            "finetune_attention_modules": False,
            "finetune_mlp_modules": True,
            "model_name": args.model,
            "load_in_4bit": not args.no_4bit,
            "dataset_size": len(full_dataset),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        })

        trainer_stats = trainer.train()

        # Show final stats
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

            print(f"\nTraining completed!")
            print(f"Time: {trainer_stats.metrics['train_runtime']:.1f} seconds")
            print(f"Peak memory: {used_memory} GB ({used_percentage}% of GPU)")
            print(f"Peak memory for LoRA: {used_memory_for_lora} GB ({lora_percentage}% of GPU)")

        # The best model is already loaded due to load_best_model_at_end=True
        print(f"\nSaving best model to {args.output_dir}/final_model")
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"Best eval loss: {trainer.state.best_metric:.4f}")
        model.save_pretrained(f"{args.output_dir}/final_model")
        tokenizer.save_pretrained(f"{args.output_dir}/final_model")

        # Log the final model artifacts
        mlflow.log_artifacts(f"{args.output_dir}/final_model", "model")

        # Log final training stats
        mlflow.log_metrics({
            "final_train_loss": trainer_stats.metrics.get('train_loss', 0),
            "total_train_time": trainer_stats.metrics.get('train_runtime', 0),
            "train_samples_per_second": trainer_stats.metrics.get('train_samples_per_second', 0),
        })

    print("\nTraining complete! To use the model, run inference.py")
    print(f"MLflow tracking at: http://localhost:5000")


if __name__ == "__main__":
    main()
