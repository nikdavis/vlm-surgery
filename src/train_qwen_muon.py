#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-VL-7B for OCR structured output extraction using Muon optimizer.
Following best practices from CLAUDE.md.
"""

import os
from pathlib import Path
import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import sys
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import click
from muon import MuonWithAuxAdam

from src.expanded_dataset_loader import ExpandedOCRDataset
from src.cot_dataset_loader import OCRCoTDataset
from src.unified_dataset_loader import UnifiedOCRDataset


LORA_RANK = 16

def setup_model(model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct", load_in_4bit: bool = True):
    """Load and prepare Qwen2.5-VL model with LoRA."""
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
        finetune_attention_modules=True,
        finetune_mlp_modules=True,

        r=LORA_RANK,  # Rank - higher for more complex tasks
        lora_alpha=LORA_RANK,  # Usually same as r
        lora_dropout=0.05,  # Small dropout for regularization
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


@click.command()
@click.option("--model", default="unsloth/Qwen2.5-VL-7B-Instruct", help="Model to fine-tune")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("outputs_qwen"), help="Output directory")
@click.option("--cot", is_flag=True, help="Use CoT dataset for training")
@click.option("--batch-size", type=int, default=1, help="Batch size per device")
@click.option("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
@click.option("--learning-rate", type=float, default=1e-4, help="Learning rate")
@click.option("--num-epochs", type=int, default=5, help="Number of epochs")
@click.option("--max-steps", type=int, default=None, help="Max steps (overrides epochs)")
@click.option("--warmup-steps", type=int, default=20, help="Warmup steps")
@click.option("--logging-steps", type=int, default=5, help="Logging frequency")
@click.option("--save-steps", type=int, default=10, help="Save checkpoint frequency")
@click.option("--val-ratio", type=float, default=0.15, help="Validation set ratio")
@click.option("--eval-steps", type=int, default=10, help="Eval frequency (set same as save-steps for eval at checkpoints)")
@click.option("--patience", type=int, default=3, help="Early stopping patience (number of evals without improvement)")
@click.option("--seed", type=int, default=3407, help="Random seed")
@click.option("--no-4bit", is_flag=True, help="Disable 4bit quantization")
@click.option("--muon-lr", type=float, default=0.02, help="Learning rate for Muon optimizer (LoRA weights)")
@click.option("--adamw-lr", type=float, default=3e-4, help="Learning rate for AdamW optimizer (other params)")
@click.option("--resume-from-checkpoint", type=click.Path(exists=True, path_type=Path), default=None, help="Resume training from checkpoint")
@click.option("--mlflow-run-id", type=str, default=None, help="Resume specific MLflow run ID")
def main(model, output_dir, cot, batch_size, gradient_accumulation, learning_rate,
         num_epochs, max_steps, warmup_steps, logging_steps, save_steps,
         val_ratio, eval_steps, patience, seed, no_4bit, muon_lr, adamw_lr, resume_from_checkpoint, mlflow_run_id):
    """Fine-tune Qwen2.5-VL for OCR structured output extraction."""

    # Set up paths
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # If resuming with a specific run ID, use its experiment
    if mlflow_run_id:
        # Get the run's experiment from MLflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(mlflow_run_id)
        mlflow.set_experiment(experiment_id=run.info.experiment_id)
        print(f"Using experiment ID {run.info.experiment_id} from run {mlflow_run_id}")
    else:
        mlflow.set_experiment("qwen-ocr-muon")

    # Load dataset
    if cot:
        print("\nLoading unified dataset from datasetv2/")
        full_dataset = UnifiedOCRDataset("datasetv2/combined_dataset.json", enable_cot=True)
    else:
        print("\nLoading expanded dataset from outputs-synth/expanded_dataset.json")
        full_dataset = ExpandedOCRDataset()

    if len(full_dataset) == 0:
        print("Error: No training examples found!")
        return

    # Split into train/val
    train_dataset, val_dataset = full_dataset.get_train_val_split(val_ratio)
    print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")

    # Load model
    model, tokenizer = setup_model(model, load_in_4bit=not no_4bit)

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
    if max_steps is not None:
        training_config = {
            "max_steps": max_steps,
            # Don't set num_train_epochs when using max_steps
        }
    else:
        training_config = {
            "num_train_epochs": num_epochs,
        }

    # Create run name
    run_name = f"Qwen-OCR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    training_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",  # Switched back from cosine to linear to disable cosine tapering
        seed=seed,
        output_dir=str(output_dir),
        report_to="mlflow",  # Enable MLflow logging
        run_name=run_name,

        # Validation and early stopping
        do_eval=True,
        eval_strategy="steps",  # Note: eval_strategy not evaluation_strategy
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
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

    # Create parameter groups for Muon optimizer
    # Get only trainable parameters (LoRA weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # LoRA matrices (A and B) are 2D, so they are the 'hidden_weights'
    lora_weights = [p for p in trainable_params if p.ndim >= 2]

    # Other parameters (e.g., biases if any) are 1D
    other_trainable_params = [p for p in trainable_params if p.ndim < 2]

    print(f"\nParameter grouping for Muon optimizer:")
    print(f"  - LoRA weight tensors (for Muon): {len(lora_weights)}")
    print(f"  - Other trainable tensors (for AdamW): {len(other_trainable_params)}")

    # Create parameter groups
    param_groups = [
        # Group for LoRA weights, to be optimized by Muon
        dict(
            params=lora_weights,
            use_muon=True,
            lr=muon_lr,
            weight_decay=0.01
        ),
        # Group for any other trainable params, to be optimized by AdamW
        dict(
            params=other_trainable_params,
            use_muon=False,
            lr=adamw_lr,
            betas=(0.9, 0.95),
            weight_decay=0.01
        ),
    ]

    # Create the Muon optimizer
    optimizer = MuonWithAuxAdam(param_groups)

    # Create trainer with custom optimizer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add validation dataset
        data_collator=UnslothVisionDataCollator(model, tokenizer, max_seq_length=8192, resize="max"),
        args=training_args,
        optimizers=(optimizer, None),  # Pass custom optimizer, None for scheduler
    )

    # Train with MLflow tracking
    print("\nStarting training...")

    # If resuming, try to find and resume the existing MLflow run
    if resume_from_checkpoint:
        # Extract run name from checkpoint path if possible
        checkpoint_name = Path(resume_from_checkpoint).name
        print(f"Resuming from checkpoint: {checkpoint_name}")

        # Check if run_id was saved in checkpoint (for future runs)
        if not mlflow_run_id:
            run_id_file = Path(resume_from_checkpoint) / "mlflow_run_id.txt"
            if run_id_file.exists():
                mlflow_run_id = run_id_file.read_text().strip()
                print(f"Found MLflow run ID in checkpoint: {mlflow_run_id}")

    # Start or resume MLflow run
    if mlflow_run_id:
        # Use the provided or found run ID
        mlflow.start_run(run_id=mlflow_run_id)
        print(f"Resuming MLflow run: {mlflow_run_id}")
    else:
        mlflow.start_run(run_name=run_name)

    with mlflow.active_run() as run:
        # Log initial parameters
        mlflow.log_params({
            "optimizer": "MuonWithAuxAdam",
            "muon_lr": muon_lr,
            "adamw_lr": adamw_lr,
            "lora_weights_count": len(lora_weights),
            "other_params_count": len(other_trainable_params),
            "lora_r": LORA_RANK,  # Updated to match actual value
            "lora_alpha": LORA_RANK,  # Updated to match actual value
            "lora_dropout": 0.05,
            "finetune_vision_layers": True,
            "finetune_language_layers": True,
            "finetune_attention_modules": False,
            "finetune_mlp_modules": True,
            "model_name": model,
            "load_in_4bit": not no_4bit,
            "dataset_size": len(full_dataset),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        })

        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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
        print(f"\nSaving best model to {output_dir}/final_model")
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"Best eval loss: {trainer.state.best_metric:.4f}")
        model.save_pretrained(f"{output_dir}/final_model")
        tokenizer.save_pretrained(f"{output_dir}/final_model")

        # Log the final model artifacts
        mlflow.log_artifacts(f"{output_dir}/final_model", "model")

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
