#!/usr/bin/env python3
"""
Train the InternVL3-DeepSeekR1 hybrid model using a masked loss for CoT.
"""
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from typing import List, Dict, Any
import click
from accelerate import init_empty_weights


# Use relative imports because all our code is in the 'surgery' module
from src.surgery.hybrid_model import InternVL3DeepSeekR1Hybrid
from src.unified_dataset_loader import UnifiedOCRDataset

class CoTDataCollator:
    """Handles vision inputs and creates masked labels for CoT training."""
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.think_start_id = tokenizer.convert_tokens_to_ids("<thk>")
        self.think_end_id = tokenizer.convert_tokens_to_ids("</thk>")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.randn(len(features), 3, 448, 448) # Placeholder

        texts_for_tokenization = []
        for feature in features:
            prompt = feature["messages"][0]["content"][1]["text"]
            response = feature["messages"][1]["content"][0]["text"]

            # Use the tokenizer's chat template to format the full input string
            # This correctly handles all special tokens like <|im_start|>, etc.
            # We replace a placeholder string with the special image marker token.
            formatted_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f"<IMAGE_PLACEHOLDER> {prompt}"},
                    {"role": "assistant", "content": response}
                ],
                tokenize=False, add_generation_prompt=False
            )
            texts_for_tokenization.append(formatted_text)

        # Tokenize the full batch at once
        tokenized = self.tokenizer(texts_for_tokenization, padding=True, truncation=True, max_length=4096, return_tensors="pt")

        # Replace the placeholder ID with the actual <IMAGE> marker ID used by the model
        placeholder_id = self.tokenizer.convert_tokens_to_ids("<IMAGE_PLACEHOLDER>")
        image_marker_id = 20006 # The model uses ' unused'
        tokenized['input_ids'][tokenized['input_ids'] == placeholder_id] = image_marker_id

        labels = tokenized.input_ids.clone()

        for i in range(len(features)):
            # Find where the assistant's response begins to mask everything before it
            assistant_prompt = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            input_id_list = tokenized.input_ids[i].tolist()

            try:
                assistant_start_index = input_id_list.index(assistant_prompt[0])
                labels[i, :assistant_start_index + len(assistant_prompt)] = -100
            except ValueError:
                continue # Skip if assistant prompt not found (should not happen)

            # Mask the thinking block
            think_starts = (labels[i] == self.think_start_id).nonzero(as_tuple=True)[0]
            think_ends = (labels[i] == self.think_end_id).nonzero(as_tuple=True)[0]

            if len(think_starts) > 0 and len(think_ends) > 0:
                labels[i, think_starts[0] : think_ends[0] + 1] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100

        return { "pixel_values": pixel_values, **tokenized, "labels": labels }

@click.command()
@click.option("--output-dir", default="./outputs_internvl_r1", help="Output directory")
@click.option("--data-path", default="./datasetv2/combined_dataset.json", help="Path to the unified dataset JSON file.")
@click.option("--batch-size", type=int, default=1, help="Batch size per device")
@click.option("--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps")
@click.option("--learning-rate", type=float, default=1e-5, help="Learning rate for the adapter")
@click.option("--num-epochs", type=int, default=1, help="Number of epochs")
def main(output_dir, data_path, batch_size, gradient_accumulation, learning_rate, num_epochs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with init_empty_weights():
        model = InternVL3DeepSeekR1Hybrid()
    tokenizer = model.tokenizer

    # Add special tokens
    special_tokens = ["<thk>", "</thk>", "<IMAGE_PLACEHOLDER>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.language_model.resize_token_embeddings(len(tokenizer))

    full_dataset = UnifiedOCRDataset(data_path, enable_cot=True)
    train_dataset, val_dataset = full_dataset.get_train_val_split(val_ratio=0.02)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        do_train=True,
        do_eval=True,
        remove_unused_columns=False,

        # --- RE-ENABLE FSDP HERE ---
        fsdp="full_shard auto_wrap",

        # --- PROVIDE THE CONFIG FOR THE AUTO WRAP POLICY ---
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_offload_params": False,  # Keep parameters on GPU
            "fsdp_cpu_ram_efficient_loading": True, # This is the key!
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CoTDataCollator(tokenizer),
    )

    trainer.train()
    model.save_pretrained(str(output_dir / "final_model"))

if __name__ == "__main__":
    main()
