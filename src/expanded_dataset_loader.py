"""
Dataset loader for pre-expanded synthetic data from outputs-synth.
"""

import json
from pathlib import Path
from typing import List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


class ExpandedOCRDataset(Dataset):
    """Load pre-expanded OCR examples from outputs-synth."""
    
    def __init__(self, expanded_file: Path = Path("outputs-synth/expanded_dataset.json")):
        """Load the expanded dataset."""
        with open(expanded_file, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)
        
        print(f"Loaded {len(self.examples)} expanded examples from {expanded_file}")
        
        # Count examples with/without OCR
        with_ocr = sum(1 for ex in self.examples if ex.get('ocr_text', ''))
        print(f"  - {with_ocr} with OCR text")
        print(f"  - {len(self.examples) - with_ocr} without OCR text")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get an example by index."""
        example = self.examples[idx]
        
        # Load image
        image = Image.open(example["image_path"]).convert("RGB")
        
        # Build the prompt - only include OCR section if we have OCR text
        prompt = example["prompt"]
        if example.get("ocr_text", ""):
            full_prompt = f"{prompt}\n\nOCR Text:\n{example['ocr_text']}"
        else:
            full_prompt = prompt
        
        # Format for training
        formatted_example = {
            "id": example["id"],
            "images": [image],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": full_prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example["output"]}
                    ]
                }
            ]
        }
        
        return formatted_example
    
    def get_train_val_split(self, val_ratio: float = 0.15, seed: int = 42):
        """Split dataset into train and validation sets with random sampling."""
        # Set random seed for consistent splits
        random.seed(seed)
        
        # Create a copy of all examples and shuffle
        all_examples = self.examples.copy()
        random.shuffle(all_examples)
        
        # Calculate split point
        n_val = max(1, int(len(all_examples) * val_ratio))
        n_train = len(all_examples) - n_val
        
        # Split into train and validation
        train_examples = all_examples[:n_train]
        val_examples = all_examples[n_train:]
        
        # Create new dataset instances
        train_dataset = ExpandedOCRDataset.__new__(ExpandedOCRDataset)
        train_dataset.examples = train_examples
        
        val_dataset = ExpandedOCRDataset.__new__(ExpandedOCRDataset)
        val_dataset.examples = val_examples
        
        # Print split statistics
        print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Count types in each split for visibility
        train_enc = len([e for e in train_examples if 'enc_brit' in e['id']])
        val_enc = len([e for e in val_examples if 'enc_brit' in e['id']])
        train_hn = len([e for e in train_examples if 'hackernews' in e['id']])
        val_hn = len([e for e in val_examples if 'hackernews' in e['id']])
        train_other = len(train_examples) - train_enc - train_hn
        val_other = len(val_examples) - val_enc - val_hn
        
        print(f"  Encyclopedia: {train_enc} train, {val_enc} val")
        print(f"  Hackernews: {train_hn} train, {val_hn} val")
        if train_other + val_other > 0:
            print(f"  Other: {train_other} train, {val_other} val")
        
        return train_dataset, val_dataset