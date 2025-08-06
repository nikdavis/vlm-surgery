"""
Dataset loader for pre-formatted CoT data from data-cot directory.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset


class OCRCoTDataset(Dataset):
    """Load CoT examples directly from data-cot directory."""
    
    def __init__(self, data_dir: str = "data-cot"):
        self.data_dir = Path(data_dir)
        self.examples = self._load_examples()
        
        if not self.examples:
            raise ValueError(f"No examples found in {data_dir}")
            
        print(f"Loaded {len(self.examples)} CoT examples from {data_dir}")
    
    def _load_examples(self) -> List[Dict]:
        """Load all examples from data-cot directory."""
        examples = []
        
        if not self.data_dir.exists():
            print(f"Warning: {self.data_dir} does not exist")
            return examples
        
        for example_dir in self.data_dir.iterdir():
            if not example_dir.is_dir():
                continue
                
            stem = example_dir.name
            
            # Find required files
            cot_path = example_dir / f"{stem}_cot.txt"
            prompt_path = example_dir / f"{stem}_prompt.txt"
            
            # Skip if missing CoT or prompt
            if not cot_path.exists() or not prompt_path.exists():
                continue
            
            # Find image file
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                candidate = example_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if not image_path:
                print(f"Warning: No image found for {stem}")
                continue
            
            # Find output file (XML or JSON)
            output_path = None
            output_format = None
            for ext, fmt in [('_done.xml', 'xml'), ('_done.json', 'json')]:
                candidate = example_dir / f"{stem}{ext}"
                if candidate.exists():
                    output_path = candidate
                    output_format = fmt
                    break
            
            if not output_path:
                print(f"Warning: No output file found for {stem}")
                continue
            
            # Read all files
            with open(cot_path, 'r', encoding='utf-8') as f:
                cot_text = f.read().strip()
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            with open(output_path, 'r', encoding='utf-8') as f:
                output = f.read().strip()
            
            # OCR is optional
            ocr_text = ""
            ocr_path = example_dir / f"{stem}_ocr.txt"
            if ocr_path.exists():
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read().strip()
            
            # Add only CoT version
            examples.append({
                "id": f"{stem}_cot",
                "image_path": str(image_path),
                "prompt": prompt,
                "cot_text": cot_text,
                "output": output,
                "ocr_text": ocr_text,
                "format": output_format,
                "has_cot": True
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a single example, with or without CoT based on has_cot flag."""
        example = self.examples[idx]
        
        # Read and prepare image
        image = Image.open(example["image_path"]).convert("RGB")
        
        # All examples now have CoT
        # WITH CoT: add "think" and format with <thk> tags
        prompt = self._add_think_to_prompt(example["prompt"], idx)
        output = f"<thk>\n{example['cot_text']}\n</thk>\n{example['output']}"
        
        # Format the full prompt with OCR if available
        if example.get("ocr_text", ""):
            full_prompt = f"{prompt}\n\nOCR Text:\n{example['ocr_text']}"
        else:
            full_prompt = prompt
        
        # Format as messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image", "image": image}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": output}
                ]
            }
        ]
        
        return {
            "id": example["id"],
            "messages": messages,
            "images": [image]
        }
    
    def _add_think_to_prompt(self, prompt: str, seed: int) -> str:
        """Randomly add 'think' to the beginning or end of prompt."""
        random.seed(seed)
        position = random.choice(['start', 'end'])
        
        if position == 'start':
            return f"think\n{prompt}"
        else:
            return f"{prompt}\nthink"
    
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
        train_dataset = OCRCoTDataset.__new__(OCRCoTDataset)
        train_dataset.data_dir = self.data_dir
        train_dataset.examples = train_examples
        
        val_dataset = OCRCoTDataset.__new__(OCRCoTDataset)
        val_dataset.data_dir = self.data_dir
        val_dataset.examples = val_examples
        
        print(f"Split CoT dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_dataset, val_dataset