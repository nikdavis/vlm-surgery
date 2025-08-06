"""
Unified dataset loader that supports multiple data formats.
"""

import json
import os
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset


class UnifiedOCRDataset(Dataset):
    """
    Unified dataset loader supporting:
    - Single JSON files with all data
    - Directory structure with data.json
    - Legacy data-cot format (separate files)
    """
    
    def __init__(self, data_paths: Union[str, List[str]], enable_cot: bool = True):
        """
        Args:
            data_paths: Single path or list of paths to data directories/files
            enable_cot: Whether to include CoT in training (can be toggled)
        """
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        self.data_paths = [Path(p) for p in data_paths]
        self.enable_cot = enable_cot
        self.examples = self._load_all_examples()
        
        if not self.examples:
            raise ValueError(f"No examples found in {data_paths}")
            
        print(f"Loaded {len(self.examples)} examples")
        self._print_stats()
    
    def _load_all_examples(self) -> List[Dict]:
        """Load examples from all provided paths."""
        examples = []
        
        for path in self.data_paths:
            if path.is_file() and path.suffix == '.json':
                # Single JSON file
                examples.extend(self._load_json_file(path))
            elif path.is_dir():
                # Directory - check for different formats
                examples.extend(self._load_directory(path))
            else:
                print(f"Warning: Skipping {path} - not a file or directory")
        
        return examples
    
    def _load_json_file(self, json_path: Path) -> List[Dict]:
        """Load examples from a single JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, dict):
            # Check if it's a dataset format with examples array
            if 'examples' in data and isinstance(data['examples'], list):
                return [self._normalize_example(ex, json_path.parent) for ex in data['examples']]
            else:
                # Single example
                return [self._normalize_example(data, json_path.parent)]
        elif isinstance(data, list):
            # List of examples
            return [self._normalize_example(ex, json_path.parent) for ex in data]
        else:
            return []
    
    def _load_directory(self, dir_path: Path) -> List[Dict]:
        """Load examples from a directory."""
        examples = []
        
        # Check for unified format (data.json files)
        for data_file in dir_path.glob("**/data.json"):
            examples.extend(self._load_json_file(data_file))
        
        # Check for legacy data-cot format
        if not examples:  # Only if no data.json files found
            examples.extend(self._load_legacy_cot_format(dir_path))
        
        return examples
    
    def _load_legacy_cot_format(self, dir_path: Path) -> List[Dict]:
        """Load examples in legacy data-cot format."""
        examples = []
        
        for example_dir in dir_path.iterdir():
            if not example_dir.is_dir():
                continue
            
            stem = example_dir.name
            
            # Find required files
            cot_path = example_dir / f"{stem}_cot.txt"
            prompt_path = example_dir / f"{stem}_prompt.txt"
            
            if not prompt_path.exists():
                continue
            
            # Find image
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                candidate = example_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if not image_path:
                continue
            
            # Find output file
            output_path = None
            output_format = "text"
            for ext, fmt in [('_done.xml', 'xml'), ('_done.json', 'json')]:
                candidate = example_dir / f"{stem}{ext}"
                if candidate.exists():
                    output_path = candidate
                    output_format = fmt
                    break
            
            if not output_path:
                continue
            
            # Read files
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            with open(output_path, 'r', encoding='utf-8') as f:
                output = f.read().strip()
            
            # CoT is optional
            thinking = ""
            if cot_path.exists():
                with open(cot_path, 'r', encoding='utf-8') as f:
                    thinking = f.read().strip()
            
            # OCR is optional
            ocr_text = ""
            ocr_path = example_dir / f"{stem}_ocr.txt"
            if ocr_path.exists():
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read().strip()
            
            # Convert to unified format
            example = {
                "id": stem,
                "images": [str(image_path)],
                "prompt": prompt,
                "ocr_text": ocr_text,
                "response": {
                    "thinking": thinking,
                    "output": output,
                    "format": output_format
                }
            }
            
            examples.append(self._normalize_example(example, example_dir))
        
        return examples
    
    def _normalize_example(self, example: Dict, base_path: Path) -> Dict:
        """Normalize example to ensure consistent format."""
        # Ensure all required fields exist
        normalized = {
            "id": example.get("id", base_path.name),
            "images": example.get("images", []),
            "prompt": example.get("prompt", ""),
            "schema": example.get("schema", None),
            "ocr_text": example.get("ocr_text", ""),
            "response": example.get("response", {}),
            "metadata": example.get("metadata", {})
        }
        
        # Handle image paths - convert relative to absolute
        abs_images = []
        for img_path in normalized["images"]:
            if not Path(img_path).is_absolute():
                img_path = base_path / img_path
            abs_images.append(str(img_path))
        normalized["images"] = abs_images
        
        # Ensure response has required fields
        if "output" not in normalized["response"]:
            normalized["response"]["output"] = ""
        if "thinking" not in normalized["response"]:
            normalized["response"]["thinking"] = ""
        if "format" not in normalized["response"]:
            # Try to detect format
            output = normalized["response"]["output"]
            if output.strip().startswith("<") and output.strip().endswith(">"):
                normalized["response"]["format"] = "xml"
            elif output.strip().startswith("{") and output.strip().endswith("}"):
                normalized["response"]["format"] = "json"
            else:
                normalized["response"]["format"] = "text"
        
        return normalized
    
    def _extract_between_tags(self, text: str, tag: str) -> str:
        """Extract content between <tag> and </tag>."""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _print_stats(self):
        """Print dataset statistics."""
        # Count by response type
        has_cot = sum(1 for ex in self.examples if ex["response"]["thinking"])
        has_schema = sum(1 for ex in self.examples if ex["schema"])
        formats = {}
        for ex in self.examples:
            fmt = ex["response"]["format"]
            formats[fmt] = formats.get(fmt, 0) + 1
        
        print(f"Dataset stats:")
        print(f"  - Examples with CoT: {has_cot}/{len(self.examples)}")
        print(f"  - Examples with schema: {has_schema}/{len(self.examples)}")
        print(f"  - Output formats: {formats}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a single training example."""
        example = self.examples[idx]
        
        # Load images
        images = []
        for img_path in example["images"]:
            # Handle absolute paths that need remapping for Docker
            if img_path.startswith("/home/") and os.path.exists("/app/"):
                # We're in Docker - remap the path
                img_path = img_path.replace("/home/nik/code/langgraph-work-dir/fine-pixtral/", "/app/")
            
            img = Image.open(img_path).convert("RGB")
            
            # Resize very large images to prevent token mismatch
            # Qwen typically handles up to 2048x2048 well
            max_size = 2048
            if img.width > max_size or img.height > max_size:
                # Keep aspect ratio
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            images.append(img)
        
        # Build prompt
        prompt = example["prompt"]
        
        # Add schema if present
        if example.get("schema"):
            schema_def = example["schema"]["definition"]
            if example["schema"]["type"] == "json":
                prompt = f"{prompt}\n\n{schema_def}"
            else:  # xml
                prompt = f"{prompt}\n\n{schema_def}"
        
        # Add OCR if present
        if example.get("ocr_text"):
            prompt = f"{prompt}\n\nOCR Text:\n{example['ocr_text']}"
        
        # Build response
        response = example["response"]
        if self.enable_cot and response.get("thinking"):
            # Use native thinking tokens when CoT is enabled
            output = f"<think>\n{response['thinking']}\n</think>\n{response['output']}"
        else:
            # When CoT is disabled, DON'T add thinking tokens - Qwen3 tokenizer adds them automatically
            # The tokenizer will add empty <think>\n\n</think>\n\n before the output
            output = response['output']
        
        # Format as messages - images MUST come before text for Qwen
        user_content = []
        for img in images:
            user_content.append({"type": "image"})  # Just placeholder, actual image in top-level
        user_content.append({"type": "text", "text": prompt})
        
        messages = [
            {
                "role": "user",
                "content": user_content
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
            "images": images
        }
    
    def get_train_val_split(self, val_ratio: float = 0.15, seed: int = 42):
        """Split dataset into train and validation sets."""
        random.seed(seed)
        
        # Shuffle examples
        all_examples = self.examples.copy()
        random.shuffle(all_examples)
        
        # Split
        n_val = max(1, int(len(all_examples) * val_ratio))
        n_train = len(all_examples) - n_val
        
        # Create new instances
        train_dataset = UnifiedOCRDataset.__new__(UnifiedOCRDataset)
        train_dataset.data_paths = self.data_paths
        train_dataset.enable_cot = self.enable_cot
        train_dataset.examples = all_examples[:n_train]
        
        val_dataset = UnifiedOCRDataset.__new__(UnifiedOCRDataset)
        val_dataset.data_paths = self.data_paths
        val_dataset.enable_cot = self.enable_cot
        val_dataset.examples = all_examples[n_train:]
        
        print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_dataset, val_dataset


# Convenience function to convert new format examples
def convert_new_cot_example(old_format: Dict, image_mappings: Dict[str, str]) -> Dict:
    """
    Convert from the new CoT format to unified format.
    
    Args:
        old_format: Dict with question, generated_response, etc.
        image_mappings: Dict mapping placeholder to actual image path
            e.g., {"[problem_image_1]": "problem_image.jpg"}
    """
    # Extract prompt without image placeholders
    prompt = old_format["question"]
    for placeholder, _ in image_mappings.items():
        prompt = prompt.replace(f"<image_start>{placeholder}<image_end>", "")
    prompt = prompt.strip()
    
    # Extract thinking and output from generated_response
    response_text = old_format["generated_response"]
    
    # Extract content between <thk> tags
    thinking_match = re.search(r'<thk>(.*?)</thk>', response_text, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else ""
    
    # Get output (everything after </thk>)
    if thinking_match:
        output = response_text[thinking_match.end():].strip()
    else:
        output = response_text
    
    # Detect output format
    if output.startswith("<") and ">" in output:
        output_format = "xml"
    else:
        output_format = "text"
    
    return {
        "id": old_format.get("id", "unknown"),
        "images": list(image_mappings.values()),
        "prompt": prompt,
        "response": {
            "thinking": thinking,
            "output": output,
            "format": output_format
        },
        "metadata": {
            "ground_truth": old_format.get("ground_truth"),
            "model_used": old_format.get("model_used"),
            "response_time": old_format.get("response_time"),
            "processed_at": old_format.get("processed_at")
        }
    }