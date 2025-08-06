# Dataset Documentation

This document describes the dataset structure, formats, and procedures for adding new data to the fine-tuning pipeline.

## Overview

Our training pipeline uses a unified dataset format that supports multiple types of vision-language tasks:
- OCR structured extraction (JSON/XML output)
- Visual reasoning with Chain-of-Thought (CoT)
- General visual question answering

## Dataset Structure

### Unified Dataset Format

All datasets are converted to a unified JSON format stored in `datasetv2/combined_dataset.json`. The schema is defined in `src/models/dataset.py` using Pydantic models.

```json
{
  "version": "1.0",
  "name": "Dataset Name",
  "description": "Dataset description",
  "created_at": "ISO timestamp",
  "examples": [
    {
      "id": "unique_example_id",
      "images": ["/absolute/path/to/image.jpg"],
      "prompt": "User question or instruction",
      "schema_": {  // Optional, for structured extraction
        "type": "json|xml",
        "definition": "JSON Schema or XSD"
      },
      "ocr_text": "Pre-extracted OCR text (optional)",
      "response": {
        "thinking": "Chain of thought reasoning (optional)",
        "output": "Final answer or structured output",
        "format": "text|json|xml"
      },
      "metadata": {
        "ground_truth": "Expected answer",
        "model_used": "Model that generated response",
        "response_time": 12.34,
        "processed_at": "ISO timestamp",
        "tags": ["ocr", "cot", "visual-reasoning"],
        "notes": "Additional notes"
      },
      "provenance": {
        "source": "manual|synthetic|ocr|converted",
        "original_format": "Original dataset format",
        "conversion_notes": "Conversion details"
      }
    }
  ]
}
```

### Physical Data Layout

Images are stored in their original dataset directories:

```
fine-pixtral/
├── datasetv2/                  # Unified JSON datasets
│   ├── combined_dataset.json   # Main training dataset
│   ├── dataset.json           # Legacy OCR examples
│   ├── test_data_200_unified.json
│   └── cot_unified.json
├── data-cot/                   # Legacy OCR dataset images
│   ├── example_1/
│   │   ├── example_1.jpg
│   │   ├── example_1_prompt.txt
│   │   └── example_1_done.xml
│   └── ...
├── test_data_200/              # Visual reasoning dataset
│   ├── example_001/
│   │   ├── problem_image.jpg
│   │   ├── reasoning_image.jpg  # Not used in training
│   │   ├── data.json
│   │   └── cot_response.json
│   └── ...
└── cot/                        # New CoT batch (1000 examples)
    ├── example_0001/
    │   ├── problem_image.jpg
    │   ├── data.json
    │   └── cot_response.json
    └── ...
```

## Dataset Interface

### Core Components

1. **Schema Definition**: `src/models/dataset.py`
   - Defines Pydantic models for all dataset components
   - Ensures data validation and consistency
   - Key classes: `DatasetExample`, `Response`, `Metadata`, `Provenance`

2. **Dataset Loader**: `src/unified_dataset_loader.py`
   - `UnifiedOCRDataset` class that implements PyTorch Dataset interface
   - Handles image loading with automatic resizing for large images (>2048px)
   - Supports Docker path remapping for containerized training
   - Formats data for Qwen/Pixtral models with proper message structure

3. **Training Integration**: `src/train_qwen_muon.py`
   ```python
   full_dataset = UnifiedOCRDataset("datasetv2/combined_dataset.json", enable_cot=True)
   ```

## Adding New Datasets

### Step 1: Prepare Your Data

Your dataset should have:
- Images (JPEG, PNG supported)
- Questions/prompts
- Answers (with optional CoT reasoning)
- Any metadata (ground truth, model used, etc.)

### Step 2: Create Conversion Script

Create a conversion script following this template:

```python
#!/usr/bin/env python3
"""Convert [dataset_name] to unified format."""

import json
from pathlib import Path
from datetime import datetime
from src.models.dataset import DatasetExample, Response, Metadata, Provenance, DataSource

def convert_example(raw_example, base_path):
    """Convert a single example to unified format."""
    
    # Extract image path(s) - use absolute paths
    image_path = (base_path / raw_example['image_file']).absolute()
    
    # Clean CoT if present (remove <thk> tags)
    thinking = clean_cot_text(raw_example.get('cot', ''))
    
    # Create unified example
    example = DatasetExample(
        id=raw_example['id'],
        images=[str(image_path)],
        prompt=raw_example['question'],
        response=Response(
            thinking=thinking,  # Optional CoT
            output=raw_example['answer'],
            format="text"  # or "json"/"xml"
        ),
        metadata=Metadata(
            ground_truth=raw_example.get('ground_truth'),
            model_used=raw_example.get('model'),
            tags=["your-tags-here"]
        ),
        provenance=Provenance(
            source=DataSource.SYNTHETIC,
            original_format="your_format"
        )
    )
    
    return example.model_dump(mode='json')

# See convert_test_data_200.py or convert_cot_dataset.py for full examples
```

### Step 3: Run Conversion

```bash
uv run python convert_[dataset_name].py
```

This creates `datasetv2/[dataset_name]_unified.json`

### Step 4: Combine Datasets

Update `combine_datasets.py` to include your new dataset:

```python
# Load your dataset
with open("datasetv2/your_dataset_unified.json", 'r') as f:
    your_data = json.load(f)
your_examples = your_data['examples']

# Add to combination
all_examples = legacy_examples + test_examples + cot_examples + your_examples
```

Run: `uv run python combine_datasets.py`

### Step 5: Update Docker Mounts

If your images are in a new directory, add it to `docker-compose.train.yml`:

```yaml
volumes:
  - ./your_image_dir:/app/your_image_dir  # Your dataset images
```

### Step 6: Verify Dataset

```bash
# Check images are valid
uv run python check_images.py

# Verify dataset integrity
uv run python verify_dataset.py

# Sample random examples
uv run python sample_dataset.py
```

## Important Notes

### Image Handling

1. **Always use absolute paths** in the dataset JSON
2. **Large images** (>2048px) are automatically resized during training
3. **Multiple images** per example are supported but rarely used
4. **Only problem images** are used (not reasoning/solution images)

### Chain of Thought

- CoT text should have `<thk>` tags removed before storage
- The training pipeline adds `<thk>` tags automatically when formatting
- Store clean reasoning text in `response.thinking`

### Docker Compatibility

The dataset loader automatically remaps paths when running in Docker:
- Local: `/home/nik/code/langgraph-work-dir/fine-pixtral/`
- Docker: `/app/`

### Quality Control

- All images must exist and be readable
- Empty prompts/outputs are not allowed
- JSON/XML outputs should match their declared format
- Use meaningful example IDs for debugging

## Utility Scripts

- `sample_dataset.py` - Randomly sample and display dataset examples
- `verify_dataset.py` - Comprehensive dataset validation
- `check_images.py` - Verify all images are valid and check sizes
- `convert_test_data_200.py` - Example conversion script
- `convert_cot_dataset.py` - Example for CoT data
- `combine_datasets.py` - Merge multiple datasets

## Current Dataset Statistics

As of the last update:
- **Total examples**: 1266
- **Sources**: 
  - Legacy OCR (data-cot): 66 examples
  - test_data_200: 200 examples
  - New CoT batch: 1000 examples
- **All examples have CoT reasoning**
- **Output formats**: 1200 text, 33 JSON, 33 XML