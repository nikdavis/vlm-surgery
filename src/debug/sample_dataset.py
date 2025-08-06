#!/usr/bin/env python3
"""
Sample a random example from the unified dataset to see exactly what the LLM receives.
Shows the formatted messages and verifies image files exist.
"""

import json
import random
from pathlib import Path
from src.unified_dataset_loader import UnifiedOCRDataset


def format_message_content(content):
    """Format message content for display."""
    formatted = []
    for item in content:
        if item['type'] == 'text':
            # Just show the text without the [TEXT] label
            formatted.append(item['text'])
        elif item['type'] == 'image':
            formatted.append("[Image provided to model]")
    return "\n\n".join(formatted)


def verify_images(image_paths):
    """Verify all image files exist and are readable."""
    results = []
    for img_path in image_paths:
        path = Path(img_path)
        if path.exists():
            size = path.stat().st_size
            results.append(f"✓ {path.name} ({size:,} bytes)")
        else:
            results.append(f"✗ {path.name} (NOT FOUND)")
    return results


def main():
    print("Loading dataset...")
    dataset = UnifiedOCRDataset("datasetv2/combined_dataset.json", enable_cot=True)
    print(f"Total examples: {len(dataset)}\n")
    
    # Get a random example
    idx = random.randint(0, len(dataset) - 1)
    example = dataset[idx]
    
    print("="*80)
    print(f"RANDOM SAMPLE: Example {idx} - {example['id']}")
    print("="*80)
    
    # Show what the LLM sees
    messages = example['messages']
    
    print("\n### USER MESSAGE ###")
    print(format_message_content(messages[0]['content']))
    
    print("\n### ASSISTANT RESPONSE ###")
    print(format_message_content(messages[1]['content']))
    
    # Verify images
    print("\n### IMAGE VERIFICATION ###")
    # Get the raw example to access image paths
    raw_example = dataset.examples[idx]
    image_results = verify_images(raw_example['images'])
    for result in image_results:
        print(result)
    
    # Show metadata if available
    if raw_example.get('metadata'):
        print("\n### METADATA ###")
        print(json.dumps(raw_example['metadata'], indent=2))
    
    # Show response breakdown
    response = raw_example['response']
    print("\n### RESPONSE BREAKDOWN ###")
    print(f"Format: {response.get('format', 'unknown')}")
    print(f"Has thinking: {'Yes' if response.get('thinking') else 'No'}")
    if response.get('thinking'):
        print(f"Thinking length: {len(response['thinking'])} chars")
    print(f"Output length: {len(response.get('output', ''))} chars")
    
    # Show data source
    if raw_example.get('provenance'):
        print("\n### PROVENANCE ###")
        print(f"Source: {raw_example['provenance'].get('source', 'unknown')}")
        if raw_example['provenance'].get('original_dataset'):
            print(f"Original dataset: {raw_example['provenance']['original_dataset']}")


if __name__ == "__main__":
    main()