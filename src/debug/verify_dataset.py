#!/usr/bin/env python3
"""
Verify dataset integrity - check all images exist, data formats are correct, etc.
"""

import json
from pathlib import Path
from collections import defaultdict
from src.unified_dataset_loader import UnifiedOCRDataset


def verify_dataset(dataset_path="datasetv2/combined_dataset.json"):
    """Comprehensive dataset verification."""
    print(f"Verifying dataset: {dataset_path}\n")
    
    # Load dataset
    dataset = UnifiedOCRDataset(dataset_path, enable_cot=True)
    
    # Track issues
    issues = defaultdict(list)
    stats = {
        'total_examples': len(dataset),
        'missing_images': 0,
        'empty_prompts': 0,
        'empty_outputs': 0,
        'missing_thinking': 0,
        'format_issues': 0,
        'image_sizes': defaultdict(int),
        'output_formats': defaultdict(int),
        'sources': defaultdict(int),
    }
    
    print("Checking all examples...")
    for i, raw_example in enumerate(dataset.examples):
        example_id = raw_example['id']
        
        # Check images exist
        for img_path in raw_example.get('images', []):
            path = Path(img_path)
            if not path.exists():
                issues['missing_images'].append(f"{example_id}: {img_path}")
                stats['missing_images'] += 1
            else:
                # Track image size distribution
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb < 0.1:
                    stats['image_sizes']['<100KB'] += 1
                elif size_mb < 0.5:
                    stats['image_sizes']['100KB-500KB'] += 1
                elif size_mb < 1:
                    stats['image_sizes']['500KB-1MB'] += 1
                elif size_mb < 5:
                    stats['image_sizes']['1MB-5MB'] += 1
                else:
                    stats['image_sizes']['>5MB'] += 1
        
        # Check prompt
        if not raw_example.get('prompt', '').strip():
            issues['empty_prompts'].append(example_id)
            stats['empty_prompts'] += 1
        
        # Check response
        response = raw_example.get('response', {})
        output = response.get('output', '')
        if not output.strip():
            issues['empty_outputs'].append(example_id)
            stats['empty_outputs'] += 1
        
        # Check thinking (for CoT examples)
        if not response.get('thinking'):
            stats['missing_thinking'] += 1
        
        # Check format
        fmt = response.get('format', 'unknown')
        stats['output_formats'][fmt] += 1
        
        # Verify format matches content
        if fmt == 'json' and output.strip():
            try:
                json.loads(output)
            except:
                issues['format_issues'].append(f"{example_id}: Claims JSON but invalid")
                stats['format_issues'] += 1
        elif fmt == 'xml' and output.strip():
            if not (output.strip().startswith('<') and '>' in output):
                issues['format_issues'].append(f"{example_id}: Claims XML but doesn't look like XML")
                stats['format_issues'] += 1
        
        # Track sources
        source = raw_example.get('provenance', {}).get('source', 'unknown')
        stats['sources'][source] += 1
        
        # Test that we can actually load this example
        try:
            loaded_example = dataset[i]
            # Verify it has the expected structure
            assert 'messages' in loaded_example
            assert len(loaded_example['messages']) == 2
            assert loaded_example['messages'][0]['role'] == 'user'
            assert loaded_example['messages'][1]['role'] == 'assistant'
        except Exception as e:
            issues['loading_errors'].append(f"{example_id}: {str(e)}")
    
    # Print report
    print("\n" + "="*60)
    print("DATASET VERIFICATION REPORT")
    print("="*60)
    
    print(f"\nTotal examples: {stats['total_examples']}")
    print(f"Examples with thinking: {stats['total_examples'] - stats['missing_thinking']}")
    
    print("\nOutput formats:")
    for fmt, count in sorted(stats['output_formats'].items()):
        print(f"  {fmt}: {count}")
    
    print("\nData sources:")
    for source, count in sorted(stats['sources'].items()):
        print(f"  {source}: {count}")
    
    print("\nImage size distribution:")
    for size_range, count in sorted(stats['image_sizes'].items()):
        print(f"  {size_range}: {count}")
    
    print("\nISSUES FOUND:")
    if stats['missing_images'] > 0:
        print(f"  ❌ Missing images: {stats['missing_images']}")
        for issue in issues['missing_images'][:5]:
            print(f"     - {issue}")
        if len(issues['missing_images']) > 5:
            print(f"     ... and {len(issues['missing_images']) - 5} more")
    
    if stats['empty_prompts'] > 0:
        print(f"  ❌ Empty prompts: {stats['empty_prompts']}")
    
    if stats['empty_outputs'] > 0:
        print(f"  ❌ Empty outputs: {stats['empty_outputs']}")
    
    if stats['format_issues'] > 0:
        print(f"  ❌ Format mismatches: {stats['format_issues']}")
        for issue in issues['format_issues'][:5]:
            print(f"     - {issue}")
    
    if 'loading_errors' in issues:
        print(f"  ❌ Loading errors: {len(issues['loading_errors'])}")
        for issue in issues['loading_errors'][:5]:
            print(f"     - {issue}")
    
    # Overall status
    total_issues = sum([
        stats['missing_images'],
        stats['empty_prompts'],
        stats['empty_outputs'],
        stats['format_issues'],
        len(issues.get('loading_errors', []))
    ])
    
    if total_issues == 0:
        print("\n✅ Dataset verification PASSED - No issues found!")
    else:
        print(f"\n❌ Dataset verification FAILED - {total_issues} total issues found")
    
    return total_issues == 0


if __name__ == "__main__":
    verify_dataset()