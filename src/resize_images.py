#!/usr/bin/env python3
"""
Resize images in ready directory to max 1024x1024 maintaining aspect ratio.
"""

from pathlib import Path
from PIL import Image
import argparse


def resize_image(image_path: Path, max_size: int = 1024):
    """Resize image to fit within max_size x max_size, maintaining aspect ratio."""
    img = Image.open(image_path)
    
    # Skip if already smaller than max size
    if img.width <= max_size and img.height <= max_size:
        print(f"Skipping {image_path.name} - already within size limits")
        return
    
    # Calculate new size maintaining aspect ratio
    ratio = min(max_size / img.width, max_size / img.height)
    new_width = int(img.width * ratio)
    new_height = int(img.height * ratio)
    
    # Resize using high quality
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save resized version
    resized_path = image_path.parent / f"{image_path.stem}_resized.jpg"
    resized.save(resized_path, "JPEG", quality=95)
    
    print(f"Resized {image_path.name} from {img.width}x{img.height} to {new_width}x{new_height}")
    return resized_path


def main():
    parser = argparse.ArgumentParser(description="Resize images in ready directory")
    parser.add_argument("--ready-dir", type=Path, default=Path("ready"), help="Directory with training examples")
    parser.add_argument("--max-size", type=int, default=1024, help="Maximum dimension size")
    args = parser.parse_args()
    
    # Find all original jpg files (not _resized)
    jpg_files = []
    for example_dir in sorted(args.ready_dir.iterdir()):
        if not example_dir.is_dir():
            continue
        
        # Find original jpg (not resized)
        for jpg_path in example_dir.glob("*.jpg"):
            if not jpg_path.stem.endswith("_resized"):
                jpg_files.append(jpg_path)
    
    print(f"Found {len(jpg_files)} images to process")
    
    # Resize each image
    for jpg_path in jpg_files:
        resize_image(jpg_path, args.max_size)
    
    print(f"\nResizing complete! Created {len(jpg_files)} resized images")


if __name__ == "__main__":
    main()