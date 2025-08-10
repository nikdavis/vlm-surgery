"""
Transform implementations for data augmentation pipeline.

These transforms implement the Transform protocol and can be
composed in a pipeline for data preprocessing and augmentation.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
from io import BytesIO
import random
from loguru import logger

from .protocols import Transform


class DecodeImage:
    """
    Decode image bytes to PIL Image.
    This should be the first transform in any image pipeline.
    """

    def __init__(self, mode: str = "RGB"):
        """
        Initialize decoder.

        Args:
            mode: PIL image mode to convert to (default: "RGB")
        """
        self.mode = mode

    def __call__(self, sample: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Decode image bytes to PIL Image.

        Args:
            sample: Dictionary with 'image' key containing bytes
            rng: Random generator (unused for this transform)

        Returns:
            Sample with 'image' converted to PIL Image
        """
        if 'image' not in sample:
            raise KeyError("Sample must have 'image' key")

        image_data = sample['image']

        # If already a PIL Image, just ensure correct mode
        if hasattr(image_data, 'mode'):
            if image_data.mode != self.mode:
                sample['image'] = image_data.convert(self.mode)
            return sample

        # Decode bytes to PIL Image
        image = Image.open(BytesIO(image_data))
        if image.mode != self.mode:
            image = image.convert(self.mode)

        sample['image'] = image
        return sample


class RandomResize:
    """
    Randomly resize image while preserving aspect ratio.
    """
    
    def __init__(self, scale: Tuple[float, float] = (0.8, 1.1)):
        """
        Initialize random resize.
        
        Args:
            scale: Range of resize scale (0.8 = 80% of original, 1.1 = 110% of original)
        """
        self.scale = scale
    
    def __call__(self, sample: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Apply random resize to image, preserving aspect ratio.
        
        Args:
            sample: Dictionary with 'image' key containing PIL Image
            rng: Random generator for reproducible randomness
            
        Returns:
            Sample with resized image
        """
        if 'image' not in sample:
            raise KeyError("Sample must have 'image' key")
        
        img = sample['image']
        if not hasattr(img, 'resize'):
            raise ValueError("Image must be PIL Image. Apply DecodeImage first.")
        
        # Random scale factor
        scale_factor = rng.uniform(self.scale[0], self.scale[1])
        
        # Calculate new size preserving aspect ratio
        width, height = img.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        
        sample['image'] = img
        return sample


class RandomCrop:
    """
    Random crop from image, with each edge independently cropped.
    """
    
    def __init__(self, min_crop_ratio: float = 0.85):
        """
        Initialize random crop.
        
        Args:
            min_crop_ratio: Minimum ratio to keep (0.85 = each edge crops between 85% and 100%)
        """
        if not 0 < min_crop_ratio <= 1.0:
            raise ValueError(f"min_crop_ratio must be between 0 and 1, got {min_crop_ratio}")
        self.min_crop_ratio = min_crop_ratio
    
    def __call__(self, sample: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Apply random crop to image.
        
        Args:
            sample: Dictionary with 'image' key containing PIL Image
            rng: Random generator for reproducible randomness
            
        Returns:
            Sample with cropped image
        """
        if 'image' not in sample:
            raise KeyError("Sample must have 'image' key")
        
        img = sample['image']
        if not hasattr(img, 'crop'):
            raise ValueError("Image must be PIL Image. Apply DecodeImage first.")
        
        width, height = img.size
        
        # Each edge gets its own random ratio between min_crop_ratio and 1.0
        width_ratio = rng.uniform(self.min_crop_ratio, 1.0)
        height_ratio = rng.uniform(self.min_crop_ratio, 1.0)
        
        # Calculate crop size
        crop_width = int(width * width_ratio)
        crop_height = int(height * height_ratio)
        
        # Random position for crop within the image
        left = rng.integers(0, width - crop_width + 1) if width > crop_width else 0
        top = rng.integers(0, height - crop_height + 1) if height > crop_height else 0
        
        # Crop
        cropped = img.crop((left, top, left + crop_width, top + crop_height))
        
        sample['image'] = cropped
        return sample



class MildColorJitter:
    """
    Apply mild color jittering for robust training.
    Uses small perturbations to add noise without fundamentally altering the image.
    """

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05
    ):
        """
        Initialize color jitter.

        Args:
            brightness: Max brightness change (0.1 = ±10%)
            contrast: Max contrast change (0.1 = ±10%)
            saturation: Max saturation change (0.1 = ±10%)
            hue: Max hue shift (0.05 = ±5% of color wheel)
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def adjust_brightness(self, img: Image.Image, factor: float) -> Image.Image:
        """Adjust brightness of image."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def adjust_contrast(self, img: Image.Image, factor: float) -> Image.Image:
        """Adjust contrast of image."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def adjust_saturation(self, img: Image.Image, factor: float) -> Image.Image:
        """Adjust saturation of image."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    def adjust_hue(self, img: Image.Image, factor: float) -> Image.Image:
        """Adjust hue of image."""
        import numpy as np
        img_array = np.array(img)

        # Convert RGB to HSV
        img_hsv = Image.fromarray(img_array).convert('HSV')
        h, s, v = img_hsv.split()

        # Shift hue
        h_array = np.array(h, dtype=np.float32)
        h_array = (h_array + factor * 255) % 255
        h = Image.fromarray(h_array.astype(np.uint8))

        # Merge back and convert to RGB
        img_hsv = Image.merge('HSV', (h, s, v))
        return img_hsv.convert('RGB')

    def __call__(self, sample: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Apply mild color jittering to image.

        Args:
            sample: Dictionary with 'image' key containing PIL Image
            rng: Random generator for reproducible randomness

        Returns:
            Sample with color-jittered image
        """
        if 'image' not in sample:
            raise KeyError("Sample must have 'image' key")

        img = sample['image']
        if not hasattr(img, 'mode'):
            raise ValueError("Image must be PIL Image. Apply DecodeImage first.")

        # Random order of transforms
        transforms = []

        if self.brightness > 0:
            brightness_factor = rng.uniform(
                max(0, 1 - self.brightness),
                1 + self.brightness
            )
            transforms.append(('brightness', brightness_factor))

        if self.contrast > 0:
            contrast_factor = rng.uniform(
                max(0, 1 - self.contrast),
                1 + self.contrast
            )
            transforms.append(('contrast', contrast_factor))

        if self.saturation > 0:
            saturation_factor = rng.uniform(
                max(0, 1 - self.saturation),
                1 + self.saturation
            )
            transforms.append(('saturation', saturation_factor))

        if self.hue > 0:
            hue_factor = rng.uniform(-self.hue, self.hue)
            transforms.append(('hue', hue_factor))

        # Shuffle order
        rng.shuffle(transforms)

        # Apply transforms
        for transform_name, factor in transforms:
            if transform_name == 'brightness':
                img = self.adjust_brightness(img, factor)
            elif transform_name == 'contrast':
                img = self.adjust_contrast(img, factor)
            elif transform_name == 'saturation':
                img = self.adjust_saturation(img, factor)
            elif transform_name == 'hue':
                img = self.adjust_hue(img, factor)

        sample['image'] = img
        return sample


class SelectRandomPrompt:
    """
    Select a random prompt from available prompts.
    """

    def __init__(self, prompt_key: str = "prompt"):
        """
        Initialize prompt selector.

        Args:
            prompt_key: Key to store selected prompt (default: "prompt")
        """
        self.prompt_key = prompt_key

    def __call__(self, sample: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Select random prompt from prompts list.

        Args:
            sample: Dictionary with 'prompts' key containing list of prompts
            rng: Random generator for reproducible randomness

        Returns:
            Sample with selected prompt in self.prompt_key
        """
        if 'prompts' not in sample:
            raise KeyError("Sample must have 'prompts' key")

        prompts = sample['prompts']
        if not prompts:
            raise ValueError("Prompts list is empty")

        # Select random prompt
        idx = rng.integers(0, len(prompts))
        sample[self.prompt_key] = prompts[idx]

        return sample


class PrepareForModel:
    """
    Final transform to prepare sample for model/collator.
    Converts to the format expected by the training pipeline.
    """

    def __init__(self, include_image_id: bool = False):
        """
        Initialize model preparation transform.

        Args:
            include_image_id: Whether to include image_id in output
        """
        self.include_image_id = include_image_id

    def __call__(self, sample: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Prepare sample for model consumption.

        Args:
            sample: Dictionary with processed image and prompt
            rng: Random generator (unused for this transform)

        Returns:
            Sample in format expected by collator:
            - 'images': List containing single PIL Image
            - 'messages': List of message dictionaries
        """
        if 'image' not in sample or 'prompt' not in sample:
            raise KeyError("Sample must have 'image' and 'prompt' keys")

        # Create messages format expected by collator
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample['prompt']}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""}  # Will be filled by dataset
                ]
            }
        ]

        output = {
            'images': [sample['image']],
            'messages': messages
        }

        if self.include_image_id and 'image_id' in sample:
            output['image_id'] = sample['image_id']

        return output


class ComposeTransforms:
    """
    Compose multiple transforms into a pipeline.
    """

    def __init__(self, transforms: List[Transform]):
        """
        Initialize transform pipeline.

        Args:
            transforms: List of transforms to apply in sequence
        """
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Apply all transforms in sequence.

        Args:
            sample: Input sample dictionary
            rng: Random generator for reproducible randomness

        Returns:
            Transformed sample
        """
        for transform in self.transforms:
            sample = transform(sample, rng)
        return sample
