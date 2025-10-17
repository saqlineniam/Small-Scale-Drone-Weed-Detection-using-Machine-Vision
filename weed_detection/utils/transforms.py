import random
import torch
import torchvision.transforms.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from PIL import Image
import torchvision.transforms as T


class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image and numpy arrays to PyTorch tensors."""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    """Randomly flip the image and bboxes horizontally with a given probability."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    """Randomly flip the image and bboxes vertically with a given probability."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target["boxes"] = boxes
        return image, target


class RandomBrightness:
    """Randomly change the brightness of the image."""
    def __init__(self, min_factor=0.8, max_factor=1.2, prob=0.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            factor = random.uniform(self.min_factor, self.max_factor)
            image = F.adjust_brightness(image, factor)
        return image, target


class RandomContrast:
    """Randomly change the contrast of the image."""
    def __init__(self, min_factor=0.8, max_factor=1.2, prob=0.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            factor = random.uniform(self.min_factor, self.max_factor)
            image = F.adjust_contrast(image, factor)
        return image, target


class RandomSaturation:
    """Randomly change the saturation of the image."""
    def __init__(self, min_factor=0.8, max_factor=1.2, prob=0.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob and image.shape[0] == 3:  # Only for RGB images
            factor = random.uniform(self.min_factor, self.max_factor)
            image = F.adjust_saturation(image, factor)
        return image, target


class RandomHue:
    """Randomly change the hue of the image."""
    def __init__(self, min_factor=-0.1, max_factor=0.1, prob=0.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob and image.shape[0] == 3:  # Only for RGB images
            factor = random.uniform(self.min_factor, self.max_factor)
            image = F.adjust_hue(image, factor)
        return image, target


class RandomGrayscale:
    """Randomly convert image to grayscale with a given probability."""
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.rgb_to_grayscale(image, num_output_channels=3)
        return image, target


class Normalize:
    """Normalize the image with mean and standard deviation."""
    def __init__(self, mean=None, std=None, inplace=False):
        self.mean = mean or [0.485, 0.456, 0.406]  # ImageNet mean
        self.std = std or [0.229, 0.224, 0.225]  # ImageNet std
        self.inplace = inplace

    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return image, target


def get_transform(train: bool = True) -> T.Compose:
    """
    Get the appropriate data transformations for training or evaluation.
    
    Args:
        train: If True, apply data augmentation for training
        
    Returns:
        A composition of data transformations
    """
    transforms = [ToTensor()]
    
    if train:
        # Training transformations
        transforms.extend([
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            RandomBrightness(min_factor=0.8, max_factor=1.2, prob=0.5),
            RandomContrast(min_factor=0.8, max_factor=1.2, prob=0.5),
            RandomSaturation(min_factor=0.8, max_factor=1.2, prob=0.5),
            RandomHue(min_factor=-0.1, max_factor=0.1, prob=0.5),
            RandomGrayscale(prob=0.1),
        ])
    
    # Always normalize
    transforms.append(Normalize())
    
    return Compose(transforms)
