import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class WeedDataset(Dataset):
    """
    Custom dataset for weed detection.
    
    Args:
        root (str): Root directory containing 'images' and 'annotations' folders
        transforms (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, root: str, transforms: Optional[callable] = None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        
        # Simple validation
        if len(self.imgs) != len(self.annotations):
            raise ValueError("Number of images and annotations do not match!")
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Load images and annotations
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annotations", self.annotations[idx])
        
        img = Image.open(img_path).convert("RGB")
        
        with open(annot_path) as f:
            annot_data = json.load(f)
        
        # Get image dimensions
        width, height = img.size
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for region in annot_data.get('regions', []):
            # Get bounding box coordinates
            x = region['shape_attributes']['x']
            y = region['shape_attributes']['y']
            w = region['shape_attributes']['width']
            h = region['shape_attributes']['height']
            
            # Convert to [x0, y0, x1, y1] format
            box = [x, y, x + w, y + h]
            boxes.append(box)
            
            # Get label (1 for weed, 0 for background)
            label = 1 if region['region_attributes']['class'] == 'weed' else 0
            labels.append(label)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Apply transforms if any
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self) -> int:
        return len(self.imgs)


def get_transform(train: bool = True) -> callable:
    """
    Returns a composition of data transformations.
    
    Args:
        train (bool): If True, applies data augmentation
        
    Returns:
        callable: A function that applies the transformations
    """
    transforms = []
    
    if train:
        # Add data augmentation transforms during training
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    
    # Convert PIL image to tensor
    transforms.append(T.ToTensor())
    
    def transform(image, target):
        for t in transforms:
            if isinstance(t, T.ToTensor):
                image = t(image)
            else:
                image = t(image)
        return image, target
    
    return transform
