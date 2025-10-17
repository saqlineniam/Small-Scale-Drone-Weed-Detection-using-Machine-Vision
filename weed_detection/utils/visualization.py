import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Union, Any
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
import colorsys
import random

def plot_predictions(
    image: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    score_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> plt.Figure:
    """
    Plot predictions on an image.
    
    Args:
        image: Input image tensor (C, H, W)
        predictions: Dictionary containing 'boxes', 'labels', and 'scores'
        score_threshold: Minimum score for a prediction to be displayed
        class_names: List of class names (including background)
        colors: List of RGB colors for each class
        
    Returns:
        Matplotlib figure with the image and predictions
    """
    # Convert image to numpy array and change from CxHxW to HxWxC
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.shape[0] == 3:  # CxHxW to HxWxC
        image = image.transpose(1, 2, 0)
    
    # Denormalize if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # Default class names
    if class_names is None:
        num_classes = len(torch.unique(predictions['labels'])) if 'labels' in predictions else 1
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Generate distinct colors if not provided
    if colors is None:
        colors = generate_colors(len(class_names))
    
    # Plot each prediction
    if 'boxes' in predictions and len(predictions['boxes']) > 0:
        boxes = predictions['boxes'].detach().cpu().numpy()
        labels = predictions.get('labels', torch.zeros(len(boxes))).detach().cpu().numpy()
        scores = predictions.get('scores', torch.ones(len(boxes))).detach().cpu().numpy()
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if score < score_threshold:
                continue
                
            # Get box coordinates
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Get class color and name
            class_id = int(label) % len(colors)
            color = colors[class_id]
            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with score
            label_text = f'{class_name}: {score:.2f}'
            ax.text(
                x1, y1 - 5,
                label_text,
                color='white',
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=0)
            )
    
    ax.axis('off')
    plt.tight_layout()
    return fig

def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n visually distinct colors.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of RGB tuples
    """
    random.seed(42)  # For reproducibility
    hsv_tuples = [(x / n, 1., 1.) for x in range(n)]
    random.shuffle(hsv_tuples)
    colors = []
    for h, s, v in hsv_tuples:
        h = h + (random.random() * 0.3 - 0.1)  # Add some randomness
        h = max(0, min(1, h))  # Clamp to [0, 1]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

def plot_image_with_boxes(
    image: Union[torch.Tensor, np.ndarray],
    target: Dict[str, torch.Tensor],
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    figsize: Tuple[int, int] = (12, 9)
) -> plt.Figure:
    """
    Plot an image with ground truth bounding boxes.
    
    Args:
        image: Input image (C, H, W) or (H, W, C)
        target: Dictionary containing 'boxes' and 'labels'
        class_names: List of class names
        colors: List of RGB colors for each class
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure with the image and ground truth boxes
    """
    # Convert image to numpy array and change from CxHxW to HxWxC if needed
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.shape[0] == 3:  # CxHxW to HxWxC
        image = image.transpose(1, 2, 0)
    
    # Denormalize if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Default class names
    if class_names is None and 'labels' in target:
        num_classes = int(target['labels'].max().item() + 1)
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Generate distinct colors if not provided
    if colors is None and class_names is not None:
        colors = generate_colors(len(class_names))
    
    # Plot ground truth boxes
    if 'boxes' in target:
        boxes = target['boxes'].detach().cpu().numpy()
        labels = target.get('labels', torch.zeros(len(boxes))).detach().cpu().numpy()
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            # Get box coordinates
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Get class color and name
            class_id = int(label) % len(colors) if colors is not None else 0
            color = colors[class_id] if colors is not None else 'red'
            class_name = class_names[class_id] if class_names is not None and class_id < len(class_names) else str(class_id)
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                x1, y1 - 5,
                class_name,
                color='white',
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=0)
            )
    
    ax.axis('off')
    plt.tight_layout()
    return fig
