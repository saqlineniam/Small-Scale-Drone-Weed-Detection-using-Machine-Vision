#!/usr/bin/env python3
"""
Prediction script for the weed detection model.
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weed_detection.models.faster_rcnn import get_faster_rcnn
from weed_detection.utils.visualization import plot_predictions


def load_model(
    model_path: str,
    num_classes: int = 2,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        num_classes: Number of classes in the model
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Create model
    model = get_faster_rcnn(num_classes=num_classes)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    return model


def predict_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: str = 'cuda',
    score_threshold: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Run prediction on a single image.
    
    Args:
        model: Trained model
        image: Input image tensor (C, H, W)
        device: Device to run inference on
        score_threshold: Minimum score for a prediction to be included
        
    Returns:
        Dictionary containing predictions
    """
    # Move image to device
    image = image.to(device)
    
    # Add batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image)
    
    # Filter predictions by score
    filtered_predictions = []
    for pred in predictions:
        keep = pred['scores'] >= score_threshold
        filtered_pred = {
            'boxes': pred['boxes'][keep],
            'labels': pred['labels'][keep],
            'scores': pred['scores'][keep]
        }
        filtered_predictions.append(filtered_pred)
    
    return filtered_predictions[0] if len(filtered_predictions) == 1 else filtered_predictions


def process_image(image_path: str, target_size: int = 800) -> torch.Tensor:
    """
    Load and preprocess an image for the model.
    
    Args:
        image_path: Path to the image file
        target_size: Size to resize the smaller dimension to
        
    Returns:
        Preprocessed image tensor (C, H, W)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor and normalize
    image = F.to_tensor(image)
    
    # Resize the smaller dimension to target_size while maintaining aspect ratio
    _, h, w = image.shape
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = F.resize(image, [new_h, new_w])
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    
    return image


def save_predictions(
    image: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    output_path: str,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.5
) -> None:
    """
    Save predictions as an image and JSON file.
    
    Args:
        image: Input image tensor (C, H, W)
        predictions: Model predictions
        output_path: Path to save the output
        class_names: List of class names
        score_threshold: Minimum score for a prediction to be included
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save visualization
    fig = plot_predictions(
        image,
        predictions,
        score_threshold=score_threshold,
        class_names=class_names
    )
    
    # Save figure
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Save predictions as JSON
    pred_dict = {
        'boxes': predictions['boxes'].cpu().numpy().tolist(),
        'labels': predictions['labels'].cpu().numpy().tolist(),
        'scores': predictions['scores'].cpu().numpy().tolist()
    }
    
    json_path = os.path.splitext(output_path)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(pred_dict, f, indent=2)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--model', required=True, help='path to model checkpoint')
    parser.add_argument('--input', required=True, help='path to input image or directory')
    parser.add_argument('--output', default='outputs/predictions', help='output directory')
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='minimum score for detection')
    parser.add_argument('--device', default='cuda', help='device to use for inference')
    parser.add_argument('--class-names', nargs='+', default=['background', 'weed'], help='list of class names')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print('Loading model...')
    model = load_model(
        args.model,
        num_classes=args.num_classes,
        device=device
    )
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image
        image_paths = [input_path]
    else:
        # Directory of images
        image_paths = list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg')) + list(input_path.glob('*.png'))
    
    print(f'Found {len(image_paths)} images to process')
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f'Processing {i+1}/{len(image_paths)}: {image_path.name}')
        
        try:
            # Load and preprocess image
            image = process_image(str(image_path))
            
            # Run prediction
            predictions = predict_image(
                model,
                image,
                device=device,
                score_threshold=args.score_threshold
            )
            
            # Save results
            output_path = os.path.join(
                args.output,
                f'pred_{image_path.stem}.jpg'
            )
            
            save_predictions(
                image,
                predictions,
                output_path,
                class_names=args.class_names,
                score_threshold=args.score_threshold
            )
            
        except Exception as e:
            print(f'Error processing {image_path}: {e}')
    
    print(f'Predictions saved to {args.output}')


if __name__ == '__main__':
    main()
