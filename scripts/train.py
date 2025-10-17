#!/usr/bin/env python3
"""
Training script for the weed detection model.
"""
import os
import sys
import argparse
import json
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weed_detection.data.dataset import WeedDataset, get_transform
from weed_detection.models.faster_rcnn import get_faster_rcnn
from weed_detection.training.train import train_model, evaluate
from weed_detection.training.evaluate import evaluate_coco


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a weed detection model')
    
    # Dataset parameters
    parser.add_argument('--data-dir', default='data/weed_detection',
                        help='dataset root directory (default: data/weed_detection)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='number of classes including background (default: 2)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--lr-step-size', type=int, default=3,
                        help='step size for learning rate scheduler (default: 3)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='gamma for learning rate scheduler (default: 0.1)')
    
    # Model parameters
    parser.add_argument('--resume', default='',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model')
    
    # Output parameters
    parser.add_argument('--output-dir', default='outputs',
                        help='directory to save outputs (default: outputs)')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='save checkpoint every N epochs (default: 1)')
    
    # Device parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command line arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    print('Creating model...')
    model = get_faster_rcnn(num_classes=args.num_classes)
    model.to(device)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.resume:
        print(f'Loading checkpoint from {args.resume}...')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f'Resuming training from epoch {start_epoch}')
    
    # Create datasets
    print('Creating datasets...')
    train_dataset = WeedDataset(
        os.path.join(args.data_dir, 'train'),
        get_transform(train=True)
    )
    
    val_dataset = WeedDataset(
        os.path.join(args.data_dir, 'val'),
        get_transform(train=False)
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Train the model
    print('Starting training...')
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Evaluate the model on the test set
    print('Evaluating on test set...')
    test_dataset = WeedDataset(
        os.path.join(args.data_dir, 'test'),
        get_transform(train=False)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Run COCO evaluation
    metrics = evaluate_coco(
        model,
        test_loader,
        device=device,
        output_dir=os.path.join(args.output_dir, 'eval')
    )
    
    # Save evaluation metrics
    with open(os.path.join(args.output_dir, 'eval', 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print('Training complete!')


if __name__ == '__main__':
    main()
