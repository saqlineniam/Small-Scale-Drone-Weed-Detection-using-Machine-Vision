import torch
import torch.utils.data
from tqdm import tqdm
from typing import Dict, List, Tuple
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

from ..data.dataset import WeedDataset
from .train import evaluate, reduce_dict


def evaluate_coco(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str = 'outputs',
    iou_thresholds: List[float] = None
) -> Dict[str, float]:
    """
    Evaluate the model on the validation set using COCO evaluation metrics.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation data
        device: Device to evaluate on (cuda or cpu)
        output_dir: Directory to save evaluation results
        iou_thresholds: List of IoU thresholds to evaluate on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75, 0.5, 0.95, 0.5, 0.5]
    
    model.eval()
    
    # Initialize COCO ground truth and results
    coco_gt = COCO()
    coco_gt.dataset = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'weed'}]
    }
    
    coco_results = []
    image_id = 0
    annotation_id = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            
            # Get model predictions
            outputs = model(images)
            
            # Process each image in the batch
            for img_idx, (img, target, output) in enumerate(zip(images, targets, outputs)):
                # Add image to COCO dataset
                img_info = {
                    'id': image_id,
                    'width': img.shape[2],
                    'height': img.shape[1]
                }
                coco_gt.dataset['images'].append(img_info)
                
                # Add ground truth annotations
                for box, label in zip(target['boxes'], target['labels']):
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    
                    annotation = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': label.item(),
                        'bbox': [x1, y1, width, height],
                        'area': width * height,
                        'iscrowd': 0
                    }
                    coco_gt.dataset['annotations'].append(annotation)
                    annotation_id += 1
                
                # Add predictions to results
                for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    
                    result = {
                        'image_id': image_id,
                        'category_id': label.item(),
                        'bbox': [x1, y1, width, height],
                        'score': score.item()
                    }
                    coco_results.append(result)
                
                image_id += 1
    
    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'coco_results.json')
    with open(results_file, 'w') as f:
        json.dump(coco_results, f)
    
    # Load ground truth and results into COCO API
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results_file)
    
    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Set custom IoU thresholds if provided
    if iou_thresholds is not None:
        coco_eval.params.iouThrs = np.array(iou_thresholds)
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Get evaluation metrics
    metrics = {
        'mAP': coco_eval.stats[0],  # mAP @[0.5:0.95]
        'mAP_50': coco_eval.stats[1],  # mAP @0.5
        'mAP_75': coco_eval.stats[2],  # mAP @0.75
        'mAP_small': coco_eval.stats[3],  # mAP for small objects
        'mAP_medium': coco_eval.stats[4],  # mAP for medium objects
        'mAP_large': coco_eval.stats[5],  # mAP for large objects
        'AR_1': coco_eval.stats[6],  # AR @1
        'AR_10': coco_eval.stats[7],  # AR @10
        'AR_100': coco_eval.stats[8],  # AR @100
        'AR_small': coco_eval.stats[9],  # AR for small objects
        'AR_medium': coco_eval.stats[10],  # AR for medium objects
        'AR_large': coco_eval.stats[11]  # AR for large objects
    }
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics


def main():
    import argparse
    from ..models.faster_rcnn import get_faster_rcnn
    from torch.utils.data import DataLoader
    from ..data.dataset import get_transform
    
    parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN on Weed Detection Dataset')
    parser.add_argument('--data-dir', default='data/weed_detection', help='dataset root directory')
    parser.add_argument('--model', default='', help='path to model checkpoint')
    parser.add_argument('--batch-size', default=4, type=int, help='batch size')
    parser.add_argument('--num-classes', default=2, type=int, help='number of classes (including background)')
    parser.add_argument('--output-dir', default='outputs/eval', help='directory to save evaluation results')
    parser.add_argument('--device', default='cuda', help='device to use for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_faster_rcnn(num_classes=args.num_classes)
    
    if args.model:
        checkpoint = torch.load(args.model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    
    # Load dataset
    dataset = WeedDataset(
        args.data_dir,
        get_transform(train=False)
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Run evaluation
    metrics = evaluate_coco(
        model,
        data_loader,
        device=device,
        output_dir=args.output_dir
    )
    
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
