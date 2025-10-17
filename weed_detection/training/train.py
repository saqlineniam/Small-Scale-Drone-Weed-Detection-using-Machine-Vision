import os
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, List, Tuple
import time
import numpy as np

from ..data.dataset import WeedDataset, get_transform
from ..models.faster_rcnn import get_faster_rcnn


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
    scaler=None
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        optimizer: The optimizer to use
        data_loader: DataLoader for training data
        device: Device to train on (cuda or cpu)
        epoch: Current epoch number
        print_freq: How often to print training progress
        scaler: Gradient scaler for mixed precision training
        
    Returns:
        Dict containing the average loss for this epoch
    """
    model.train()
    metric_logger = MetricLogger()
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Reduce losses over all GPUs for logging
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
            
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
            
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return metric_logger.avg_metrics()


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation data
        device: Device to evaluate on (cuda or cpu)
        
    Returns:
        Tuple of (mAP, mAP_50)
    """
    model.eval()
    metric_logger = MetricLogger()
    header = 'Test:'
    
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get model predictions
            outputs = model(images)
            
            # Convert detections to COCO format
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            
            # TODO: Calculate mAP using COCO evaluation
            # This requires implementing or using an existing COCO evaluation function
    
    # Return dummy values for now - implement proper evaluation
    return 0.0, 0.0


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    output_dir: str = 'outputs',
    checkpoint_interval: int = 1
) -> None:
    """
    Train the model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda or cpu)
        num_epochs: Number of epochs to train for
        lr: Learning rate
        momentum: Momentum for SGD
        weight_decay: Weight decay for SGD
        output_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N epochs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    best_mAP = 0.0
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=10, scaler=scaler
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        mAP, mAP_50 = evaluate(model, val_loader, device)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('mAP/val', mAP, epoch)
        writer.add_scalar('mAP_50/val', mAP_50, epoch)
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP,
                'mAP_50': mAP_50,
            }
            
            # Save latest checkpoint
            torch.save(
                checkpoint,
                os.path.join(output_dir, 'checkpoint_latest.pth')
            )
            
            # Save best checkpoint
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_best.pth')
                )
    
    writer.close()
    return model


# Utility classes and functions
class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window."""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue(fmt='{value:.4f}')
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg.append('time: {time}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str}')
    
    def avg_metrics(self):
        return {k: v.global_avg for k, v in self.meters.items()}


def reduce_dict(input_dict, average=True):
    """Reduce the values in the dictionary from all processes."""
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
