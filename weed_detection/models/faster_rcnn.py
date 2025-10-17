import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def get_faster_rcnn(num_classes: int = 2) -> torch.nn.Module:
    """
    Create a Faster R-CNN model with a ResNet-50 backbone.
    
    Args:
        num_classes (int): Number of output classes (including background)
        
    Returns:
        torch.nn.Module: Faster R-CNN model
    """
    # Load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    # Remove the last fully connected layer
    modules = list(backbone.children())[:-2]
    backbone = torch.nn.Sequential(*modules)
    
    # Create the anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Define the feature map that the model will use for region proposals
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Put the pieces together inside a FasterRCNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model


def get_model_instance_segmentation(num_classes: int) -> torch.nn.Module:
    """
    Get an instance segmentation model for weed detection.
    
    Args:
        num_classes (int): Number of classes (including background)
        
    Returns:
        torch.nn.Module: Faster R-CNN model with instance segmentation head
    """
    # Load a pre-trained model for instance segmentation
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
