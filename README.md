# Weed Detection using Faster R-CNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-EE4C2C.svg?logo=PyTorch)](https://pytorch.org/)

This project implements a **Weed Detection System** using **Faster R-CNN**, a state-of-the-art object detection model. The goal is to identify and classify various types of weeds from images of crops, enabling automatic weed control for agricultural applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Metrics](#metrics)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project uses the **Faster R-CNN** model to detect weeds from images of crops. We train the model on a custom dataset of images labeled with weed and crop annotations, and evaluate its performance using metrics like **IoU (Intersection over Union)**, **Precision**, **Recall**, and **Average Precision**.

### Key Features
- 🚀 **State-of-the-art Model**: Utilizes Faster R-CNN with ResNet-50 backbone
- 📊 **Comprehensive Metrics**: Implements IoU, Precision, Recall, and mAP for evaluation
- 🔄 **Data Augmentation**: Includes preprocessing and augmentation for robust training
- ⚡ **GPU Support**: Accelerated training with CUDA support
- 📈 **Easy Training Pipeline**: Simple command-line interface for training and evaluation

## Prerequisites

- Python >= 3.6
- PyTorch >= 1.7
- TorchVision >= 0.8.0
- OpenCV >= 4.5.1
- tqdm >= 4.41.1
- CUDA (Optional, for GPU acceleration)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/weed-detection.git
   cd weed-detection
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install individually:
   ```bash
   pip install torch torchvision opencv-python tqdm
   ```

## Dataset

### Dataset Structure
```
dataset/
    images/
        image1.jpg
        image2.jpg
        ...
    annotations/
        image1.json
        image2.json
        ...
```

### Annotation Format
```json
{
    "image": "image1.jpg",
    "regions": [
        {
            "shape_attributes": {"name": "rect", "x": 100, "y": 120, "width": 150, "height": 200},
            "region_attributes": {"class": "weed"}
        }
    ]
}
```

## Model Architecture

The model uses Faster R-CNN with the following components:
- **Backbone**: Pre-trained ResNet-50
- **RPN (Region Proposal Network)**: Generates region proposals
- **RoI Pooling**: Extracts fixed-size features from proposals
- **Box Predictor**: Classifies objects and refines bounding boxes

## Training the Model

To train the model, run:
```bash
python train.py --epochs 10 --batch-size 8 --lr 0.005
```

### Training Parameters
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 0.005)
- `--data-dir`: Path to dataset (default: 'dataset/')
- `--output-dir`: Directory to save checkpoints (default: 'output/')

## Evaluation

Evaluate the model on the test set:
```bash
python evaluate.py --model-path path/to/model.pth --data-dir dataset/
```

## Metrics

- **IoU (Intersection over Union)**: Measures localization accuracy
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual positives
- **mAP**: Mean Average Precision across all classes

## Usage

### Training
```bash
python train.py --epochs 20 --batch-size 16 --lr 0.001
```

### Evaluation
```bash
python evaluate.py --model-path output/model_best.pth
```

### Inference on Single Image
```bash
python predict.py --image path/to/image.jpg --model output/model_best.pth
```

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [PyTorch](https://pytorch.org/) - The machine learning framework used
- [TorchVision](https://pytorch.org/vision/stable/index.html) - For model architectures and datasets
- [COCO API](https://cocodataset.org/) - For evaluation metrics implementation
