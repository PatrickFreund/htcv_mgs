from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    VGG16_Weights,
)

def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Creates a model based on the configuration provided. The model can be either ResNet18, ResNet50, or VGG16.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - "pretrained": bool, whether to use pretrained weights
            - "num_classes": int, number of output classes
            - "model_name": str, name of the model to use (e.g., "resnet18", "resnet50", "vgg16")

    Returns:
        nn.Module: An instance of the specified model with the appropriate number of output classes.
    """
    pretrained = config.get("pretrained", None)
    num_classes = config.get("num_classes", None)  
    model_name = config.get("model_name", None)

    if num_classes is None:
        raise ValueError("Number of classes must be specified in the config")
    if pretrained is None:
        raise ValueError("Pretrained flag must be specified in the config")
    if model_name is None:
        raise ValueError("Model name must be specified in the config")
    
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        # has to be removed if input images are not grayscaled 
        model.conv1 = nn.Conv2d(in_channels=1, out_channels=model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.conv1 = nn.Conv2d(in_channels=1, out_channels=model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        weights = VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        model.features[0] = nn.Conv2d(in_channels=1, out_channels=model.features[0].out_channels, kernel_size=model.features[0].kernel_size, stride=model.features[0].stride, padding=model.features[0].padding, bias=model.features[0].bias)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported yet")
    
    return model

