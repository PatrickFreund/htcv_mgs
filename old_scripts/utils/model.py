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


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = config.get("optim", None)
    learning_rate = config.get("learning_rate", None)
    
    if optimizer_name is None or learning_rate is None:
        raise ValueError("Optimizer name and learning rate must be specified in the config")
    
    if optimizer_name == "Adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        momentum = config.get("momentum", 0.9)
        optimizer: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported yet")
    
    return optimizer