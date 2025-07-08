from typing import Dict, Any

import torch
from torch import nn

def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Creates an optimizer for the given model based on the configuration provided.

    Args:
        model (nn.Module): The model for which the optimizer is to be created.
        config (Dict[str, Any]): Configuration dictionary containing:
            - "optim": str, name of the optimizer (e.g., "Adam", "SGD")
            - "learning_rate": float, learning rate for the optimizer
            - "momentum": float, optional, momentum for SGD (default is 0.9)
            - "weight_decay": float, weight decay for the optimizer (default is 0.0)

    Returns:
        torch.optim.Optimizer: An instance of the optimizer specified in the config.
    """
    optimizer_name = config.get("optim", None)
    learning_rate = config.get("learning_rate", None)
    weight_decay = config.get("weight_decay", None)
    
    if optimizer_name is None or learning_rate is None:
        raise ValueError("Optimizer name and learning rate must be specified in the config")
    if weight_decay is None:
        raise ValueError("Weight decay must be specified in the config, even if it is 0.0")
    
    if optimizer_name == "Adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = config.get("momentum", 0.9)
        optimizer: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported yet")
    
    return optimizer