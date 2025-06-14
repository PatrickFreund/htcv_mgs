from typing import Dict, Any

import torch
from torch import nn

def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
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