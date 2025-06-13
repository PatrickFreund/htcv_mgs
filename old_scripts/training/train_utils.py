import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (str): Device to use ('cuda' or 'cpu').
        
    Returns:
        float: Average loss for the epoch.
        float: Average accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    epoch_loss = running_loss / total_samples   
    epoch_accuracy = correct / total_samples
    return epoch_loss, epoch_accuracy

def validate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float]:
    """
    Validate the model on the validation set.
    
    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): DataLoader for the val data.
        criterion (nn.Module): Loss function.
        device (str): Device to use ('cuda' or 'cpu').
        
    Returns:
        float: Average loss for the validation set.
        float: Average accuracy for the validation set.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct = (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct / total_samples
    return epoch_loss, epoch_accuracy