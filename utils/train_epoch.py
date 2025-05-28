from typing import Tuple, List, Dict
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import _Loss

def train_func(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
    scheduler: _LRScheduler
) -> Tuple[float, Dict[str, float]]:
    model.train()
    train_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if scheduler:
        scheduler.step()

    avg_loss = train_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

    return avg_loss, metrics

def val_func(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: _Loss
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim = 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader) #the average loss of batch
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    val_metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    return avg_val_loss, val_metrics