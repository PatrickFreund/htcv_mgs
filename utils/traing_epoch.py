from typing import Tuple, List, Dict, Any
import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import _Loss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from typing import Callable

def train_func(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epochs: int,
) -> Tuple[List[float], List[Dict[str, float]]]:
    model.train()
    train_losses = []
    train_metrics = []

    for epoch in range(epochs):
        all_preds = []
        all_labels = []
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(torch.device), labels.to(torch.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_loss = train_loss / len(train_loader)
        train_losses.append(avg_loss)

        if scheduler:
            scheduler.step()
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average= 'weighted')
        precision = precision_score(all_labels, all_preds, average = 'weighted')
        recall = recall_score(all_labels, all_preds, average = 'weighted')
        metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
        }
        train_metrics.append(metrics)

    return train_losses, train_metrics


def val_func(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: _Loss,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(torch.device), labels.to(torch.device)
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

def cross_validation(
    dataset: torch.utils.data.Dataset,
    model_fn: Callable,
    model_args: Dict[str, Any],
    config: Dict[str, Any],
    k_folds: int,
    train_func: Callable,
    val_func: Callable,
    device: str,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Dict[str, float]:
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=config["General"]["seed"])
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== Fold {fold + 1}/{k_folds} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = model_fn(**model_args).to(device)
        criterion = torch.nn.CrossEntropyLoss() if config["loss_fn"] == "CrossEntropy" else torch.nn.MSELoss()

        if config["optim"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        elif config["optim"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer: {config['optim']}")

        scheduler = None
        if config["scheduler"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step_size"],
                                                        gamma=config["scheduler_gamma"])

        train_func(model, train_loader, criterion, optimizer, scheduler, config["epochs"], device)
        val_loss, metrics = val_func(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Metrics: {metrics}")

        metrics["val_loss"] = val_loss
        fold_metrics.append(metrics)

        final_results = {}
        for key in fold_metrics[0].keys():
            final_results[key] = sum(d[key] for d in fold_metrics) / k_folds

        print("\n===== Cross Validation Results =====")
        for k, v in final_results.items():
            print(f"{k}: {v:.4f}")

        return final_results