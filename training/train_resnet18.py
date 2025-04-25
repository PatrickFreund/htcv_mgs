import optuna
import optuna.visualization as vis
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader


sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.dataset import get_dataset

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"



def train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scheduler = None):
    model.train()
    total_loss = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return model


def evaluate(model: nn.Module, dataloader: DataLoader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    return accuracy


def init_weights(model, init_type):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if init_type == "he":
                nn.init.kaiming_normal_(m.weight)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif init_type == "uniform":
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            # Bias initialisieren, wenn vorhanden:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



def objective(trial):
    config = {
        "batch_size": trial.suggest_int("batch_size", 16, 128, step=16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"]),
        "init_type": trial.suggest_categorical("init_type", ["he", "xavier", "normal", "uniform"]),
        "scheduler": trial.suggest_categorical("scheduler", ["none", "step_lr", "cosine_annealing"]),
    }

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Grayscale(num_output_channels=1),
        # transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
    ])
    
    dataset = get_dataset(data_foldername="MGS_data", transform=transform)
    train_split = int(len(dataset) * 0.8)
    test_split = len(dataset) - train_split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_split, test_split])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=10)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=10)

    model = resnet18(weights=None, num_classes=3)
    init_weights(model, config["init_type"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], momentum=0.9)

    scheduler = None
    if config["scheduler"] == "step_lr":
        step_size = trial.suggest_int("step_size", 1, 10)
        gamma = trial.suggest_float("gamma", 0.1, 0.9)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif config["scheduler"] == "cosine_annealing":
        t_max = trial.suggest_int("t_max", 5, 20)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)


    print(f"\n\nTraining follows config:")
    print(f"Batch size: {config['batch_size']}, Learning rate: {config['learning_rate']}, Weight decay: {config['weight_decay']}, Optimizer: {config['optimizer']}, Init type: {config['init_type']}, Scheduler: {config['scheduler']}, Optimizer: {optimizer}")
    
    best_acc = 0
    patience = 5
    patience_counter = 0
    for epoch in range(300):
        model = train(model, train_dataloader, criterion, optimizer, scheduler)
        accuracy = evaluate(model, test_dataloader)

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        print(f"Epoch {epoch+1}/{300}, Accuracy: {accuracy:.4f}")
        if accuracy > best_acc:
            best_acc = accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break 

    return best_acc



if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available()) 
    print(torch.cuda.current_device())  
    print(torch.cuda.get_device_name(0))  
    
    
    
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=3, min_early_stopping_rate=1)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=1)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    
    records = []
    for trial in study.trials:
        record = {
            "number": trial.number,
            "value": trial.value,
            "state": trial.state
        }
        # Die Param-Werte als eigene Spalten:
        record.update(trial.params)
        records.append(record)

    # In ein DataFrame packen:
    df_trials = pd.DataFrame(records)

    # Als CSV speichern:
    df_trials.to_csv("optuna_trials1.csv", index=False)
    
    
    fig = vis.plot_optimization_history(study)
    fig.write_image("optuna_optimization_history.png") 
    
    fig = vis.plot_param_importances(study)
    fig.write_image("optuna_param_importance.png")
    
    fig = vis.plot_parallel_coordinate(study)
    fig.write_image("optuna_parallel_coordinate.png")