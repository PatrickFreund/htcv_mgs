import sys
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Union, Tuple

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.balancing import (
    BalancingStrategy, NoBalancingStrategy, WeightedLossBalancing, OversamplingBalancing
)
from training.logger import TensorBoardLogger


class EarlyStopping:
    def __init__(self, mode: str,  patience: int = 10):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_metric: float = None
    
    def __call__(self, metric: float) -> bool:
        if self.best_metric is None:
            self.best_metric = metric
            return False
        
        if self._is_improvement(metric):
            self.best_metric = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_metric
        elif self.mode == "max":
            return metric > self.best_metric 
        else:
            raise ValueError(f"Invalid mode '{self.mode}' for EarlyStopping. Use 'min' or 'max'.")

    def reset(self):
        self.counter = 0
        self.best_metric = None

class ModelTrainer:
    def __init__(
        self,
        model_builder: Callable,
        optimizer_builder: Callable,
        **kwargs, 
    ) -> None:
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder
        self.logger = None
        
        self.main_metric, self.mode = self._resolve_main_metric(kwargs.get("main_metric", "loss"))
        self.shuffle = kwargs.get("shuffle", True)
        early_stopp = kwargs.get("early_stopping", False)
        patience = kwargs.get("patience", None)
        self.early_stopping: EarlyStopping = self._resolve_early_stopping(early_stopping=early_stopp, mode = self.mode, patience=patience)
        strategy = kwargs.get("balancing_strategy", NoBalancingStrategy())
        class_weights = kwargs.get("class_weights", None)
        self._check_class_weights(class_weights, strategy)
        self.balancing_strategy = self._resolve_strategy(strategy = strategy, class_weights = class_weights)
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _resolve_lr_scheduler(self, config: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Optional[_LRScheduler]:
        scheduler_type = config.get("lr_scheduler", "none")
        if scheduler_type == "none":
            return None

        elif scheduler_type == "step":
            step_size = config.get("scheduler_step_size", 10)
            gamma = config.get("scheduler_gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_type == "cosine":
            t_max = config.get("scheduler_t_max", 50)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _resolve_strategy(
        self,
        strategy: Union[str, BalancingStrategy],
        class_weights: Optional[Dict[int, float]] = None
    ) -> BalancingStrategy:
        if isinstance(strategy, str):
            if strategy == "no_balancing":
                return NoBalancingStrategy()
            elif strategy == "weighted_loss":
                return WeightedLossBalancing(class_weights)
            elif strategy == "oversampling":
                return OversamplingBalancing(class_weights)
            else:
                raise ValueError(f"Unknown balancing strategy: {strategy}")
        elif isinstance(strategy, BalancingStrategy):
            return strategy
        else:
            raise TypeError("Balancing strategy must be a string or an instance of BalancingStrategy.")

    def _resolve_early_stopping(self, early_stopping: Union[bool, EarlyStopping], mode: str, patience: Optional[int] = None) -> EarlyStopping:
        if patience is not None and not isinstance(patience, int):
            raise TypeError("Patience must be an integer if specified.")
        
        if isinstance(early_stopping, bool):
            if early_stopping:
                if not patience:
                    raise ValueError("Patience must be specified if early stopping is enabled.")
                return EarlyStopping(patience = patience, mode = mode)
            else:
                return None
        elif isinstance(early_stopping, EarlyStopping):
            early_stopping.reset()
            early_stopping.mode = mode
            early_stopping.patience = patience if patience is not None else early_stopping.patience
            return early_stopping
        else:
            raise TypeError("Early stopping must be an boolean or an instance of EarlyStopping.")
    
    def _resolve_main_metric(self, main_metric: str) -> Tuple[str, str]:
        valid_metrics = ["loss", "acc", "f1", "precision", "recall"]
        if main_metric not in valid_metrics:
            raise ValueError(f"Invalid main metric '{main_metric}' provided. Valid options are: {valid_metrics}")
        if main_metric in ["loss"]:
            mode = "min"
        else:
            mode = "max"
        return main_metric, mode
    
    def _check_class_weights(self, class_weights: Optional[Dict[int, float]], balancing_strategy: Union[str, BalancingStrategy]) -> None:
        strategy_name = getattr(balancing_strategy, "name", balancing_strategy)
        if strategy_name == "no_balancing":
            return
        
        if class_weights is None and strategy_name != "no_balancing":
            raise ValueError(f"Class weights are required for balancing strategy '{strategy_name}'.")
        
        if not isinstance(class_weights, dict):
            raise ValueError("Class weights must be a dictionary mapping integer class indices to float weights.")

        if not all(isinstance(k, int) for k in class_weights):
            raise ValueError("All class weight keys must be integers (class indices).")

        if not all(isinstance(v, float) for v in class_weights.values()):
            raise ValueError("All class weight values must be floats.")

        sorted_keys = sorted(class_weights.keys())
        if sorted_keys != list(range(len(class_weights))):
            raise ValueError("Class weight keys must be consecutive integers starting from 0.")

        if not abs(sum(class_weights.values()) - 1.0) < 1e-6:
            raise ValueError("Class weights must sum to 1.0.")
    
    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
        scheduler: Optional[_LRScheduler] = None,
    ) -> Dict[str, float]:
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        return {
            "loss": avg_loss,
            "acc": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average='weighted'),
            "precision": precision_score(all_labels, all_preds, average='weighted'),
            "recall": recall_score(all_labels, all_preds, average='weighted'),
        }

    def _validate_one_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion
    ) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        metrics = {
            "loss": avg_loss,
            "acc": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="weighted"),
            "precision": precision_score(all_labels, all_preds, average='weighted'),
            "recall": recall_score(all_labels, all_preds, average='weighted'),
        }
        return metrics

    def set_logger(self, log_path: Union[str, Path]) -> None:
        """
        Sets the logger for the trainer. This method should be called before training starts to
        ensure that all training metrics are logged correctly.

        Args:
            log_path (Path): The path where the logs will be saved. This should be a valid directory path.
        """
        if self.logger is not None:
            self.logger.close()
            self.logger = None # Reset logger to avoid memory leaks
        
        if not isinstance(log_path, (str, Path)):
            raise TypeError(f"log_path must be a string or a Path object, not {type(log_path)}")
        if not isinstance(log_path, str):
            log_path = Path(log_path).resolve()
        if not log_path.exists():
            raise FileNotFoundError(f"The specified log path does not exist: {log_path}")
        self.log_path = log_path
        self.logger = TensorBoardLogger(log_path)
        
    def train(self, config: Dict[str, Any], train_data: Dataset, val_data: Dataset) -> Dict[str, Any]:
        model: nn.Module = self.model_builder(config).to(self.device)
        optimizer: torch.optim.Optimizer = self.optimizer_builder(model, config)
        
        train_loader, train_criterion = self.balancing_strategy.prepare(train_data, config, device = self.device)
        val_loader = DataLoader(val_data, batch_size = config["batch_size"], shuffle = self.shuffle)
        val_criterion = nn.CrossEntropyLoss()
        lr_scheduler = self._resolve_lr_scheduler(config, optimizer)

        history: Dict[str, Any] = {
            "train": {"loss": [], "acc": [], "f1": [], "precision": [], "recall": []},
            "val": {"loss": [], "acc": [], "f1": [], "precision": [], "recall": []}
        }
        best_epoch = 0
        if self.logger:
            self.logger.log_params(config)

        for epoch in range(config['epochs']):
            train_metrics = self._train_one_epoch(model, train_loader, optimizer, train_criterion, lr_scheduler)
            val_metrics = self._validate_one_epoch(model, val_loader, val_criterion)
            
            for key in history["train"]:
                history["train"][key].append(train_metrics[key])
                history["val"][key].append(val_metrics[key])
                
                if self.logger:
                    self.logger.log_scalar(f"train/{key}", train_metrics[key], epoch)
                    self.logger.log_scalar(f"val/{key}", val_metrics[key], epoch)

            # Check for the best epoch based on the main metric
            if self.mode == "min":
                if history["val"][self.main_metric][-1] < history["val"][self.main_metric][best_epoch]:
                    best_epoch = epoch
            elif self.mode == "max":
                if history["val"][self.main_metric][-1] > history["val"][self.main_metric][best_epoch]:
                    best_epoch = epoch
            else:
                raise ValueError(f"Invalid mode '{self.mode}' for main metric. Use 'min' or 'max'.")
            
            # Early stopping check if early stopping is enabled
            if self.early_stopping and self.early_stopping(val_metrics[self.main_metric]):
                break

        if self.early_stopping:
            self.early_stopping.reset()
        if self.logger:
            self.logger.close()
            self.logger = None  # Reset logger to avoid memory leaks

        try:
            torch.save(model.state_dict(), self.log_path / "best_model.pth")
        except Exception as e:
            print(f"Error saving model: {e}")
            print("Model will not be saved. Please check the log path and permissions.")
            

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_metrics": {key: history["val"][key][best_epoch] for key in history["val"]},
        }


