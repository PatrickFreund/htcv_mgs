import copy
from pathlib import Path
import sys
from typing import Callable, Dict, Any, Optional, Union, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.balancing import (
    BalancingStrategy, NoBalancingStrategy, WeightedLossBalancing, OversamplingBalancing
)
from training.logger import TensorBoardLogger


class EarlyStopping:
    """
    EarlyStopping is a utility class to stop training when a monitored metric has stopped improving.
    And to save the best model state when an improvement is detected.
    """
    def __init__(self, mode: str,  patience: int = 10):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_metric: float = None
        self.best_model_state: Optional[nn.Module] = None
    
    def __call__(self, metric: float, model: nn.Module) -> bool:
        if self.best_metric is None:
            self.best_metric = metric
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        
        if self._is_improvement(metric):
            self.best_metric = metric
            self.best_model_state = copy.deepcopy(model.state_dict())
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
        self.best_model_state = None

class ModelTrainer:
    """
    ModelTrainer is responsible for training a model using a specified training configuration 
    and dataset split. It initializes the model and optimizer, sets up the training and validation data loaders,
    and manages the training loop, including early stopping and logging.
    """
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
        """
        Resolves the learning rate scheduler based on the provided configuration and optimizer.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing scheduler type and parameters.
            optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is to be created.

        Returns:
            Optional[_LRScheduler]: An instance of a learning rate scheduler or None if no scheduler is specified.
        """
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
        """
        Resolves the balancing strategy based on the provided string or instance.

        Args:
            strategy (Union[str, BalancingStrategy]): The balancing strategy to be used, either as a string identifier or an instance of BalancingStrategy.
            class_weights (Optional[Dict[int, float]], optional): Class weights to be used for balancing. Required if strategy is not "no_balancing". Defaults to None.

        Returns:
            BalancingStrategy: An instance of the specified balancing strategy.
        """
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
        """
        Resolves the early stopping mechanism based on the provided parameters.

        Args:
            early_stopping (Union[bool, EarlyStopping]): If True, enables early stopping with the specified patience; if an instance of EarlyStopping, uses that instance.
            mode (str): The mode for early stopping, either "min" or "max", indicating whether to stop training when the monitored metric stops decreasing or increasing.
            patience (Optional[int], optional): The number of epochs with no improvement after which training will be stopped. If None, it will not apply early stopping. Defaults to None.

        Returns:
            EarlyStopping: An instance of EarlyStopping configured with the specified parameters.
        """
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
        """
        Resolves the main metric for training and validation, ensuring it is valid and setting the mode accordingly.

        Args:
            main_metric (str): The main metric to monitor during training. Valid options are "loss", "acc", "f1", "precision", and "recall".

        Returns:
            Tuple[str, str]: A tuple containing the main metric and the mode ("min" or "max") for monitoring.
        """
        valid_metrics = ["loss", "acc", "f1", "precision", "recall"]
        if main_metric not in valid_metrics:
            raise ValueError(f"Invalid main metric '{main_metric}' provided. Valid options are: {valid_metrics}")
        if main_metric in ["loss"]:
            mode = "min"
        else:
            mode = "max"
        return main_metric, mode
    
    def _check_class_weights(self, class_weights: Optional[Dict[int, float]], balancing_strategy: Union[str, BalancingStrategy]) -> None:
        """
        Checks the validity of class weights based on the balancing strategy and raises appropriate errors if they are not valid.

        Args:
            class_weights (Optional[Dict[int, float]]): A dictionary mapping class indices (integers) to their corresponding weights (floats).
            balancing_strategy (Union[str, BalancingStrategy]): The balancing strategy to be used, either as a string identifier or an instance of BalancingStrategy.
        """
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
        """
        Trains the model for one epoch using the provided training data loader, optimizer, and loss criterion.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (DataLoader): The data loader for the training dataset.
            optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
            criterion: The loss function to compute the training loss.
            scheduler (Optional[_LRScheduler], optional): The learning rate scheduler to adjust the learning rate during training. Defaults to None.

        Returns:
            Dict[str, float]: A dictionary containing the average training loss and various metrics such as accuracy, F1 score, precision, and recall.
        """
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
        """
        Validates the model for one epoch using the provided validation data loader and loss criterion.

        Args:
            model (nn.Module): The model to be validated.
            val_loader (DataLoader): The data loader for the validation dataset.
            criterion: The loss function to compute the validation loss.

        Returns:
            Dict[str, float]: A dictionary containing the average validation loss and various metrics such as accuracy, F1 score, precision, and recall.
        """
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
        """
        Interface for training the model. It initializes the model and optimizer, prepares the data loaders,
        and runs the training loop for the specified number of epochs. It also handles early stopping and logging.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing training parameters such as:
            train_data (Dataset): Dataset for training the model.
            val_data (Dataset): Dataset for validating the model during training.

        Returns:
            Dict[str, Any]: A dictionary containing the training history, best epoch, and best validation metrics.
        """
        model: nn.Module = self.model_builder(config).to(self.device)
        optimizer: torch.optim.Optimizer = self.optimizer_builder(model, config)
        
        pin_memory = True if self.device == torch.device("cuda") else False
        train_loader, train_criterion = self.balancing_strategy.prepare(train_data, config, device = self.device)
        val_loader = DataLoader(val_data, batch_size = config["batch_size"], shuffle = self.shuffle, pin_memory=pin_memory)
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
            if self.early_stopping and self.early_stopping(val_metrics[self.main_metric], model = model):
                break

        try:
            if self.early_stopping and self.log_path:
                best_metric = self.early_stopping.best_metric
                torch.save(self.early_stopping.best_model_state, self.log_path / f"best_model_epoch{best_epoch}_metric{best_metric}.pth")
        except Exception as e:
            print(f"Error saving model: {e}")
            print("Model will not be saved. Please check the log path and permissions.")
            

        if self.early_stopping:
            self.early_stopping.reset()
        if self.logger:
            self.logger.close()
            self.logger = None  # Reset logger to avoid memory leaks

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_metrics": {key: history["val"][key][best_epoch] for key in history["val"]},
        }


