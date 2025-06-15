import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Union

import torch
from torch import nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.trainer import EarlyStopping
from training.balancing import BalancingStrategy



@dataclass
class TrainingConfig:
    """
    Configuration class for model training, covering all key aspects including model construction,
    optimization, class balancing, training control, and evaluation.

    Attributes:
        model_builder (Callable[[Dict[str, Any]], nn.Module]):
            Function that constructs the model for each fold during cross-validation in the training function.

        optimizer_builder (Callable):
            Function that creates the optimizer instance for each fold in the training function.

        fold_seeds (List[int]):
            Random seeds used for each fold in cross-validation. The number of folds is determined by the length of this list.

        shuffle (bool):
            Whether to shuffle the training data before each epoch. Default is True.

        early_stopping (Union[bool, EarlyStopping]):
            Enables early stopping based on the validation performance. Can be set to a boolean or an EarlyStopping instance.

        patience (Optional[int]):
            Number of epochs with no improvement in the main metric before training is stopped (used with early stopping).

        main_metric (str):
            Primary validation metric used for:
              - Early stopping (monitored for improvements),
              - Selecting the best model per fold,
              - Averaging metrics across folds.
            The final reported results per fold come from the epoch with the best value of this metric disregarding that the 
            other metrics might have a better value in another epoch.
            Options include: "loss", "acc", "precision", "recall", "f1".

        balancing_strategy (Union[str, BalancingStrategy]):
            Strategy used to handle class imbalance:
              - "no_balancing": Standard `CrossEntropyLoss` without any weighting (default).
              - "weighted_loss": `CrossEntropyLoss` with manually specified class weights.
              - "oversampling": Use `WeightedRandomSampler` to oversample minority classes. Requires `class_weights`.

        class_weights (Optional[Dict[int, float]]):
            Dictionary specifying the weight for each class. Required for "weighted_loss" and "oversampling".
            Minor classes should be assigned higher weights. For example, for class distribution {0: 0.1, 1: 0.9},
            weights should be inverted: {0: 1/0.1 = 10.0, 1: 1/0.9 ≈ 1.11}.
            Normalization to sum to 1.0 is optional but not necessary.

        device (Union[str, torch.device]):
            Computation device to use for training (e.g., "cuda" or "cpu"). Default is "cuda".
    """
    # Required config parameters
    model_builder: Callable[[Dict[str, Any]], nn.Module] 
    optimizer_builder: Callable
    fold_seeds: List[int] 

    # General training behavior
    shuffle: bool = True
    early_stopping: Union[bool, EarlyStopping] = False 
    patience: Optional[int] = None 
    main_metric: str = "loss" 
    balancing_strategy: Union[str, BalancingStrategy] = "no_balancing" 
    class_weights: Optional[Dict[int, float]] = None 
    device: Union[str, torch.device] = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the TrainingConfig instance to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the TrainingConfig.
        """
        return {
            "model_builder": self.model_builder,
            "optimizer_builder": self.optimizer_builder,
            "fold_seeds": self.fold_seeds,
            "shuffle": self.shuffle,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "main_metric": self.main_metric,
            "balancing_strategy": self.balancing_strategy,
            "class_weights": self.class_weights,
            "device": self.device
        }


@dataclass
class GridSearchSpaceConfig:
    """
    Configuration class for defining the hyperparameter search space during model training.
    Each attribute represents a list of possible values to explore during hyperparameter tuning.

    Attributes:
        batch_size (List[int]):
            List of batch sizes to try during training.

        optim (List[str]):
            Optimizers to evaluate (currently ["Adam", "SGD"] are supported).

        learning_rate (List[float]):
            Learning rates to be tested (e.g., [0.001, 0.0001]).

        epochs (List[int]):
            Number of training epochs. Default: [200].

        model_name (List[str]):
            Names of model architectures to evaluate (e.g., ["resnet18"]).

        pretrained (List[bool]):
            Whether to use pretrained weights for the models.

        num_classes (List[int]):
            Number of output classes for the classification task.

        lr_scheduler (List[str]):
            Learning rate scheduler options (currently ["none", "step", "cosine"] are supported).

        scheduler_step_size (Optional[List[int]]):
            Step size for the "step" scheduler (only used when selected).

        scheduler_gamma (Optional[List[float]]):
            Decay factor for the "step" scheduler.

        scheduler_t_max (Optional[List[int]]):
            Maximum number of iterations for the "cosine" scheduler.

        momentum (Optional[List[float]]):
            Momentum values to use with SGD optimizer.

    Methods:
        to_grid_dict() -> Dict[str, List[Any]]:
            Converts the instance into a dictionary suitable for grid search,
            omitting any fields that are set to None.
    """
    # Each of these field must be a list of at least one element or there will be an error
    batch_size: List[int] # Batch size for training
    optim: List[str] # Optimizer to use, e.g., ["Adam", "SGD"]
    weight_decay: List[float]
    learning_rate: List[float] # Learning rate for the optimizer, e.g., [0.001, 0.0001]
    epochs: List[int] = field(default_factory=lambda: [200]) # Number of epochs to train, e.g., [30]
    model_name: List[str] = field(default_factory=lambda: ["resnet18"]) # Model architecture, e.g., ["resnet18"]
    pretrained: List[bool] = field(default_factory=lambda: [False]) # Whether to use pretrained weights, e.g., [False]
    num_classes: List[int] = field(default_factory=lambda: [2]) # Number of classes in the dataset, e.g., [2] for binary classification

    # Optional scheduler-related
    lr_scheduler: List[str] = field(default_factory=lambda: ["none"])
    scheduler_step_size: Optional[List[int]] = None # will only be used if lr_scheduler is "step"
    scheduler_gamma: Optional[List[float]] = None # will only be used if lr_scheduler is "step"
    scheduler_t_max: Optional[List[int]] = None # will only be used if lr_scheduler is "cosine"
    momentum: Optional[List[float]] = None  # will only be used if optim is "SGD"

    def to_grid_dict(self) -> Dict[str, List[Any]]:
        """
        Converts the dataclass into a parameter grid dictionary suitable for sklearn's ParameterGrid.
        Skips None values.
        """
        grid = {}
        for k, v in self.__dict__.items():
            if v is not None:
                grid[k] = v
        return grid


@dataclass
class ParamRange:
    """Defines a continuous range for float or int parameters."""
    type: str  # "int" or "float"
    low: float
    high: float
    log: bool = False  # whether to sample on a log scale

@dataclass
class OptunaSearchSpaceConfig:
    """
    Configuration class for Optuna-based hyperparameter search.
    Allows for both discrete (categorical) and continuous (range-based) definitions.

    Each parameter is either a list of discrete options or a ParamRange object.
    """

    batch_size: Union[List[int], ParamRange]
    optim: List[str]
    weight_decay: Union[List[float], ParamRange]
    learning_rate: Union[List[float], ParamRange]
    epochs: Union[List[int], ParamRange]
    
    lr_scheduler: List[str] # categorical options for learning rate scheduler
    scheduler_step_size: Optional[Union[List[int], ParamRange]] = None
    scheduler_gamma: Optional[Union[List[float], ParamRange]] = None
    scheduler_t_max: Optional[Union[List[int], ParamRange]] = None
    momentum: Optional[Union[List[float], ParamRange]] = None

    model_name: List[str] = field(default_factory=lambda: ["resnet18"])  # Model architecture options
    pretrained: List[bool] = field(default_factory=lambda: [False])
    num_classes: List[int] = field(default_factory=lambda: [2])  # Default for binary classification

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the dataclass to a dictionary for easier handling in Optuna parameter suggestion.
        """
        return self.__dict__