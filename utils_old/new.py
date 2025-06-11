from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
import sys
import os
from typing import Optional, Dict, Any, List, Union, Tuple, Callable, Type

import numpy as np
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.model_selection import ParameterGrid, KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.model import get_model
from utils.optimizer import get_optimizer
from datamodule.dataset import ImageCSVDataset, TransformSubset
from datamodule.transforms import get_train_transforms, get_val_transforms


# For testing purposes:
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.utils.data import Subset
from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path

def debug_visualize_transform(transform_subset:TransformSubset, config: dict, sample_idx: int = 3):
    """
    Visualisiert ein Originalbild und seine transformierte Version
    aus einem TransformSubset.

    Args:
        transform_subset (TransformSubset): Subset mit definierter transform-Funktion.
        config (dict): Dict mit "mean" und "std" (für Denormalisierung).
        sample_idx (int): Index im Subset, der visualisiert werden soll.
    """
    # Index und zugrundeliegender Datensatz
    original_dataset = transform_subset.base_dataset
    original_idx = transform_subset.indices[sample_idx]

    # Bildinformationen abrufen
    row = original_dataset.labels.iloc[original_idx]
    img_name = row["filename"]
    img_path = original_dataset.img_dir / img_name

    if not img_path.exists():
        img_stem = Path(img_name).stem
        possible_files = list(original_dataset.img_dir.glob(f"{img_stem}.*"))
        if not possible_files:
            raise FileNotFoundError(f"Bild {img_name} nicht gefunden.")
        img_path = possible_files[0]

    # Lade Originalbild
    original_img = Image.open(img_path)
    if original_img.mode == "L":
        original_img = original_img.convert("L")
        cmap = "gray"
    else:
        original_img = original_img.convert("RGB")
        cmap = None

    # Transformiertes Bild abrufen
    transformed_img, label = transform_subset[sample_idx]

    # Denormalisierung
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        return (tensor * std + mean).clamp(0, 1)

    mean = config.get("mean", [0.5])
    std = config.get("std", [0.5])
    denorm_img = denormalize(transformed_img.clone(), mean, std)

    # Plot erstellen
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original_img, cmap=cmap)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Transformiert (Denormalisiert)")
    if denorm_img.shape[0] == 1:
        plt.imshow(denorm_img.squeeze(), cmap="gray")
    else:
        plt.imshow(denorm_img.permute(1, 2, 0))  # CHW -> HWC
    plt.axis("off")

    plt.suptitle(f"Label: {label} | Datei: {img_name}")
    plt.tight_layout()
    plt.show()

def test_image_is_grayscale():
    dataset = ImageCSVDataset(data_dir=r"C:\Users\Freun\Desktop\htcv_mgs\data\MGS_data")
    train_transform = get_train_transforms(mean = 0.36995071172714233, std = 0.21818380057811737)
    for or_img, label in dataset:
        img = train_transform(or_img)
        assert isinstance(img, torch.Tensor), "Bild sollte ein Tensor sein"
        assert img.dim() == 3, "Bild sollte 3 Dimensionen haben (Kanal, Höhe, Breite)"
        assert img.shape[0] == 1, "Bild sollte 1 Kanal haben (Grayscale)"
        assert img.shape[1] == 224 and img.shape[2] == 224, "Bildgröße sollte 224x224 sein"
        print(f"Bildgröße: {or_img.size}, Label: {label}")
        print(f"Transformiertes Bildgröße: {img.shape}")
        print(f"Transformiertes Bild dtype: {img.dtype}")
    
    or_img, label = dataset[0]
    print(f"Bildgröße: {or_img.size}, Label: {label}")
    print(f"Transformiertes Bildgröße: {img.shape}")
    print(f"Transformiertes Bild dtype: {img.dtype}")

    #plot original and transformed image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(or_img, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Transformiert")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def test_train_transforms_randomness():
    # Dummy-Graustufenbild
    img = torch.rand(1, 224, 224)
    train_transform = get_train_transforms(mean = 0.36995071172714233, std = 0.21818380057811737)
    val_transform = get_val_transforms(mean = 0.36995071172714233, std = 0.21818380057811737)
    
    
    img1 = transform(img)
    img2 = transform(img)

    # Bei Random Transforms sollten die Ausgaben unterschiedlich sein
    assert not torch.equal(img1, img2), "Random Transforms sollten unterschiedliche Ergebnisse liefern"

def test_resnet18_input_output():
    config = {
        "model_name": "resnet18",
        "pretrained": False,
        "num_classes": 2,
    }
    model = get_model(config)
    
    # Eingabe: Batch aus 1-Kanal-Bildern mit 224x224
    assert isinstance(model, nn.Module), "Modell sollte eine Instanz von nn.Module sein"
    print(f"Modellarchitektur:\n{model}")
    dummy_input = torch.randn(4, 1, 224, 224)
    output = model(dummy_input)
    print(f"Modellausgabeform: {output.shape}")

    assert output.shape == (4, 2), "Modell muss 2 Outputneuronen liefern für binary classification"


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across various libraries.
    
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== Data Related Classes ==========

class SplitStrategy(ABC):
    @abstractmethod
    def get_splits(self, dataset: Dataset) -> List[Tuple[List[int], List[int]]]:
        pass

    def set_seed(self, seed: int):
        """
        Sets the random seed for reproducibility.
        """
        self.seed = seed

class KFoldSplit(SplitStrategy):
    def __init__(self, k: int, seed: int = 42):
        self.k = k
        self.seed = seed

    def get_splits(self, dataset: Dataset):
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        indices = list(range(len(dataset)))
        return [(list(train), list(val)) for train, val in kf.split(indices)]

class RandomSplit(SplitStrategy):
    def __init__(self, val_size: float = 0.2, seed: int = 42):
        self.val_size = val_size
        self.seed = seed

    def get_splits(self, dataset: Dataset):
        total_size = len(dataset)
        indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(self.seed)).tolist()
        val_count = int(total_size * self.val_size)
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        return [(train_indices, val_indices)]


# ========== Balancing Related Strategies ==========
class BalancingStrategy(ABC):
    """
    Abstract Base Class for balancing strategies in data loading and training.
    This class defines the interface for preparing data loaders and any custom loss functions
    that may be required for balancing classes in a dataset.
    """ 
    def __init__(self, class_weights: Optional[Dict[int, float]] = None):
        """
        Initializes the balancing strategy with optional class weights.

        Args:
            class_weights (Optional[Dict[int, float]], optional): 
                A dictionary mapping class indices to their corresponding weights.
        """
        self.class_weights = class_weights
        self.name = ""

    @abstractmethod
    def prepare(
        self,
        train_subset: Dataset,
        config: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[DataLoader, Optional[torch.nn.Module]]:
        pass

class NoBalancingStrategy(BalancingStrategy):
    """
    This stategy is a placeholder for scenarios where no balancing strategy is requested.
    It simply returns a DataLoader and a simple CrossEntropyLoss without any class balancing.
    """
    def __init__(self):
        super().__init__(class_weights = None)
        self.name = "no_balancing"
        
    def prepare(
        self, 
        train_subset: Dataset,
        config: Dict[str, Any],
        device: torch.device
    ) -> Tuple[DataLoader, nn.CrossEntropyLoss]:
        lossfn = nn.CrossEntropyLoss()
        loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=config.get("shuffle", True))
        return loader, lossfn

class WeightedLossBalancing(BalancingStrategy):
    """
    This strategy computes class weights based on the frequency of each class in the dataset
    and applies these weights to a CrossEntropyLoss function.
    """
    def __init__(self, class_weights: Optional[Dict[int, float]] = None):
        super().__init__(class_weights)
        self.name = "weighted_loss"

    def prepare(
        self, 
        train_subset: Dataset,
        config: Dict[str, Any],
        device: torch.device
    ) -> Tuple[DataLoader, nn.CrossEntropyLoss]:
        """
        Prepares a DataLoader and a weighted CrossEntropyLoss function based on the class distribution in the dataset.
        Args:
            train_subset (Dataset): The train_subset containing the training data.
            config (Dict[str, Any]): Configuration dictionary containing parameters like batch size.
            device (torch.device): The device to which the tensors should be moved (e.g., "cuda" or "cpu").
        
        Returns:
            Tuple[DataLoader, nn.CrossEntropyLoss]: A DataLoader for the dataset and a weighted CrossEntropyLoss function.
        """
        weights: List[float]
        if self.class_weights is None:
            raise ValueError("Class weights must be provided for WeightedLossBalancing strategy.")
        _, weights = zip(*sorted(self.class_weights.items()))
        weights: torch.Tensor = torch.tensor(weights, dtype = torch.float32, device = device)
        lossfn = nn.CrossEntropyLoss(weight = weights)
        loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=config.get("shuffle", True))
        return loader, lossfn

class OversamplingBalancing(BalancingStrategy):
    """
    This strategy oversamples the minority classes in the dataset to balance the class distribution.
    """
    def __init__(self, class_weights: Optional[Dict[int, float]] = None):
        super().__init__(class_weights)
        self.name = "oversampling"
    
    def prepare(
        self, 
        train_subset: Dataset,
        config: Dict[str, Any],
        device: torch.device
    ) -> Tuple[DataLoader, nn.CrossEntropyLoss]:
        weights: List[float]
        if self.class_weights is None:
            raise ValueError("Class weights must be provided for OversamplingBalancing strategy.")
        _, weights = zip(*sorted(self.class_weights.items()))
        weights: torch.Tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        labels = torch.tensor([label for _, label in train_subset], dtype=torch.long, device=device)
        
        sampling_weights = weights[labels]
        sampler = torch.utils.data.WeightedRandomSampler(sampling_weights, len(sampling_weights), replacement=True)
        loader = DataLoader(train_subset, batch_size=config["batch_size"], sampler=sampler)
        lossfn = nn.CrossEntropyLoss()
        return loader, lossfn




# ========== Logging Related Classes ==========
class Logger(ABC):
    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int):
        pass

    @abstractmethod
    def log_params(self, params: dict):
        pass

    @abstractmethod
    def log_model(self, model, name: str):
        pass
    
    @abstractmethod
    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        pass
    
    @abstractmethod
    def close(self):
        pass

class TensorBoardLogger(Logger):
    def __init__(self, log_dir: Union[str, Path]):
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def log_params(self, params):
        for key, value in params.items():
            self.writer.add_text(f"param/{key}", str(value))

    def log_model(self, model, name):
        self.writer.add_graph(model)

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]): 
        self.writer.add_hparams(
            hparams,
            metrics
        )
    
    def close(self):
        self.writer.close()


# ========== Training Related Classes ==========
class EarlyStopping:
    def __init__(self, mode: str,  patience: int = 10):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_metric = None
    
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
        criterion: _Loss,
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
        criterion: _Loss
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

class ModelEvaluator:
    def __init__(
        self, 
        dataset: Dataset, 
        trainer: ModelTrainer, 
        trainer_cfg: Dict[str, Any],
        data_splitter: SplitStrategy,
        transforms: Dict[str, Callable]
    ) -> None:
        self.dataset = dataset
        self.trainer = trainer
        self.splitter = data_splitter
        self.log_path: Optional[Path] = None
        self.transforms = transforms
        self.fold_seeds = trainer_cfg.get("fold_seeds", None)
        self._check_fold_seeds()
        
    def _check_fold_seeds(self):
        if self.fold_seeds is None:
            raise ValueError("Fold seeds must be provided in the trainer configuration.")
        if not isinstance(self.fold_seeds, list):
            raise TypeError("Fold seeds must be a list of integers.")
        if len(self.fold_seeds) != self.splitter.k:
            raise ValueError(f"Number of fold seeds ({len(self.fold_seeds)}) does not match the number of folds ({self.splitter.k}).")

    def set_log_path(self, log_path: Union[str, Path]) -> None:
        if not isinstance(log_path, (str, Path)):
            raise TypeError(f"log_path must be a string or a Path object, not {type(log_path)}")
        if isinstance(log_path, str):
            log_path = Path(log_path).resolve()
        self.log_path = log_path

    def run(self, config: Dict[str, Any]) -> Tuple[float, float]:
        scores = []
        fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(self.splitter.get_splits(self.dataset)):
            set_seed(self.fold_seeds[fold_idx])
            config["used_seed"] = self.fold_seeds[fold_idx]

            train_data = TransformSubset(self.dataset, train_indices, self.transforms["train"])
            val_data = TransformSubset(self.dataset, val_indices, self.transforms["val"])
            
            # testing
            # debug_visualize_transform(train_data, config)
            
            try:
                if self.log_path:
                    fold_log_path = self.log_path / f"fold_{fold_idx}"
                    fold_log_path.mkdir(parents=True, exist_ok=True)
                    self.trainer.set_logger(fold_log_path)
                
                results = self.trainer.train(config, train_data, val_data)
                best_score = results["best_val_metrics"][self.trainer.main_metric]
                scores.append(best_score)
                fold_results.append(results["best_val_metrics"])

            except Exception as e:
                print(f"Error during training fold {fold_idx}: {e}")
        
        if self.log_path:
            hparam_log_path = self.log_path / "hparams_summary"
            hparam_log_path.mkdir(parents=True, exist_ok=True)
            logger = TensorBoardLogger(hparam_log_path)
                    
            all_keys = fold_results[0].keys()
            mean_metrics = {
                key: float(np.mean([fold[key] for fold in fold_results])) for key in all_keys
            }
            std_metrics = {
                key: float(np.std([fold[key] for fold in fold_results])) for key in all_keys
            }

            hparams = {k: config[k] for k in config if isinstance(config[k], (int, float, str))}

            final_metrics = {f"mean_{k}": mean_metrics[k] for k in mean_metrics}
            final_metrics.update({f"std_{k}": std_metrics[k] for k in std_metrics})
            
            try:
                hparams_csv_path = self.log_path / "hparams_summary.csv"
                with open(hparams_csv_path, "w") as f:
                    # Header
                    headers = list(hparams.keys()) + list(final_metrics.keys())
                    f.write(",".join(headers) + "\n")

                    # Werte
                    values = [str(hparams[k]) for k in hparams] + [f"{final_metrics[k]:.4f}" for k in final_metrics]
                    f.write(",".join(values) + "\n")
            except Exception as e:
                print(f"Error saving hyperparameters to CSV: {e}")
                
            
            logger.log_hparams(hparams=hparams, metrics=final_metrics)
            logger.close()

            return mean_metrics[self.trainer.main_metric], std_metrics[self.trainer.main_metric]
        else:
            return float(np.mean(scores)), float(np.std(scores))


# ========== Search Strategy Implementation ==========
class SearchStrategy(ABC):
    def __init__(self, search_space: Dict, model_validator: ModelEvaluator, log_base_path: Optional[Path] = None):
        self.search_space = search_space
        self.model_validator = model_validator
        self.log_base_path = Path(log_base_path) if log_base_path else None

    @abstractmethod
    def search(self):
        """
        Implements the search algorithm to find the best configuration.
        """

    @abstractmethod
    def evaluate_config(self, config):
        """
        Evaluates one given configuration.
        """

class GridSearch(SearchStrategy):
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in config.items() if isinstance(v, (str, int, float))}
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """
        Takes the search space dictionary of form {param_name: [value1, value2, ...], ...}
        and generates all combinations of hyperparameters and filters out pointless combinations.

        Returns:
            List[Dict[str, Any]]: List of configurations with valid hyperparameter combinations.
        """
        raw_combinations = ParameterGrid(self.search_space)
        configs = []
        
        for cfg in raw_combinations:
            if cfg.get("optim") != "SGD" and "momentum" in cfg and cfg["momentum"] != 0.0:
                continue
            
            scheduler = cfg.get("lr_scheduler", "none")

            # Entferne unnötige scheduler-Parameter je nach Scheduler-Typ
            cfg_copy = dict(cfg)  # mache Kopie, um Original nicht zu verändern

            # # Muss wieder entfernt werden !!!!!!!!!!!!!!!
            # if scheduler == "step" and cfg.get("learning_rate") == 0.001:
            #     continue
            
            if scheduler == "none":
                # Remove all scheduler params
                for key in ["scheduler_step_size", "scheduler_gamma", "scheduler_t_max"]:
                    cfg.pop(key, None)
                # Add defaults (optional if downstream expects keys)
                cfg["scheduler_step_size"] = 0
                cfg["scheduler_gamma"] = 0
                cfg["scheduler_t_max"] = 0

            elif scheduler == "step":
                # Remove unused param
                cfg.pop("scheduler_t_max", None)
                # Ensure required ones exist
                cfg["scheduler_step_size"] = cfg.get("scheduler_step_size", 0)
                cfg["scheduler_gamma"] = cfg.get("scheduler_gamma", 0)
                cfg["scheduler_t_max"] = 0

            elif scheduler == "cosine":
                cfg.pop("scheduler_step_size", None)
                cfg.pop("scheduler_gamma", None)
                cfg["scheduler_t_max"] = cfg.get("scheduler_t_max", 0)
                cfg["scheduler_step_size"] = 0
                cfg["scheduler_gamma"] = 0

            else:
                raise ValueError(f"Unbekannter lr_scheduler: {scheduler}")

            configs.append(cfg_copy)

        return configs

    def _get_config_log_path(self, config: Dict[str, Any]) -> Optional[Path]:
        if not self.log_base_path:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dir = self.log_base_path / f"config_{timestamp}"
        config_dir.mkdir(parents=True, exist_ok=True)

        # YAML speichern
        # config_yaml_path = config_dir / "config.yaml"
        # with open(config_yaml_path, "w") as f:
        #     yaml.dump(self._sanitize_config(config), f)

        return config_dir

    def search(self) -> Tuple[Dict[str, Any], float, float]:
        """
        Searches for the best configuration by evaluating all combinations of hyperparameters
        in the search space. It uses the model validator to evaluate each configuration and 
        keeps track of the best configuration based on the mean score across folds.

        Returns:
            Tuple[Dict[str, Any], float, float]: Best configuration, its mean score, and std score.
        """
        best_mean_score = float('-inf')
        best_std_score = None
        best_config = None

        for config in self._generate_grid():
            print(config)
        print(len(self._generate_grid()), "configurations found in search space.")

        for config in self._generate_grid():
            log_path = self._get_config_log_path(config)
            if log_path:
                self.model_validator.set_log_path(log_path)

            mean_score, std_score = self.evaluate_config(config)
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_std_score = std_score
                best_config = config

        return best_config, best_mean_score, best_std_score

    def evaluate_config(self, config):
        return self.model_validator.run(config)

@dataclass
class SearchSpaceConfig:
    # Each of these field must be a list of at least one element or there will be an error
    batch_size: List[int] # Batch size for training
    optim: List[str] # Optimizer to use, e.g., ["Adam", "SGD"]
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

# ========== Experiment Class ==========
class Experiment:
    def __init__(
        self,
        dataset: Dataset,
        search_strategy_cls: Type[SearchStrategy],
        search_space: Dict[str, List[Any]],
        trainer_cfg: Dict[str, Any],
        split_strategy: SplitStrategy,
        transforms: Dict[str, Callable],
        log_base_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.dataset = dataset
        self.search_strategy_cls = search_strategy_cls
        self.search_space = search_space
        self.trainer_cfg = trainer_cfg
        self.split_strategy = split_strategy
        self.log_base_path = Path(log_base_path).resolve() if log_base_path else None
        self.transforms = transforms

    def _save_configs(self, search_space: Dict[str, List[Any]], train_cfg: Dict[str, Any]) -> None:
        """
        Saves the search space and training configuration to a YAML file in the specified log path.
        
        Args:
            search_space (Dict[str, List[Any]]): The search space dictionary.
            train_cfg (Dict[str, Any]): The training configuration dictionary.
        """
        config = {
            "search_space": search_space,
            "trainer_config": train_cfg
        }
        config_path = self.log_base_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def run(self) -> Tuple[Dict[str, Any], float, float]:
        self.log_base_path.mkdir(parents=True, exist_ok=True) if self.log_base_path else None
        self._save_configs(self.search_space, self.trainer_cfg)
        
        # Setup Trainer
        trainer = ModelTrainer(
            **self.trainer_cfg,
        )

        # Setup ModelEvaluator
        model_validator = ModelEvaluator(
            dataset=self.dataset,
            trainer=trainer,
            trainer_cfg=self.trainer_cfg,
            transforms = self.transforms,
            data_splitter=self.split_strategy,
        )

        # Setup SearchStrategy
        strategy = self.search_strategy_cls(
            search_space=self.search_space,
            model_validator=model_validator,
            log_base_path=self.log_base_path
        )

        return strategy.search()

if __name__ == "__main__":
    # Search Space Definition
    # search_space = SearchSpaceConfig(
    #     batch_size=[32],
    #     optim=["Adam", "SGD"],
    #     learning_rate=[0.01, 0.05, 0.001, 0.0005, 0.0001],
    #     epochs=[200],
    #     lr_scheduler=["step", "cosine", "none"],
    #     scheduler_step_size=[5],
    #     scheduler_gamma=[0.5],
    #     scheduler_t_max=[50],
    #     momentum=[0.9, 0.8, 0.0]
    # ).to_grid_dict()
    
    search_space = SearchSpaceConfig(
        batch_size=[32],
        optim=["SGD"],
        learning_rate=[0.001],
        epochs=[10],
        lr_scheduler=["cosine"],
        scheduler_t_max=[50],
        momentum=[0.8]
    ).to_grid_dict()
    
    
    # Split Strategy Definition
    split_strategy = KFoldSplit(k=2)
    
    # Training Configuration
    class_weights = {0: 0.5744292237442923, 1: 0.42557077625570777}  # 0: no_pain, 1: pain
    class_weights = {k: 1 / v for k, v in class_weights.items()}  # Invert weights for CrossEntropyLoss
    class_weights = {k: v / sum(class_weights.values()) for k, v in class_weights.items()}  # Normalize to sum to 1.0
    print(f"Class weights: {class_weights}")
    trainer_config = TrainingConfig(
        model_builder=get_model,  # Funktion zum Erstellen des Modells
        optimizer_builder=get_optimizer,  # Funktion zum Erstellen des Optimizers
        fold_seeds=[42, 43],  # Folds für K-Fold Cross-Validation
        shuffle=False,
        early_stopping=True,
        patience=30,
        main_metric="loss",
        balancing_strategy="weighted_loss",  # oder "oversampling" oder "no_balancing"
        class_weights=class_weights,  # darf nicht None sein, wenn balancing_strategy != "no_balancing"
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).to_dict()
    
    # Dataset Definition
    dataset_path = Path(r"C:\Users\Freun\Desktop\htcv_mgs\data\MGS_data")  # Struktur: data/ & labels/labels.csv
    dataset = ImageCSVDataset(data_dir=dataset_path)
    transform = {
        "train": get_train_transforms(mean = 0.37203550954887965, std = 0.21801310757916936),  # Funktion zum Erstellen der Trainings-Transforms
        "val": get_val_transforms(mean = 0.37203550954887965, std = 0.21801310757916936)  # Funktion zum Erstellen der Validierungs-Transforms
    }
    
    
    # 6. Loggingpfad setzen
    log_base_path = Path(r"C:\Users\Freun\Desktop\htcv_mgs\results\run_2")

    # 7. Experiment starten
    experiment = Experiment(
        dataset=dataset,
        search_strategy_cls=GridSearch,
        search_space=search_space,
        trainer_cfg=trainer_config,
        transforms= transform,
        split_strategy=split_strategy,
        log_base_path=log_base_path
    )

    
    best_config, best_mean, best_std = experiment.run()
    print("✅ Best configuration found:")
    print(best_config)
    print(f"Mean score: {best_mean:.4f} ± {best_std:.4f}")