from abc import ABC, abstractmethod
from datetime import datetime
import sys
from typing import Optional, Dict, Any, List, Union, Tuple, Callable, Type

import numpy as np
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms as transforms
from sklearn.model_selection import ParameterGrid, KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.model import get_model
from datamodule.dataset import ImageCSVDataset
from datamodule.transforms import get_train_transforms, get_val_transforms


# ========== Data Related Classes ==========

class SplitStrategy(ABC):
    @abstractmethod
    def get_splits(self, dataset: Dataset) -> List[Tuple[List[int], List[int]]]:
        pass

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
        device: torch.device = None,
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
            model (torch.nn.Module): The model for which the loss is computed. 
            config (Dict[str, Any]): Configuration dictionary containing parameters like batch size, shuffle, weighenings for the classes, etc.
            transforms (Dict[str, Any]): Dictionary of transforms to apply to the dataset.
        
        Returns:
            Tuple[DataLoader, nn.CrossEntropyLoss]: A DataLoader for the dataset and a weighted CrossEntropyLoss function.
        """
        class_weights: Dict[int, float] = config.get("class_weights")
        _, weights = zip(*sorted(class_weights.items()))   

        weights = torch.tensor(weights, dtype = torch.float32, device = device)
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
        device: torch.device = None
    ) -> Tuple[DataLoader, nn.CrossEntropyLoss]:
        class_weights: Dict[int, float] = config.get("class_weights")
        _, weights = zip(*sorted(class_weights.items()))
        
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
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
        self.mode = None
        self.patience = 10

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
        
        self.main_metric, self.mode = self._resolve_main_metric(kwargs.get("main_metric", "f1"))
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
        scheduler_type = config.get("lr_scheduler" "none")
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
                    return EarlyStopping(mode = mode)
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
    
    def _resolve_main_metric(self, main_metric: str) -> str:
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
                inputs, labels = inputs.to(model.device), labels.to(model.device)
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
            "precision": precision_score(all_preds, all_labels, average="weighted"),
            "recall": recall_score(all_preds, all_labels, average="weighted")
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
        self.logger = TensorBoardLogger(log_path)
        
    def train(self, config: Dict[str, Any], train_data: Dataset, val_data: Dataset) -> Dict[str, Any]:
        model: nn.Module = self.model_builder(config).to(self.device)
        optimizer: torch.optim.Optimizer = self.optimizer_builder(model.parameters(), config)
        
        train_loader, train_criterion = self.balancing_strategy.prepare(train_data, config, device = self.device)
        val_loader = DataLoader(val_data, batch_size = config["batch_size"], shuffle = self.shuffle)
        val_criterion = nn.CrossEntropyLoss()
        lr_scheduler = self._resolve_lr_scheduler(config, optimizer)
        

        history = {
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
            else:
                if history["val"][self.main_metric][-1] > history["val"][self.main_metric][best_epoch]:
                    best_epoch = epoch
            
            # Early stopping check if early stopping is enabled
            if self.early_stopping and self.early_stopping(val_metrics[self.main_metric]):
                break

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

class ModelEvaluator:
    def __init__(
        self, 
        dataset: Dataset, 
        trainer: ModelTrainer, 
        data_splitter: SplitStrategy
    ) -> None:
        self.dataset = dataset
        self.trainer = trainer
        self.splitter = data_splitter
        self.log_path: Optional[Path] = None

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
            train_data, val_data = Subset(self.dataset, train_indices), Subset(self.dataset, val_indices)
            
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
            logger = TensorBoardLogger(self.log_path)
            
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

            logger.log_hparams(hparam_dict=hparams, metric_dict=final_metrics)
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
            if cfg["optim"] != "SGD" and "momentum" in cfg and cfg["momentum"] != 0.0:
                continue
            if cfg["scheduler"] == "none" and any(k in cfg for k in ["scheduler_step_size", "scheduler_gamma"]):
                continue
            configs.append(cfg)

        return configs

    def _get_config_log_path(self, config: Dict[str, Any]) -> Optional[Path]:
        if not self.log_base_path:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dir = self.log_base_path / f"config_{timestamp}"
        config_dir.mkdir(parents=True, exist_ok=True)

        # YAML speichern
        config_yaml_path = config_dir / "config.yaml"
        with open(config_yaml_path, "w") as f:
            yaml.dump(self._sanitize_config(config), f)

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


# ========== Experiment Class ==========
class Experiment:
    def __init__(
        self,
        dataset: Dataset,
        search_strategy_cls: Type[SearchStrategy],
        search_space: Dict[str, List[Any]],
        trainer_cfg: Dict[str, Any],
        split_strategy: SplitStrategy,
        log_base_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.dataset = dataset
        self.search_strategy_cls = search_strategy_cls
        self.search_space = search_space
        self.trainer_cfg = trainer_cfg
        self.split_strategy = split_strategy
        self.log_base_path = Path(log_base_path).resolve() if log_base_path else None

    def run(self) -> Tuple[Dict[str, Any], float, float]:
        # Setup Trainer
        trainer = ModelTrainer(
            **self.trainer_cfg,
        )

        # Setup ModelEvaluator
        model_validator = ModelEvaluator(
            dataset=self.dataset,
            trainer=trainer,
            data_splitter=self.split_strategy
        )

        # Setup SearchStrategy
        strategy = self.search_strategy_cls(
            search_space=self.search_space,
            model_validator=model_validator,
            log_base_path=self.log_base_path
        )

        return strategy.search()
    
    
    
if __name__ == "__main__":
    # Generell müssen die Seeds noch irgendwo gesetzt werden, damit in jedem Fold? der Seed geändert wird oder so (oder random?) es muss auf jeden fall eine lösung dafür gefunden werden, dass die Ergebnisse repoduierbar sind
    
    # !!!! nicht so viele auf einmal zu beginn
    search_space = {
        "batch_size": [16, 32],
        "epochs": [30],
        "model_name": ["resnet18"],
        "optim": ["Adam", "SGD"],
        "lr_scheduler": ["step", "cosine", "none"],
        "scheduler_step_size": [10],
        "scheduler_gamma": [0.1],
        "scheduler_t_max": [30],
    }

    # 2. Trainer-Konfiguration definieren
    # !!!! geht dass, dass man hier model_builder und optimizier_builder so übergibt mit und *train_config alles richtig für die initialisierung von ModelTrainer übergeben wird oder müssen die callables explizit definiert werden?
    # !!!! get_model muss daran angepasst werden eine config zu erhalten
    # !!!! optimizer_builder muss noch geschrieben werden
    trainer_config = {
        "model_builder": get_model,
        "optimizer_builder": Callable, 
        "shuffle": True,
        "early_stopping": True,
        "patience": 7,
        "main_metric": "f1",
        "balancing_strategy": "weighted_loss",  # oder "no_balancing", "oversampling"
        "device": "cuda",
        "class_weights": {0: 0.5, 1: 0.5},  # oder None wenn nicht benötigt
    }

    # 3. Dataset vorbereiten
    # !!!! Trainings und Validierungstransforms müssen eigentlich in den model validator oder so und dort an die subsets übergeben werden oder so
    dataset_path = Path("path/to/dataset/train")  # Struktur: data/ & labels/labels.csv
    dataset = ImageCSVDataset(data_dir=dataset_path, transform=)


    # 5. Splitstrategie auswählen
    split_strategy = KFoldSplit(k=5, seed=42)

    # 6. Loggingpfad setzen
    log_base_path = Path("logs/gridsearch_run")

    # 7. Experiment starten
    experiment = Experiment(
        dataset=dataset,
        search_strategy_cls=GridSearch,
        search_space=search_space,
        trainer_cfg=trainer_config,
        transforms=transforms,
        split_strategy=split_strategy,
        log_base_path=log_base_path
    )

    if __name__ == "__main__":
        best_config, best_mean, best_std = experiment.run()
        print("✅ Best configuration found:")
        print(best_config)
        print(f"Mean score: {best_mean:.4f} ± {best_std:.4f}")