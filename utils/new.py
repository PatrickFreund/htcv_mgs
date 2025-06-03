from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Tuple, Callable

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

# Muss noch so angepasst werden, dass der Trainer nicht immer den Pfad korrekt angeben muss oder zumindest nur foldx/paramx/... oder so
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



# ========== Training Related Classes ==========
class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_metric = None

    def should_stop(self, metric):
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# Da kein neuer Trainer erstellt wird muss EarlyStopping entweder immer neu erstellt werden oder am ende von train gelöscht werden
class ModelTrainer:
    def __init__(
        self,
        model_builder: Callable,
        optimizer_builder: Callable,
        **kwargs, 
    ) -> None:
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder

        self.shuffle = kwargs.get("shuffle", True)
        early_stopp = kwargs.get("early_stopping", False)
        patience = kwargs.get("patience", None)
        self.early_stopping = self._resolve_early_stopping(early_stopping=early_stopp, patience=patience)
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

    def _resolve_early_stopping(self, early_stopping: Union[bool, EarlyStopping], patience: Optional[int] = None) -> Optional[EarlyStopping]:
        if patience is not None and not isinstance(patience, int):
            raise TypeError("Patience must be an integer if specified.")
        
        if isinstance(early_stopping, bool):
            if early_stopping:
                if not patience:
                    raise ValueError("Patience must be specified if early stopping is enabled.")
                return EarlyStopping(patience = patience)
            else:
                return None
        elif isinstance(early_stopping, EarlyStopping):
            if patience is not None:
                raise ValueError("Patience should not be specified if an EarlyStopping instance is provided.")
            return early_stopping
        else:
            raise TypeError("Early stopping must be an boolean or an instance of EarlyStopping.")
    
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

    def _val_func(
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

    def train(self, config: Dict[str, Any], train_data: Dataset, val_data: Dataset) -> float:
        model: nn.Module = self.model_builder(config)
        model = model.to(self.device)
        optimizier: torch.optim.Optimizer = self.optimizer_builder(model.parameters(), config)
        
        train_loader, train_criterion = self.balancing_strategy.prepare(train_data, config, device = self.device)
        val_loader = DataLoader(val_data, batch_size = config["batch_size"], shuffle = self.shuffle)
        val_criterion = nn.CrossEntropyLoss()
        lr_scheduler = self._resolve_lr_scheduler(config, optimizier)
        
        history = {
            "train": {"loss": [], "acc": [], "f1": [], "precision": [], "recall": []},
            "val": {"loss": [], "acc": [], "f1": [], "precision": [], "recall": []}
        }

        for epoch in range(config['epochs']):
            train_metrics = self._train_one_epoch(model, train_loader, optimizier, train_criterion, lr_scheduler)
            val_metrics = self._validate(model, val_loader, val_criterion)
            
            for key in history["train"]:
                history["train"][key].append(train_metrics[key])
                history["val"][key].append(val_metrics[key])

            if self.early_stopping and self.early_stopping.should_stop(val_metrics["loss"]):
                break

        return history



class CrossValidator:
    def __init__(
        self, 
        dataset: Dataset, 
        trainer: ModelTrainer, 
        data_splitter: SplitStrategy,  
        transforms: Dict[str, transforms.Compose], 
        k: int = 5
    ) -> None:
        self.dataset = dataset
        self.trainer = trainer
        self.splitter = data_splitter
        self.transforms = transforms
        self.k = k

    def run(self, config: Dict[str, Any]) -> float:
        scores = []
        for train_indices, val_indices in self.splitter.get_splits(self.dataset):
            train_data, val_data = Subset(self.dataset, train_indices), Subset(self.dataset, val_indices)
            score = self.trainer.train(config, train_data, val_data)
            scores.append(score)
        return sum(scores) / len(scores)



# ========== Search Strategy Implementation ==========
class SearchStrategy(ABC):
    def __init__(self, search_space: Dict, cross_validator: CrossValidator):
        self.search_space = search_space
        self.cross_validator = cross_validator

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
    def search(self) -> Tuple[Dict[str, Any], float]:
        best_score = float('-inf')
        best_config = None
        for config in self._generate_grid():
            score = self.evaluate_config(config)
            if score > best_score:
                best_score = score
                best_config = config
        return best_config, best_score

    def evaluate_config(self, config):
        return self.cross_validator.run(config)

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









class Experiment:
    def __init__(
        self,
        dataset,
        search_strategy_cls,
        search_space,
        trainer_cfg,
        transforms,
        k_folds=5
    ) -> None:
        self.dataset = dataset
        self.search_strategy_cls = search_strategy_cls
        self.search_space = search_space
        self.trainer_cfg = trainer_cfg
        self.transforms = transforms
        self.k_folds = k_folds

    def run(self) -> Tuple[Dict[str, Any], float]:
        splitter = DataSplitter()
        early_stopping = EarlyStopping(patience=self.trainer_cfg['patience'])
        trainer = ModelTrainer(**self.trainer_cfg, early_stopping=early_stopping)
        cross_validator = CrossValidator(
            dataset=self.dataset,
            trainer=trainer,
            data_splitter=splitter,
            transforms=self.transforms,
            k=self.k_folds
        )
        strategy = self.search_strategy_cls(
            search_space=self.search_space,
            cross_validator=cross_validator
        )
        return strategy.search()

experiment = Experiment(
    dataset=my_dataset,
    search_strategy_cls=GridSearch,
    search_space=search_space,
    trainer_cfg=trainer_config,
    transforms=TransformProvider(),
    k_folds=5
)
best_config, best_score = experiment.run()




class ModelTrainer:
    def __init__(self, ..., logger: Optional[Logger] = None):
        self.logger = logger

    def train(self, config, train_loader, val_loader):
        ...
        for epoch in range(config['epochs']):
            train_loss = self._train_one_epoch(...)
            val_score = self._validate(...)

            if self.logger:
                self.logger.log_scalar("train/loss", train_loss, epoch)
                self.logger.log_scalar("val/score", val_score, epoch)

            if self.early_stopping.should_stop(val_score):
                break
        return val_score

log_dir = f"runs/config_{hashlib.md5(str(config).encode()).hexdigest()[:8]}"
logger = TensorBoardLogger(log_dir)
trainer = ModelTrainer(..., logger=logger)


# ========== trainer_cfg ==========
trainer_cfg = {
    "model_builder": build_model,
    "optimizer_builder": build_optimizer,
    "balancing_strategy": WeightedLossBalancing(),
    # kein 'shuffle', kein 'batch_size' hier
}

# ========== config (Hyperparameter-Suchraum) ==========
search_space = {
    "lr": [0.001, 0.01],
    "batch_size": [32, 64],
    "momentum": [0.0, 0.9],
    "epochs": [10],
    "scheduler": ["none", "step"],
    ...
}