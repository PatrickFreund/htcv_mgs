from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler



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
        """
        Prepares a DataLoader and a CrossEntropyLoss function without any class balancing.

        Args:
            train_subset (Dataset): The train_subset containing the training data.
            config (Dict[str, Any]): Configuration dictionary containing parameters like batch size.
            device (torch.device): The device to which the tensors should be moved (e.g., "cuda" or "cpu").

        Returns:
            Tuple[DataLoader, nn.CrossEntropyLoss]: _description_
        """
        pin_memory = True if device == torch.device("cuda") else False
        lossfn = nn.CrossEntropyLoss()
        loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=config.get("shuffle", True), pin_memory=pin_memory)
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
        pin_memory = True if device == torch.device("cuda") else False
        weights: List[float]
        if self.class_weights is None:
            raise ValueError("Class weights must be provided for WeightedLossBalancing strategy.")
        _, weights = zip(*sorted(self.class_weights.items()))
        weights: torch.Tensor = torch.tensor(weights, dtype = torch.float32, device = device)
        lossfn = nn.CrossEntropyLoss(weight = weights)
        loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=config.get("shuffle", True), pin_memory=pin_memory)
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
        """
        Prepares a DataLoader with oversampling of minority classes and a CrossEntropyLoss function.

        Args:
            train_subset (Dataset): The train_subset containing the training data.
            config (Dict[str, Any]): Configuration dictionary containing parameters like batch size.
            device (torch.device): The device to which the tensors should be moved (e.g., "cuda" or "cpu").

        Returns:
            Tuple[DataLoader, nn.CrossEntropyLoss]: A DataLoader for the dataset with oversampling and a CrossEntropyLoss function.
        """
        pin_memory = True if device == torch.device("cuda") else False
        weights: List[float]
        if self.class_weights is None:
            raise ValueError("Class weights must be provided for OversamplingBalancing strategy.")
        _, weights = zip(*sorted(self.class_weights.items()))
        weights: torch.Tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        labels = torch.tensor([label for _, label in train_subset], dtype=torch.long, device=device)
        
        sampling_weights: torch.Tensor = weights[labels]
        sampler = WeightedRandomSampler(sampling_weights, len(sampling_weights), replacement=True)
        loader = DataLoader(train_subset, batch_size=config["batch_size"], sampler=sampler, pin_memory=pin_memory)
        lossfn = nn.CrossEntropyLoss()
        return loader, lossfn

