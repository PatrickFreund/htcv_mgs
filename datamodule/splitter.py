from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

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
