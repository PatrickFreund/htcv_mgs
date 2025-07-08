from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset


class SplitStrategy(ABC):
    def __init__(self, seed: int = 42):
        self.seed = seed 
        self.k: int
    
    @abstractmethod
    def get_splits(self, dataset: Dataset, seed: Optional[int] = None) -> List[Tuple[List[int], List[int]]]:
        pass

class KFoldSplit(SplitStrategy):
    def __init__(self, k: int, seed: int = 42):
        self.k = k
        self.seed = seed

    def get_splits(self, dataset: Dataset, seed: Optional[int] = None) -> List[Tuple[List[int], List[int]]]:
        """
        Generate k-fold splits for the given dataset.

        Args:
            dataset (Dataset): The dataset to split.
            seed (Optional[int], optional): Random seed for reproducibility. Defaults to None, which uses the instance's seed.

        Returns:
            List[Tuple[List[int], List[int]]]: A list of tuples (one tuple = one fold), each containing two lists:
                - The first list contains indices for the training set.
                - The second list contains indices for the validation set.
        """
        seed = seed if seed is not None else self.seed
        kf = KFold(n_splits=self.k, shuffle=True, random_state=seed)
        indices = list(range(len(dataset)))
        return [(list(train), list(val)) for train, val in kf.split(indices)]

class RandomSplit(SplitStrategy):
    def __init__(self, val_size: float = 0.2, seed: int = 42):
        self.val_size = val_size
        self.seed = seed
        self.k = 1

    def get_splits(self, dataset: Dataset, seed: Optional[int] = None) -> List[Tuple[List[int], List[int]]]:
        """
        Generate a single random split of the dataset into training and validation sets.

        Args:
            dataset (Dataset): The dataset to split.
            seed (Optional[int], optional): Random seed for reproducibility. Defaults to None, which uses the instance's seed.

        Returns:
            List[Tuple[List[int], List[int]]]: A list of one tuple with two lists:
                - The first list contains indices for the training set.
                - The second list contains indices for the validation set.
        """
        seed = seed if seed is not None else self.seed
        total_size = len(dataset)
        indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(seed)).tolist()
        val_count = int(total_size * self.val_size)
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        return [(train_indices, val_indices)]