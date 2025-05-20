import sys
from pathlib import Path
from typing import Dict, Callable, Tuple

import torch
import torch.utils.data as data
from sklearn.model_selection import ParameterGrid

sys.path.append(str(Path(__file__).resolve().parents[1]))


class GridSearch:
    def __init__(self, model: torch.nn.Module, hyperparameters: Dict, model_savepath: str):
        if not isinstance(hyperparameters, Dict):
            raise ValueError("'parameter'-Parameter should be type dict.")
        self.model = model
        self.params: Dict = hyperparameters
        self.model_savepath: str = model_savepath
    
    def _get_loss_fn(self, loss_fn: str):
        if loss_fn == "CrossEntropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif loss_fn == "KLDivLoss":
            criterion = torch.nn.KLDivLoss()
        else:
            raise ValueError(f"{loss_fn} is not supported for this gridsearch implementation")
        return criterion
    
    def _get_optimizer(self, optim: str):
        if optim == "SGD":
            optimizer = torch.optim.SGD()
        elif optim == "Adam": 
            optimizer = torch.optim.Adam()
        else:
            raise ValueError(f"{optim} is not supported for this gridsearch implementation")
        return optimizer

    def gridsearch(self, train_func: Callable, val_func: Callable, train_loader: data.DataLoader, val_loader: data.DataLoader) -> Tuple[float, Dict]:
        """Function performs exhaustiv gridsearch of all possible combinations of the given 
        hyperparameter grid.

        Args:
            train_func (Callable): Function that takes in the model, train_loader, loss_fn, optimizier and epochs and returns the training loss
            val_func (Callable): Function that takes in the trained model, val_loader, loss_fn and returns the val_loss
            train_loader (data.DataLoader): DataLoader for the training subset 
            val_loader (data.DataLoader): DataLoader of the validation subset
        
        Return:
            best_loss (float): best validation loss 
            best_params (Dict): best hyperparameters
        """
        
        param_grid = ParameterGrid(self.params)
        best_params: Dict = {}
        best_loss: float = float("inf")
        
        for i, params in enumerate(param_grid):
            print(f"{i}. Testing Parameters: {params}")
            
            criterion = self._get_loss_fn(params["loss_fn"])
            optimizer = self._get_optimizer(params["optim"])
            model = self.model
            
            epochs = params["epochs"]
            train_loss = train_func(model, train_loader, criterion, optimizer, epochs)
            val_loss   = val_func(model, val_loader, criterion)
            
            print(f"{i}. Testing Parameters: Train-Loss: {train_loss:.2f} -- Val-Loss: {val_loss:.2f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
            
        return best_loss, best_params
            
            
        
    
    def simple_gridsearch(self, train_func: Callable, val_func: Callable, train_loader: data.DataLoader, val_loader: data.DataLoader):
        pass