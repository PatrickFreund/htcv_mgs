import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable, Tuple, Union, List, Any

import numpy as np
import yaml
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid

from old_scripts.training.EarlyStopping import EarlyStopping

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.model import get_model

class GridSearch:
    def __init__(
        self,  
        config_filename: str, 
        model_fn: Callable = get_model,
        log_savepath: Union[str, Path, None] = None,
        experiment_name: str = "gridsearch"
    ):
        # Set standard directory if no other save directories are given
        self.root_proj_dir: Path = Path(__file__).resolve().parents[1]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        experiment_id = f"run_{timestamp}_{experiment_name}"
        log_savepath: Path = Path(log_savepath or self.root_proj_dir / "results" / "logs") / experiment_id
        self.writer = SummaryWriter(log_dir=str(log_savepath))
        
        self.config = self._load_config(config_filename)
        self.param_grid = self._build_filtered_grid(baseline=self.config["Baseline"], hyperparameters=self.config["Hyperparameters"])
        self.model_args = self.config["Model"]
        self.model_fn = model_fn
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(self.config["General"]["seed"])

    def __del__(self):
        self._close()

    def _close(self):
        if self.writer:
            self.writer.close()

    def _load_config(self, config_filename: str) -> Dict[str, Any]:
        config_dir = self.root_proj_dir / "configs"
        config_path: Path = config_dir / config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "Baseline" not in config or "Hyperparameters" not in config:
            raise ValueError("Config file must contain 'Baseline' and 'Hyperparameters' sections.")
        return config
        
    def _build_filtered_grid(self, baseline: Dict, hyperparameters: Dict) -> List[Dict]:
        raw_combinations = ParameterGrid(hyperparameters)
        configs = []

        for override in raw_combinations:
            cfg = baseline.copy()
            cfg.update(override)

            if cfg["optim"] != "SGD" and "momentum" in override and cfg["momentum"] != 0.0:
                continue

            if cfg["scheduler"] == "none" and any(k in override for k in ["scheduler_step_size", "scheduler_gamma"]):
                continue

            configs.append(cfg)
        return configs
    
    def _get_loss_fn(self, loss_fn: str):
        if loss_fn == "CrossEntropy":
            return torch.nn.CrossEntropyLoss()
        elif loss_fn == "MSELoss":
            return torch.nn.MSELoss()
        else:
            raise ValueError(f"{loss_fn} is not supported for this gridsearch implementation")

    def _get_optimizer(self, params: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
        if params["optim"] == "SGD":
            return torch.optim.SGD(
                params=model.parameters(),
                lr=params["lr"],
                momentum=params["momentum"]
            )
        elif params["optim"] == "Adam":
            return torch.optim.Adam(
                params=model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"]
            )
        else:
            raise ValueError(f"{params['optim']} is not supported for this gridsearch implementation")

    def _get_lr_scheduler(self, params: Dict, optimizer: torch.optim.Optimizer):
        if params["scheduler"] == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=params["scheduler_step_size"],
                gamma=params["scheduler_gamma"]
            )
        elif params["scheduler"] == "none":
            return None
        else:
            raise ValueError(f"{params['scheduler']} is not supported for this gridsearch implementation")

    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def gridsearch(
            self,
            train_func: Callable,
            val_func: Callable,
            train_loader: List[data.DataLoader],
            val_loader: List[data.DataLoader],
            save_best_model_path: Union[str, Path, None] = None
    ) -> Tuple[float, Dict]:
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
        best_params: Dict = {}
        best_loss: float = float("inf")
        best_model_state = None

        for i, params in enumerate(self.param_grid):
            print(f"{i}. Testing Parameters: {params}")

            model = self.model_fn(**self.model_args)  # new model for each kombination
            model.to(self.device)
            criterion = self._get_loss_fn(params["loss_fn"])
            optimizer = self._get_optimizer(params, model)
            scheduler = self._get_lr_scheduler(params, optimizer)

            early_stopping = EarlyStopping(patience=10, delta=0.01)

            epochs = params["epochs"]
            train_losses, train_metrics = [], []

            # for fold in dataloader_list:
            #      train_loader, val_loader = fold
            for epoch in range(epochs):
                train_loss, train_metric = train_func(model, train_loader, criterion, optimizer, scheduler)
                val_loss, val_metrics = val_func(model, val_loader, criterion)

                train_losses.append(train_loss)
                train_metrics.append(train_metric)

                # Logging
                model_name = self.model_args.get("name", model.__class__.__name__)
                self.writer.add_text(f"Run_{i}/Model_Name", model_name)
                self.writer.add_text("Seed", self.config["General"]["seed"])
                self.writer.add_scalar(f"Run_{i}/Train_Loss", train_loss, epoch)

                for metric_name, value in train_metric.items():
                    self.writer.add_scalar(f"Run_{i}/Train_{metric_name}", value, epoch)
                self.writer.add_scalar(f"Run_{i}/Val_Loss", val_loss, epoch)
                for metric_name, value in val_metrics.items():
                    self.writer.add_scalar(f"Run_{i}/{metric_name}", value, epoch)

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            self.writer.add_scalar(f"Run_{i}/Final_Val_Loss", val_loss, len(train_losses) - 1)
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f"Run_{i}/{metric_name}", value, len(train_losses) - 1)
            self.writer.add_hparams(params, {"val_loss": val_loss, **val_metrics})

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
                best_model_state = model.state_dict()

        if save_best_model_path and best_model_state is not None:
            torch.save(best_model_state, save_best_model_path)

        return best_loss, best_params


# Fold erstellung auslagern

# das training und loggin auslagern und nun in gridsearch aufrufen

# neue Klasse für Dataset erstellen die idx und df nimmt und daraus ein torch.utils.data.Dataset macht 
# und somit transforms für train und val separat anwendbar sind

# in jeder Iteration der Parameterkombinationen die folds neu erstellen aber mit fixen (train_idx, val_idx) in splits

