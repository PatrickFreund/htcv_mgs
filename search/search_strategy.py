import sys
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import ParameterGrid

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.evaluator import ModelEvaluator



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