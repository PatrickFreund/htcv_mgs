from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple, Optional, Union

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import ParameterGrid
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.evaluator import ModelEvaluator
from configs.config import ParamRange



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

class OptunaSearch(SearchStrategy):
    def __init__(self, search_space: Dict, model_validator: ModelEvaluator, log_base_path: Optional[Path] = None, 
                 n_trials: int = 100, timeout: Optional[int] = None, startup_trials: int = 10, seed: Optional[int] = None):
        super().__init__(search_space, model_validator, log_base_path)
        self.n_trials = n_trials
        self.timeout = timeout
        self.start_up_trials = startup_trials
        self.seed = seed
        
    def _get_config_log_path(self, trial_number: int) -> Optional[Path]:
        if not self.log_base_path:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dir = self.log_base_path / f"optuna_trial_{trial_number}_{timestamp}"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
      
    def _suggest_single_param(self, trial: optuna.Trial, name: str, value: Union[List[Any], ParamRange]):
        if isinstance(value, ParamRange):
            if value.type == "float":
                return trial.suggest_float(name, value.low, value.high, log=value.log)
            elif value.type == "int":
                return trial.suggest_int(name, int(value.low), int(value.high), log=value.log)
            else:
                raise ValueError(f"Unsupported ParamRange type: {value.type}")
        elif isinstance(value, list):
            return trial.suggest_categorical(name, value)
        else:
            raise ValueError(f"Unsupported parameter format for {name}: {value}")
  
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        config = {}

        # 1. Suggest most important parameters first which have higher dependencies
        optims_params = self.search_space.get("optim")
        lr_schedulers_params = self.search_space.get("lr_scheduler")
        assert optims_params is not None, "Parameter 'optim' must be defined in the search space."
        assert lr_schedulers_params is not None, "Parameter 'lr_scheduler' must be defined in the search space."
        config["optim"] = trial.suggest_categorical("optim", optims_params)
        config["lr_scheduler"] = trial.suggest_categorical("lr_scheduler", lr_schedulers_params)

        # 2. Only suggest momemtum if SGD is chosen
        if config["optim"] == "SGD":
            val = self.search_space.get("momentum")
            assert val is not None, "Parameter 'momentum' must be defined in the search space for SGD."
            config["momentum"] = self._suggest_single_param(trial, "momentum", val)
        else:
            config["momentum"] = 0.0

        # 3. Suggest scheduler parameters based on the chosen scheduler
        if config["lr_scheduler"] == "step":
            val_step = self.search_space.get("scheduler_step_size")
            val_gamma = self.search_space.get("scheduler_gamma")
            assert val_step is not None, "Parameter 'scheduler_step_size' must be defined in the search space for step scheduler."
            assert val_gamma is not None, "Parameter 'scheduler_gamma' must be defined in the search space for step scheduler."
            config["scheduler_step_size"] = self._suggest_single_param(trial, "scheduler_step_size", val_step)
            config["scheduler_gamma"] = self._suggest_single_param(trial, "scheduler_gamma", val_gamma)
            config["scheduler_t_max"] = 0

        elif config["lr_scheduler"] == "cosine":
            val = self.search_space.get("scheduler_t_max")
            assert val is not None, "Parameter 'scheduler_t_max' must be defined in the search space for cosine scheduler."
            config["scheduler_t_max"] = self._suggest_single_param(trial, "scheduler_t_max", val)
            config["scheduler_step_size"] = 0
            config["scheduler_gamma"] = 0

        else:  # "none"
            config["scheduler_step_size"] = 0
            config["scheduler_gamma"] = 0
            config["scheduler_t_max"] = 0

        # 3. Rest of the parameters can be suggested directly
        for param_name in ["batch_size", "epochs", "model_name", "pretrained", "num_classes"]:
            val = self.search_space.get(param_name)
            assert val is not None, f"Parameter {param_name} must be defined in the search space."
            config[param_name] = val[0] if isinstance(val, list) else val
        
        for param_name in ["learning_rate", "weight_decay"]:
            val = self.search_space.get(param_name)
            assert val is not None, f"Parameter {param_name} must be defined in the search space."
            config[param_name] = self._suggest_single_param(trial, param_name, val)
        
        return config

    def _resolve_objective_direction(self) -> str:
        """
        Resolves the objective directive for Optuna based on the main metric.
        This is used to determine whether to maximize or minimize the objective function.
        """
        main_metric = self.model_validator.trainer_config.get("main_metric")
        if main_metric is None:
            raise ValueError("Trainer configuration must contain 'main_metric' to resolve objective directive.")
        if main_metric.startswith("loss") or main_metric.startswith("error"):
            return "minimize"
        else:
            return "maximize"

    def objective(self, trial: optuna.Trial) -> float:
        """Objective-Funktion für Optuna"""
        # Parameter für diesen Trial vorschlagen
        config = self._suggest_params(trial)
        
        # Log-Pfad für diese Konfiguration erstellen
        log_path = self._get_config_log_path(trial.number)
        if log_path:
            self.model_validator.set_log_path(log_path)
            
            # Optional: Konfiguration als YAML speichern
            config_yaml_path = log_path / "config.yaml"
            with open(config_yaml_path, "w") as f:
                yaml.dump({k: v for k, v in config.items() if isinstance(v, (str, int, float))}, f)
        
        # Konfiguration evaluieren
        mean_score, std_score = self.evaluate_config(config)
        
        # Speichere Standardabweichung als Trial-Attribut
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("std_score", std_score)
        
        if self.direction == "maximize":
            # if main metric is chosen as main metric, we want to maximize the objective function therefore solutions
            # with a high mean score and low std score are preferred
            return mean_score - std_score 
        else:
            # if loss is chosen as main metric, we want to minimize the objective function therefore solutions
            # with a low mean score and low std score are preferred
            return mean_score + std_score  # Beispiel: Minimieren der Summe zwischen Mean und Std Score
    
    def search(self) -> Tuple[Dict[str, Any], float, float]:
        """
        Führt eine Optuna-Optimierung durch, um die beste Hyperparameter-Konfiguration zu finden.
        
        Returns:
            Tuple[Dict[str, Any], float, float]: Beste Konfiguration, deren Mean Score und Std Score.
        """
        print(f"Starte Optuna-Suche... with n_trials={self.n_trials}, start_up_trials={self.start_up_trials}, seed={self.seed}")
        self.direction = self._resolve_objective_direction()
        print(f"Optuna objective direction: {self.direction}")
        sampler = TPESampler(n_startup_trials=self.start_up_trials, seed=self.seed)  # Setze den Sampler mit dem Seed
        study = optuna.create_study(direction=self.direction, sampler=sampler, study_name="MGS_Optuna_Search")
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        
        best_trial = study.best_trial
        best_config = best_trial.params
        best_mean_score = best_trial.value
        best_std_score = best_trial.user_attrs.get("std_score", None)
        
        print(f"Beste Konfiguration gefunden mit Score: {best_mean_score:.4f} ± {best_std_score:.4f}")
        print(f"Parameter: {best_config}")
        
        return best_config, best_mean_score, best_std_score
    
    def evaluate_config(self, config):
        """Evaluiert eine einzelne Konfiguration mit dem Model-Validator"""
        return self.model_validator.run(config)