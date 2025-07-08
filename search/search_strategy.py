import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import optuna
import yaml
from optuna.samplers import TPESampler
from sklearn.model_selection import ParameterGrid

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import ParamRange
from training.evaluator import ModelEvaluator


class SearchStrategy(ABC):
    """
    Abstract base class for search strategies used in hyperparameter optimization.
    
    Attributes:
        search_space (Dict): The parameter space to explore.
        model_validator (ModelEvaluator): Object responsible for training and evaluating models.
        log_base_path (Optional[Path]): Directory where search results and logs will be saved.
    """
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
    """
    GridSearch implements a grid search strategy for hyperparameter optimization.
    """
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes sure that the configuration values are of types that can be serialized to 
        YAML for logging purposes. If the values are not of type str, int, or float, they are excluded.

        Args:
            config (Dict[str, Any]): configuration dictionary to sanitize.

        Returns:
            Dict[str, Any]: Sanitized configuration dictionary with only serializable values.
        """
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
            cfg_copy = dict(cfg)
        
            # Filter/normalize momentum
            if cfg_copy.get("optim") != "SGD":
                cfg_copy["momentum"] = 0.0

            # Handle lr_scheduler-specific params
            scheduler = cfg_copy.get("lr_scheduler", "none")

            if scheduler == "none":
                for key in ["scheduler_step_size", "scheduler_gamma", "scheduler_t_max"]:
                    cfg_copy.pop(key, None)
                cfg_copy["scheduler_step_size"] = 0
                cfg_copy["scheduler_gamma"] = 0
                cfg_copy["scheduler_t_max"] = 0

            elif scheduler == "step":
                cfg_copy.pop("scheduler_t_max", None)
                cfg_copy["scheduler_step_size"] = cfg_copy.get("scheduler_step_size", 0)
                cfg_copy["scheduler_gamma"] = cfg_copy.get("scheduler_gamma", 0)
                cfg_copy["scheduler_t_max"] = 0

            elif scheduler == "cosine":
                cfg_copy.pop("scheduler_step_size", None)
                cfg_copy.pop("scheduler_gamma", None)
                cfg_copy["scheduler_t_max"] = cfg_copy.get("scheduler_t_max", 0)
                cfg_copy["scheduler_step_size"] = 0
                cfg_copy["scheduler_gamma"] = 0

            else:
                raise ValueError(f"Unbekannter lr_scheduler: {scheduler}")

            configs.append(cfg_copy)

        # Remove duplicates
        seen = set()
        unique_configs = []
        for cfg in configs:
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                unique_configs.append(cfg)

        return unique_configs

    def _get_config_log_path(self, config: Dict[str, Any]) -> Optional[Path]:
        """
        Creates a directory for logging the configuration details. The directory is named with a timestamp
        and contains a YAML file with the sanitized configuration.

        Args:
            config (Dict[str, Any]): The configuration dictionary to log.

        Returns:
            Optional[Path]: Path to the directory where the configuration is logged, or None if no log base path is set.
        """
        if not self.log_base_path:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dir = self.log_base_path / f"config_{timestamp}"
        config_dir.mkdir(parents=True, exist_ok=True)

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

    def evaluate_config(self, config: Dict[str, Any]) -> Tuple[float, float]:
        """
        Auxiliary method to evaluate a single configuration using the model validator.

        Args:
            config (_type_): Configuration dictionary to evaluate.

        Returns:
            Tuple[float, float]: Mean score and standard deviation of the target metric across folds (if Kfold is used).
        """
        return self.model_validator.run(config)

class OptunaSearch(SearchStrategy):
    """
    OptunaSearch implements a hyperparameter optimization strategy using the Optuna library.
    
    Attributes:
        search_space (Dict): The parameter space to explore.
        model_validator (ModelEvaluator): Object responsible for training and evaluating models.
        log_base_path (Optional[Path]): Directory where search results and logs will be saved.
    Args:
        search_space (Dict): The parameter space to explore.
        model_validator (ModelEvaluator): Object responsible for training and evaluating models.
        log_base_path (Optional[Path]): Directory where search results and logs will be saved.
        n_trials (int): Number of trials to run for hyperparameter optimization.
        timeout (Optional[int]): Maximum time in seconds to run the optimization.
        startup_trials (int): Number of initial trials to run before switching to TPE sampler.
    """
    def __init__(self, search_space: Dict, model_validator: ModelEvaluator, log_base_path: Optional[Path] = None, 
                 n_trials: int = 100, timeout: Optional[int] = None, startup_trials: int = 10, seed: Optional[int] = None):
        super().__init__(search_space, model_validator, log_base_path)
        self.n_trials = n_trials
        self.timeout = timeout
        self.start_up_trials = startup_trials
        self.seed = seed
        
    def _get_config_log_path(self, trial_number: int) -> Optional[Path]:
        """
        Creates a directory for logging the configuration details of a specific trial.

        Args:
            trial_number (int): The trial number for which the log directory is created.

        Returns:
            Optional[Path]: Path to the directory where the trial configuration is logged, or None if no log base path is set.
        """
        if not self.log_base_path:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dir = self.log_base_path / f"optuna_trial_{trial_number}_{timestamp}"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
      
    def _suggest_single_param(self, trial: optuna.Trial, name: str, value: Union[List[Any], ParamRange]):
        """
        Suggests a single parameter for the Optuna trial based on its type.
        
        Args:
            trial (optuna.Trial): The current Optuna trial.
            name (str): The name of the parameter to suggest.
            value (Union[List[Any], ParamRange]): The value or range of values for the parameter.
        Returns:
            Any: The suggested value for the parameter.
        """
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
        """
        Suggests hyperparameters for the Optuna trial based on the search space.
        This method ensures that parameters with dependencies are only suggested when their prerequisites are met.
        For example, it only suggests momentum if SGD is chosen as the optimizer, and it suggests scheduler parameters based on the chosen scheduler type.

        Args:
            trial (optuna.Trial): The current Optuna trial for which parameters are suggested.

        Returns:
            Dict[str, Any]: A dictionary containing the suggested hyperparameters for the trial.
        """
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
        main_metric:str = self.model_validator.trainer_config.get("main_metric")
        if main_metric is None:
            raise ValueError("Trainer configuration must contain 'main_metric' to resolve objective directive.")
        if main_metric.startswith("loss") or main_metric.startswith("error"):
            return "minimize"
        else:
            return "maximize"

    def objective(self, trial: optuna.Trial) -> float:
        """Objective-Funktion für Optuna"""
        
        config = self._suggest_params(trial)
        log_path = self._get_config_log_path(trial.number)
        
        if log_path:
            self.model_validator.set_log_path(log_path)
            config_yaml_path = log_path / "config.yaml"
            with open(config_yaml_path, "w") as f:
                yaml.dump({k: v for k, v in config.items() if isinstance(v, (str, int, float))}, f)
        
        mean_score, std_score = self.evaluate_config(config)
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("std_score", std_score)
        
        return mean_score
    
    
    def search(self) -> Tuple[Dict[str, Any], float, float]:
        """
        The main search method that runs the Optuna optimization process.
        
        Returns:
            Tuple[Dict[str, Any], float, float]: Best configuration found, its mean score
        """
        print(f"Start Optuna search... with n_trials={self.n_trials}, start_up_trials={self.start_up_trials}, seed={self.seed}")
        self.direction = self._resolve_objective_direction()
        print(f"Optuna objective direction: {self.direction}")
        sampler = TPESampler(n_startup_trials=self.start_up_trials, seed=self.seed)
        study = optuna.create_study(direction=self.direction, sampler=sampler, study_name="MGS_Optuna_Search")
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        
        best_trial = study.best_trial
        best_config = best_trial.params
        best_mean_score = best_trial.value
        best_std_score = best_trial.user_attrs.get("std_score", None)
        
        print(f">> best parameter combination has: {best_mean_score:.4f} ± {best_std_score:.4f}")
        print(f">> best parameter combination: {best_config}")
        
        return best_config, best_mean_score, best_std_score
    
    def evaluate_config(self, config):
        """
        Evaluates a single configuration using the model validator.
        """
        return self.model_validator.run(config)