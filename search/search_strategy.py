from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple, Optional

import optuna
from sklearn.model_selection import ParameterGrid
import yaml

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

class OptunaSearch(SearchStrategy):
    def __init__(self, search_space: Dict, model_validator: ModelEvaluator, log_base_path: Optional[Path] = None, 
                 n_trials: int = 100, timeout: Optional[int] = None):
        super().__init__(search_space, model_validator, log_base_path)
        self.n_trials = n_trials
        self.timeout = timeout
        
    def _get_config_log_path(self, trial_number: int) -> Optional[Path]:
        if not self.log_base_path:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dir = self.log_base_path / f"optuna_trial_{trial_number}_{timestamp}"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
        
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        config = {}
        
        # Für jeden Parameter im Suchraum entsprechende Optuna suggest_* Methode verwenden
        for param_name, param_values in self.search_space.items():
            if isinstance(param_values, list):
                if all(isinstance(x, int) for x in param_values):
                    # Wähle aus einer Liste von Integer-Werten
                    config[param_name] = trial.suggest_categorical(param_name, param_values)
                elif all(isinstance(x, float) for x in param_values):
                    # Wähle aus einer Liste von Float-Werten
                    config[param_name] = trial.suggest_categorical(param_name, param_values)
                elif all(isinstance(x, str) for x in param_values):
                    # Wähle aus einer Liste von String-Optionen
                    config[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    # Gemischte Typen als kategoriale Werte behandeln
                    config[param_name] = trial.suggest_categorical(param_name, param_values)
            
        # Ähnliche Logik wie in GridSearch anwenden, um ungültige Konfigurationen zu vermeiden
        if config.get("optim") != "SGD" and "momentum" in config and config["momentum"] != 0.0:
            # Setze momentum auf 0 für non-SGD Optimierer
            config["momentum"] = 0.0
            
        # Scheduler-spezifische Parameter setzen
        scheduler = config.get("lr_scheduler", "none")
        if scheduler == "none":
            # Default-Werte für nicht verwendete Parameter
            config["scheduler_step_size"] = 0
            config["scheduler_gamma"] = 0
            config["scheduler_t_max"] = 0
        elif scheduler == "step":
            # Nur relevante Parameter für StepLR
            config["scheduler_t_max"] = 0
        elif scheduler == "cosine":
            # Nur relevante Parameter für CosineAnnealingLR
            config["scheduler_step_size"] = 0
            config["scheduler_gamma"] = 0
            
        return config
    
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
        trial.set_user_attr("std_score", std_score)
        
        return mean_score
    
    def search(self) -> Tuple[Dict[str, Any], float, float]:
        """
        Führt eine Optuna-Optimierung durch, um die beste Hyperparameter-Konfiguration zu finden.
        
        Returns:
            Tuple[Dict[str, Any], float, float]: Beste Konfiguration, deren Mean Score und Std Score.
        """
        study = optuna.create_study(direction="maximize")
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