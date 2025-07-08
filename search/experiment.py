import sys
from pathlib import Path
from typing import Any, Dict, List, Type, Tuple, Optional, Union

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))
from datamodule.splitter import SplitStrategy
from search.search_strategy import SearchStrategy
from training.evaluator import ModelEvaluator
from training.trainer import ModelTrainer



class Experiment:
    """
    Encapsulates a complete hyperparameter search experiment, including dataset configuration,
    search strategy, search space, trainer settings, and data splitting.

    This class provides a unified interface for executing different search strategies (e.g., GridSearch, Optuna),
    leveraging the abstract SearchStrategy interface for flexibility.
    """
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        search_strategy_cls: Type[SearchStrategy],
        search_space: Dict[str, List[Any]],
        trainer_cfg: Dict[str, Any],
        split_strategy: SplitStrategy,
        search_strategy_params: Dict[str, Any] = {},
        log_base_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.dataset_config = dataset_config
        self.search_strategy_cls = search_strategy_cls
        self.search_strategy_params = search_strategy_params
        self.search_space = search_space
        self.trainer_cfg = trainer_cfg
        self.split_strategy = split_strategy
        self.log_base_path = Path(log_base_path).resolve() if log_base_path else None

    def _save_configs(self, search_space: Dict[str, List[Any]], train_cfg: Dict[str, Any]) -> None:
        """
        Saves the search space and training configuration to a YAML file in the specified log path.
        
        Args:
            search_space (Dict[str, List[Any]]): The search space dictionary.
            train_cfg (Dict[str, Any]): The training configuration dictionary.
        """
        if not self.log_base_path:
            return
        
        config = {
            "search_space": search_space,
            "trainer_config": train_cfg
        }
        config_path = self.log_base_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def run(self) -> Tuple[Dict[str, Any], float, float]:
        self.log_base_path.mkdir(parents=True, exist_ok=True) if self.log_base_path else None
        self._save_configs(self.search_space, self.trainer_cfg)
        
        # Setup Trainer
        trainer = ModelTrainer(
            **self.trainer_cfg,
        )

        # Setup ModelEvaluator
        model_validator = ModelEvaluator(
            dataset_config=self.dataset_config,
            trainer=trainer,
            trainer_cfg=self.trainer_cfg,
            data_splitter=self.split_strategy,
        )

        # Setup SearchStrategy
        strategy = self.search_strategy_cls(
            search_space=self.search_space,
            model_validator=model_validator,
            log_base_path=self.log_base_path,
            **self.search_strategy_params
        )

        return strategy.search()
