import sys
import yaml
from typing import Any, Dict, List, Callable, Type, Tuple, Optional, Union
from pathlib import Path

from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from search.search_strategy import SearchStrategy
from datamodule.splitter import SplitStrategy
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator


class Experiment:
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        search_strategy_cls: Type[SearchStrategy],
        search_space: Dict[str, List[Any]],
        trainer_cfg: Dict[str, Any],
        split_strategy: SplitStrategy,
        # transforms: Dict[str, Callable],
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
        # self.transforms = transforms

    def _save_configs(self, search_space: Dict[str, List[Any]], train_cfg: Dict[str, Any]) -> None:
        """
        Saves the search space and training configuration to a YAML file in the specified log path.
        
        Args:
            search_space (Dict[str, List[Any]]): The search space dictionary.
            train_cfg (Dict[str, Any]): The training configuration dictionary.
        """
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
            # transforms = self.transforms,
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
