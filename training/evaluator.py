from pathlib import Path
import sys
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from datamodule.dataset import get_dataset_and_subsets
from datamodule.splitter import SplitStrategy
from training.trainer import ModelTrainer
from training.logger import TensorBoardLogger
from utils.utility import set_seed


class ModelEvaluator:
    """
    ModelEvaluator is responsible for evaluating a model using cross-validation.
    It initializes the dataset with a dataset configuration, a model trainer, and a data splitter strategy.
    It runs all splits of the dataset and returns the mean and standard deviation of the best validation scores across all folds.
    
    Attributes:
        dataset_config (Dict[str, Any]): Configuration for the dataset.
        trainer (ModelTrainer): The model trainer instance used for training the model.
        trainer_config (Dict[str, Any]): Configuration for the model trainer.
        data_splitter (SplitStrategy): Strategy for splitting the dataset into training and validation sets.
        fold_seeds (Optional[List[int]]): Seeds for each fold in cross-validation.
        log_path (Optional[Path]): Path to save logs and results.  
    """
    def __init__(
        self, 
        dataset_config: Dict[str, Any],
        trainer: ModelTrainer, 
        trainer_cfg: Dict[str, Any],
        data_splitter: SplitStrategy,
    ) -> None:
        self.dataset = None
        self.subset_factory = None
        self.dataset_config = dataset_config
        self.trainer = trainer
        self.log_path: Optional[Path] = None
        self.trainer_config = trainer_cfg
        self.fold_seeds = trainer_cfg.get("fold_seeds", None) 

        self.splitter = data_splitter
        self._check_fold_seeds()
        self._create_dataset()
    
    def _create_dataset(self) -> None:
        """
        Auxiliary method to create the dataset and subset factory based on the dataset configuration.
        """
        self.dataset, self.subset_factory = get_dataset_and_subsets(config=self.dataset_config)
    
    def _check_fold_seeds(self) -> None:
        """
        Checks if the fold seeds are provided, if they are a list of integers, and if their length matches the number of folds in the splitter.
        """
        if self.fold_seeds is None:
            raise ValueError("Fold seeds must be provided in the trainer configuration.")
        if not isinstance(self.fold_seeds, list):
            raise TypeError("Fold seeds must be a list of integers.")
        if len(self.fold_seeds) != self.splitter.k:
            raise ValueError(f"Number of fold seeds ({len(self.fold_seeds)}) does not match the number of folds ({self.splitter.k}).")

    def set_log_path(self, log_path: Union[str, Path]) -> None:
        """
        Sets the log path for saving training logs and results. The log path can be a string or a Path object.

        Args:
            log_path (Union[str, Path]): Path to the directory where logs will be saved.
        """
        if not isinstance(log_path, (str, Path)):
            raise TypeError(f"log_path must be a string or a Path object, not {type(log_path)}")
        if isinstance(log_path, str):
            log_path = Path(log_path).resolve()
        self.log_path = log_path

    def run(self, config: Dict[str, Any]) -> Tuple[float, float]:
        """
        Runs the evaluation of the model using cross-validation. It trains the model on each fold and 
        returns the mean and standard deviation of the best validation scores across all folds.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model training. It should include parameters like learning rate, batch size, etc.

        Returns:
            Tuple[float, float]: A tuple containing the mean and standard deviation of the best validation scores across all folds.
        """
        scores = []
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.splitter.get_splits(self.dataset)):
            set_seed(self.fold_seeds[fold_idx])
            config["used_seed"] = self.fold_seeds[fold_idx]
            
            train_data = self.subset_factory(train_idx, train=True)
            val_data = self.subset_factory(val_idx, train=False)

            try:
                if self.log_path:
                    fold_log_path = self.log_path / f"fold_{fold_idx}"
                    fold_log_path.mkdir(parents=True, exist_ok=True)
                    self.trainer.set_logger(fold_log_path)
                
                results = self.trainer.train(config, train_data, val_data)
                best_score = results["best_val_metrics"][self.trainer.main_metric]
                scores.append(best_score)
                fold_results.append(results["best_val_metrics"])

            except Exception as e:
                print(f"Error during training fold {fold_idx}: {e}")
        
        if self.log_path:
            hparam_log_path = self.log_path / "hparams_summary"
            hparam_log_path.mkdir(parents=True, exist_ok=True)
            logger = TensorBoardLogger(hparam_log_path)
                    
            all_keys = fold_results[0].keys()
            mean_metrics = {
                key: float(np.mean([fold[key] for fold in fold_results])) for key in all_keys
            }
            std_metrics = {
                key: float(np.std([fold[key] for fold in fold_results])) for key in all_keys
            }

            hparams = {k: config[k] for k in config if isinstance(config[k], (int, float, str))}

            final_metrics = {f"mean_{k}": mean_metrics[k] for k in mean_metrics}
            final_metrics.update({f"std_{k}": std_metrics[k] for k in std_metrics})
            
            try:
                hparams_csv_path = self.log_path / "hparams_summary.csv"
                with open(hparams_csv_path, "w") as f:
                    headers = list(hparams.keys()) + list(final_metrics.keys())
                    f.write(",".join(headers) + "\n")

                    values = [str(hparams[k]) for k in hparams] + [f"{final_metrics[k]:.4f}" for k in final_metrics]
                    f.write(",".join(values) + "\n")
            except Exception as e:
                print(f"Error saving hyperparameters to CSV: {e}")
                
            
            logger.log_hparams(hparams=hparams, metrics=final_metrics)
            logger.close()

            return mean_metrics[self.trainer.main_metric], std_metrics[self.trainer.main_metric]
        else:
            return float(np.mean(scores)), float(np.std(scores))
