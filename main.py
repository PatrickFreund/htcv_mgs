import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from datamodule.dataset import ImageCSVDataset, CachedImageCSVDataset
from datamodule.transforms import get_train_transforms, get_val_transforms
from utils.model import get_model
from utils.optimizer import get_optimizer
from configs.config import TrainingConfig, GridSearchSpaceConfig, OptunaSearchSpaceConfig, ParamRange
from search.experiment import Experiment
from search.search_strategy import GridSearch, OptunaSearch, SearchStrategy
from datamodule.splitter import KFoldSplit, RandomSplit, SplitStrategy
from configs.constants import CLASS_WEIGHTS, DATASET_MEAN, DATASET_STD

def run_grid_search() -> Tuple[Dict[str, Any], Dict[str, Any], SplitStrategy, type[SearchStrategy], Dict[str, Any]]:
    # 1. Define the search space for hyperparameter tuning 
    search_space = GridSearchSpaceConfig(
        batch_size=[32],
        optim=["Adam", "SGD"],
        learning_rate=[0.1, 0.01, 0.001, 0.0001],
        epochs=[200],
        lr_scheduler=["none", "step", "cosine"],
        scheduler_step_size=[5, 30],
        scheduler_gamma=[0.5, 0.1],
        scheduler_t_max=[200],
        momentum=[0.0, 0.8, 0.9]
    ).to_grid_dict()
    
    # 2. Define the training configuration
    class_weights = {k: 1 / v for k, v in CLASS_WEIGHTS.items()}  # Invert weights for CrossEntropyLoss
    class_weights = {k: v / sum(class_weights.values()) for k, v in class_weights.items()} 
    fold_seeds = [42, 43, 44]  # Seeds for each cross-validation fold
    trainer_config = TrainingConfig(
        model_builder=get_model,
        optimizer_builder=get_optimizer,
        fold_seeds=fold_seeds,  # seed for each cross-validation fold
        shuffle=True,
        early_stopping=True,
        patience=20,
        main_metric="loss",
        balancing_strategy="weighted_loss",
        class_weights=class_weights,  # darf nicht None sein, wenn balancing_strategy != "no_balancing"
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).to_dict()
    
    # 3. Define the split strategy
    # Only one seed for splitting the data since the trails are better comparable
    split_strategy = KFoldSplit(k=3, seed=fold_seeds[0])
    # split_strategy = RandomSplit(val_size=0.2, seed=fold_seeds[0])  
    
    # 4. Define search strategy
    search_strategy_cls = GridSearch
    
    return search_space, trainer_config, split_strategy, search_strategy_cls, {}

def run_optuna_search() -> Tuple[Dict[str, Any], Dict[str, Any], SplitStrategy, type[SearchStrategy], Dict[str, Any]]:
    search_space = OptunaSearchSpaceConfig(
        batch_size=[32],
        optim=["Adam", "SGD"],
        learning_rate=ParamRange(type="float", low=1e-5, high=1e-1, log=True),
        epochs=[100],
        lr_scheduler=["none", "step", "cosine"],
        scheduler_step_size= ParamRange(type="int", low=5, high=20, log=False),
        scheduler_gamma = ParamRange(type="float", low=0.1, high=0.9, log=False),
        scheduler_t_max=[100],
        momentum= ParamRange(type="float", low=0.5, high=0.9, log=False)
    ).to_dict()
    
    class_weights = {k: 1 / v for k, v in CLASS_WEIGHTS.items()}  # Invert weights for CrossEntropyLoss
    class_weights = {k: v / sum(class_weights.values()) for k, v in class_weights.items()} 
    fold_seeds = [42]
    trainer_config = TrainingConfig(
        model_builder=get_model,
        optimizer_builder=get_optimizer,
        fold_seeds=fold_seeds,  # seed for each cross-validation fold
        shuffle=True,
        early_stopping=True,
        patience=15,
        main_metric="loss",
        balancing_strategy="weighted_loss",
        class_weights=class_weights,  # darf nicht None sein, wenn balancing_strategy != "no_balancing"
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).to_dict()
    
    # Only one seed for splitting the data since the trails are better comparable
    # splitter = KFoldSplit(k=3, seed=fold_seeds[0])
    splitter = RandomSplit(val_size=0.2, seed=fold_seeds[0])
    
    search_strategy_cls = OptunaSearch
    search_strategy_params = {
        "n_trials": 100, # Number of trials for Optuna
        # "timeout": 3600,  # Timeout in seconds for the search
    }
    
    return search_space, trainer_config, splitter, search_strategy_cls, search_strategy_params
    
    
    

if __name__ == "__main__":
    
    # search_space, trainer_config, split_strategy, search_strategy_cls, search_strategy_params = run_grid_search()
    search_space, trainer_config, split_strategy, search_strategy_cls, search_strategy_params = run_optuna_search()
    
    # 3. Prepare the dataset
    data_dir = Path(__file__).resolve().parent / "data" / "MGS_data"
    # dataset = ImageCSVDataset(data_dir = data_dir)
    dataset = CachedImageCSVDataset(data_dir = data_dir) # caches the whole dataset in ram to speed up training
    transform = {
        "train": get_train_transforms(mean = DATASET_MEAN, std = DATASET_STD), 
        "val": get_val_transforms(mean = DATASET_MEAN, std = DATASET_STD) 
    }
    
        # 4. Set path for loggin if wished 
    log_dir = Path(__file__).resolve().parent / "results"
    run_number = len(list(Path.glob(log_dir, "run_*"))) + 1
    log_dir = log_dir / f"run_{run_number}"
    
    
        # 5. Initialize experiment 
    experiment = Experiment(
        dataset=dataset,
        search_strategy_cls=search_strategy_cls,
        search_strategy_params=search_strategy_params,
        search_space=search_space,
        trainer_cfg=trainer_config,
        transforms= transform,
        split_strategy=split_strategy,
        log_base_path=log_dir
    )

    
    best_config, best_mean, best_std = experiment.run()
    print("✅ Best configuration found:")
    print(best_config)
    print(f"Mean score: {best_mean:.4f} ± {best_std:.4f}")