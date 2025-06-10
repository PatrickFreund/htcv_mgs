import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from datamodule.dataset import ImageCSVDataset
from datamodule.transforms import get_train_transforms, get_val_transforms
from utils.model import get_model
from utils.optimizer import get_optimizer
from configs.config import TrainingConfig, SearchSpaceConfig
from search.experiment import Experiment
from search.search_strategy import GridSearch
from datamodule.splitter import KFoldSplit, RandomSplit

# Konstants
CLASS_WEIGHTS = {0: 0.5744292237442923, 1: 0.42557077625570777}
DATASET_MEAN = 0.36995071172714233
DATASET_STD = 0.21818380057811737


if __name__ == "__main__":
    
    # 1. Define the search space for hyperparameter tuning 
    search_space = SearchSpaceConfig(
        batch_size=[32],
        optim=["Adam", "SGD"],
        learning_rate=[0.01, 0.05, 0.001, 0.0005, 0.0001],
        epochs=[200],
        lr_scheduler=["step", "cosine", "none"],
        scheduler_step_size=[5],
        scheduler_gamma=[0.5],
        scheduler_t_max=[50],
        momentum=[0.9, 0.8, 0.0]
    ).to_grid_dict()
    
    # 2. Define the training configuration
    class_weights = {k: 1 / v for k, v in CLASS_WEIGHTS.items()}  # Invert weights for CrossEntropyLoss
    class_weights = {k: v / sum(class_weights.values()) for k, v in class_weights.items()} 
    trainer_config = TrainingConfig(
        model_builder=get_model,
        optimizer_builder=get_optimizer,
        fold_seeds=[42, 43, 44],  # seed for each cross-validation fold
        shuffle=True,
        early_stopping=True,
        patience=30,
        main_metric="loss",
        balancing_strategy="weighted_loss",
        class_weights=class_weights,  # darf nicht None sein, wenn balancing_strategy != "no_balancing"
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).to_dict()
    
    # 3. Prepare the dataset
    data_dir = Path(__file__).resolve().parent / "data" / "MGS_data"
    dataset = ImageCSVDataset(data_dir = data_dir)
    transform = {
        "train": get_train_transforms(mean = DATASET_MEAN, std = DATASET_STD), 
        "val": get_val_transforms(mean = DATASET_MEAN, std = DATASET_STD) 
    }
    
    # 4. Define the split strategy
    split_strategy = KFoldSplit(k=3)
    # split_strategy = RandomSplit(val_size=0.2)  
    
    # 5. Define search strategy
    search_strategy_cls = GridSearch
    
    # 4. Set path for loggin if wished 
    log_dir = Path(__file__).resolve().parent / "results"
    run_number = len(list(Path.glob(log_dir, "run_*"))) + 1
    log_dir = log_dir / f"run_{run_number}"
    
    
    # 5. Initialize experiment 
    experiment = Experiment(
        dataset=dataset,
        search_strategy_cls=search_strategy_cls,
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