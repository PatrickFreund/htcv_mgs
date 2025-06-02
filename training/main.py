import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.grid_search import GridSearch
from datamodule.dataloader import get_kfold_dataloaders
from utils.model import get_model
from utils.train_epoch import train_func, val_func



if __name__ == "__main__":
    grid_search = GridSearch(config_filename="config.yml", model_fn=get_model)

    dataloaders = get_kfold_dataloaders()
    train_loader, val_loader = dataloaders["train"], dataloaders["val"]
    
    grid_search.gridsearch(
        train_func = train_func,
        val_func = val_func,
        train_loader=train_loader,
        val_loader=val_loader,
    )
