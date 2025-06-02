import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.grid_search import GridSearch
# from utils.train_utils import train, val
# from utils.data import dataloader, dataset

def experiment():
    config_filename = "config.yml"
    data_path = Path("data\MGS_data\grayscaled_data")
    label_path = Path("data\MGS_data\labels")
    
    dataset = None
    dataloader = None
    
    gridsearch = GridSearch(config_filename = config_filename)
    best_loss, best_parameters = gridsearch.gridsearch(
        train_func = None,
        val_func = None,
        train_loader = None,
        val_loader = None,
    )
    
    with open("results.txt", "w") as file:
        file.write(f"Best Loss: {best_loss}")
        for key, value in best_parameters.items():
            file.write(f"{key}: {value}\n")