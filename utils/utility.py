import os
import shutil
from pathlib import Path

def get_unique_path(base_path: Path) -> Path:
    """
    Function to get a unique path by appending an incrementing number to the base path.
    If the base path does not exist, it returns the base path as is.
    """
    if not isinstance(base_path, Path):
        base_path = Path(base_path)
    
    if not base_path.exists():
        return base_path
    else:
        counter = 0
        while True:
            new_path = Path(f"{base_path}_{counter}")
            if not new_path.exists():
                return new_path
            counter += 1
            
def save_args_to_file(args, save_dir: Path, filename="config.json"):
    """
    Saves the argparse arguments into a JSON file for reproducibility.
    
    Args:
        args (argparse.Namespace): The parsed arguments.
        save_dir (Path): Directory where the config file will be saved.
        filename (str): Name of the config file (default 'config.json').
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    args_dict = vars(args)  # Namespace -> dict
    
    with open(save_dir / filename, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    print(f"Saved config to {save_dir / filename}")