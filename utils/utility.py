import os
import random
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across various libraries.
    
    Args:
        seed (int): The seed value to set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

def process_and_summarize_results(
    results_dir: Path,
    output_csv_name: str = "combined_results.csv",
    metrics: list = ['loss', 'acc', 'f1', 'precision', 'recall'],
    show_markdown: bool = True
) -> str:
    """
    Loads all .csv result files from a given directory, merges them into one DataFrame,
    formats metrics as 'mean ± std', and returns a markdown table string.

    Args:
        results_dir (Path): Directory containing the CSV result files.
        output_csv_name (str): Name of the combined output file to save.
        metrics (list): List of metric names to format.
        show_markdown (bool): Whether to print the markdown table.

    Returns:
        str: Markdown-formatted summary table.
    """
    # Load and merge all CSV files
    csv_files = list(results_dir.rglob("*.csv"))
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    raw_combined_path = results_dir / f"{output_csv_name.replace('.csv', '_raw.csv')}"
    df.to_csv(raw_combined_path, index=False)

    for metric in metrics:
        mean_col = f'mean_{metric}'
        std_col = f'std_{metric}'
        if mean_col in df.columns and std_col in df.columns:
            df[metric] = df[mean_col].round(4).astype(str) + ' ± ' + df[std_col].round(4).astype(str)
    drop_cols = [f'mean_{m}' for m in metrics] + [f'std_{m}' for m in metrics]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    columns_to_keep = [
        'batch_size', 'epochs', 'learning_rate', 'lr_scheduler',
        'scheduler_step_size', 'scheduler_gamma', 'scheduler_t_max',
        'weight_decay', 'model_name', 'optim', 'momentum', 'pretrained'
    ] + metrics

    df_short = df[[col for col in columns_to_keep if col in df.columns]]
    final_output_path = results_dir / output_csv_name
    df_short.to_csv(final_output_path, index=False)

    markdown_table = df_short.to_markdown(index=False)
    if show_markdown:
        print(markdown_table)

    return markdown_table

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