import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('future.no_silent_downcasting', True)

def safe_mean(row, columns):
    values = row[columns]
    str_values = values.astype(str)
    
    if "-" in str_values.values:
        if any(v in ["0", "1", "2"] for v in str_values.values):
            raise ValueError(f"Invalid mix of numbers and '-' in row {row.name}")
        values = values.replace("-", np.nan)

    
    values = pd.to_numeric(values, errors='coerce')
    if values.isna().all():
        return np.nan
    else:
        values = values.replace(9, np.nan)
        return values.mean(axis=0, skipna=True)

def create_mgs_labels(
    mgs_path: Path,
) -> None:
    """
    The function tasks a CSV file containing the five facial action unit (FAU) values from different reviewers on the the mice images in the dataset 
    and creates a new CSV file ('labels.csv') which only contains the filename of the images and their corresponding label (accumulated and averaged MGS score) 
    
    | filename   | label |
    |------------|-------|
    | img0.jpg   | 0     |
    | img1.png   | 1     |
    | ...        | ...   |

    """
    
    reviewer_columns = [
        ['ot1', 'nb1', 'cb1', 'ep1', 'wc1'],
        ['ot2', 'nb2', 'cb2', 'ep2', 'wc2'],
        ['ot3', 'nb3', 'cb3', 'ep3', 'wc3'],
        ['ot4', 'nb4', 'cb4', 'ep4', 'wc4'],
        ['ot5', 'nb5', 'cb5', 'ep5', 'wc5'],
        ['ot6', 'nb6', 'cb6', 'ep6', 'wc6'],
        ['ot7', 'nb7', 'cb7', 'ep7', 'wc7'],
        ['ot9', 'nb9', 'cb9', 'ep9', 'wc9'],
        ['ot10', 'nb10', 'cb10', 'ep10', 'wc10'],
        ['ot11', 'nb11', 'cb11', 'ep11', 'wc11'],
        ['ot12', 'nb12', 'cb12', 'ep12', 'wc12']
    ]
    
    df = pd.read_csv(mgs_path)
    for i, columns in enumerate(reviewer_columns):
        df[f'mgs_reviewer{i+1}'] = df.apply(lambda row: safe_mean(row, columns), axis=1)
    df = df[['index'] + [f'mgs_reviewer{i+1}' for i in range(len(reviewer_columns))]]

    
    reviewer_cols = [col for col in df.columns if col.startswith('mgs_reviewer')]
    df['mgs_mean'] = df[reviewer_cols].mean(axis=1, skipna=True)
    df['mgs_mean_rounded'] = df['mgs_mean'].round()
    df = df[['index', 'mgs_mean_rounded']]
    df.to_csv(
        mgs_path.parent / "labels.csv",
        index=False,
        header=["filename", "label"],
    )

if __name__ == "__main__":
    mgs_path = Path(r"C:\Users\Freun\Desktop\htcv_mgs\data\MGS_data\labels\v3_mgs_01.csv")
    create_mgs_labels(mgs_path)
    