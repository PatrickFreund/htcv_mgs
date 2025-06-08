from typing import Callable, Tuple, List
import pandas as pd
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImageCSVDataset(Dataset):
    """
    ImageCSVDataset is a custom dataset class for loading images and their corresponding labels from a CSV file.
    It is designed for datasets structured as follows:

    data_dir/
    ├── data/             # contains image files (e.g., .jpg, .png)
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── labels/
        └── labels.csv    # CSV file with 'filename,label' format

    Example of labels.csv:
    ----------------------
    filename,label
    img1.jpg,0
    img2.jpg,1
    img3.jpg,0
    ...

    Args:
        data_dir (Path): Path to the dataset split directory (e.g., 'train', 'test').
        transform (torchvision.transforms, optional): Transformations to apply to the images.

    Returns:
        A tuple (image, label) for each sample in the dataset.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.img_dir = Path(data_dir) / "data"
        self.labels = pd.read_csv(Path(data_dir) / "labels" / "labels.csv")
        self.delete_missing_images_from_labels()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        img_name = row["filename"]
        img_stem = img_name.split(".")[0]
        label = int(row["label"])

        possible_files = list(self.img_dir.glob(f"{img_stem}.*"))
        if len(possible_files) == 0:
            raise FileNotFoundError(f"No image found for {img_name} in {self.img_dir}.")
        
        img_path = possible_files[0]
        img = Image.open(img_path).convert("L")  # Grayscale

        return img, label

    def delete_missing_images_from_labels(self):
        missing_files = []
        for index, row in self.labels.iterrows():
            img_name = row["filename"]
            img_stem = img_name.split(".")[0]
            possible_files = list(self.img_dir.glob(f"{img_stem}.*"))
            if len(possible_files) == 0:
                missing_files.append(index)

        self.labels.drop(missing_files, inplace=True)
        self.labels.reset_index(drop=True, inplace=True)
        print(f"Deleted {len(missing_files)} missing images from labels.")

class TransformSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int], transform: Callable = None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataset(data_foldername: str) -> ImageCSVDataset:
    """
    Load dataset corresponding to the specified folder name in the data directory that follows the structure 
    described in the ImageCSVDataset class.

    Args:
        data_foldername (str): Folder name in the data folder
        
    Returns:
        ImageCSVDataset: Dataset object containing images and labels of the specified folder in data directory.
    """
    root_dir = Path(__file__).resolve().parent.parent # htcv_mgs directory path
    data_dir = root_dir / "data" / data_foldername
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    dataset = ImageCSVDataset(data_dir = data_dir) 
    return dataset

def split_dataset(dataset, save_dir: Path, train_ratio=0.8, seed=42):
    """
    Splits the given dataset into train and test splits using random_split 
    and saves the resulting labels (filename, label) as CSV files.
    
    Args:
        dataset (ImageCSVDataset): The full dataset.
        save_dir (Path): Directory where the CSVs will be saved.
        train_ratio (float): Ratio of training data (e.g., 0.8 for 80% train).
        seed (int): Random seed for reproducibility.
    """
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, test_subset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

    labels_df = dataset.labels
    train_df = labels_df.iloc[train_subset.indices].reset_index(drop=True)
    test_df = labels_df.iloc[test_subset.indices].reset_index(drop=True)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(save_dir / "train_labels.csv", index=False)
    test_df.to_csv(save_dir / "test_labels.csv", index=False)

    print(f"Saved train_labels.csv and test_labels.csv to {save_dir}")
    return train_subset, test_subset


