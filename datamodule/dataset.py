from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Tuple
import sys

import pandas as pd
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utility import set_seed


class TransformedSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int], transform: transforms):
        """
        Initializes the TransformedSubset with a base dataset, indices, and a transformation function.
        
        Args:
            base_dataset (Dataset): The original dataset from which this subset is derived.
            indices (List[int]): List of indices to select from the base dataset.
            transform (transforms): A transformation function to apply to the images.
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        """
        Returns the length of the subset, which is the number of indices.
        
        Returns:
            int: The number of items in the subset.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

class TransformedNoBgSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int], transform: transforms, train: bool = True):
        """
        Initializes the TransformedNoBgSubset with a base dataset, indices, and a transformation function.
        Args:
            base_dataset (Dataset): The original dataset from which this subset is derived.
            indices (List[int]): List of indices to select from the base dataset.
            transform (transforms): A transformation function to apply to the images.
            train (bool):  Train = nutzt NoBG + Maske, Val = nutzt Original
        """
        self.base_dataset = base_dataset
        if not isinstance(self.base_dataset, CachedImageCSVNoBgDataset):
            raise TypeError("base_dataset must be an instance of CachedImageCSVNoBgDataset")
        self.indices = indices
        self.transform = transform
        self.train = train
    
    def __len__(self):
        """
        Returns the length of the subset, which is the number of indices.
        
        Returns:
            int: The number of items in the subset.
        """
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        bg_img, nobg_img, mask, label = self.base_dataset[self.indices[idx]]

        if self.train:
            if self.transform:
                nobg_img = self.transform(nobg_img)
            return nobg_img, label
        else:
            if self.transform:
                bg_img = self.transform(bg_img)
            return bg_img, label


class CachedImageCSVNoBgDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.bg_dir = self.data_dir / "data"
        self.nobg_img_dir = self.data_dir / "data_nobg"
        self.mask_dir = self.data_dir / "masks"
        self.label_path = self.data_dir / "labels" / "labels.csv"
        self.labels = pd.read_csv(self.label_path)
        self.delete_missing_images_from_labels()
        self.data = []  # (bg_image, nobg_image, mask, label) tuples

        print("Caching all images into RAM...")
        for _, row in self.labels.iterrows():
            filename = row["filename"]
            stem = Path(filename).stem
            label = int(row["label"])
            try:
                # Suche nach passenden Dateien mit beliebiger Endung
                bg_files = list(self.bg_dir.glob(f"{stem}.*"))
                nobg_files = list(self.nobg_img_dir.glob(f"{stem}.*"))
                mask_files = list(self.mask_dir.glob(f"{stem}.*"))
                if not (bg_files and nobg_files and mask_files):
                    raise FileNotFoundError(f"⚠️ One or more files missing for {stem}")
                bg_img = Image.open(bg_files[0]).convert("L")
                nobg_img = Image.open(nobg_files[0]).convert("L")
                mask_img = Image.open(mask_files[0]).convert("L")
                self.data.append((bg_img.copy(), nobg_img.copy(), mask_img.copy(), label))
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

        print(f"Cached {len(self.data)} image sets into RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def delete_missing_images_from_labels(self):
        missing_files = []
        for index, row in self.labels.iterrows():
            stem = Path(row["filename"]).stem
            if not list(self.bg_dir.glob(f"{stem}.*")):
                print(f"Missing background image for {stem}")
                missing_files.append(index)
                continue
            if not list(self.nobg_img_dir.glob(f"{stem}.*")):
                print(f"Missing no-background image for {stem}")
                missing_files.append(index)
                continue
            if not list(self.mask_dir.glob(f"{stem}.*")):
                print(f"Missing mask image for {stem}")
                missing_files.append(index)

        self.labels.drop(missing_files, inplace=True)
        self.labels.reset_index(drop=True, inplace=True)
        if missing_files:
            print(f"Deleted {len(missing_files)} missing image entries from labels.")

class CachedImageCSVDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.img_dir = Path(data_dir) / "data"
        self.labels = pd.read_csv(Path(data_dir) / "labels" / "labels.csv")
        self.delete_missing_images_from_labels()

        self.data = []  # (PIL image, label) tuples

        print("Caching all images into RAM...")
        for _, row in self.labels.iterrows():
            img_name = row["filename"]
            img_stem = img_name.split(".")[0]

            possible_files = list(self.img_dir.glob(f"{img_stem}.*"))
            if not possible_files:
                raise FileNotFoundError(f"No image found for {img_name}")

            img = Image.open(possible_files[0]).convert("L")
            self.data.append((img.copy(), int(row["label"])))
        print(f"Cached {len(self.data)} images into RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def delete_missing_images_from_labels(self):
        missing_files = []
        for index, row in self.labels.iterrows():
            img_stem = row["filename"].split(".")[0]
            if not list(self.img_dir.glob(f"{img_stem}.*")):
                missing_files.append(index)
        self.labels.drop(missing_files, inplace=True)
        self.labels.reset_index(drop=True, inplace=True)
        if missing_files:
            print(f"⚠️ Deleted {len(missing_files)} missing image entries from labels.")

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




def get_dataset_and_subsets(config: Dict) -> Tuple[Dataset, Callable]:
    """
    Initialisiert das Dataset und gibt eine Funktion zurück, die passende Subsets erzeugt.
    
    Args:
        config (Dict): Muss Schlüssel enthalten:
            - "data_dir": str oder Path
            - "dataset_type": "default", "cached", "nobg"
            - "transforms": Dict[str, Callable], mit "train" und "val"
    
    Returns:
        Tuple[Dataset, subset_factory]: 
            dataset: das vollständige Dataset (Cached oder NoBg)
            subset_factory: Funktion (indices: List[int], train: bool) -> Dataset-Subset
    """
    data_dir = Path(config["data_dir"])
    dataset_type = config.get("dataset_type", "default")  # "default" oder "nobg"
    transforms = config.get("transforms", {})

    if dataset_type == "cached":
        dataset = CachedImageCSVDataset(data_dir=data_dir)

        def subset_factory(indices: List[int], train: bool) -> Dataset:
            return TransformedSubset(
                base_dataset=dataset,
                indices=indices,
                transform=transforms["train"] if train else transforms["val"]
            )

    elif dataset_type == "nobg":
        dataset = CachedImageCSVNoBgDataset(data_dir=data_dir)

        def subset_factory(indices: List[int], train: bool) -> Dataset:
            return TransformedNoBgSubset(
                base_dataset=dataset,
                indices=indices,
                transform=transforms["train"] if train else transforms["val"],
                train=train
            )

    elif dataset_type == "default":
        dataset = CachedImageCSVDataset(data_dir=data_dir)

        def subset_factory(indices: List[int], train: bool) -> Dataset:
            return TransformedSubset(
                base_dataset=dataset,
                indices=indices,
                transform=transforms["train"] if train else transforms["val"]
            )

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return dataset, subset_factory

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

def calculate_pain_status(data_dir: Path):
        img_dir = Path(data_dir) / "data" # Assuming images are in a subfolder named 'data'
        mgs = pd.read_csv(Path(data_dir) / "labels" / "v3_mgs_01.csv")
        labels = pd.read_csv(Path(data_dir) / "labels" / "labels.csv") # Assuming labels are in a subfolder named 'labels'
        df = mgs.copy()
        index_col = df['index'].copy()

        for col in df.columns:
            if col != 'index':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.replace(9, np.nan, inplace = True)
        df['index'] = index_col

        fau_types = ['ot', 'nb', 'cb', 'ep', 'wc']
        reviewer_indices = range(1, 13)
        result = []

        for idx, row in df.iterrows():
            reviewer_avgs = []

            for i in reviewer_indices:
                reviewer_scores = [row.get(f"{fau}{i}", np.nan) for fau in fau_types]
                valid_scores = [s for s in reviewer_scores if pd.notna(s)]

                if pd.notna(row.get(f'ot{i}', np.nan)) and len(valid_scores) >= 3:
                    reviewer_avgs.append(np.nanmean(valid_scores))

            if not reviewer_avgs:
                result.append("no data")
            else:
                total_avg = np.mean(reviewer_avgs)
                result.append(1 if total_avg >= 0.6 else 0)

        df['pain_status'] = result

        df = df[df['pain_status'] != 'no data']

        output_path = Path(data_dir) / "labels" / "pain_nopain.csv"
        df[['index', 'pain_status']].to_csv(output_path, index=False)
        return df[['index', 'pain_status']]


if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).resolve().parent.parent / "data" / "MGS_data_nobg"
    
    config = {
        "data_dir": data_dir,
        "dataset_type": "nobg",  # or "default"
        "transforms": {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]),
            "val": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        }
    }
    
    dataset, subset_factory = get_dataset_and_subsets(config)
    print(f"Dataset length: {len(dataset)}")
    
    from splitter import RandomSplit
    rs = RandomSplit(val_size=0.2, seed=42)
    train_idx, test_idx = rs.get_splits(dataset)[0]
    
    train_subset = subset_factory(train_idx, train=True)
    test_subset = subset_factory(test_idx, train=False)

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from PIL import Image
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
    
    for images, labels in train_loader:
        # Display the first image in the batch
        img = images[0].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {labels[0]}")
        plt.axis('off')
        plt.show()
        
        print(f"Batch images shape: {images[0].size}, Labels: {labels}")
        break
