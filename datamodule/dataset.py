from typing import Callable, List, Dict, Tuple

import pandas as pd
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TransformedSubset(Dataset):
    """
    A PyTorch-compatible dataset wrapper that applies image transformations to a subset of samples
    from a base dataset. This subset is defined by a list of indices and is used during
    training or validation. The transformation function is applied to the images when they are accessed.   
    """
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
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

class TransformedNoBgSubset(Dataset):
    """
    A PyTorch-compatible dataset wrapper that applies image transformations to a subset of samples
    from a CachedImageCSVNoBgDataset. This subset is defined by a list of indices and is used during
    training or validation.
    During training, the "no-background" version of the image is used. During validation, the original
    (background) image is used instead. This allows separate augmentation strategies or data views
    depending on the training phase.
    """
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
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        bg_img, nobg_img, label = self.base_dataset[self.indices[idx]]

        if self.train:
            if self.transform:
                nobg_img = self.transform(nobg_img)
            return nobg_img, label
        else:
            if self.transform:
                bg_img = self.transform(bg_img)
            return bg_img, label

class CachedImageCSVNoBgDataset(Dataset):
    """    
    CachedImageCSVNoBgDataset is a custom dataset class for loading images, their corresponding binary 
    segmentation mask (no-background image), and labels from a CSV file, and caching them in memory for 
    faster access during training increasing efficiency tremendously.
    It is designed for datasets structured as follows:

    data_dir/
    ├── data/             # contains image files (e.g., .jpg, .png)
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── data_nobg/        # contains no-background image files (e.g., .jpg, .png)
    │   ├── img1.jpg
    │   ├── img2.jpg
    |   └── ...
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
        data_dir (Path): Path to the dataset directory containing the 'data' and 'labels' folders.

    Returns:
        A tuple (image, label) for each sample in the dataset.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.bg_dir = self.data_dir / "data"
        self.nobg_img_dir = self.data_dir / "data_nobg"
        self.label_path = self.data_dir / "labels" / "labels.csv"
        self.labels = pd.read_csv(self.label_path)
        self.delete_missing_images_from_labels()
        self.data = []  # (bg_image, nobg_image, label) tuples

        print("Caching all images into RAM...")
        for _, row in self.labels.iterrows():
            filename = row["filename"]
            stem = Path(filename).stem
            label = int(row["label"])
            try:
                bg_files = list(self.bg_dir.glob(f"{stem}.*"))
                nobg_files = list(self.nobg_img_dir.glob(f"{stem}.*"))
                if not (bg_files and nobg_files):
                    raise FileNotFoundError(f"⚠️ One or more files missing for {stem}")
                bg_img = Image.open(bg_files[0]).convert("L")
                nobg_img = Image.open(nobg_files[0]).convert("L")
                self.data.append((bg_img.copy(), nobg_img.copy(), label))
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

        print(f"Cached {len(self.data)} image sets into RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def delete_missing_images_from_labels(self):
        """
        Auxiliary function that removes entries from the labels DataFrame for which no
        corresponding image or segmentation mask (no-background image) exists in the directories.
        """
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

        self.labels.drop(missing_files, inplace=True)
        self.labels.reset_index(drop=True, inplace=True)
        if missing_files:
            print(f"Deleted {len(missing_files)} missing image entries from labels.")

class CachedImageCSVDataset(Dataset):
    """
    CachedImageCSVDataset is a custom dataset class for loading images and their corresponding labels from a CSV file
    and caching them in memory for faster access during training increasing efficiency tremendously.
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
        data_dir (Path): Path to the dataset directory containing the 'data' and 'labels' folders.

    Returns:
        A tuple (image, label) for each sample in the dataset.
    """
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
        """
        Auxiliary function that removes entries from the labels DataFrame for which no
        corresponding image file exists in the image directory.
        """
        missing_files = []
        for index, row in self.labels.iterrows():
            img_stem = row["filename"].split(".")[0]
            if not list(self.img_dir.glob(f"{img_stem}.*")):
                missing_files.append(index)
        self.labels.drop(missing_files, inplace=True)
        self.labels.reset_index(drop=True, inplace=True)
        if missing_files:
            print(f"Deleted {len(missing_files)} missing image entries from labels.")

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
        data_dir (Path): Path to the dataset directory containing the 'data' and 'labels' folders.

    Returns:
        A tuple (image, label) for each sample in the dataset.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.img_dir = Path(data_dir) / "data"
        self.labels = pd.read_csv(Path(data_dir) / "labels" / "labels.csv")
        self.delete_missing_images_from_labels()
        self.filenames = self.labels["filename"].tolist()

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
        img = Image.open(img_path).convert("L")

        return img, label

    def delete_missing_images_from_labels(self):
        """
        Auxiliary functions that removes entries from the labels DataFrame for which no 
        corresponding image file exists in the image directory. 
        """
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
        dataset = ImageCSVDataset(data_dir=data_dir)

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