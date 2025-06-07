import pandas as pd
from pathlib import Path
import torch
import numpy as np
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
    def __init__(self, data_dir: Path, transform: transforms = None):
        self.data_dir = data_dir
        self.img_dir = Path(data_dir) / "data" # Assuming images are in a subfolder named 'data'
        self.mgs = pd.read_csv(Path(data_dir) / "labels" / "v3_mgs_01.csv")
        self.labels = pd.read_csv(Path(data_dir) / "labels" / "labels.csv") # Assuming labels are in a subfolder named 'labels'
        self.delete_missing_images_from_labels()
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        img_name = row["filename"]
        img_stem = img_name.split(".")[0] 
        label = int(row["label"]) 
        
        # Check for the image with the same stem but maybe different extension
        possible_files = list(self.img_dir.glob(f"{img_stem}.*")) 
        if len(possible_files) == 0:
            raise FileNotFoundError(f"No image found for {img_name} in {self.img_dir}.")
        
        img_name = possible_files[0]        
        img_path = self.img_dir / img_name
        
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label

    def delete_missing_images_from_labels(self):
        """
        Deletes rows from the labels DataFrame where the corresponding image file does not exist.
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


    def calculate_pain_status(self):
        df = self.mgs.copy()
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
            total_score = 0
            valid_reviewer_count = 0

            for i in reviewer_indices:
                reviewer_scores = [row.get(f"{fau}{i}", np.nan) for fau in fau_types]
                reviewer_scores = [pd.to_numeric(s, errors='coerce') for s in reviewer_scores]
                valid_scores = [s for s in reviewer_scores if pd.notna(s)]

                ot_score = pd.to_numeric(row.get(f'ot{i}', np.nan), errors='coerce')
                if pd.notna(ot_score) and len(valid_scores) >= 3:
                    total_score += sum(valid_scores)
                    valid_reviewer_count += 1

            if valid_reviewer_count == 0:
                result.append("no data")
            else:
                avg_score = total_score / valid_reviewer_count
                result.append('pain' if avg_score > 3 else 'no pain')

        df['pain_status'] = result

        output_path = Path(self.data_dir) / "labels" / "pain_nopain.csv"
        df.to_csv(output_path, index = False)
        return df[['index', 'pain_status']]

def get_dataset(data_foldername: str, transform:transforms) -> ImageCSVDataset:
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

    dataset = ImageCSVDataset(data_dir = data_dir, transform = transform) 
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



if __name__ == "__main__":
    # Example usage
    data_foldername = "MGS_data"
    dataset = get_dataset(data_foldername, transform=None)
    from torch.utils.data import DataLoader
    pain_df = dataset.calculate_pain_status()
    print(f"Number of samples in {data_foldername} dataset: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for images, labels in dataloader:
        print(images.shape, labels.shape, labels)