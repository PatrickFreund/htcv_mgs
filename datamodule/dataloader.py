import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader
from datamodule.dataset import get_dataset
from datamodule.transforms import get_train_transforms, get_val_transforms, load_dataset_stats


def get_dataloaders(
    train_foldername: str,
    val_foldername: str = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 4,
    stats_path: str = "data/dataset_stats.json",
    image_size=(128, 128),
    flip_prob=0.5,
    angle_range=(0, 7)
):
    """
    Returns PyTorch DataLoaders for training and validation.

    Args:
        train_foldername (str): Folder name inside `data/` for training (or full dataset if val is None)
        val_foldername (str): Optional. Folder name inside `data/` for validation
        batch_size (int): Batch size for training/validation
        shuffle_train (bool): Whether to shuffle training data
        num_workers (int): Number of workers for DataLoader
        stats_path (str): Path to dataset normalization statistics (JSON)
        image_size (tuple): Output image size (H, W)
        flip_prob (float): Probability of horizontal flip (train only)
        angle_range (tuple): Rotation range (train only)

    Returns:
        Tuple[train_loader, val_loader]
    """

    # Load dataset stats
    mean, std = load_dataset_stats(stats_path)

    # Create transforms
    train_transform = get_train_transforms(output_size=image_size, flip_prob=flip_prob, angle_range=angle_range, mean=mean, std=std)
    val_transform = get_val_transforms(output_size=image_size, mean=mean, std=std)

    # Load training dataset
    train_dataset = get_dataset(train_foldername, transform=train_transform)

    # Validation from separate folder
    if val_foldername:
        val_dataset = get_dataset(val_foldername, transform=val_transform)
    else:
        # Split train_dataset into train/val if no val_folder is given
        from torch.utils.data import random_split
        total_size = len(train_dataset)
        val_size = int(0.2 * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders("MGS_data")
    for batch in train_loader:
        x, y = batch
        print("Train batch:", x.shape, y.shape)
        break