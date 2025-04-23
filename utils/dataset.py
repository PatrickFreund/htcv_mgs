import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def get_dataset(data_foldername: str, batch_size: int) -> DataLoader:
    """
    Load dataset from the specified path and return a DataLoader.
    
    Args:
        data_foldername (str): Folder name in the data folder
        batch_size (int): Batch size for DataLoader.
        
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = torch.utils.data.TensorDataset(torch.randn(100, 3, 224, 224), torch.randint(0, 2, (100,)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader