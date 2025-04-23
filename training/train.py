import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.dataset import get_dataset
from utils.train_utils import train_one_epoch, evaluate

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN models with PyTorch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=Path("./checkpoints/"))
    parser.add_argument("--data_path", type=str, default="")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Training {args.model} on device {args.device}")
    
    # Load dataset
    
    
    
if __name__ == "__main__":
    main()