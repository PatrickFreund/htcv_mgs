import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from datamodule.dataset import get_dataset, split_dataset
from utils.model import get_model
from utils.train_utils import train_one_epoch, validate_model
from utils.utility import get_unique_path, save_args_to_file

# Constants
ROOT_DIR = Path(__file__).resolve().parent.parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN models with PyTorch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=ROOT_DIR / "models")
    parser.add_argument("--data_folder_train", type=str, required=True, help="Path to the training dataset folder (mandatory)")
    parser.add_argument("--data_folder_test", type=str, default=None, help="Path to the test dataset folder (optional)")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_classes", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Training {args.model} on device {device}")
    
    # Save path for the model
    save_path = get_unique_path(args.save_path / args.model)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # save args to file
    save_args_to_file(args, save_path, filename="config.json")
    
    # Check if test dataset was provided
    if args.data_folder_test is None:
        print("No separate test folder provided. Splitting training dataset into train and test sets.")
        dataset = get_dataset(args.data_folder_train, transform=None)
        train_dataset, test_dataset = split_dataset(dataset, save_dir=save_path, train_ratio=0.8, seed=42)
    else:
        print(f"Loading separate train and test datasets from {args.data_folder_train} and {args.data_folder_test}.")
        train_dataset = get_dataset(args.data_folder_train, transform=None)
        test_dataset = get_dataset(args.data_folder_test, transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    
    # 
    
    # load model and initialize optimizer and loss function
    model = get_model(args.model, args.num_classes, args.pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # run training and validation loop
    print(f"Model will be saved to {save_path}")
    best_accuracy = 0.0
    checkpoints = 0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        test_loss, test_acc = validate_model(model, test_dataloader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | Val Loss {test_loss:.4f}, Val Acc {test_acc:.4f}")
        
        # Save the model if it has the best accuracy so far
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), save_path / f"checkpoint_{checkpoints}_{args.model}_best.pth")
            print(f"Model saved to {save_path / f'checkpoint_{checkpoints}_{args.model}_best.pth'}")
            checkpoints += 1

if __name__ == "__main__":
    main()