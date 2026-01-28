from utils.train_engine import Trainer
from models.baselines import get_resnet50, get_vit, get_mobilenet_v3
from models.green_mamba import GreenMamba
from data.dataset import create_dataloaders
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_model(model_name, num_classes, device):
    if model_name == 'green-mamba':
        # Attempt to use CUDA kernel for training if GPU is available
        use_cuda = torch.cuda.is_available()
        return GreenMamba(num_classes=num_classes, use_cuda_kernel=use_cuda).to(device)
    elif model_name == 'resnet':
        return get_resnet50(num_classes).to(device)
    elif model_name == 'vit':
        return get_vit(num_classes).to(device)
    elif model_name == 'mobilenet':
        return get_mobilenet_v3(num_classes).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train RF Signal Classifiers")
    parser.add_argument('--model', type=str, required=True,
                        choices=['green-mamba', 'resnet', 'vit', 'mobilenet'],
                        help="Model architecture to train")
    parser.add_argument('--data_dir', type=str,
                        default='dataset', help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='outcomes')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {args.model} on {device}")

    # Data
    train_loader, val_loader, class_names = create_dataloaders(
        args.data_dir, args.model, args.batch_size)
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Model
    model = get_model(args.model, num_classes, device)

    # Optimizer settings
    criterion = nn.CrossEntropyLoss()

    if args.model == 'green-mamba':
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Train
    save_path = os.path.join(args.output_dir, args.model)
    trainer = Trainer(model, train_loader, val_loader,
                      criterion, optimizer, scheduler, device, save_path)
    trainer.fit(args.epochs)


if __name__ == "__main__":
    main()
