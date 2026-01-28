from models.baselines import get_resnet50, get_vit, get_mobilenet_v3
from models.green_mamba import GreenMamba
from data.dataset import create_dataloaders
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_noise(tensor, snr_db):
    if snr_db is None:
        return tensor
    signal_power = torch.mean(tensor ** 2)
    if signal_power == 0:
        return tensor
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(tensor) * torch.sqrt(noise_power)
    return tensor + noise


def test_snr(model, loader, device, snr_levels):
    model.eval()
    accuracies = []

    for snr in snr_levels:
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Add noise
                inputs = torch.stack([add_noise(img, snr) for img in inputs])

                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        accuracies.append(acc)
        print(f"SNR {snr}dB: {acc*100:.2f}%")

    return accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', type=str, default='outcomes')
    parser.add_argument('--data_dir', type=str, default='dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snr_levels = [-10, -5, 0, 5, 10, 15, 20]
    results = {'SNR': snr_levels}

    models_config = [
        ('Green-Mamba', 'green-mamba', 3),
        ('MobileNetV3', 'mobilenet', 3),
        ('ResNet50', 'resnet', 3),
        ('ViT', 'vit', 3)
    ]

    for pretty_name, model_name, num_classes in models_config:
        print(f"\nTesting {pretty_name}...")

        # Load Model
        if model_name == 'green-mamba':
            model = GreenMamba(
                num_classes, use_cuda_kernel=torch.cuda.is_available())
        elif model_name == 'resnet':
            model = get_resnet50(num_classes)
        elif model_name == 'vit':
            model = get_vit(num_classes)
        elif model_name == 'mobilenet':
            model = get_mobilenet_v3(num_classes)

        model = model.to(device)

        # Load Weights
        weight_path = os.path.join(
            args.weights_dir, model_name, 'best_model.pth')
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
        else:
            print(
                f"Warning: Weights not found at {weight_path}, using random init.")

        # Get Loader
        _, val_loader, _ = create_dataloaders(
            args.data_dir, model_name, batch_size=32)

        # Test
        accs = test_snr(model, val_loader, device, snr_levels)
        results[pretty_name] = accs

    # Plot
    df = pd.DataFrame(results)
    df.to_csv("snr_robustness.csv", index=False)

    plt.figure(figsize=(10, 6))
    for col in df.columns:
        if col != 'SNR':
            plt.plot(df['SNR'], df[col], marker='o', label=col)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Noise Robustness Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('snr_robustness.png')
    print("Saved snr_robustness.png")


if __name__ == "__main__":
    main()
