Green-Mamba: Linear-Complexity Spectrogram Sensing for 6G Edge Networks

This repository contains the official PyTorch implementation for the paper "Green-Mamba: Linear-Complexity Spectrogram Sensing for Energy-Constrained 6G Edge Networks".

We propose Green-Mamba, an ultra-lightweight Selective State Space Model (SSM) designed for RF signal classification on resource-constrained edge devices (e.g., Raspberry Pi). It matches the accuracy of Vision Transformers (ViT) while reducing inference latency by 14x and memory usage by 92%.

📂 Repository Structure

.
├── data/ # Dataset loading and transforms (128px vs 224px)
├── models/ # Model architectures (Green-Mamba, ResNet, ViT, MobileNetV3)
├── scripts/ # Experiment scripts for training, benchmarking, and analysis
├── utils/ # Training engine and plotting utilities
└── outcomes/ # (Auto-generated) Stores weights, logs, and result plots

🚀 Setup & Installation

Clone the repository:

git clone [https://github.com/your-username/Green-Mamba.git](https://github.com/your-username/Green-Mamba.git)
cd Green-Mamba

Install dependencies:

pip install -r requirements.txt

Note: For GPU acceleration of the Mamba kernel, install vision-mamba and causal-conv1d. If not found, the code automatically falls back to a pure PyTorch CPU implementation.

Prepare Dataset:
Place your RF spectrogram images in a folder named dataset/ with the following structure:

dataset/
├── LTE/
├── NR/
└── LTE_NR/

📊 Reproducing Paper Results

Follow these three steps to generate all figures and tables presented in the paper.

Step 1: Model Training

Train the proposed model and baselines to generate Accuracy Curves (Fig. 2b) and Confusion Matrices (Fig. 2a).

# 1. Train Proposed Green-Mamba

python scripts/train.py --model green-mamba --epochs 30 --batch_size 32

# 2. Train Baselines

python scripts/train.py --model vit --epochs 30 --batch_size 8
python scripts/train.py --model resnet --epochs 30 --batch_size 32
python scripts/train.py --model mobilenet --epochs 30 --batch_size 32

Output: Check outcomes/{model_name}/ for best_model.pth, training_curves.png, and confusion_matrix.png.

Step 2: Hardware Efficiency Benchmark (Table I)

Run this script on the target edge device (e.g., Raspberry Pi 4) to measure Latency, FPS, Memory, and Energy.

python scripts/benchmark_pi.py

Output: Generates benchmark_results.csv comparing all models.

Step 3: Noise Robustness Analysis (Fig. 4)

Evaluate model resilience by injecting Gaussian noise at varying SNR levels (-10dB to +20dB).

python scripts/robustness.py

Output: Generates snr_robustness.png plotting Accuracy vs. SNR.

📈 Results Summary

Model

Params (M)

Accuracy (%)

Latency (Pi 4)

Memory (MB)

Green-Mamba (Ours)

0.2

~99.4%

~48ms

~28 MB

MobileNetV3-Small

2.5

~97.0%

~55ms

~45 MB

ResNet50

25.6

~95.1%

~110ms

~140 MB

ViT-Base

86.4

~99.5%

~680ms

~340 MB

📜 Citation

If you use this code in your research, please cite:

@article{singh2025greenmamba,
title={Green-Mamba: Linear-Complexity Spectrogram Sensing for Energy-Constrained 6G Edge Networks},
author={Singh, Deependra},
journal={arXiv preprint},
year={2025}
}
