# 🟢 Green-Mamba: Linear-Complexity Spectrogram Sensing for 6G Edge Networks

This repository contains the **official PyTorch implementation** for the paper:

**“Green-Mamba: Linear-Complexity Spectrogram Sensing for Energy-Constrained 6G Edge Networks.”**

Green-Mamba is an **ultra-lightweight Selective State Space Model (SSM)** designed for **RF signal classification** on **resource-constrained edge devices** (e.g., Raspberry Pi). It achieves **Vision Transformer–level accuracy** while reducing:

- ⚡ **Inference latency by 14×**
- 💾 **Memory usage by 92%**

---

## 📂 Repository Structure

```
.
├── data/        # Dataset loading and transforms (128px vs 224px)
├── models/      # Model architectures (Green-Mamba, ResNet, ViT, MobileNetV3)
├── scripts/     # Training, benchmarking, and analysis scripts
├── utils/       # Training engine and plotting utilities
└── outcomes/    # (Auto-generated) Weights, logs, and result plots
```

---

## 🚀 Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Green-Mamba.git
cd Green-Mamba
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional (Recommended for GPU acceleration):**

```bash
pip install vision-mamba causal-conv1d
```

If these packages are unavailable, the code **automatically falls back** to a pure PyTorch CPU implementation.

---

## 📁 Dataset Preparation

Place your RF spectrogram images in a folder named `dataset/` with the following structure:

```
dataset/
├── LTE/
├── NR/
└── LTE_NR/
```

Each folder should contain spectrogram images corresponding to that class.

---

## 📊 Reproducing Paper Results

Follow these **three steps** to reproduce all major figures and tables from the paper.

---

### 🔹 Step 1: Model Training

Generates **Accuracy Curves (Fig. 2b)** and **Confusion Matrices (Fig. 2a)**

#### Train Proposed Green-Mamba

```bash
python scripts/train.py --model green-mamba --epochs 30 --batch_size 32
```

#### Train Baselines

```bash
python scripts/train.py --model vit --epochs 30 --batch_size 8
python scripts/train.py --model resnet --epochs 30 --batch_size 32
python scripts/train.py --model mobilenet --epochs 30 --batch_size 32
```

**Outputs saved in:** `outcomes/{model_name}/`

- `best_model.pth`
- `training_curves.png`
- `confusion_matrix.png`

---

### 🔹 Step 2: Hardware Efficiency Benchmark (Table I)

Run this on the **target edge device** (e.g., Raspberry Pi 4):

```bash
python scripts/benchmark_pi.py
```

**Output:** `benchmark_results.csv` containing latency, FPS, memory usage, and energy consumption.

---

### 🔹 Step 3: Noise Robustness Analysis (Fig. 4)

Evaluates performance under Gaussian noise at varying SNR levels (-10 dB to +20 dB).

```bash
python scripts/robustness.py
```

**Output:** `snr_robustness.png` showing Accuracy vs. SNR.

---

## 📈 Results Summary

| Model                  | Params (M) | Accuracy (%) | Latency (Pi 4) | Memory (MB) |
| ---------------------- | ---------- | ------------ | -------------- | ----------- |
| **Green-Mamba (Ours)** | **0.2**    | **~99.4%**   | **~48 ms**     | **~28 MB**  |
| MobileNetV3-Small      | 2.5        | ~97.0%       | ~55 ms         | ~45 MB      |
| ResNet50               | 25.6       | ~95.1%       | ~110 ms        | ~140 MB     |
| ViT-Base               | 86.4       | ~99.5%       | ~680 ms        | ~340 MB     |

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@article{singh2025greenmamba,
  title={Green-Mamba: Linear-Complexity Spectrogram Sensing for Energy-Constrained 6G Edge Networks},
  author={Deependra},
  journal={arXiv preprint},
  year={2025}
}
```

---

## ⭐ Support

If you find this work useful, consider giving the repo a ⭐ and sharing it with the research community!
