# This repo is for RF signal classification using advanced architectures such as VIT and VIM models

Here is a concise, structured report suitable for a GitHub `README.md` file.

---

# Model Architectures & Experimental Setup

This repository explores three distinct deep learning architectures for RF signal and image classification: **ResNet50** (CNN), **Vision Transformer** (ViT), and a custom **Ultra-Light Vision Mamba** (Vim).

## Architecture Comparison

| Feature          | [cite_start]ResNet50 [cite: 8, 127]                     | [cite_start]Vision Transformer (ViT) [cite: 136] | Ultra-Light Vim                  |
| :--------------- | :------------------------------------------------------ | :----------------------------------------------- | :------------------------------- |
| **Type**         | Convolutional Neural Network (CNN)                      | Transformer (Self-Attention)                     | State Space Model (SSM)          |
| **Backbone**     | ResNet50 (w/ Residual Connections)                      | `vit_base_patch16_224`                           | Custom `Vim` Block (Lightweight) |
| **Input Size**   | [cite_start]$180 \times 180 \times 3$ [cite: 140]       | $224 \times 224 \times 3$                        | $128 \times 128 \times 3$        |
| **Depth/Size**   | 50 Layers                                               | 12 Layers (Base)                                 | 6 Layers (Depth), 96 Dim         |
| **Pre-training** | [cite_start]**ImageNet** [cite: 141]                    | [cite_start]**ImageNet** [cite: 136]             | **None** (Trained from Scratch)  |
| **Head**         | [cite_start]Dense (512, ReLU) $\to$ Softmax [cite: 143] | Linear Layer (Class Token)                       | Linear Layer                     |

---

## Training Hyperparameters

| Parameter         | ResNet50 (Paper)                                            | ViT (Code Implementation) | Vim (Code Implementation)        |
| :---------------- | :---------------------------------------------------------- | :------------------------ | :------------------------------- |
| **Optimizer**     | [cite_start]Adam [cite: 172]                                | Adam                      | AdamW                            |
| **Learning Rate** | [cite_start]$1 \times 10^{-4}$ [cite: 172]                  | $1 \times 10^{-4}$        | $5 \times 10^{-4}$               |
| **Scheduler**     | None Specified                                              | None                      | Cosine Annealing ($T_{max}=200$) |
| **Batch Size**    | [cite_start]~75 (derived from 32 batches/epoch) [cite: 185] | 8                         | 32                               |
| **Epochs**        | [cite_start]10 [cite: 297]                                  | 10                        | 200 (w/ Early Stopping)          |
| **Loss Function** | [cite_start]Cross-Entropy [cite: 80]                        | Cross-Entropy             | Cross-Entropy                    |

---

## Model Specifics

### 1. ResNet50 (CNN)

- [cite_start]**Source:** _Deep Learning-Driven Classification of 5G and LTE Signals_[cite: 2].
- [cite_start]**Customization:** Top layers removed; spatial dimensions reduced via average pooling [cite: 139-140].
- [cite_start]**Classifier:** Flatten $\to$ Dense(512, ReLU) $\to$ Dense(3, Softmax) [cite: 142-143].

### 2. Vision Transformer (ViT)

- [cite_start]**Source:** `vit.py` / _ICOIN Paper_[cite: 136].
- **Implementation:** `timm` library (`vit_base_patch16_224`).
- [cite_start]**Patching:** $16 \times 16$ patch size[cite: 170].
- **Augmentation:** Resize to $224 \times 224$, ToTensor.

### 3. Ultra-Light Vision Mamba (Vim)

- **Source:** `vim-lightweight.py`.
- **Implementation:** Custom `Vim` class built for efficiency.
- **Configuration:**
  - **Patch Size:** 16
  - **Embedding Dim:** 96
  - **State Dim ($d_{state}$):** 16
  - **Rank ($dt\_rank$):** 6
- **Augmentation:** `TrivialAugmentWide`, `RandomCrop(128)`, `RandomHorizontalFlip`.
