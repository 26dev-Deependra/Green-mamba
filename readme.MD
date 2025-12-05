# This repo is for RF signal classification using advanced architectures such as VIT and VIM models

This repository explores three distinct deep learning architectures for RF signal and image classification: **ResNet50** (CNN), **Vision Transformer** (ViT), and a custom **Ultra-Light Vision Mamba** (Vim).

## Architecture Comparison

| Feature          | ResNet50                           | Vision Transformer (ViT)     | Ultra-Light Vim                  |
| :--------------- | :--------------------------------- | :--------------------------- | :------------------------------- |
| **Type**         | Convolutional Neural Network (CNN) | Transformer (Self-Attention) | State Space Model (SSM)          |
| **Backbone**     | ResNet50 (w/ Residual Connections) | `vit_base_patch16_224`       | Custom `Vim` Block (Lightweight) |
| **Input Size**   | $180 \times 180 \times 3$          | $224 \times 224 \times 3$    | $128 \times 128 \times 3$        |
| **Depth/Size**   | 50 Layers                          | 12 Layers (Base)             | 6 Layers (Depth), 96 Dim         |
| **Parameters**   | **\~24.5 Million**                 | **\~86 Million**             | **\~0.3 Million**                |
| **Pre-training** | ImageNet                           | ImageNet                     | None (Trained from Scratch)      |
| **Head**         | Dense (512, ReLU) $\to$ Softmax    | Linear Layer (Class Token)   | Linear Layer                     |

---

## Training Hyperparameters

| Parameter         | ResNet50           | ViT                | Vim                              |
| :---------------- | :----------------- | :----------------- | :------------------------------- |
| **Optimizer**     | Adam               | Adam               | AdamW                            |
| **Learning Rate** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $5 \times 10^{-4}$               |
| **Scheduler**     | None Specified     | None               | Cosine Annealing ($T_{max}=200$) |
| **Batch Size**    | \~75               | 8                  | 32                               |
| **Epochs**        | 10                 | 10                 | 200 (w/ Early Stopping)          |
| **Loss Function** | Cross-Entropy      | Cross-Entropy      | Cross-Entropy                    |

---

## Model Specifics

### 1\. ResNet50 (CNN)

- **Source:** Based on the methodology for identifying 5G and LTE signals.
- **Customization:** The standard top layers were removed and spatial dimensions were reduced via average pooling.
- **Classifier:** A custom head was attached consisting of a Flatten layer, a Dense layer with 512 units (ReLU), and a final Softmax output layer.

### 2\. Vision Transformer (ViT)

- **Source:** `vit.py`
- **Implementation:** Utilizes the `timm` library implementation of `vit_base_patch16_224`.
- **Patching:** Images are processed in $16 \times 16$ patches.
- **Augmentation:** Includes resizing to standard $224 \times 224$ dimensions and tensor conversion.

### 3\. Ultra-Light Vision Mamba (Vim)

- **Source:** `vim-lightweight.py`
- **Implementation:** A custom, highly efficient `Vim` class designed for training from scratch on smaller datasets.
- **Configuration:**
  - **Patch Size:** 16
  - **Embedding Dim:** 96
  - **State Dim ($d_{state}$):** 16
  - **Rank ($dt\_rank$):** 6
- **Augmentation:** Uses `TrivialAugmentWide`, `RandomCrop(128)`, and `RandomHorizontalFlip` for robust training.

## Inference Speed & Resource Cost

When selecting a model for deployment, balancing accuracy with computational cost is just as important as the model's predictive power. Our experiments reveal a massive difference in "cost-to-serve" between the three architectures.

| Architecture                 | Speed Profile   | Resource Intensity | Complexity                                                                                                                                                                        |
| :--------------------------- | :-------------- | :----------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Vision Transformer (ViT)** | **Heavyweight** | **High**           | **$O(N^2)$ Quadratic** <br> The self-attention mechanism is resource-hungry. In our tests, it proved \~14x more expensive to run than ResNet50.                                   |
| **ResNet50**                 | **Balanced**    | **Medium**         | **Efficient ConvNet** <br> A reliable standard. It processes data significantly faster (\~50ms/epoch vs. ViT's \~700ms), making it a solid choice for general server deployments. |
| **Ultra-Light Vim**          | **Ultra-Fast**  | **Low**            | **$O(N)$ Linear** <br> Designed for speed. With linear complexity and a tiny footprint (\~0.3M params), it is ideal for real-time applications and edge devices.                  |

### 💡 Key Takeaways

- **The Cost of Attention:** While **ViT** offers powerful global context, its quadratic scaling means it requires substantially more GPU VRAM and compute time.
- **The Efficiency of Vim:** The **Ultra-Light Vim** model shines in resource-constrained environments. Its state-space backbone allows it to process long sequences with linear scaling, making it orders of magnitude cheaper to run than the Transformer.
