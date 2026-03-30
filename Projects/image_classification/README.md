# Image classification: MLP vs ResNet-18 benchmark

This project benchmarks two deep learning approaches for image classification on **MNIST**: a **fully connected MLP** built from scratch and a **convolutional ResNet-18** adapted for grayscale input. The goal is to compare accuracy, per-class behavior, and model complexity side by side using PyTorch.

## What gets compared

| Aspect | MLP (`SimpleMLP`) | CNN (ResNet-18) |
|--------|-------------------|-----------------|
| Structure | Three linear layers (784 → 128 → 128 → 10) with ReLU | `torchvision.models.resnet18`, modified for 1-channel input and 10 classes |
| Input | Flattened 28×28 pixels | 1×28×28 tensors with convolutions over spatial structure |

ResNet-18 is adjusted for MNIST: the first convolution accepts a single channel, the final layer outputs 10 classes, and the initial max-pooling layer is replaced with an identity so small 28×28 feature maps are preserved.

## Dataset and preprocessing

- **[MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)** via `torchvision.datasets`, stored under `./datasets` on first download.
- Transforms: resize to 28×28, tensor conversion, normalization with mean `0.1307` and std `0.3081` (standard MNIST statistics).

## Project layout

| File | Role |
|------|------|
| `image_classification.ipynb` | End-to-end pipeline: data loading, model definitions, training, test predictions, sklearn metrics, confusion-matrix plots |
| `custom_torchinfo.py` | Lightweight layer-wise summary (parameter counts and shapes), similar in spirit to `torchinfo` |

## Metrics and analysis

The notebook reports **test accuracy**, **classification reports** (precision, recall, F1 per digit), and **confusion matrices** for both models so you can see where each architecture succeeds or confuses classes.

## How to run

1. Install Python 3 with **PyTorch**, **torchvision**, and common scientific stack packages, for example:

   ```bash
   pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

   Adjust the `torch` / `torchvision` install command for your platform and CUDA/MPS needs from [pytorch.org](https://pytorch.org/get-started/locally/).

2. Open and run the notebook from this directory so `custom_torchinfo` imports resolve:

   ```bash
   jupyter notebook image_classification.ipynb
   ```

The notebook selects **MPS** (Apple Silicon), then **CUDA**, then **CPU** automatically.

## Reproducibility

Training uses `torch.manual_seed(42)`. Training hyperparameters (epochs, learning rate, batch size) are set in the notebook cells; change them there for different benchmark runs.
