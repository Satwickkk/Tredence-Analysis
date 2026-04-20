# Tredence-Analysis
# Self-Pruning Neural Network on CIFAR-10

> A neural network that **learns to prune itself during training** using learnable gate parameters and L1 sparsity regularization.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

##  Overview

Traditional neural network pruning is a **post-training** step — you train the model first, then remove unimportant weights. This project takes a different approach:

The network has a built-in mechanism to identify and **dynamically remove its own weakest connections during training**, by learning which weights are unnecessary through a custom `PrunableLinear` layer and a sparsity regularization loss.

<img width="1494" height="496" alt="image" src="https://github.com/user-attachments/assets/793daafd-dc09-414d-ab0a-220acfa49fcc" />

<img width="715" height="338" alt="image" src="https://github.com/user-attachments/assets/dd9a116c-06a5-49a6-aebe-c777b4f0ae1a" />
<img width="687" height="302" alt="image" src="https://github.com/user-attachments/assets/c200bde3-8517-4f5c-b36e-818d8d2b5c27" />
<img width="710" height="463" alt="image" src="https://github.com/user-attachments/assets/73673197-78cb-405b-9456-79e73cccbce9" />


### Key Idea

Each weight in the network is associated with a learnable **gate parameter**. This gate (a scalar between 0 and 1) multiplies the weight's output. When a gate value approaches 0, the corresponding weight is effectively *pruned* from the network. The training process is incentivized to push most gates to zero — leaving only the most important connections active.



##  Architecture

```
Input (3×32×32)
      │
  Conv Block 1   [64 filters, BN, ReLU, MaxPool]
      │
  Conv Block 2   [128 filters, BN, ReLU, MaxPool]
      │
  Conv Block 3   [256 filters, BN, ReLU, MaxPool]
      │
   Flatten  →  4096-dim
      │
PrunableLinear(4096 → 1024)  ← gates here
      │  ReLU + Dropout
PrunableLinear(1024 → 512)   ← gates here
      │  ReLU + Dropout
PrunableLinear(512 → 10)     ← gates here
      │
   Logits (10 classes)
```

---

##  Core Components

### 1. `PrunableLinear` Layer


- `gate_scores` has the **exact same shape** as `weight`
- `sigmoid` keeps gates in (0, 1) — differentiable end-to-end
- Initialized to `2.0` so `sigmoid(2) ≈ 0.88` — all weights start active

### 2. Sparsity Loss (L1 on gates)

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss

SparsityLoss = Σ sigmoid(gate_scores)   [sum over all PrunableLinear layers]
```

The **L1 norm** is used because its gradient is a constant `+λ` regardless of the gate's current value. This constant pull toward zero is strong enough to force gates to *exactly* zero — unlike L2 which only shrinks large values.

---


### Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/self-pruning-neural-network.git
cd self-pruning-neural-network

# Install dependencies
pip install torch torchvision matplotlib numpy

# Run training
python train.py
```



## 📊 Results

Results from training 30 epochs per lambda value on CIFAR-10:

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|-----------|------------------|--------------------|
| `0.5` (low) | ~84% | 99.7% |
| `2.0` (medium) | ~82% | 99.84% |
| `5.0` (high) | ~81% | 99.93% |



---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
```

Install with:
```bash
pip install torch torchvision matplotlib numpy
```

---

## 💡 How Sparsity is Measured

After training, the **sparsity level** is computed as:

```python
def get_sparsity_level(model, threshold=1e-2):
    all_gates = torch.cat([
        torch.sigmoid(layer.gate_scores).flatten()
        for layer in model.prunable_layers
    ])
    return (all_gates < threshold).float().mean().item() * 100
```

A gate value below `0.01` means the corresponding weight contributes less than 1% of its potential output — effectively pruned.

