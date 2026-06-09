# Generative Adversarial Networks — Vanilla GAN · LSGAN · DCGAN

<img width="478" height="343" alt="image" src="https://github.com/user-attachments/assets/631c5fbd-08aa-4c02-8360-e380292b6c28" />
<img width="926" height="405" alt="image" src="https://github.com/user-attachments/assets/ef0d336e-3818-47eb-84ff-6eb76ba26284" />


![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

Three GAN architectures implemented from scratch in TensorFlow on MNIST — progressing from a fully-connected baseline to a convolutional architecture, with two loss formulations compared side-by-side.

---

## What's Implemented

| Model | Architecture | Loss | Notes |
|-------|-------------|------|-------|
| **Vanilla GAN** | FC Discriminator · FC Generator | Binary Cross-Entropy | Baseline minimax formulation |
| **LSGAN** | FC Discriminator · FC Generator | Least Squares | More stable gradients, avoids vanishing |
| **DCGAN** | Conv Discriminator · ConvTranspose Generator | Binary Cross-Entropy | Spatial reasoning, BatchNorm |

---

## Architecture Details

### Vanilla GAN / LSGAN

```
Discriminator (267,009 params)
  Input(784) → Dense(256) → LeakyReLU(0.01)
             → Dense(256) → LeakyReLU(0.01)
             → Dense(1)

Generator (1,858,320 params)
  Input(96)  → Dense(1024) → ReLU
             → Dense(1024) → ReLU
             → Dense(784)  → Tanh
```

### DCGAN

```
Discriminator (1,102,721 params)
  Input(784) → Reshape(28,28,1)
             → Conv2D(32, 5×5) → LeakyReLU → MaxPool(2×2)
             → Conv2D(64, 5×5) → LeakyReLU → MaxPool(2×2)
             → Flatten → Dense(1024) → LeakyReLU → Dense(1)

Generator (6,595,521 params)
  Input(96)  → Dense(1024)    → ReLU → BatchNorm
             → Dense(7×7×128) → ReLU → BatchNorm → Reshape(7,7,128)
             → ConvTranspose(64, 4×4, s=2) → ReLU → BatchNorm
             → ConvTranspose(1,  4×4, s=2) → Tanh
```

---

## Loss Formulations

**Vanilla GAN (Binary Cross-Entropy)**
```
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
L_G = -E[log D(G(z))]
```

**LSGAN (Least Squares)**
```
L_D = 0.5 * E[(D(x) - 1)²] + 0.5 * E[D(G(z))²]
L_G = 0.5 * E[(D(G(z)) - 1)²]
```

LSGAN addresses vanishing gradients by penalizing samples based on their distance from the decision boundary rather than using a saturating log loss.

---

## Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Noise dimension | 96 |
| Batch size | 128 |
| Epochs | 20 |
| Optimizer | Adam (lr=1e-3, β₁=0.5) |
| Image preprocessing | Scale to [−1, 1] |

Separate optimizers for Generator and Discriminator. Custom `tf.GradientTape` training loop — no `model.fit()`.

---

## Project Structure

```
Generative-Adversarial-Networks-Implementation/
├── notebook/
│   └── gan_implementation.ipynb   # Full implementation: Vanilla GAN → LSGAN → DCGAN
├── src/
│   ├── models.py                  # discriminator() and generator() definitions
│   ├── losses.py                  # BCE and LSGAN loss functions
│   ├── train.py                   # run_a_gan() training loop
│   └── utils.py                   # show_images(), preprocess_img(), count_params()
├── results/
│   └── samples/                   # Generated image grids per epoch
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/FatinIshraq/Generative-Adversarial-Networks-Implementation
cd Generative-Adversarial-Networks-Implementation
pip install -r requirements.txt
jupyter notebook notebook/gan_implementation.ipynb
```

**requirements.txt**
```
tensorflow>=2.8
numpy
matplotlib
```

---

## Key Concepts Demonstrated

- Minimax adversarial training with separate `GradientTape` passes per network
- Leaky ReLU in discriminator to prevent dead neurons
- Tanh output activation with `[−1, 1]` image normalization
- BatchNormalization in DCGAN generator for training stability
- LSGAN vs BCE loss: gradient behavior comparison
- Spatial upsampling via `Conv2DTranspose` (DCGAN generator)
- Parameter count verification for reproducibility

---

## Roadmap

- [x] Vanilla GAN (MNIST, FC)
- [x] LSGAN (Least Squares loss variant)
- [x] DCGAN (Convolutional architecture)
- [ ] Conditional GAN (cGAN) — class-conditioned generation
- [ ] Training curves & FID score logging
- [ ] CIFAR-10 extension

---

## Author

**Fatin Ishraq** 

[![GitHub](https://img.shields.io/badge/GitHub-FatinIshraq-black?logo=github)](https://github.com/FatinIshraq)
