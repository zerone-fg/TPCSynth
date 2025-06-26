# Triple-Prompt Controllable Diffusion for Universal Data Augmentation in Medical Image Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2024.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

> **Abstract:** This repository contains the official implementation of "Triple-Prompt Controllable Diffusion for Universal Data Augmentation in Medical Image Segmentation". Our method introduces a novel triple-prompt mechanism that enables controllable diffusion models to generate high-quality synthetic medical images for data augmentation, addressing the challenge of limited labeled data in medical image segmentation.

## 🏆 Key Features

- **Triple-Prompt Control**: Novel three-level prompting mechanism for precise image generation
- **Universal Augmentation**: Works across multiple medical imaging modalities (CT, MRI, Ultrasound, etc.)
- **High Fidelity**: Generates realistic medical images that preserve anatomical structures
- **Plug-and-Play**: Easy integration with existing segmentation frameworks
- **SOTA Performance**: Achieves state-of-the-art results on multiple medical segmentation benchmarks

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/triple-prompt-diffusion.git
cd triple-prompt-diffusion

# Create conda environment
conda create -n triple-prompt python=3.8
conda activate triple-prompt

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Download and prepare datasets
python scripts/prepare_data.py --dataset [acdc/promise12/isic2018] --data_root /path/to/data

# For custom datasets
python scripts/prepare_custom_data.py --input_dir /path/to/your/data --output_dir /path/to/processed/data
```

### Training

#### 1. Train Diffusion Model

```bash
# Train the triple-prompt diffusion model
python train_diffusion.py \
    --config configs/diffusion_config.yaml \
    --dataset acdc \
    --batch_size 16 \
    --epochs 1000 \
    --gpu 0
```

#### 2. Generate Augmented Data

```bash
# Generate synthetic images for augmentation
python generate_augmented_data.py \
    --model_path checkpoints/diffusion_model.pth \
    --num_samples 1000 \
    --output_dir augmented_data/ \
    --prompts_config configs/prompts.yaml
```

#### 3. Train Segmentation Model

```bash
# Train segmentation model with augmented data
python train_segmentation.py \
    --config configs/segmentation_config.yaml \
    --data_root /path/to/data \
    --augmented_data augmented_data/ \
    --model unet \
    --batch_size 12
```

### Inference

```bash
# Run inference on test set
python inference.py \
    --model_path checkpoints/best_model.pth \
    --test_data /path/to/test/data \
    --output_dir results/
```

---

<div align="center">
<p>⭐ If you find this repository helpful, please give it a star! ⭐</p>
</div>
