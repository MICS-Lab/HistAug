# What is HistAug?

## Overview

**HistAug** is a research project that provides **controllable latent-space augmentation** for digital pathology. It's a machine learning tool designed to make data augmentation faster and more efficient when working with whole slide images (WSIs) in medical imaging, specifically for computational pathology.

## The Problem It Solves

In digital pathology, researchers work with gigapixel whole slide images (WSIs) of tissue samples. These images are typically:
- **Extremely large** (billions of pixels)
- **Split into millions of patches** (small image tiles)
- **Processed by foundation models** that convert patches into numerical embeddings (feature vectors)
- **Analyzed using Multiple Instance Learning (MIL)** for slide-level predictions

Traditional data augmentation (rotating, color shifting, etc.) is applied to raw images, which is:
- ⏱️ **Slow**: Processing millions of patches takes significant time
- 💾 **Memory-intensive**: Storing augmented images requires massive storage
- 🔄 **Redundant**: Images are augmented, then encoded—doing the same work repeatedly

## The HistAug Solution

Instead of augmenting **images** and then encoding them, HistAug augments **embeddings** (the numerical representations) directly. This is:

1. **Faster**: No need to process pixels—work directly with compact embeddings
2. **Memory-efficient**: Process up to 200,000 patches in parallel on a single GPU
3. **Controllable**: Specify exactly what augmentation to apply (hue shift, rotation, etc.)
4. **Semantically meaningful**: Augmented embeddings mimic what you'd get from augmenting the original image

### How It Works

```
Traditional Pipeline:
Image → Augment (slow) → Encode with Foundation Model → Embedding

HistAug Pipeline:
Image → Encode with Foundation Model → Embedding → HistAug (fast) → Augmented Embedding
```

HistAug is a **lightweight transformer model** that learns to transform embeddings as if the original image had been augmented. It conditions on explicit transformation parameters, allowing users to control exactly what augmentation is applied.

## Key Components

### 1. **Foundation Model Integration** (`src/histaug/models/foundation_models.py`)
- Integrates with popular pathology foundation models:
  - CONCH and CONCH v1.5
  - Virchow2
  - H-optimus-1
  - UNI
- These models convert tissue image patches into embeddings

### 2. **HistAug Transformer Model** (`src/histaug/models/histaug_model.py`)
- A custom transformer architecture with:
  - Multi-head attention mechanism
  - Cross-attention between embeddings and transformation parameters
  - Learnable positional encodings for transformation types
- Takes input embeddings and augmentation parameters
- Outputs augmented embeddings

### 3. **Augmentation Pipeline** (`src/histaug/utils/transform_factory.py`)
- Supports various augmentations:
  - **Geometric**: rotation, horizontal/vertical flip, crop
  - **Morphological**: erosion, dilation
  - **Color**: brightness, contrast, saturation, hue, gamma
  - **Stain normalization**: HED (Hematoxylin-Eosin-DAB) transforms
  - **Blur**: Gaussian blur

### 4. **Training System** (`src/histaug/train.py`)
- PyTorch Lightning-based training framework
- Configurable via YAML files
- Supports multi-GPU training

### 5. **Data Processing** (`src/histaug/datasets/`)
- Loads WSI patches from disk
- Applies augmentations to images during training
- Extracts both original and augmented embeddings
- HistAug learns to predict the augmented embedding from the original

## Use Cases

### Primary Use Case: MIL Training Acceleration
When training Multiple Instance Learning models for slide-level classification:
- Extract embeddings once from all WSI patches
- During training, use HistAug to generate augmented embeddings on-the-fly
- Benefit from data augmentation without re-encoding images

### Secondary Use Case: Low-Data Scenarios
- In settings with limited training data
- Augmentation helps prevent overfitting
- HistAug makes this practical even with massive WSI datasets

## Technical Specifications

### Model Architecture
- **Input**: Embeddings from foundation models (typically 512-2048 dimensions)
- **Output**: Augmented embeddings (same dimension)
- **Design**: Transformer with 8-16 layers, 8 attention heads
- **Parameters**: ~10-50M depending on configuration
- **Training**: Supervised learning to match ground-truth augmented embeddings

### Performance
The models achieve high fidelity in recreating augmented embeddings:
- CONCH: 93.0% cosine similarity
- CONCH v1.5: 91.9% cosine similarity
- Virchow2: 90.5% cosine similarity
- H-optimus-1: 81.7% cosine similarity
- UNI: 80.9% cosine similarity

### Augmentation Modes
- **WSI-wise**: Same augmentation applied to all patches in a slide (maintains spatial consistency)
- **Instance-wise**: Different augmentation per patch (maximum variability)

## Project Structure

```
HistAug/
├── src/histaug/
│   ├── models/
│   │   ├── histaug_model.py       # Main transformer model
│   │   ├── foundation_models.py    # Foundation model wrappers
│   │   └── model_interface.py      # PyTorch Lightning interface
│   ├── datasets/
│   │   ├── patch_dataset.py        # WSI patch data loading
│   │   └── data_interface.py       # Data module
│   ├── utils/
│   │   ├── transform_factory.py    # Augmentation implementations
│   │   ├── hed.py                  # HED stain normalization
│   │   ├── loss_factory.py         # Loss functions
│   │   └── optim_factory.py        # Optimizers
│   ├── config/                     # YAML configuration files
│   └── train.py                    # Training/testing script
├── README.md                       # User documentation
└── pyproject.toml                  # Package dependencies
```

## Getting Started

### For Users
If you want to **use** pretrained HistAug models:

```python
from transformers import AutoModel
import torch

# Load pretrained model
model = AutoModel.from_pretrained(
    "sofieneb/histaug-conch_v15",
    trust_remote_code=True
)

# Your embeddings from a foundation model
embeddings = torch.randn(50000, 768)  # 50k patches, 768-dim

# Sample augmentation parameters
aug_params = model.sample_aug_params(
    batch_size=50000,
    device="cuda",
    mode="wsi_wise"  # Same augmentation for all patches
)

# Get augmented embeddings
augmented = model(embeddings, aug_params)
```

### For Researchers
If you want to **train** HistAug on a new foundation model:

1. Add your foundation model to `src/histaug/models/foundation_models.py`
2. Create a config file in `src/histaug/config/`
3. Prepare WSI patches (recommended: use [CLAM toolbox](https://github.com/mahmoodlab/CLAM/))
4. Run training:
   ```bash
   python src/histaug/train.py --stage=train --config path/to/config.yaml
   ```

## Research Context

This work was published at ICCV 2025:

> Boutaj, S., Scalbert, M., Marza, P., Couzinie-Devy, F., Vakalopoulou, M., & Christodoulidis, S. (2025). 
> Controllable Latent Space Augmentation for Digital Pathology. 
> *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 22165-22174.

**Paper**: [arXiv:2508.14588](https://arxiv.org/abs/2508.14588)

## License

This project is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, which means:
- ✅ You can use, modify, and distribute the code
- ✅ You can use it for commercial purposes
- ⚠️ You must give appropriate credit to the authors
- ⚠️ You must indicate if changes were made

## Key Innovations

1. **Latent-space operation**: First to systematically apply controllable augmentations in embedding space for pathology
2. **Semantic preservation**: Maintains the medical meaning of tissue features while adding variability
3. **Explicit conditioning**: Unlike random noise-based methods, allows precise control over augmentation type and strength
4. **Scalability**: Handles hundreds of thousands of patches efficiently
5. **Foundation model compatibility**: Works with multiple popular pathology encoders

## Dependencies

Key Python packages:
- PyTorch and PyTorch Lightning (deep learning)
- Transformers and Hugging Face Hub (model distribution)
- Kornia (image augmentations)
- Timm (vision transformer components)
- OpenSlide (WSI reading)
- CONCH (foundation model)

## Summary

**HistAug is a tool for making data augmentation in computational pathology fast, efficient, and controllable.** Rather than augmenting millions of image patches, it augments their compact numerical representations (embeddings), achieving the same effect with orders of magnitude less computation. This enables better model training, especially when working with large-scale whole slide image datasets.
