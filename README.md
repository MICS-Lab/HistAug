

<div align="center">
<h1>[ICCV 2025] HistAug : Controllable Latent-Space Augmentation for Digital Pathology</h1>
</div>

<div align="center">


[![Paper](https://img.shields.io/badge/HistAug-arXiv.2508.14588-purple.svg)](https://arxiv.org/abs/2508.14588)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-page-yellow.svg)](https://huggingface.co/collections/sofieneb/histaug-models-68a334437f71d35c7037a54e)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](./LICENSE)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)
</div>

 
**HistAug** is a lightweight transformer-based generator for **controllable feature-space augmentations** in digital pathology. Instead of performing costly image-space transformations on millions of WSI patches, HistAug operates **directly on patch embeddings** extracted from foundation models.

By conditioning on explicit transformation parameters (e.g., hue shift, erosion, HED color transform), HistAug produces realistic augmented embeddings that **preserve semantic content** while mimicking the effect of corresponding image-level augmentations.

In practice, HistAug variants faithfully reconstruct ground-truth augmented embeddings with high cosine similarity at **10X, 20X, and 40X magnifications** (see the [performance table](#average-performance-across-models)).

This enables Multiple Instance Learning (MIL) models to benefit from:

* ⚡ **Faster augmentation** (orders of magnitude faster than image-space transforms)
* 🧠 **Low memory usage** (up to 200k patches in parallel on a single V100 32GB)
* 🎛 **Controllable, WSI-consistent augmentations** (bag-wise or patch-wise conditioning)

> **Abstract**: Whole slide image (WSI) analysis in digital pathology presents unique challenges due to the gigapixel resolution of WSIs and the scarcity of dense supervision signals. While Multiple Instance Learning (MIL) is a natural fit for slide-level tasks, training robust models requires large and diverse datasets. Even though image augmentation techniques could be utilized to increase data variability and reduce overfitting, implementing them effectively is not a trivial task. Traditional patch-level augmentation is prohibitively expensive due to the large number of patches extracted from each WSI, and existing feature-level augmentation methods lack control over transformation semantics. We introduce **HistAug**, a fast and efficient generative model for controllable augmentations in the latent space for digital pathology. By conditioning on explicit patch-level transformations (e.g., hue, erosion), **HistAug** generates realistic augmented embeddings while preserving initial semantic information. Our method allows the processing of a large number of patches in a single forward pass efficiently, while at the same time consistently improving MIL model performance. Experiments across multiple slide-level tasks and diverse organs show that **HistAug** outperforms existing methods, particularly in low-data regimes. Ablation studies confirm the benefits of learned transformations over noise-based perturbations and highlight the importance of uniform WSI-wise augmentation.

👉 Want to use a pretrained HistAug model? Check out the full collection here: **[🤗 HistAug models collection](https://huggingface.co/collections/sofieneb/histaug-models-68a334437f71d35c7037a54e)**


---

# Pretrained Models on Hugging Face

## How to Access Models

You can load pretrained HistAug models from the Hugging Face Hub. We provide variants aligned with several foundation models:

* [CONCH](https://www.nature.com/articles/s41591-024-02856-4) → [`sofieneb/histaug-conch`](https://huggingface.co/sofieneb/histaug-conch)
* [CONCH v1.5](https://arxiv.org/abs/2411.19666) → [`sofieneb/histaug-conch_v15`](https://huggingface.co/sofieneb/histaug-conch_v15)
* [Virchow2](https://arxiv.org/abs/2408.00738) → [`sofieneb/histaug-virchow2`](https://huggingface.co/sofieneb/histaug-virchow2)
* [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) → [`sofieneb/histaug-hoptimus1`](https://huggingface.co/sofieneb/histaug-hoptimus1)
* [UNI](https://www.nature.com/articles/s41591-024-02857-3) → [`sofieneb/histaug-uni`](https://huggingface.co/sofieneb/histaug-uni)

Each repo includes a short README and a config embedded in the model card. Load any model with `transformers`. Example with **Virchow2**:

```python
import torch
from transformers import AutoModel

# Load HistAug
model_id = "sofieneb/histaug-virchow2"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

# Example: patch embeddings from Virchow2
num_patches = 50000
embedding_dim = 2560
patch_embeddings = torch.randn((num_patches, embedding_dim), device="cuda")

# Sample augmentation parameters
# mode="wsi_wise" applies the same transformation across the whole slide
# mode="instance_wise" applies different transformations per patch
aug_params = model.sample_aug_params(
    batch_size=num_patches,
    device=patch_embeddings.device,
    mode="wsi_wise"
)

# Apply augmentation in latent space
augmented_embeddings = model(patch_embeddings, aug_params)
print(augmented_embeddings.shape)  # (num_patches, embedding_dim)
```

> **Tip:** Each model repository (linked above) includes additional usage examples, configuration files, and details on how to adapt HistAug to your workflow.
## Average Performance Across Models


The table reports **mean cosine similarity (%)** between HistAug-generated embeddings and their image-augmented counterparts, with **95% bootstrap confidence intervals**. Results are averaged over **10X, 20X, 40X magnifications** and across the **BLCA, BRCA, and LUSC** datasets (total of **10,000 patches per foundation model**).

| Foundation model | Mean (%) | 95% CI (bootstrap, %) |
| ---------------- | -------: | :-------------------: |
| CONCH            |     93.0 |     \[92.9, 93.1]     |
| CONCH v1.5       |     91.9 |     \[91.8, 92.0]     |
| Virchow2         |     90.5 |     \[90.3, 90.6]     |
| H-optimus-1      |     81.7 |     \[81.5, 81.9]     |
| UNI              |     80.9 |     \[80.7, 81.1]     |



# Train Your Own HistAug Model

## Installation

```bash
# 1) Create and activate the conda environment
conda create -n histaug_env python=3.10 -y
conda activate histaug_env

# 2) Install OpenSlide (library + Python bindings)
conda install -c conda-forge openslide openslide-python -y

# 3) Install this project in editable mode
pip install -e .
```

## Training & Evaluation

```bash
cd src/histaug

# Training
python train.py --stage='train' --config config/Histaug_conch.yaml

# Testing
python train.py --stage='test' --config config/Histaug_conch.yaml
```

* The config files for all pretrained models are in **[`src/histaug/config`](src/histaug/config/)**.
* Please set your own paths for WSI preprocessing (patching) under the `Data.patching` key in your chosen config.
* We recommend using the **[CLAM toolbox](https://github.com/mahmoodlab/CLAM/)** so that no modifications in the code are required. Otherwise, small changes will be needed in **[`src/histaug/datasets/patch_dataset.py`](src/histaug/datasets/patch_dataset.py)**.
* To train HistAug on a new foundation model, simply add your model implementation to **[`src/histaug/models/foundation_models.py`](src/histaug/models/foundation_models.py)**.

> **Note:** If you encounter disk errors, they may be related to your `/tmp` folder. In that case, you can specify a custom temporary directory by doing the following:
>
> ```bash
> export TMPDIR=/path/to/your/tmp
> ```

---

## Offline usage (HPC clusters without internet)

If compute nodes don’t have internet, **always** run jobs with the offline flags to **prevent unnecessary network calls** and force local loads:

```bash
# On your compute job (no internet):
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Prepare the model **in advance** on a front-end/login node (with internet), then choose **either** approach below.

### Option — Warm the cache (simplest)

```bash
# On the front-end/login node (with internet):
python -c "from transformers import AutoModel; AutoModel.from_pretrained('sofieneb/histaug-conch_v15', trust_remote_code=True)"
```

Then in your offline job/script:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "sofieneb/histaug-conch_v15",
    trust_remote_code=True,
    local_files_only=True,  # uses local cache only
)
```

### Option — Download to a local folder with `hf download`

```bash
# On the front-end/login node (with internet):
hf download sofieneb/histaug-conch_v15 --local-dir ./histaug-conch_v15
```

Then in your offline job/script:

```python
from transformers import AutoModel
cross_transformer = AutoModel.from_pretrained(
    "./histaug-conch_v15",   # local path instead of hub ID
    trust_remote_code=True,
    local_files_only=True,  # uses local files only
)
```

---

# Citation

If our work contributes to your research, or if you incorporate part of this code, please consider citing our paper:

```bibtex
@misc{boutaj2025controllablelatentspaceaugmentation,
      title={Controllable Latent Space Augmentation for Digital Pathology}, 
      author={Sofiène Boutaj and Marin Scalbert and Pierre Marza and Florent Couzinie-Devy and Maria Vakalopoulou and Stergios Christodoulidis},
      year={2025},
      eprint={2508.14588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14588}, 
}
```
