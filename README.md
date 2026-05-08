# MRI Super-Resolution Tutorial

This repository provides a step-by-step tutorial and a reusable training framework for MRI super-resolution using deep learning.

It is designed for students and researchers who want to learn the complete workflow of medical image super-resolution, including environment setup, data preparation, model training, and evaluation.

The repository demonstrates three model types:

- **UNet**: a deterministic residual-learning baseline
- **DDPM**: a diffusion-based super-resolution model
- **Flow Matching**: a generative model with efficient ODE-style sampling

The project is educational by design. The notebooks explain the workflow step by step, while the codebase is organized so that users can extend it to their own datasets, models, and experiments.

---

## Updates

`tutorial/04_full_sr_downstream_synthseg.ipynb` is now available.

It demonstrates full-SR inference with larger checkpoints and downstream segmentation analysis with SynthSeg.

---

## Repository Structure

```text
MRI_SR_Tutorial/
├── checkpoints/              # Local model checkpoints, not uploaded to GitHub
├── config/
│   ├── ddpm_2p5d.yaml
│   ├── ddpm_2p5d_full.yaml
│   ├── fm_2p5d.yaml
│   ├── fm_2p5d_full.yaml
│   ├── unet_2p5d.yaml
│   └── unet_2p5d_full.yaml
│
├── data/                     # Local raw and processed data, not uploaded to GitHub
├── logs/                     # Local training logs, not uploaded to GitHub
│
├── model/
│   ├── ddpm.py
│   ├── flow_matching.py
│   └── unet.py
│
├── results/                  # Local evaluation outputs, not uploaded to GitHub
│
├── tutorial/
│   ├── 00_environment_setup.ipynb
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_full_sr_downstream_synthseg.ipynb
│
├── dataset.py
├── loss.py
├── README.md
├── requirements.txt
├── test.py
├── train.py
└── utils.py
```

The folders `data/`, `logs/`, `checkpoints/`, and `results/` are generated locally and are not included in the repository.

---

## Installation

Create and activate a Conda environment:

```bash
conda create -n mri_sr python=3.10 -y
conda activate mri_sr
```

Install `ipykernel` and register the environment as a Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name mri_sr --display-name "Python (mri_sr)"
```

Install PyTorch according to your system configuration. Please use the official PyTorch installation selector:

```text
https://pytorch.org/get-started/locally/
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

After installation, open Jupyter, VS Code, or PyCharm and select the kernel:

```text
Python (mri_sr)
```

---

## Tutorial Track

If you are new to this repository, we recommend starting with the notebooks under the `tutorial/` folder.

Follow them in order:

```text
tutorial/00_environment_setup.ipynb
tutorial/01_data_preparation.ipynb
tutorial/02_training.ipynb
tutorial/03_evaluation.ipynb
tutorial/04_full_sr_downstream_synthseg.ipynb
```

### 00. Environment Setup

This notebook helps users create the Python environment, install dependencies, register the Jupyter kernel, check CUDA availability, and verify that the required packages are correctly installed.

### 01. Data Preparation

This notebook introduces the data preparation workflow for MRI super-resolution. It guides users through preparing [Neurite-OASIS](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) MRI data, simulating low-resolution inputs, and saving training-ready `.npy` files.

It also introduces two common settings:

- **2D super-resolution**, where one slice is used as input
- **2.5D super-resolution**, where neighboring slices provide additional context

The generated `.npy` files are used by the later training and evaluation steps.

### 02. Training

This notebook introduces residual learning for MRI super-resolution and demonstrates how to train UNet, DDPM, and Flow Matching models.

It covers:

- residual-learning formulation
- a simple UNet training example
- YAML-based training configuration
- 2.5D training for different model types
- checkpointing and metric logging

### 03. Evaluation

This notebook evaluates trained models using quantitative metrics and visual inspection.

The evaluation includes:

- PSNR
- SSIM
- MAE
- inference speed
- visualization of reconstructed MRI volumes

### 04. Full SR Inference and Downstream SynthSeg

This notebook extends the evaluation track to a stronger, more practical setting.

It covers:

- running full 2.5D SR checkpoints (`config/*_full.yaml`) for UNet, DDPM, and Flow Matching
- visualizing complete 3D results from `examples/` (`LR`, `HR`, `unet_2p5d_full`, `ddpm_2p5d_full`, `fm_2p5d_full`)
- using SynthSeg outputs for LR/SR and Neurite-OASIS labels in `HR/seg` as ground truth
- mapping SynthSeg labels to Neurite seg4 and reporting Dice by structure (`Cortex`, `SubcorticalGM`, `WhiteMatter`, `CSF_Ventricles`) plus mean Dice

---

## Reusable Training Framework

Besides the tutorial notebooks, this repository also provides a reusable framework for MRI super-resolution experiments.

The main scripts are:

```text
train.py
test.py
```

The main configuration files are stored under:

```text
config/
```

The core model files are stored under:
```text
model/
```

A typical training command is:

```bash
python train.py --config config/fm_2p5d.yaml
```

A typical testing command is:

```bash
python test.py --config config/fm_2p5d.yaml --ckpt your_ckpt.ckpt
```

The same YAML config file used for training can also be used for testing.

You can define your own model and configs in a similar style.

### Dataset and DataModule

The data pipeline is defined in `dataset.py`.

This file contains the dataset and datamodule classes used to load processed `.npy` data, build training/validation/test dataloaders, and provide batches to the model.

A typical batch contains fields such as:

```text
src
dst
residual
name
```

where:

- `src` is the low-resolution or upsampled low-resolution input
- `dst` is the high-resolution target
- `residual` is the target residual used for residual learning
- `name` is the sample name used when saving predictions

To use a new dataset, you can define your own dataset or datamodule class in `dataset.py`, then point to it in the YAML config file. This keeps the training and testing scripts unchanged while allowing the data pipeline to be customized.

### PyTorch Lightning

This project uses **PyTorch Lightning** to organize the training pipeline. Lightning helps manage:

- training and validation loops
- optimizer and scheduler configuration
- mixed-precision training
- checkpoint saving
- metric logging
- reproducible experiment setup
- multi-GPU or distributed training when needed

Each model is implemented as a LightningModule, and each experiment is controlled through a YAML config. This makes the codebase easier to maintain, reproduce, and extend.

### diffusers

This project uses **diffusers** components for DDPM and Flow Matching models.

The diffusers library provides highly integrated interfaces for modern generative modeling, including reusable UNet backbones, scheduler implementations, sampling utilities, and conditioning mechanisms.

Although this tutorial focuses on lightweight generative super-resolution models, the same design can be extended to more advanced conditional generation settings, such as:

- class-conditioned generation
- cross-attention conditioning
- text- or image-conditioned generation
- ControlNet-style structural conditioning
- multi-condition medical image synthesis

This makes the framework useful not only for basic MRI super-resolution, but also for more advanced medical image generation and restoration tasks.

### YAML Configuration

Experiments are controlled by YAML config files.

A config file defines:

- model class and model parameters
- dataset/datamodule class and data parameters
- trainer settings
- checkpoint paths
- logging paths
- testing output paths

Example config files include:

```text
config/unet_2p5d.yaml
config/ddpm_2p5d.yaml
config/fm_2p5d.yaml
```

The `_full.yaml` files are intended for longer or more complete training runs, while the shorter configs are more suitable for demonstration and testing.

---

## Supported Models

### UNet

UNet is used as a deterministic baseline. It directly predicts the residual between the upsampled low-resolution image and the high-resolution target.

This model is fast, stable, and useful as a strong baseline for MRI super-resolution.

### DDPM

DDPM is a diffusion-based model. It learns a denoising process and reconstructs the high-resolution residual through iterative sampling.

Compared with UNet, DDPM is generative and can model uncertainty, but it usually requires more sampling steps during inference.

### Flow Matching

Flow Matching learns a velocity field that maps noise to the target residual. It provides a generative alternative to DDPM and can often generate results with fewer sampling steps.

In this repository, Flow Matching is implemented with a diffusers-style UNet and scheduler interface.

---

## Outputs

During training and evaluation, the following folders are generated locally:

```text
logs/
checkpoints/
results/
```

These folders are not uploaded to GitHub.

- `logs/` stores training logs
- `checkpoints/` stores model checkpoints
- `results/` stores evaluation outputs and reconstructed volumes

---

## Notes

This repository is intended for teaching and demonstration. The default settings are lightweight and easy to run.

For research-level experiments, you may need larger models, longer training schedules, more data, and more extensive validation.

Quantitative metrics such as PSNR and SSIM are useful, but they should not be the only criteria for evaluating medical image quality. Visual inspection is also important, especially for checking anatomical structures, artifacts, and over-smoothing.

---

## Acknowledgements

This tutorial uses the [Neurite-OASIS](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) dataset, which is based on the OASIS-I brain MRI dataset.

The implementation is built with PyTorch, PyTorch Lightning, MONAI and diffusers.
