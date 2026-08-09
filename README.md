# NMSRG: Reproducibility Package

This repository contains the implementation and reproducibility materials for NMSRG, a two-stage framework for unsupervised domain adaptation.

## Overview

NMSRG consists of two stages:

1. **Stage 1** uses PyTorch to learn view-specific representations with masked learning, EMA teacher supervision, pseudo-label supervision, and domain-adversarial learning.
2. **Stage 2** uses MATLAB for nonconvex multi-view graph learning, latent-space projection, target clustering, and decision-level fusion.

Target-domain ground-truth labels are used only to compute final evaluation metrics. They are not used during Stage-1 training or Stage-2 graph learning, projection, clustering, pseudo-label refinement, or decision fusion.

## Repository Structure

```text
stage1_pytorch/
├── common/                               # Shared utilities
├── dalib/                                # Domain-adaptation modules
├── examples/
│   └── cdan_mcc_sdat_masking_modified.py # Stage-1 training implementation
├── extract_features.py                   # Export features for Stage 2
└── utils.py                              # Utilities

stage2_matlab/
├── OfficeHome/resnet50/                  # Stage-2 input organization for Office-Home
├── function/                             # NMSRG optimization and evaluation functions
├── utils/                                # LPP, clustering, and auxiliary functions
├── DA_LPP_MV_GLR_quick.m                 # Main Stage-2 NMSRG procedure
├── distmeans.m                           # Prototype-distance decision procedure
├── office_home.m                         # Office-Home main experiment script
├── select_params_source_validation.m     # Source-validation parameter selection
├── Run_Ablation_Table9.m                 # Table 9 ablation experiment
└── Run_GraphReplacement_Table8.m         # Table 8 graph-module replacement experiment

LICENSE
requirements.txt
environment.yml
```

## Environment

### Stage 1

The Stage-1 implementation was developed with:

```text
Python 3.10.18
PyTorch 2.1.0+cu118
CUDA 11.8
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
If the CUDA-enabled PyTorch packages cannot be resolved from the default package index, install the PyTorch CUDA 11.8 build following the official PyTorch installation instructions, and then install the remaining dependencies.
```

Alternatively, create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate nmsrg
```

### Stage 2

The Stage-2 implementation was developed with:

```text
MATLAB R2023b, version 23.2.0.2365128
Required toolbox: Statistics and Machine Learning Toolbox
```

## Experimental Settings

### Stage 1

For ViT experiments on Office-Home and VisDA-2017:

```text
Backbone: vit_base_patch16_224
Pretrained weights: enabled
Input resolution: 224 x 224
Epochs: 20
Batch size: 64
Initial learning rate: 0.002
Training seed: 0
Mask block size: 64
Mask ratio: 0.7
EMA coefficient: 0.9
Pseudo-label confidence threshold: 0.88
Loss weights: lambda_adv=1.0, lambda_cc=1.0, lambda_cm=0.01, lambda_pl=1.0
```

### Stage 2

```text
Latent dimension: r=128
Projection regularization: alpha=1
ADMM: mu0=1e-3, rho=1.5, mu_max=1e10, epsilon=1e-4, maximum iterations=100
K-means: K equals the number of source classes; source-prototype initialization; one run; maximum iterations=100; stop when SSE decrease is smaller than eps
```

### Random Seeds

```text
Stage-1 training: 0
Stage-2 main results: rng(2024, 'twister')
Source-validation split: 2024
Ablation experiments: 2024, 2025, 2026
Graph-module replacement experiments: 2024, 2025, 2026
```

## Data Preparation

Office-Home, ImageCLEF-DA, and VisDA-2017 are public benchmark datasets and are not redistributed in this repository. Download the datasets from their official sources and configure the corresponding local paths before running the code.

Stage 2 requires the view-specific source and target features exported by `stage1_pytorch/extract_features.py`, together with source-domain labels and the associated Stage-1 outputs. Target-domain labels are read from the official benchmark only for final evaluation.

## Reproduction Procedure

### 1. Run Stage 1

Run the Stage-1 training implementation using the desired dataset and transfer-task settings:

```bash
python stage1_pytorch/examples/cdan_mcc_sdat_masking_modified.py
```

Then export the Stage-2 inputs:

```bash
python stage1_pytorch/extract_features.py
```

This step exports the view-specific features required by Stage 2.

### 2. Add Stage-2 code to the MATLAB path

Open MATLAB from the repository root and run:

```matlab
addpath(genpath('stage2_matlab'));
```

### 3. Run main experiments

For Office-Home experiments, run:

```matlab
run('stage2_matlab/office_home.m');
```

The script performs source-validation parameter selection and runs the Stage-2 NMSRG procedure through `DA_LPP_MV_GLR_quick.m`.


```matlab
### 4. Reproduce Table 8 and Table 9

Load the three view-specific Stage-1 feature matrices and the corresponding labels following the data-loading procedure in `stage2_matlab/office_home.m`. Then call:
matlab Run_GraphReplacement_Table8( ... pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, ... so_feat_S, so_lb_S, so_feat_T, so_lb_T, ... pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T);

Run_Ablation_Table9( ... pre_feat_S, pre_lb_S, pre_feat_T, pre_lb_T, ... so_feat_S, so_lb_S, so_feat_T, so_lb_T, ... pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T);
The scripts use seeds 2024, 2025, and 2026 and report the mean and standard deviation.
```

## Numerical Reproducibility

Small numerical differences may arise across GPU models, CUDA versions, MATLAB platforms, and floating-point implementations. The reported performance trends should remain consistent when using the listed configurations and random seeds.

## Third-Party Components

Some auxiliary MATLAB functions retain their original authorship notices and license information. These files remain subject to their original license terms.
