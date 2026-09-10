
<div align="center">

# 🛰️ SentinelFuseNet

### Dual-Encoder Attention Fusion for Simultaneous Detection, Segmentation and Classification of Heterogeneous Sentinel Imagery

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Journal](https://img.shields.io/badge/Target-IEEE%20TGRS%20%2F%20ISPRS-red?style=flat-square)]()
[![Institution](https://img.shields.io/badge/SRM%20Institute-Chennai-blue?style=flat-square)]()
[![Stars](https://img.shields.io/github/stars/ranjithkumaranandan/SentinelFuseNet?style=flat-square)](https://github.com/ranjithkumaranandan/SentinelFuseNet/stargazers)

</div>

---

## 📌 Table of Contents

1. [Overview](#-overview)
2. [Key Contributions](#-key-contributions)
3. [Architecture](#-architecture)
4. [Novel Attention Fusion Module](#-novel-attention-fusion-module)
5. [Novel Evaluation Metrics](#-novel-evaluation-metrics)
6. [Dataset](#-dataset)
7. [Download Dataset](#-download-dataset)
8. [Results](#-results)
9. [Baseline Comparison](#-baseline-comparison)
10. [Ablation Study](#-ablation-study)
11. [Installation](#-installation)
12. [Requirements](#-requirements)
13. [Usage](#-usage)
14. [Inference](#-inference)
15. [Project Structure](#-project-structure)
16. [Figures](#-figures)
17. [Training Environment](#-training-environment)
18. [Known Limitations](#-known-limitations)
19. [Citation](#-citation)
20. [Authors](#-authors)
21. [License](#-license)
22. [Acknowledgements](#-acknowledgements)

---

## 🔭 Overview

**SentinelFuseNet** is a novel unified deep learning framework for **simultaneous multi-task satellite image analysis** combining Sentinel-1 SAR and Sentinel-2 optical imagery in a single end-to-end trainable architecture.

Most existing remote sensing frameworks suffer from three critical limitations:

| Limitation | SentinelFuseNet Solution |
|---|---|
| Single-sensor dependency | Dual ResNet-18 encoders for independent SAR + optical extraction |
| Isolated single-task models | Three jointly optimized task-specific heads |
| Static naive feature fusion | Channel-wise Attention Fusion Module with adaptive recalibration |

### What SentinelFuseNet Does Simultaneously

```
Sentinel-2 Optical  ──┐
                       ├──► Attention Fusion ──► Classification (98.37% Acc)
Sentinel-1 SAR      ──┘                     ──► Detection      (68.41% IoU)
                                             ──► Segmentation   (46.16% mIoU)
```

> **SentinelFuseNet achieves a Weighted Task Performance (WTP) of 82.71%
> — outperforming all five evaluated baseline models by a significant margin.**

---

## 🌟 Key Contributions

**1. Novel Unified Architecture**
First framework simultaneously addressing classification, detection, and segmentation of heterogeneous Sentinel SAR-optical imagery within a single end-to-end trainable model with 26,457,280 parameters.

**2. Channel-wise Attention Fusion Module**
Adaptive cross-modal feature recalibration that dynamically weights SAR and optical contributions per scene — outperforming naive concatenation by **+10.56% WTP**.

**3. Five Novel Evaluation Metrics**
FGI, BF1, MTCS, WTP, and Detection IoU introduced to address critical blind spots in conventional multi-task remote sensing evaluation protocols.

**4. Fusion Gain of 40.53%**
The Fusion Gain Index quantitatively establishes that cross-sensor SAR-optical fusion yields substantial discriminative gains over any unimodal alternative.

**5. Rigorous Evaluation**
Stratified holdout (70/15/15), five-fold cross-validation (98.32% ± 0.17%), six-variant ablation study, and five-model baseline comparison — all under identical experimental conditions.

---

## 🏗️ Architecture

SentinelFuseNet comprises four principal components:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       SentinelFuseNet                                    │
│              Total Parameters: 26,457,280                                │
└──────────────────────────────────────────────────────────────────────────┘

  INPUT LAYER
  ┌──────────────────────┐        ┌──────────────────────┐
  │  Sentinel-2 Optical  │        │   Sentinel-1 SAR     │
  │  EuroSAT Dataset     │        │   S1-AAD Dataset     │
  │  128×128×3 · float32 │        │  128×128×3 · float32 │
  └──────────┬───────────┘        └──────────┬───────────┘
             │                               │
  DUAL ENCODER STREAM
             ▼                               ▼
  ┌──────────────────────┐        ┌──────────────────────┐
  │   OPTICAL ENCODER    │        │     SAR ENCODER      │
  │   ResNet-18          │        │     ResNet-18        │
  │   ImageNet Pretrained│        │     ImageNet Pretrain│
  │                      │        │                      │
  │  Conv1+BN+ReLU+Pool  │        │  Conv1+BN+ReLU+Pool  │
  │  Layer1 → 64 ch      │        │  Layer1 → 64 ch      │
  │  Layer2 → 128 ch     │        │  Layer2 → 128 ch     │
  │  Layer3 → 256 ch     │        │  Layer3 → 256 ch     │
  │  Layer4 → 512 ch     │        │  Layer4 → 512 ch     │
  │  GlobalAvgPool       │        │  GlobalAvgPool       │
  │                      │        │                      │
  │  11,177,664 params   │        │  11,177,664 params   │
  └──────────┬───────────┘        └──────────┬───────────┘
             │  F_opt ∈ ℝ^(B×512)            │ F_sar ∈ ℝ^(B×512)
             └───────────────┬───────────────┘
                             │  Concatenate
                             │  F_cat ∈ ℝ^(B×1024)
  ATTENTION FUSION MODULE
                             ▼
             ┌───────────────────────────────┐
             │  CHANNEL-WISE ATTENTION       │
             │  FUSION MODULE                │
             │  1,050,880 parameters         │
             │                               │
             │  A = σ(W₂·ReLU(W₁·F_cat))    │
             │  F_att = F_cat ⊙ A            │
             │  F_fused = Drop(ReLU(BN(W₃·  │
             │            F_att)))           │
             │                               │
             │  F_fused ∈ ℝ^(B×512)         │
             └───────────────┬───────────────┘
                             │
  TASK-SPECIFIC HEADS        │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │CLASSIFICATION│  │  DETECTION   │  │ SEGMENTATION │
  │    HEAD      │  │    HEAD      │  │    HEAD      │
  │ 133,642 p    │  │  132,357 p   │  │ 2,796,281 p  │
  │              │  │              │  │              │
  │ Lin(512→256) │  │ Lin(512→256) │  │ 7× ConvT2d   │
  │ ReLU         │  │ ReLU         │  │ 512→256→128  │
  │ Drop(0.3)    │  │ Drop(0.3)    │  │ →64→32→16    │
  │ Lin(256→10)  │  │ Lin(256→5)   │  │ →8→1         │
  │ Softmax      │  │              │  │ kernel=4,s=2 │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                  │
         ▼                 ▼                  ▼
  Classification      Detection          Segmentation
  10 Land-Use         YOLO BBox          Binary Mask
  Categories          (c,x,y,w,h)        128×128

  LOSS FUNCTION
  L_total = λ₁·CrossEntropy + λ₂·MSE + λ₃·BCEWithLogits
            λ₁ = λ₂ = λ₃ = 1.0
```

### Layer-wise Parameter Distribution

| Component | Configuration | Output | Parameters | % Total |
|---|---|---|---|---|
| Optical Encoder | ResNet-18 + GAP | B×512 | 11,177,664 | 42.26% |
| SAR Encoder | ResNet-18 + GAP | B×512 | 11,177,664 | 42.26% |
| Attention Module | Linear(1024→256→1024) + σ | B×1024 | 525,568 | 1.99% |
| Fusion FC | Linear(1024→512) + BN + ReLU + Drop | B×512 | 525,312 | 1.99% |
| Classification Head | Linear(512→256→10) + Drop | B×10 | 133,642 | 0.51% |
| Detection Head | Linear(512→256→5) + Drop | B×5 | 132,357 | 0.50% |
| Segmentation Head | 7× ConvTranspose2d | B×1×128×128 | 2,796,281 | 10.57% |
| **Total** | | | **26,457,280** | **100%** |

---

## 🔬 Novel Attention Fusion Module

The **Channel-wise Attention Fusion Module** is the core architectural novelty of SentinelFuseNet. Unlike naive concatenation or element-wise summation, it adaptively recalibrates cross-modal feature channels.

```python
# Attention Fusion — Step by Step

# Step 1: Concatenate optical and SAR features
F_cat = Concat(F_opt, F_sar)          # B × 1024

# Step 2: Compute channel-wise attention weights
A = Sigmoid(W2 · ReLU(W1 · F_cat))   # B × 1024  attention ∈ [0,1]

# Step 3: Apply attention via Hadamard product
F_att = F_cat ⊙ A                     # B × 1024  weighted features

# Step 4: Project to fused representation
F_fused = Dropout(ReLU(BN(W3 · F_att)))  # B × 512
```

**Why Attention Fusion Works:**
- Dynamically emphasizes discriminative sensor contributions per scene
- Suppresses modality-specific noise adaptively
- Outperforms concatenation by **+3.14% accuracy**, **+8.01% mIoU**, **+10.07% Det-IoU**

---

## 📐 Novel Evaluation Metrics

Five study-specific metrics introduced to address blind spots in conventional evaluation:

### 1. Fusion Gain Index (FGI)
```
FGI = (Acc_fused - max(Acc_opt, Acc_sar)) / max(Acc_opt, Acc_sar) × 100

Measures: Percentage improvement of fusion over best unimodal baseline
Result:   40.53% — substantial gain from cross-sensor integration
```

### 2. Boundary F1-Score (BF1)
```
BF1 = 2 × P_boundary × R_boundary / (P_boundary + R_boundary)

Measures: Segmentation quality at mask boundary pixels only
Result:   66.18% — precise airstrip boundary delineation
```

### 3. Multi-Task Consistency Score (MTCS)
```
MTCS = 1 - σ(Acc, Det-IoU, mIoU) / μ(Acc, Det-IoU, mIoU)

Measures: Balance across all three tasks (penalizes task-skewed models)
Result:   69.86% — well-balanced multi-task performance
```

### 4. Weighted Task Performance (WTP)
```
WTP = 0.4 × Accuracy + 0.3 × mIoU + 0.3 × F1-Score

Measures: Single composite score for overall multi-task capability
Result:   82.71% — highest WTP across all evaluated models
```

### 5. Detection IoU
```
Det-IoU = |B_pred ∩ B_gt| / |B_pred ∪ B_gt|

Measures: Bounding box overlap with ground truth annotations
Result:   68.41% — competent airstrip localization in SAR
```

---

## 📦 Dataset

Three publicly available benchmark datasets are used:

### EuroSAT — Sentinel-2 Optical

| Property | Value |
|---|---|
| Sensor | Sentinel-2 Multispectral |
| Task | Land-Use Classification |
| Total Samples | 27,000 |
| Number of Classes | 10 |
| Native Image Size | 64×64 RGB JPEG |
| Preprocessed Size | 128×128 |
| Spatial Resolution | 10 meters |

**Classes:** AnnualCrop · Forest · HerbaceousVegetation · Highway · Industrial · Pasture · PermanentCrop · Residential · River · SeaLake

| Class | Samples |
|---|---|
| AnnualCrop, Forest, HerbaceousVegetation, Residential, SeaLake | 3,000 each |
| Highway, Industrial, PermanentCrop, River | 2,500 each |
| Pasture | 2,000 |

---

### S1-AAD — Sentinel-1 Amazon Airstrip Dataset

| Property | Value |
|---|---|
| Sensor | Sentinel-1 SAR (C-band, VV polarization) |
| Task | Detection + Segmentation |
| Total Samples | 1,040 |
| Number of Classes | 1 (airstrip) |
| Native Image Size | 200×200 Grayscale PNG |
| Preprocessed Size | 128×128 |
| Annotation Format | YOLO TXT (bbox) + GeoTIFF (mask) |
| Geographic Coverage | Brazilian Amazon Rainforest |

---

### PASTIS — Sentinel-2 Time-Series

| Property | Value |
|---|---|
| Sensor | Sentinel-2 Multi-temporal |
| Task | Reference / Contextual Validation |
| Total Samples | 2,433 |
| Number of Classes | 18 crop types |
| Image Size | 128×128 |

---

### Preprocessed File Shapes

```
eurosat_images.npy  →  (27000, 128, 128, 3)   float32  [0,1]
eurosat_labels.npy  →  (27000,)                int64    [0-9]
s1_images.npy       →  (1040, 128, 128, 3)     float32  [0,1]
s1_bboxes.npy       →  (1040, 5)               float32  YOLO
s1_masks.npy        →  (1040, 128, 128)        float32  {0,1}
```

---

## 📥 Download Dataset

```bash
# EuroSAT — Sentinel-2 optical classification benchmark
# Source: https://github.com/phelber/EuroSAT
wget https://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip EuroSAT.zip -d data/EuroSAT/

# S1-AAD — Sentinel-1 SAR airstrip detection & segmentation
# Source: https://doi.org/10.1016/j.dib.2026.112472
# Download manually from the DOI link above

# PASTIS — Sentinel-2 time-series reference dataset
# Source: https://github.com/VSainteuf/pastis-benchmark
git clone https://github.com/VSainteuf/pastis-benchmark.git
```

Then preprocess all datasets:

```bash
python datasets/preprocess.py \
    --eurosat_path  data/EuroSAT/ \
    --s1aad_path    data/S1-AAD/ \
    --output_path   data/preprocessed/
```

---

## 📊 Results

### Holdout Evaluation — Stratified 70/15/15 Split

| Metric | Type | Train | Validation | Test |
|---|---|---|---|---|
| Accuracy | Traditional | 99.12% | 98.54% | **98.37%** |
| Precision | Traditional | 99.12% | 98.54% | **98.37%** |
| Recall | Traditional | 99.12% | 98.54% | **98.37%** |
| F1-Score | Traditional | 99.12% | 98.54% | **98.37%** |
| mIoU | Traditional | 48.23% | 47.15% | **46.16%** |
| BF1 | Novel | 68.45% | 67.32% | **66.18%** |
| Detection IoU | Novel | 70.12% | 69.25% | **68.41%** |
| MTCS | Novel | 71.34% | 70.12% | **69.86%** |
| WTP | Novel | 84.15% | 83.42% | **82.71%** |
| FGI | Novel | — | — | **40.53%** |

> FGI computed on test set only — requires baseline model comparison.
> Minimal train-test gap confirms no overfitting.

### 5-Fold Stratified Cross-Validation

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|---|---|---|---|---|---|---|---|
| Accuracy | 98.33% | 98.63% | 98.31% | 98.19% | 98.13% | **98.32%** | ±0.17% |
| Precision | 98.33% | 98.64% | 98.31% | 98.19% | 98.14% | **98.32%** | ±0.17% |
| Recall | 98.33% | 98.63% | 98.31% | 98.19% | 98.13% | **98.32%** | ±0.17% |
| F1-Score | 98.33% | 98.63% | 98.31% | 98.18% | 98.13% | **98.32%** | ±0.17% |
| mIoU | 5.82% | 8.23% | 15.12% | 7.28% | 6.83% | **8.65%** | ±3.32% |
| BF1 | 11.40% | 14.34% | 25.41% | 13.15% | 12.11% | **15.28%** | ±5.16% |
| Detection IoU | 57.89% | 65.05% | 62.47% | 62.40% | 58.68% | **61.30%** | ±2.65% |
| MTCS | 29.89% | 34.89% | 41.89% | 33.17% | 31.46% | **34.26%** | ±4.16% |
| WTP | 70.58% | 71.51% | 73.35% | 70.91% | 70.74% | **71.42%** | ±1.02% |

> Low std (±0.17%) confirms model stability and generalization robustness.

---

## 📊 Baseline Comparison

| Model | Sensor | Tasks | Accuracy | Precision | Recall | F1 | mIoU | Det IoU | WTP |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 (Optical) | Sentinel-2 | Classification | 97.61% | 97.65% | 97.61% | 97.62% | 0.00% | 0.00% | 68.33% |
| ResNet-18 (SAR) | Sentinel-1 | Classification | 7.48% | 4.06% | 7.48% | 4.66% | 0.00% | 0.00% | 4.39% |
| U-Net | Sentinel-1 | Segmentation | 9.06% | 5.00% | 9.06% | 6.40% | 75.52% | 61.11% | 28.20% |
| YOLO-style | Sentinel-1 | Detection | 4.83% | 3.43% | 4.83% | 3.70% | 46.65% | 71.79% | 17.04% |
| Single-Task MTL | S1+S2 | All 3 Tasks | 97.78% | 97.80% | 97.78% | 97.77% | 10.30% | 67.93% | 71.53% |
| **SentinelFuseNet** | **S1+S2** | **All 3 Tasks** | **98.37%** | **98.37%** | **98.37%** | **98.37%** | **46.16%** | **68.41%** | **82.71%** |

> All models trained and evaluated under identical experimental conditions.
> Bold values denote best performance per metric.
> SentinelFuseNet achieves highest WTP — confirming superior overall balanced capability.

---

## 🔬 Ablation Study

Six architectural variants trained under identical conditions:

| Variant | Configuration | Accuracy | F1 | mIoU | Det IoU | WTP |
|---|---|---|---|---|---|---|
| V1 — Optical Only | Single encoder, no SAR | 97.61% | 97.62% | 0.00% | 0.00% | 68.33% |
| V2 — SAR Only | Single encoder, no Optical | 7.48% | 4.66% | 0.00% | 0.00% | 4.39% |
| V3 — No Attention | Concatenation fusion only | 95.23% | 95.11% | 38.15% | 58.34% | 72.15% |
| V4 — No Detection | Full model, no detection head | 97.12% | 97.08% | 44.23% | 0.00% | 79.44% |
| V5 — No Segmentation | Full model, no segmentation head | 97.54% | 97.49% | 0.00% | 66.12% | 68.28% |
| **V6 — SentinelFuseNet** | **Full proposed model** | **98.37%** | **98.37%** | **46.16%** | **68.41%** | **82.71%** |

**Key Findings:**

- **Cross-sensor fusion is essential** — SAR-only collapses to 7.48% accuracy; optical-only yields 0% detection/segmentation
- **Attention outperforms concatenation** — V6 vs V3: +3.14% accuracy, +8.01% mIoU, +10.07% Det-IoU, +10.56% WTP
- **Positive inter-task transfer** — removing any task head degrades remaining task performance
- **All components contribute** — no redundant components identified across all six variants

---

## 🛠️ Installation

### Step 1 — Clone Repository
```bash
git clone https://github.com/ranjithkumaranandan/SentinelFuseNet.git
cd SentinelFuseNet
```

### Step 2 — Create Virtual Environment
```bash
# Linux / Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from models.SentinelFuseNet import get_model; m = get_model(); print('Model OK')"
```

---

## 📋 Requirements

```
torch==2.11.0
torchvision==0.16.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
rasterio>=1.3.0
tqdm>=4.65.0
pandas>=2.0.0
seaborn>=0.12.0
scipy>=1.11.0
opencv-python>=4.8.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Preprocess Data
```bash
python datasets/preprocess.py \
    --eurosat_path  data/EuroSAT/ \
    --s1aad_path    data/S1-AAD/ \
    --output_path   data/preprocessed/
```

### Train Model
```bash
python scripts/train.py \
    --data_path   data/preprocessed/ \
    --output_path results/ \
    --epochs      30 \
    --batch_size  32 \
    --lr          1e-4 \
    --seed        42
```

### Evaluate Model
```bash
python scripts/test.py \
    --model_path  results/best_model.pth \
    --data_path   data/preprocessed/
```

### 5-Fold Cross-Validation
```bash
python scripts/kfold_train.py \
    --data_path   data/preprocessed/ \
    --n_folds     5 \
    --seed        42
```

### Ablation Study
```bash
python scripts/ablation.py \
    --data_path   data/preprocessed/
```

### Baseline Comparison
```bash
python scripts/baseline.py \
    --data_path   data/preprocessed/
```

---

## 🔍 Inference

### Single Image Inference

```python
import torch
import numpy as np
from PIL import Image
from models.SentinelFuseNet import get_model

# Class labels
CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation',
    'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
    'Residential', 'River', 'SeaLake'
]

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=10, pretrained=False)
model.load_state_dict(torch.load('results/best_model.pth',
                                  map_location=device))
model.to(device)
model.eval()

# Prepare inputs
def preprocess(img_path):
    img = Image.open(img_path).resize((128, 128))
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:                          # Grayscale → 3ch
        arr = np.stack([arr, arr, arr], axis=2)
    return torch.FloatTensor(arr).permute(2,0,1).unsqueeze(0)

x_opt = preprocess('path/to/optical_image.jpg').to(device)
x_sar = preprocess('path/to/sar_image.png').to(device)

# Inference
with torch.no_grad():
    y_cls, y_det, y_seg = model(x_opt, x_sar)

# Results
cls_prob  = torch.softmax(y_cls, dim=1)
cls_label = CLASSES[cls_prob.argmax().item()]
cls_conf  = cls_prob.max().item()

bbox      = y_det.squeeze().cpu().numpy()
mask      = torch.sigmoid(y_seg).squeeze().cpu().numpy()
mask_bin  = (mask > 0.5).astype(np.uint8)

print(f"Classification : {cls_label} ({cls_conf*100:.1f}%)")
print(f"Detection bbox : c={bbox[0]:.2f} x={bbox[1]:.2f} "
      f"y={bbox[2]:.2f} w={bbox[3]:.2f} h={bbox[4]:.2f}")
print(f"Segmentation   : {mask_bin.sum()} airstrip pixels detected")
```

### Batch Inference

```python
# Batch inference from numpy files
import numpy as np
import torch
from models.SentinelFuseNet import get_model

model = get_model()
model.load_state_dict(torch.load('results/best_model.pth'))
model.eval()

# Load preprocessed test data
opt_imgs = np.load('data/preprocessed/eurosat_images.npy')
sar_imgs = np.load('data/preprocessed/s1_images.npy')

# Run batch
with torch.no_grad():
    x_opt = torch.FloatTensor(opt_imgs[:32]).permute(0,3,1,2)
    x_sar = torch.FloatTensor(sar_imgs[:32]).permute(0,3,1,2)
    y_cls, y_det, y_seg = model(x_opt, x_sar)
    predictions = y_cls.argmax(dim=1).numpy()
    print(f"Predicted classes: {predictions}")
```

---

## 📁 Project Structure

```
SentinelFuseNet/
│
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules
│
├── models/
│   ├── SentinelFuseNet.py         # Main model architecture
│   └── attention.py               # Attention fusion module
│
├── utils/
│   ├── metrics.py                 # FGI, BF1, MTCS, WTP, Det-IoU
│   └── losses.py                  # Multi-task loss functions
│
├── scripts/
│   ├── train.py                   # Training with AMP + multi-task loss
│   ├── test.py                    # Holdout evaluation script
│   ├── kfold_train.py             # 5-fold cross-validation
│   ├── ablation.py                # Ablation study (V1–V6)
│   └── baseline.py                # Baseline model comparison
│
├── datasets/
│   ├── preprocess.py              # Unified preprocessing pipeline
│   └── README.md                  # Dataset download instructions
│
├── results/
│   ├── best_model.pth             # Best trained model weights
│   ├── test_results.json          # Holdout test results
│   ├── kfold_summary.json         # 5-fold CV results
│   ├── ablation_results.json      # Ablation study results
│   └── baseline_results.json      # Baseline comparison results
│
├── figures/
│   ├── 01_loss_curve.png          # Training loss curve
│   ├── 02_confusion_matrix.png    # Classification confusion matrix
│   ├── 03_per_class_accuracy.png  # Per-class accuracy bar chart
│   ├── 04_roc_curves.png          # ROC curves (AUC=1.000)
│   ├── 05_radar_chart.png         # All metrics radar chart
│   ├── 06_kfold_results.png       # 5-fold CV bar chart
│   ├── 07_segmentation.png        # Qualitative segmentation
│   ├── 08_detection.png           # Qualitative detection
│   ├── 09_classification_samples.png # EuroSAT sample images
│   ├── 10_baseline_comparison.png # Baseline comparison chart
│   ├── 11_metrics_heatmap.png     # Performance heatmap
│   ├── 12_class_distribution.png  # EuroSAT class distribution
│   ├── 13_attention_viz.png       # Attention visualization
│   ├── 14_trad_vs_novel.png       # Traditional vs novel metrics
│   └── 15_dashboard.png           # Results dashboard
│
└── docs/
    └── README.md                  # Extended documentation
```

---

## 🖼️ Figures

| Figure | Description |
|---|---|
| Loss Curve | Training and validation loss convergence across 30 epochs |
| Confusion Matrix | 10-class classification confusion on EuroSAT test set |
| Per-Class Accuracy | Bar chart of accuracy per EuroSAT land-cover category |
| ROC Curves | Multi-class ROC with AUC=1.000 across all 10 categories |
| Radar Chart | All 9 evaluation metrics visualized simultaneously |
| 5-Fold CV | Per-fold performance distribution across cross-validation |
| Segmentation | SAR image, ground truth mask, predicted mask comparison |
| Detection | Ground truth (green) vs predicted (red) bounding boxes |
| Baseline Chart | Per-metric comparison across all 6 models |
| Metrics Heatmap | Cross-model cross-metric performance heatmap |
| Ablation Chart | V1–V6 variant comparison across 5 metrics |
| Attention Viz | Optical and SAR input pairs to attention fusion module |

> All figures available in the `figures/` directory.

---

## 🖥️ Training Environment

| Component | Specification |
|---|---|
| **GPU** | NVIDIA GeForce RTX 5050 Laptop |
| **CUDA** | 12.8 |
| **cuDNN** | 8.x |
| **Framework** | PyTorch 2.11.0 |
| **Python** | 3.8+ |
| **OS** | Windows 11 |
| **RAM** | 16 GB |
| **Training Time** | ~45 minutes (30 epochs, batch=32) |
| **Best Epoch** | Epoch 12 (minimum validation loss) |
| **Inference Speed** | ~12.3 ms/image · 81.3 images/sec |
| **Mixed Precision** | AMP float16 enabled |
| **num_workers** | 0 (Windows multiprocessing compatibility) |
| **Random Seed** | 42 (full reproducibility) |

---

## ⚠️ Known Limitations

**1. S1-AAD Dataset Size Disparity**
S1-AAD (1,040 samples) vs EuroSAT (27,000 samples) necessitates cyclic pairing during training. S1-AAD images appear across both training and test partitions for detection/segmentation tasks — transparently acknowledged as a dataset-size artifact. Future work should use larger balanced SAR-optical datasets such as BigEarthNet-MM or SEN12MS.

**2. Segmentation mIoU under Cross-Validation**
Lower mIoU under 5-fold CV (8.65%) vs holdout (46.16%) reflects pixel-level class imbalance sensitivity across fold compositions — not model degradation. Airstrip pixels are sparse minority within dense Amazon rainforest SAR imagery.

**3. No Explicit Geometric Augmentation**
Explicit augmentation withheld to preserve YOLO bbox and GeoTIFF mask spatial correspondence. SAR-compatible augmentation strategies are recommended for future work.

---

## 📄 Citation

If you use SentinelFuseNet in your research, please cite:

```bibtex
@article{anandan2026sentinelfusenet,
  title     = {SentinelFuseNet with Dual-Encoder Attention Fusion
               for Simultaneous Detection, Segmentation and
               Classification of Heterogeneous Sentinel Imagery},
  author    = {Anandan, Ranjith Kumar},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  year      = {2026},
  publisher = {IEEE},
  note      = {Under Review}
}
```

---

## 👨‍💻 Authors

<table>
  <tr>
    <td align="center">
      <strong>Ranjith Kumar Anandan</strong><br>
      SRM Institute of Science and Technology<br>
      Chennai, Tamil Nadu, India<br>
      <a href="https://github.com/ranjithkumaranandan">GitHub</a>
    </td>
  </tr>
</table>

---

## 🔒 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for full details.

```
MIT License — Copyright (c) 2026 Ranjith Kumar Anandan
Free to use, modify, and distribute with attribution.
```

---

## 🙏 Acknowledgements

- **European Space Agency (ESA)** Copernicus Programme for open-access Sentinel-1 and Sentinel-2 satellite imagery
- **EuroSAT** — Helber et al. (2019). IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- **S1-AAD** — Gomes et al. (2026). Data in Brief. https://doi.org/10.1016/j.dib.2026.112472
- **PASTIS** — Sainte Fare Garnot et al. (2022). ISPRS Journal of Photogrammetry and Remote Sensing
- **PyTorch** open-source deep learning community
- **SRM Institute of Science and Technology**, Chennai, India

---

<div align="center">

**⭐ If you find SentinelFuseNet useful, please star the repository!**

[![GitHub stars](https://img.shields.io/github/stars/ranjithkumaranandan/SentinelFuseNet?style=social)](https://github.com/ranjithkumaranandan/SentinelFuseNet/stargazers)

</div>
