# 🔍 Visual Anomaly Detection using PatchCore (CNN-based)

## 📌 Overview
This project implements a **research-grade visual anomaly detection pipeline** using the **PatchCore algorithm** for industrial inspection tasks.  
The method detects both **global and local anomalies** by modeling normal patterns in deep feature space, without requiring anomaly samples during training.

The project is:
- Industry-focused (non-medical)
- Unsupervised
- Interpretable (pixel-level anomaly maps)
- Aligned with international research standards

---

## 🧠 Motivation
Reconstruction-based methods such as Autoencoders often fail to detect **small localized defects**.  
PatchCore overcomes this limitation by using **pretrained CNN features** and **nearest-neighbor distance estimation**, making it more robust for industrial datasets.

---

## 📂 Dataset
**MVTec Anomaly Detection Dataset**

Structure per category:
- `train/` → normal images only
- `test/` → normal + anomalous images
- `ground_truth/` → pixel-level masks

Categories tested:
- bottle
- capsule

---

## 🏗️ Methodology

### 🔹 Backbone Network
- **ResNet-50** pretrained on ImageNet
- Weights frozen (no backpropagation)
- Feature maps extracted from:
  - `conv2_block3_out`
  - `conv3_block4_out`
  - `conv4_block6_out`

### 🔹 Feature Processing
1. Extract multi-scale CNN feature maps
2. Resize all feature maps to `64×64`
3. Concatenate along channel dimension
4. Each spatial location acts as a **patch embedding**

### 🔹 Memory Bank Construction
- Built using **only normal training images**
- Patch embeddings are sampled (coreset-style)
- Acts as reference distribution of normality

### 🔹 Anomaly Scoring
- Patch-wise anomaly score:
  - Minimum Euclidean distance to memory bank
- Image-level score:
  - Maximum patch anomaly score
- Pixel-level anomaly map:
  - Reshaped patch scores

---

## 📊 Evaluation

### Metrics
- ROC–AUC
- Confusion Matrix
- Score distribution analysis

### Visualizations
- ROC Curve
- KDE plot
- Histogram
- Box Plot
- Pixel-level anomaly heatmaps

---

## 📈 Results

| Method | ROC–AUC |
|------|--------|
| CNN Autoencoder | ~0.70 |
| **PatchCore (ResNet-50)** | **~0.90** |

PatchCore demonstrates:
- Strong separation between normal and anomalous samples
- Accurate localization of defects
- No need for training on anomaly samples

---

## 🔎 Observations
- Feature-based methods outperform reconstruction-based methods
- Local defects are better captured via patch-level embeddings
- Pretrained CNN semantics improve robustness

---

## ⚠️ Limitations
- Memory bank size increases with dataset
- kNN computation is expensive
- Threshold selection is dataset-dependent

---
