# Clustering Analysis with CNN Feature Extraction

This repository contains the implementation for **Programming Assignment 4 (Clustering Analysis)**.
The project focuses on clustering image data using deep feature representations extracted from a
pretrained **ResNet18** convolutional neural network, followed by dimensionality reduction and
classical clustering algorithms.

---

## Dataset

The **Faulty Solar Panel** image dataset is used in this project. It consists of four classes:

- Bird-drop  
- Clean  
- Dusty  
- Snow-Covered  

The same dataset was used in previous assignments, as required by the assignment instructions.
Ground-truth labels are used **only for external evaluation** and are **not used during clustering**.

---

## Feature Extraction

- Images are resized to **224 × 224** pixels.
- Image normalization is performed using **ImageNet mean and standard deviation**.
- A pretrained **ResNet18** model is used.
- Features are extracted from the **last convolutional layer (layer4)**.
- Global Average Pooling is applied to obtain a **512-dimensional feature vector** per image.

Feature extraction follows the approach described in:
> https://kozodoi.me/blog/20210527/extracting-features

---

## Dimensionality Reduction

- The extracted feature vectors are standardized.
- **Principal Component Analysis (PCA)** is applied to reduce the dimensionality to **2D**.

---

## Clustering Methods

Clustering is performed on the 2D representation using the following algorithms:

- K-Means (init = random, K = 4)
- K-Means++ (K = 4)
- Bisecting K-Means (init = random, K = 4)
- Spectral Clustering (default parameters, K = 4)
- DBSCAN (parameters selected to obtain 4 clusters)
- Agglomerative (Hierarchical) Clustering:
  - Single linkage
  - Complete linkage
  - Average linkage
  - Ward’s method

All clustering implementations use **scikit-learn**.

---

## Evaluation Metrics

Each clustering method is evaluated using:

- **Fowlkes–Mallows Index (FMI)**  
  External evaluation using ground-truth labels.

- **Silhouette Coefficient**  
  Internal evaluation based on cluster cohesion and separation.

For DBSCAN, evaluation metrics are computed on **non-noise samples only** (labels ≠ −1).

---

## Results

- FMI and Silhouette scores are computed for **all clustering methods**.
- Clustering methods are ranked from **best to worst** separately based on:
  - Fowlkes–Mallows Index
  - Silhouette Coefficient

---

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- scikit-learn
- numpy
- pandas

---

## Notes

This project is implemented strictly according to the assignment requirements and focuses on
methodology, evaluation, and comparison of clustering techniques rather than supervised learning.
