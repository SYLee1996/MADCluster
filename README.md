# MADCluster: Model-Agnostic Anomaly Detection with Self-supervised Clustering Network

**MADCluster** is a lightweight, plug-in anomaly detection framework for **unsupervised multivariate time-series anomaly detection**. It is designed to be **model-agnostic**: you can attach MADCluster on top of various deep anomaly-detection backbones (e.g., reconstruction-based, transformer-based, one-class objectives) by treating the backbone as a **Base Embedder** that produces temporal representations.

A key challenge in deep one-class anomaly detection is **hypersphere collapse**, where the network converges to a trivial representation (e.g., near-zero embeddings) and fails to form a meaningful boundary for normality. MADCluster mitigates this issue by **single-clustering normal patterns** while **jointly learning and continuously updating the cluster center**, rather than relying on a fixed centroid.

MADCluster consists of three main components:

- **Base Embedder**: extracts high-dimensional temporal dynamics from input sequences (any backbone can be used).
- **Cluster Distance Mapping**: pulls embeddings toward a learnable normal center and encourages a compact normal region.
- **Sequence-wise Clustering**: updates the center online through self-learning, producing stable single-cluster behavior.

To enable effective single clustering ($k=1$) in anomaly detection—where common clustering objectives become degenerate—MADCluster introduces a novel **One-directed Adaptive loss**, along with a **mathematical optimization proof** provided in the appendix. This loss trains a one-sided threshold parameter to progressively refine the normal cluster assignment and stabilize centroid learning.

**Results.** Across four public benchmarks (**MSL, SMAP, SMD, PSM**), MADCluster consistently improves backbone performance on both point-wise and region-aware metrics (e.g., **F1, Affiliation Precision/Recall, Range-AUC, VUS**), while remaining computationally lean and easy to integrate.

---

### Key Contributions

- **Model-agnostic plug-in**: works with diverse backbone architectures with minimal modification.
- **Prevents hypersphere collapse**: dynamic center updates preserve representational expressiveness.
- **One-directed Adaptive loss + proof**: stable single-cluster learning for one-class anomaly detection.
- **Consistent benchmark gains**: improves multiple families of baselines on standard datasets.

---

### Paper & Supplementary

- **Paper (arXiv)**: https://arxiv.org/abs/2505.16223
- **Supplementary Appendix** (proofs & extended experiments): `docs/MADCluster_Appendix.pdf` https://github.com/SYLee1996/MADCluster/blob/main/docs/MADCluster_Appendix.pdf

> Note: This repository includes an implementation for running MADCluster across multiple datasets and objectives. See **Quick Start** below for reproduction.
> 
## Project Structure

```

docs/
└── MADCluster_Appendix.pdf  
datasets/
├── MSL/
├── PSM/
├── SMAP/
└── SMD/
MADCluster/
├── RESULTS/
├── MADCluster_MAIN.py
├── MADCluster_MODEL.py
├── MADCluster_run_all_datasets.py
├── MADCluster_SOLVER.py
└── MADCluster_UTILS.py
README.md
```

---

## Quick Start

### 1. Set up the Conda environment
```python
conda create -n madcluster python=3.12
conda activate madcluster
```

### 2. Install required libraries
```python
pip install pandas numpy torch vus einops
```

### 3.1. Run with MADCluster
```python
python MADCluster_run_all_datasets.py --objective one-class --MADCluster
```

### 3.2. Run with base model only
```python
python MADCluster_run_all_datasets.py --objective one-class
```

---

## Performance Evaluation Example (MSL)

You can check the MSL dataset performance using the code below:

```python
import pandas as pd

# Replace with your actual result file path
results = pd.read_csv('./RESULTS/<your_result_file>.csv')

print('Results Summary:')
print(f'Precision: {results["precision"].mean():.5f},    Recall: {results["recall"].mean():.5f},     F1: {results["f1_score"].mean():.5f}')
print(f'AU-PR: {results["aupr"].mean():.5f},     AU-ROC: {results["roc_auc"].mean():.5f}')
print(f'R_AUC_ROC: {results["R_AUC_ROC"].mean():.5f},     R_AUC_PR: {results["R_AUC_PR"].mean():.5f}')
print(f'VUS_ROC: {results["VUS_ROC"].mean():.5f},     VUS_PR: {results["VUS_PR"].mean():.5f}')
print(f'Affiliation_Precision: {results["Affiliation_Precision"].mean():.5f},     Affiliation_Recall: {results["Affiliation_Recall"].mean():.5f}')

```

---

## Requirements

- Python 3.12
- pandas
- numpy
- torch
- vus
- einops

---

## Dataset Preparation

The following subdirectories are required under the `datasets/` folder.

- **NASA Datasets** — *Mars Science Laboratory (MSL)* and *Soil Moisture Active Passive (SMAP)*  
  Collected from NASA spacecraft, these datasets contain anomaly information based on incident reports for spacecraft monitoring systems.  
  📎 [Source](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

- **SMD (Server Machine Dataset)**  
  Gathered from 28 servers over 10 days, with normal activity observed during the first 5 days and anomalies injected sporadically in the last 5 days.  
  📎 [Source](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)

- **PSM (Pooled Server Metrics)**  
  Internally collected from multiple application server nodes at eBay with 26 monitored dimensions.  
  📎 [Source](https://github.com/eBay/RANSynCoders/tree/main/data)

---

## Supplementary Materials

Additional materials that extend the main paper are provided below:

- **Mathematical Proofs**  
  - Analysis of the One-directed Adaptive loss function.

- **Extended Experiments**  
  - Ablation studies (e.g. Multi-cluster ($k>1$) performance analysis and computational efficiency)
  - Image anomaly detection transferability (e.g., MVTec AD)

- 📄 [Download Supplementary Appendix (PDF)](./docs/MADCluster_Appendix.pdf)

These materials are referenced in the paper and are provided for transparency and reproducibility.
