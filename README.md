# MADCluster: Model-Agnostic Anomaly Detection with Self-supervised Clustering Network

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
