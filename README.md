# MADCluster: Model-Agnostic Anomaly Detection with Self-supervised Clustering Network

## Project Structure

```
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
conda create -n madcluster python=3.12
conda activate madcluster

### 2. Install required libraries
pip install pandas numpy torch vus einops

### 3.1. Run with MADCluster
python MADCluster_run_all_datasets.py --objective one-class --MADCluster

### 3.2. Run with base model only
python MADCluster_run_all_datasets.py --objective one-class

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

The following subdirectories are required under the `datasets/` folder:
- MSL
- SMAP
- SMD
- PSM

---
