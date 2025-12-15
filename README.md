# Wine Cultivar Classification with LLM Integration

This project performs a comparative analysis between **Support Vector Machines (SVM)** and **Decision Tree** algorithms to classify wine cultivars based on chemical analysis.

## Project Overview
* **Goal:** Classify wines into three distinct cultivars (0, 1, or 2).
* **Dataset:** [UCI Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine) (178 samples, 13 chemical features).

## Tech Stack
* **Language:** Python
* **Libraries:** `scikit-learn`, `pandas`, `seaborn`, `matplotlib`.
* **Environment:** Google Colab.

## Methodology
1.  **Data Loading:** Ingestion of the raw dataset.
2.  **Feature Selection:** Application of **ANOVA (SelectKBest)** to identify the top 10 most relevant chemical features.
3.  **Preprocessing:** Data normalization using **StandardScaler** (Z-score normalization).
4.  **Modeling:** Training and testing (70/30 split) of SVM (Linear Kernel) and Decision Tree.

## Results

The **Decision Tree** model achieved superior performance for this specific problem.

| Model | Accuracy | F1-Score (Weighted) |
| :--- | :--- | :--- |
| **Decision Tree** | **98.14%** | **0.98** |
| SVM (Linear) | 96.29% | 0.96 |
