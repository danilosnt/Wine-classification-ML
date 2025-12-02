# 🍷 Wine Cultivar Classification with LLM Integration

This project performs a comparative analysis between **Support Vector Machines (SVM)** and **Decision Tree** algorithms to classify wine cultivars based on chemical analysis.

Additionally, it features a **Generative AI integration (LLM)** that acts as a "Virtual Sommelier," interpreting the mathematical results into natural language descriptions.

## 📊 Project Overview
* **Goal:** Classify wines into three distinct cultivars (0, 1, or 2).
* **Dataset:** [UCI Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine) (178 samples, 13 chemical features).
* **Context:** Developed for the Artificial Intelligence course at UCSal (Universidade Católica do Salvador).

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `scikit-learn`, `pandas`, `seaborn`, `matplotlib`.
* **GenAI:** Google Gemini API (via `google-generativeai`).
* **Environment:** Google Colab.

## ⚙️ Methodology
1.  **Data Loading:** Ingestion of the raw dataset.
2.  **Feature Selection:** Application of **ANOVA (SelectKBest)** to identify the top 10 most relevant chemical features.
3.  **Preprocessing:** Data normalization using **StandardScaler** (Z-score normalization).
4.  **Modeling:** Training and testing (70/30 split) of SVM (Linear Kernel) and Decision Tree.

## 📈 Results

The **Decision Tree** model achieved superior performance for this specific problem.

| Model | Accuracy | F1-Score (Weighted) |
| :--- | :--- | :--- |
| **Decision Tree** | **98.14%** | **0.98** |
| SVM (Linear) | 96.29% | 0.96 |

## 🤖 AI Sommelier (LLM Integration)
Beyond classification, this project implements a **RAG-like approach** using Large Language Models.
* The system takes the chemical features (e.g., Alcohol, Flavanoids) and the predicted class.
* It feeds this data into an **LLM (Gemini Pro)** via prompt engineering.
* **Output:** A detailed, human-readable explanation of the wine's characteristics, simulating a real sommelier's analysis.

## 🚀 How to Run
1.  Open the notebook in **Google Colab**.
2.  (Optional) Add your Google API Key to enable the LLM features.
3.  Run all cells to execute the training pipeline and generate the graphs.

## 👥 Authors
* [Your Name]
* [Teammate Name]
* [Teammate Name]
* [Teammate Name]

---
*Project developed in November 2025.*
