# Wine Chemical Classification with Machine Learning

This project focuses on the chemical analysis and classification of wines using Machine Learning. The main goal is to compare the performance of two popular algorithms: **SVM (Support Vector Machine)** and **Decision Trees**.

## Project Overview
Using the UCI Wine Dataset, this project performs:
1.  **Exploratory Data Analysis (EDA)**: Understanding chemical distributions, means, and checking for missing values.
2.  **Feature Selection**: Using `SelectKBest` to identify the top 10 chemical predictors.
3.  **Data Standardization**: Scaling features to ensure model stability.
4.  **Model Training & Evaluation**: Training classifiers and evaluating them using Confusion Matrices and Classification Reports (Precision, Recall, F1-Score).

The repository is organized to demonstrate the transition from experimental code to production-ready scripts:

* **`/notebooks`**: Contains the original `.ipynb` file created in Google Colab. This is best for interactive visualization and step-by-step documentation.
* **`/scripts`**: Features the **standalone Python versions** of the code. These scripts demonstrate how to modularize the notebook's logic into clean `.py` files for automation and terminal execution (Mac, Windows, or Linux).

## Results
The models are evaluated based on their accuracy in identifying three different wine cultivars. Visualization through Heatmaps (Confusion Matrices) allows for a clear understanding of where each model succeeds or fails.
