# Satellite Imagery-Based Property Valuation

## 📌 Project Overview
This project implements a **Multimodal AI System** to predict real estate prices by fusing traditional housing metrics with environmental visual contexts. Unlike traditional models that rely solely on tabular data (bedrooms, sqft, etc.), this system integrates **Satellite Imagery** to capture visual "curb appeal" features such as neighborhood density, greenery, and proximity to water.

**Key Tech Stack:** Python, Sentinel Hub API, VGG16 (CNN), XGBoost, Grad-CAM.

## 📂 Repository Structure
* `data_fetcher.ipynb`: Authentication with Sentinel Hub and batch downloading of ~21,000 satellite images (L1C data).
* `preprocessing.ipynb`: Data cleaning, Exploratory Data Analysis (EDA), log-transformation of targets, and evaluating the model based on tabular data.
* `model_training.ipynb`: Visual feature extraction using VGG16, Training the Hybrid XGBoost model, evaluation metrics, and Grad-CAM visualization.
* `23116071_final.csv`: Final predictions for the test dataset.

## 🚀 Methodology
1.  **Data Acquisition:** Programmatically fetched 1km² bounding-box images for every property coordinate using Sentinel-2 satellite data.
2.  **Visual Embedding:** Utilized a pre-trained **VGG16** network to convert images into high-dimensional feature vectors (512-d).
3.  **Early Fusion:** Concatenated visual embeddings with structured features (sqft, bedrooms, year built).
4.  **Regression:** Trained a Gradient Boosting Regressor (XGBoost) on the fused dataset.

## 📊 Results Summary
| Model | RMSE ($) | R² Score |
|-------|----------|----------|
| **Baseline (Linear)** | 176,503 | 0.751 |
| **Tabular XGBoost** | 116,338 | 0.892 |
| **Multimodal Fusion** | 132,645 | 0.872 |

*Conclusion:* While tabular data remains the dominant predictor, the multimodal approach successfully integrated visual context with high accuracy (R² > 0.87), demonstrating the viability of satellite data in automated valuation models (AVM).

## 🔧 Setup & Usage
1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/Palakfs/Multimodal-Real-Estate-Price-Prediction.git
    ```
2.  **Install Requirements:**
    ```bash
    pip install pandas numpy xgboost tensorflow sentinelhub opencv-python matplotlib
    ```
3.  **Run Pipeline:**
    * Execute `data_fetcher.py` to download images (requires API keys).
    * Run `preprocessing.ipynb` to train on tabular data.
    * Run `model_training.ipynb` to train and predict on merged dataset.
