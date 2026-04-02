# 🏠 House Price Prediction — Stacked Ensemble Learning

> Predicts house prices using a stacked ensemble of Random Forest + XGBoost, achieving **88% R² score** on the Kaggle Ames Housing dataset.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Problem Statement
Predicting house sale prices based on 80+ features like location, size, quality, and amenities using an ensemble machine learning approach.

## 🧠 Approach
- **Stacked Ensemble**: RandomForest (base) + XGBoost (base) → Ridge Regression (meta-learner)
- **Feature Engineering**: 80+ features, missing value imputation, One-Hot encoding, log-transform on skewed target
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Explainability**: SHAP values for feature importance

## 📊 Results
| Metric | Score |
|--------|-------|
| R² Score | **0.88** |
| RMSE Reduction | **12% over baseline** |
| Cross-Val R² | 0.87 ± 0.02 |

---

## 📁 Project Structure
```
house-price-prediction/
├── data/
│   └── README.md           # Kaggle download instructions
├── src/
│   ├── preprocess.py       # Data cleaning & feature engineering
│   ├── train.py            # Model training & stacking
│   ├── predict.py          # Inference on new data
│   └── utils.py            # Helper functions (SHAP, metrics)
├── models/                 # Saved model artifacts (.pkl)
├── notebooks/
│   └── eda.ipynb           # Exploratory Data Analysis
├── main.py                 # Entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone & Install
```bash
git clone https://github.com/VimalN2005/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

### 2. Download Dataset (Kaggle)
```bash
# Option A — Kaggle CLI
pip install kaggle
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/

# Option B — Manual
# Go to: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
# Download train.csv and test.csv → put inside data/ folder
```

### 3. Train the Model
```bash
python main.py --mode train
```

### 4. Predict
```bash
python main.py --mode predict --input data/test.csv
```

---

## 📦 Requirements
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
shap==0.42.1
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.1
```

## 🔑 Key Features
- Handles 80+ raw features with automated preprocessing pipeline
- Log-transforms right-skewed target variable (SalePrice)
- SHAP-based model explainability plots
- Saves trained model with joblib for reuse

## 👤 Author
**Vimal Sahani** — IIIT Bhopal | [GitHub](https://github.com/VimalN2005) | [LinkedIn](https://linkedin.com/in/n-vimal-60b624379)
