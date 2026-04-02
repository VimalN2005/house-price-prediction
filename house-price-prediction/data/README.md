# 📥 Dataset Instructions

## Kaggle Dataset: Ames Housing
**Competition:** House Prices — Advanced Regression Techniques
**Link:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

---

## Option A — Kaggle CLI (Recommended)
```bash
pip install kaggle

# Put your kaggle.json API key in ~/.kaggle/
# Get it from: https://www.kaggle.com/settings → API → Create New Token

kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d .
```

## Option B — Manual Download
1. Go to https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
2. Download `train.csv` and `test.csv`
3. Place both files inside this `data/` folder

## Expected Files
```
data/
├── train.csv    (1460 rows × 81 columns)
├── test.csv     (1459 rows × 80 columns)
└── README.md    ← you are here
```
