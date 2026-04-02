"""
preprocess.py — Data Cleaning & Feature Engineering Pipeline
House Price Prediction | Vimal Sahani
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# ── Columns with meaningful NA (not missing, means "None") ─────────────────
NONE_FILL_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

ZERO_FILL_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# Ordinal quality mapping
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}


def load_data(train_path: str, test_path: str):
    """Load train and test CSVs."""
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")
    return train, test


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in training data (Ames dataset specific)."""
    df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 200000))]
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new meaningful features."""
    df = df.copy()

    # Total area
    df["TotalSF"]        = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"]   = (df["OpenPorchSF"] + df["3SsnPorch"]
                            + df["EnclosedPorch"] + df["ScreenPorch"]
                            + df["WoodDeckSF"])

    # Age-based features
    df["HouseAge"]       = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"]       = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"]    = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    # Binary flags
    df["HasPool"]        = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"]      = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"]    = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"]   = (df["Fireplaces"] > 0).astype(int)

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill all missing values appropriately."""
    df = df.copy()

    for col in NONE_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: fill with neighborhood median
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    # Remaining categoricals → mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remaining numerics → median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encode quality cols, One-Hot encode rest."""
    df = df.copy()

    # Ordinal quality columns
    quality_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                    "HeatingQC", "KitchenQual", "FireplaceQu",
                    "GarageQual", "GarageCond", "PoolQC"]
    for col in quality_cols:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)

    # One-Hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def log_transform_target(y: pd.Series) -> pd.Series:
    """Log-transform SalePrice to reduce right skew."""
    return np.log1p(y)


def inverse_transform_target(y_pred: np.ndarray) -> np.ndarray:
    """Reverse log transform for final predictions."""
    return np.expm1(y_pred)


def build_pipeline(train_path: str, test_path: str):
    """Full preprocessing pipeline. Returns X_train, y_train, X_test, test_ids."""
    train, test = load_data(train_path, test_path)

    # Save test IDs for submission
    test_ids = test["Id"]

    # Remove outliers from train only
    train = drop_outliers(train)

    # Separate target
    y_train = log_transform_target(train["SalePrice"])
    train.drop(["Id", "SalePrice"], axis=1, inplace=True)
    test.drop(["Id"], axis=1, inplace=True)

    # Combine for consistent encoding
    n_train = len(train)
    all_data = pd.concat([train, test], axis=0, ignore_index=True)

    all_data = engineer_features(all_data)
    all_data = handle_missing(all_data)
    all_data = encode_features(all_data)

    X_train = all_data.iloc[:n_train]
    X_test  = all_data.iloc[n_train:]

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, y_train, X_test, test_ids
