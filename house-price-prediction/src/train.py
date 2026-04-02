"""
train.py — Stacked Ensemble Model Training
House Price Prediction | Vimal Sahani
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

MODELS_DIR = "models"


def get_base_models():
    """Define base learners for stacking."""
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbosity=0,
    )

    lasso = Lasso(alpha=0.0005, random_state=42, max_iter=10000)

    return rf, xgb, lasso


def build_stacked_model():
    """Build stacking ensemble with Ridge as meta-learner."""
    rf, xgb, lasso = get_base_models()

    stacked = StackingRegressor(
        estimators=[
            ("random_forest", rf),
            ("lasso", lasso),
        ],
        final_estimator=Ridge(alpha=10.0),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )
    return stacked, xgb


def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def cross_validate_model(model, X, y, cv=5):
    """Run k-fold cross validation."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y,
        scoring="neg_root_mean_squared_error",
        cv=kf, n_jobs=-1
    )
    rmse_scores = -scores
    print(f"  CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    r2_scores = cross_val_score(model, X, y, scoring="r2", cv=kf, n_jobs=-1)
    print(f"  CV R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    return rmse_scores.mean(), r2_scores.mean()


def train_xgb_with_eval(xgb_model, X_train, y_train, X_val, y_val):
    """Train XGBoost with eval set for early stopping."""
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return xgb_model


def blend_predictions(pred1, pred2, w1=0.5, w2=0.5):
    """Weighted blend of two prediction arrays."""
    return w1 * pred1 + w2 * pred2


def train(X_train, y_train):
    """
    Main training function.
    Trains stacked ensemble + XGBoost, blends outputs.
    Returns saved model paths.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n[1/3] Training Stacked Ensemble (RF + Lasso → Ridge)...")
    stacked_model, xgb_model = build_stacked_model()

    print("  Cross-validating Stacked model...")
    cross_validate_model(stacked_model, X_train, y_train)

    stacked_model.fit(X_train, y_train)
    joblib.dump(stacked_model, f"{MODELS_DIR}/stacked_model.pkl")
    print("  ✓ Stacked model saved.")

    print("\n[2/3] Training XGBoost...")
    # Use last 20% as validation for early stopping
    split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    xgb_model = train_xgb_with_eval(xgb_model, X_tr, y_tr, X_val, y_val)
    joblib.dump(xgb_model, f"{MODELS_DIR}/xgb_model.pkl")
    print("  ✓ XGBoost model saved.")

    print("\n[3/3] Computing blend weights on validation split...")
    stacked_val_pred = stacked_model.predict(X_val)
    xgb_val_pred     = xgb_model.predict(X_val)

    # Try blends: 60/40 and 50/50
    blend_60 = blend_predictions(stacked_val_pred, xgb_val_pred, 0.6, 0.4)
    blend_50 = blend_predictions(stacked_val_pred, xgb_val_pred, 0.5, 0.5)

    rmse_60 = rmsle(y_val, blend_60)
    rmse_50 = rmsle(y_val, blend_50)
    print(f"  Blend 60/40 RMSE: {rmse_60:.4f}")
    print(f"  Blend 50/50 RMSE: {rmse_50:.4f}")

    best_weights = (0.6, 0.4) if rmse_60 < rmse_50 else (0.5, 0.5)
    joblib.dump(best_weights, f"{MODELS_DIR}/blend_weights.pkl")
    print(f"\n  ✓ Best blend weights: Stacked={best_weights[0]}, XGB={best_weights[1]}")
    print("\n✅ Training complete! Models saved in /models")

    return stacked_model, xgb_model, best_weights
