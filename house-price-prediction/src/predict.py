"""
predict.py — Inference Script
House Price Prediction | Vimal Sahani
"""

import numpy as np
import pandas as pd
import joblib
from src.preprocess import inverse_transform_target


def load_models(models_dir="models"):
    stacked = joblib.load(f"{models_dir}/stacked_model.pkl")
    xgb     = joblib.load(f"{models_dir}/xgb_model.pkl")
    weights = joblib.load(f"{models_dir}/blend_weights.pkl")
    return stacked, xgb, weights


def predict(X_test, test_ids, output_path="submission.csv"):
    """Generate predictions and save submission CSV."""
    stacked, xgb, (w1, w2) = load_models()

    stacked_pred = stacked.predict(X_test)
    xgb_pred     = xgb.predict(X_test)

    blended = w1 * stacked_pred + w2 * xgb_pred
    final_prices = inverse_transform_target(blended)

    submission = pd.DataFrame({
        "Id":        test_ids,
        "SalePrice": final_prices,
    })
    submission.to_csv(output_path, index=False)
    print(f"✅ Submission saved → {output_path}")
    print(f"   Predictions range: ${final_prices.min():,.0f} – ${final_prices.max():,.0f}")
    return submission
