"""
utils.py — SHAP Visualizations & Evaluation Helpers
House Price Prediction | Vimal Sahani
"""

import numpy as np
import matplotlib.pyplot as plt
import shap


def plot_shap_summary(model, X, feature_names=None, max_display=20, save_path=None):
    """Plot SHAP summary bar chart for feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names or X.columns.tolist(),
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SHAP plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Bar chart of top N feature importances (sklearn-style)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[indices], color="steelblue")
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def rmse_r2_report(y_true, y_pred, label="Model"):
    """Print RMSE and R² in a clean format."""
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*35}")
    print(f"  {label}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"{'─'*35}")
    return rmse, r2
