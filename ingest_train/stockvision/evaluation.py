"""Accuracy metrics for a set of forecasts."""

import numpy as np


def compute_metrics(y_true, y_pred):
    """Return MAE, RMSE, MAPE and directional accuracy as a plain dict."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "directional_accuracy": 0.0, "n": 0}

    error = y_true - y_pred
    # Blank out zero targets so the percentage error never divides by zero.
    denominator = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)

    # How often we got the up/down move right, which matters more than the exact price.
    directional_accuracy = 0.0
    if y_true.size > 1:
        directional_accuracy = float(
            np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
        )

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.nanmean(np.abs(error / denominator)) * 100),
        "directional_accuracy": directional_accuracy,
        "n": int(y_true.size),
    }
