from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ForecastMetrics:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    n = y_true.size
    if n == 0:
        return ForecastMetrics(0.0, 0.0, 0.0, 0.0, 0)

    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    mape = float(np.nanmean(np.abs(error / denom)) * 100)

    if n > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        directional = float(np.mean(true_dir == pred_dir))
    else:
        directional = 0.0

    return ForecastMetrics(mae=mae, rmse=rmse, mape=mape, directional_accuracy=directional, n=n)
