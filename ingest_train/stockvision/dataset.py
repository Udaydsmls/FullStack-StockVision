from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_names: tuple[str, ...]
    target_name: str
    window: int
    horizon: int

    @property
    def num_features(self) -> int:
        return self.X_train.shape[-1]


def _make_windows(
    features: np.ndarray, target: np.ndarray, window: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(features) <= window + horizon - 1:
        raise ValueError(
            f"Not enough rows ({len(features)}) for window={window}, horizon={horizon}"
        )
    n = len(features) - window - horizon + 1
    X = np.stack([features[i : i + window] for i in range(n)], axis=0)
    y = np.array([target[i + window + horizon - 1] for i in range(n)])
    return X.astype(np.float32), y.astype(np.float32)


def build_dataset(
    feature_frame: pd.DataFrame,
    *,
    target_column: str = "close",
    window: int = 30,
    horizon: int = 1,
    val_split: float = 0.15,
    test_split: float = 0.10,
) -> WindowedDataset:
    if target_column not in feature_frame.columns:
        raise KeyError(f"Target column '{target_column}' not in feature frame")
    if val_split < 0 or test_split < 0 or val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be in [0, 1)")

    feature_names = tuple(feature_frame.columns)
    target_idx = feature_names.index(target_column)

    raw = feature_frame.values.astype(np.float32)
    n = len(raw)
    n_test = max(1, int(n * test_split))
    n_val = max(1, int(n * val_split))
    n_train = n - n_val - n_test
    if n_train <= window + horizon:
        raise ValueError(
            f"Training segment too small (n_train={n_train}) for window={window}, horizon={horizon}"
        )

    train_raw = raw[:n_train]
    val_raw = raw[n_train - window : n_train + n_val]
    test_raw = raw[n_train + n_val - window :]

    feature_scaler = StandardScaler().fit(train_raw)
    target_scaler = StandardScaler().fit(train_raw[:, target_idx : target_idx + 1])

    def _scale(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = feature_scaler.transform(block)
        target = target_scaler.transform(block[:, target_idx : target_idx + 1]).ravel()
        return _make_windows(scaled, target, window, horizon)

    X_train, y_train = _scale(train_raw)
    X_val, y_val = _scale(val_raw)
    X_test, y_test = _scale(test_raw)

    return WindowedDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=feature_names,
        target_name=target_column,
        window=window,
        horizon=horizon,
    )


def transform_window(
    feature_frame: pd.DataFrame,
    feature_scaler: StandardScaler,
    window: int,
) -> np.ndarray:
    if len(feature_frame) < window:
        raise ValueError(f"Need at least {window} rows, got {len(feature_frame)}")
    block = feature_frame.values[-window:].astype(np.float32)
    scaled = feature_scaler.transform(block).astype(np.float32)
    return scaled[None, :, :]


def inverse_transform_target(value: float, target_scaler: StandardScaler) -> float:
    arr = np.asarray([[value]], dtype=np.float32)
    return float(target_scaler.inverse_transform(arr)[0, 0])
