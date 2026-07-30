"""Turn a feature frame into scaled sliding windows for supervised training."""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class Dataset:
    """Train/validation/test windows plus the scalers used to build them."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_names: list
    window: int
    horizon: int

    @property
    def num_features(self):
        return len(self.feature_names)


def make_windows(features, target, window, horizon):
    """Slice a 2-D array into inputs of shape (n, window, num_features) and their targets."""
    count = len(features) - window - horizon + 1
    if count <= 0:
        raise ValueError(
            f"Need more than {window + horizon - 1} rows for window={window}, got {len(features)}"
        )
    X = np.stack([features[i : i + window] for i in range(count)])
    y = np.array([target[i + window + horizon - 1] for i in range(count)])
    return X.astype(np.float32), y.astype(np.float32)


def build_dataset(
    frame,
    target_column="close",
    window=config.WINDOW,
    horizon=config.HORIZON,
    val_split=config.VAL_SPLIT,
    test_split=config.TEST_SPLIT,
):
    """Split chronologically, fit the scalers on the training rows only, then window."""
    if target_column not in frame.columns:
        raise KeyError(f"Target column '{target_column}' is not in the feature frame")

    feature_names = list(frame.columns)
    target_index = feature_names.index(target_column)
    values = frame.values.astype(np.float32)

    total = len(values)
    n_test = max(1, int(total * test_split))
    n_val = max(1, int(total * val_split))
    n_train = total - n_val - n_test
    if n_train <= window + horizon:
        raise ValueError(
            f"Not enough rows: {total} total leaves {n_train} for training, "
            f"which is too few for window={window}"
        )

    # The val/test blocks start `window` rows early so their first window is complete.
    train = values[:n_train]
    val = values[n_train - window : n_train + n_val]
    test = values[n_train + n_val - window :]

    feature_scaler = StandardScaler().fit(train)
    target_scaler = StandardScaler().fit(train[:, [target_index]])

    def scale_and_window(block):
        scaled = feature_scaler.transform(block)
        target = target_scaler.transform(block[:, [target_index]]).ravel()
        return make_windows(scaled, target, window, horizon)

    X_train, y_train = scale_and_window(train)
    X_val, y_val = scale_and_window(val)
    X_test, y_test = scale_and_window(test)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=feature_names,
        window=window,
        horizon=horizon,
    )


def transform_window(frame, feature_scaler, window):
    """Scale the most recent `window` rows into the (1, window, features) tensor a model wants."""
    if len(frame) < window:
        raise ValueError(f"Need at least {window} rows, got {len(frame)}")
    scaled = feature_scaler.transform(frame.values[-window:])
    return scaled.astype(np.float32)[None, :, :]


def inverse_transform_target(value, target_scaler):
    """Convert a scaled model output back into a dollar price."""
    return float(target_scaler.inverse_transform([[value]])[0][0])
