import numpy as np
import pandas as pd
import pytest

from stockvision.dataset import build_dataset, transform_window


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "close": close,
            "open": close + rng.normal(0, 0.1, n),
            "high": close + 1,
            "low": close - 1,
            "volume": rng.integers(1_000, 10_000, n),
        }
    )


def test_build_dataset_shapes():
    ds = build_dataset(_frame(), window=20, horizon=1, val_split=0.15, test_split=0.10)
    assert ds.window == 20
    assert ds.X_train.shape[1:] == (20, ds.num_features)
    assert ds.y_train.ndim == 1
    assert ds.X_val.shape[0] > 0
    assert ds.X_test.shape[0] > 0


def test_build_dataset_rejects_too_few_rows():
    with pytest.raises(ValueError):
        build_dataset(_frame(40), window=30, horizon=1)


def test_transform_window_shape_matches_dataset():
    ds = build_dataset(_frame(), window=20, horizon=1)
    arr = transform_window(_frame()[list(ds.feature_names)], ds.feature_scaler, ds.window)
    assert arr.shape == (1, 20, ds.num_features)
