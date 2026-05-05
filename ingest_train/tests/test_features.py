import numpy as np
import pandas as pd

from stockvision.features import (
    bollinger_bands,
    build_feature_frame,
    exponential_moving_average,
    relative_strength_index,
    simple_moving_average,
)


def _ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.1, n),
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1_000, 10_000, n),
        }
    )


def test_sma_matches_rolling_mean():
    s = pd.Series(np.arange(10, dtype=float))
    out = simple_moving_average(s, 3)
    assert out.iloc[2] == (0 + 1 + 2) / 3
    assert len(out) == len(s)


def test_ema_decreases_lag_with_span():
    s = pd.Series(np.arange(20, dtype=float))
    fast = exponential_moving_average(s, 3).iloc[-1]
    slow = exponential_moving_average(s, 12).iloc[-1]
    assert fast > slow


def test_rsi_in_range():
    rsi = relative_strength_index(pd.Series(np.linspace(1, 100, 50)))
    assert rsi.between(0, 100).all()


def test_bollinger_columns():
    bb = bollinger_bands(pd.Series(np.arange(50, dtype=float)))
    assert {"bb_upper", "bb_lower", "bb_width"} <= set(bb.columns)
    assert (bb["bb_upper"] >= bb["bb_lower"]).all()


def test_feature_frame_no_nans():
    frame = build_feature_frame(_ohlcv())
    assert not frame.isna().any().any()
    assert "rsi_14" in frame.columns
    assert "macd" in frame.columns
