"""Technical indicators.

The C++ server rebuilds these same 17 columns in the same order, so a model
trained here can be served by either backend.
"""

import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def simple_moving_average(series, window):
    return series.rolling(window=window, min_periods=1).mean()


def exponential_moving_average(series, span):
    return series.ewm(span=span, adjust=False).mean()


def relative_strength_index(series, period=14):
    change = series.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    strength = average_gain / average_loss.replace(0, np.nan)
    # A flat series has no losses, which leaves NaN; 50 means "neither overbought nor oversold".
    return (100 - 100 / (1 + strength)).fillna(50.0)


def macd(series, fast=12, slow=26, signal=9):
    line = exponential_moving_average(series, fast) - exponential_moving_average(series, slow)
    signal_line = exponential_moving_average(line, signal)
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line}
    )


def bollinger_bands(series, window=20, num_std=2.0):
    middle = simple_moving_average(series, window)
    spread = num_std * series.rolling(window=window, min_periods=1).std().fillna(0.0)
    return pd.DataFrame(
        {"bb_upper": middle + spread, "bb_lower": middle - spread, "bb_width": 2 * spread}
    )


def log_returns(series):
    return np.log(series / series.shift(1)).fillna(0.0)


def build_feature_frame(df, ticker=None):
    """Turn raw OHLCV bars into the feature columns the models are trained on."""
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain a 'Close' column")

    frame = pd.DataFrame(index=df.index)
    frame["close"] = df["Close"].astype(float)
    frame["open"] = df["Open"].astype(float)
    frame["high"] = df["High"].astype(float)
    frame["low"] = df["Low"].astype(float)
    frame["volume"] = df["Volume"].astype(float)
    frame["log_return"] = log_returns(frame["close"])
    frame["sma_10"] = simple_moving_average(frame["close"], 10)
    frame["sma_30"] = simple_moving_average(frame["close"], 30)
    frame["ema_12"] = exponential_moving_average(frame["close"], 12)
    frame["ema_26"] = exponential_moving_average(frame["close"], 26)
    frame["rsi_14"] = relative_strength_index(frame["close"], 14)
    frame = pd.concat([frame, macd(frame["close"]), bollinger_bands(frame["close"])], axis=1)
    frame = frame.bfill().ffill().fillna(0.0)

    if config.USE_FEAST and ticker:
        frame = _overlay_feast_row(frame, ticker)
    return frame


# Feast stores the same indicators under its own names.
_FEAST_COLUMNS = {
    "RSI_14": "rsi_14",
    "MACD_12_26": "macd",
    "BB_upper_20": "bb_upper",
    "BB_lower_20": "bb_lower",
    "SMA_5": "sma_10",
    "SMA_20": "sma_30",
}


def _overlay_feast_row(frame, ticker):
    """Replace the newest indicator row with the one served by the Feast online store."""
    try:
        from feature_store.feature_client import get_features

        row = get_features(ticker).iloc[0]
    except Exception as exc:
        log.warning("Feast lookup failed for %s, using inline features: %s", ticker, exc)
        return frame

    for feast_name, column in _FEAST_COLUMNS.items():
        if feast_name in row:
            frame.loc[frame.index[-1], column] = float(row[feast_name])
    return frame
