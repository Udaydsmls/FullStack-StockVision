from __future__ import annotations

import numpy as np
import pandas as pd


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def relative_strength_index(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = exponential_moving_average(series, fast)
    ema_slow = exponential_moving_average(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    sma = simple_moving_average(series, window)
    std = series.rolling(window=window, min_periods=1).std().fillna(0.0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower, "bb_width": upper - lower})


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)).fillna(0.0)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain a 'Close' column")

    out = pd.DataFrame(index=df.index)
    out["close"] = df["Close"].astype(float)
    out["open"] = df["Open"].astype(float)
    out["high"] = df["High"].astype(float)
    out["low"] = df["Low"].astype(float)
    out["volume"] = df["Volume"].astype(float)
    out["log_return"] = log_returns(out["close"])
    out["sma_10"] = simple_moving_average(out["close"], 10)
    out["sma_30"] = simple_moving_average(out["close"], 30)
    out["ema_12"] = exponential_moving_average(out["close"], 12)
    out["ema_26"] = exponential_moving_average(out["close"], 26)
    out["rsi_14"] = relative_strength_index(out["close"], 14)
    out = pd.concat([out, macd(out["close"]), bollinger_bands(out["close"])], axis=1)
    return out.bfill().ffill().fillna(0.0)
