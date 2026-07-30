"""Download daily price bars from yfinance and cache them as CSV."""

import logging
import time

import pandas as pd

from . import config

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataFetchError(RuntimeError):
    """Raised when a ticker has no usable price data."""


def cache_path(ticker):
    return config.DATA_DIR / f"{ticker.upper()}.csv"


def get_prices(ticker, force_refresh=False):
    """Return a DataFrame of daily bars, downloading only when the cache is stale."""
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker symbol must not be empty")

    path = cache_path(ticker)
    if not force_refresh and _is_fresh(path):
        return _clean(ticker, pd.read_csv(path, parse_dates=["Date"]))

    import yfinance as yf

    log.info("Downloading %s from yfinance", ticker)
    df = yf.download(
        ticker,
        period=config.PERIOD,
        interval=config.INTERVAL,
        progress=False,
        auto_adjust=True,
    )
    # yfinance returns a two-level column index when it downloads more than one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = _clean(ticker, df.reset_index())
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Cached %d rows for %s", len(df), ticker)
    return df


def _is_fresh(path):
    if not path.exists():
        return False
    return time.time() - path.stat().st_mtime < config.CACHE_TTL_SECONDS


def _clean(ticker, df):
    """Check the frame has the columns we need and drop rows with gaps."""
    if df is None or df.empty:
        raise DataFetchError(f"No data returned for ticker '{ticker}'")

    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise DataFetchError(f"Data for '{ticker}' is missing columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        raise DataFetchError(f"Every row returned for '{ticker}' was incomplete")
    return df
