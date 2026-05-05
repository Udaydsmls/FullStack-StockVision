from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .config import DataConfig
from .logging_setup import get_logger

log = get_logger(__name__)

_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class DataFetchError(RuntimeError):
    pass


class MarketDataSource(Protocol):
    def download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        ...


@dataclass
class YFinanceSource:
    auto_adjust: bool = True

    def download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=self.auto_adjust,
            group_by="column",
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df


def _validate_frame(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataFetchError(f"No data returned for ticker '{ticker}'")
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataFetchError(
            f"Data for '{ticker}' is missing columns: {missing}. Got: {list(df.columns)}"
        )
    df = df.dropna(subset=list(_REQUIRED_COLUMNS)).copy()
    if df.empty:
        raise DataFetchError(f"All rows for '{ticker}' contained NaNs after cleaning")
    return df


class MarketDataRepository:
    def __init__(
        self,
        config: DataConfig | None = None,
        source: MarketDataSource | None = None,
    ) -> None:
        self._config = config or DataConfig()
        self._source = source or YFinanceSource()
        self._config.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, ticker: str) -> Path:
        return self._config.cache_dir / f"{ticker.upper()}.csv"

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < self._config.cache_ttl_seconds

    def get(self, ticker: str, *, force_refresh: bool = False) -> pd.DataFrame:
        ticker = ticker.upper().strip()
        if not ticker:
            raise ValueError("Ticker symbol must not be empty")

        path = self.cache_path(ticker)
        if not force_refresh and self._is_fresh(path):
            log.debug("Cache hit for %s (path=%s)", ticker, path)
            return _validate_frame(ticker, pd.read_csv(path, parse_dates=["Date"]))

        log.info("Fetching %s (period=%s, interval=%s)", ticker, self._config.period, self._config.interval)
        df = self._source.download(ticker, self._config.period, self._config.interval)
        df = _validate_frame(ticker, df.reset_index())
        df.to_csv(path, index=False)
        log.info("Cached %d rows for %s -> %s", len(df), ticker, path)
        return df

    def load_local(self, ticker: str) -> pd.DataFrame:
        path = self.cache_path(ticker)
        if not path.exists():
            raise DataFetchError(f"No cached file for {ticker} at {path}")
        return _validate_frame(ticker, pd.read_csv(path, parse_dates=["Date"]))
