"""Compute indicators from cached OHLCV CSVs and materialise them into Feast."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ingest = repo_root / "ingest_train"
    if str(ingest) not in sys.path:
        sys.path.insert(0, str(ingest))


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _build_indicator_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    sma5 = close.rolling(5, min_periods=1).mean()
    sma20 = close.rolling(20, min_periods=1).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0)
    avg_volume = volume.rolling(20, min_periods=1).mean()

    out = pd.DataFrame(
        {
            "ticker": csv_path.stem.upper(),
            "event_timestamp": df["Date"],
            "RSI_14": _rsi(close).astype("float32"),
            "MACD_12_26": (ema12 - ema26).astype("float32"),
            "BB_upper_20": (sma20 + 2 * std20).astype("float32"),
            "BB_lower_20": (sma20 - 2 * std20).astype("float32"),
            "SMA_5": sma5.astype("float32"),
            "SMA_20": sma20.astype("float32"),
            "volume_ratio": (volume / avg_volume).astype("float32"),
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../ingest_train/data"))
    parser.add_argument("--repo", type=Path, default=Path("feature_repo"))
    args = parser.parse_args(argv)

    _ensure_path()
    from feast import FeatureStore

    csvs = sorted(args.data_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found under {args.data_dir}", file=sys.stderr)
        return 1
    frames = [_build_indicator_frame(p) for p in csvs]
    combined = pd.concat(frames, ignore_index=True)

    repo_data = args.repo / "data"
    repo_data.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(repo_data / "stock_indicators.parquet", index=False)

    store = FeatureStore(repo_path=str(args.repo))
    store.apply([])
    store.materialize_incremental(end_date=datetime.now(timezone.utc))
    print(f"Materialised {len(combined)} rows for {len(csvs)} tickers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
