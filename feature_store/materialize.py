"""Compute indicators from the cached OHLCV CSVs and load them into Feast."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def relative_strength_index(close, period=14):
    change = close.diff()
    average_gain = change.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
    average_loss = (-change.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean()
    strength = average_gain / average_loss.replace(0, np.nan)
    return (100 - 100 / (1 + strength)).fillna(50.0)


def build_indicators(csv_path):
    """One row per trading day, tagged with the ticker Feast uses as its entity."""
    prices = pd.read_csv(csv_path, parse_dates=["Date"])
    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float)

    sma_20 = close.rolling(20, min_periods=1).mean()
    deviation = close.rolling(20, min_periods=1).std().fillna(0.0)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    return pd.DataFrame(
        {
            "ticker": csv_path.stem.upper(),
            "event_timestamp": prices["Date"],
            "RSI_14": relative_strength_index(close),
            "MACD_12_26": ema_12 - ema_26,
            "BB_upper_20": sma_20 + 2 * deviation,
            "BB_lower_20": sma_20 - 2 * deviation,
            "SMA_5": close.rolling(5, min_periods=1).mean(),
            "SMA_20": sma_20,
            "volume_ratio": volume / volume.rolling(20, min_periods=1).mean(),
        }
    ).astype({name: "float32" for name in ["RSI_14", "MACD_12_26", "BB_upper_20",
                                           "BB_lower_20", "SMA_5", "SMA_20", "volume_ratio"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../ingest_train/data"))
    parser.add_argument("--repo", type=Path, default=Path("feature_repo"))
    args = parser.parse_args()

    from feast import FeatureStore

    csv_paths = sorted(args.data_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSVs found in {args.data_dir}. Run `stockvision fetch` first.")

    indicators = pd.concat([build_indicators(path) for path in csv_paths], ignore_index=True)
    (args.repo / "data").mkdir(parents=True, exist_ok=True)
    indicators.to_parquet(args.repo / "data" / "stock_indicators.parquet", index=False)

    store = FeatureStore(repo_path=str(args.repo))
    store.materialize_incremental(end_date=datetime.now(timezone.utc))
    print(f"Materialised {len(indicators)} rows for {len(csv_paths)} tickers")


if __name__ == "__main__":
    main()
