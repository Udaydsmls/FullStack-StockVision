"""Read the latest indicator row for a ticker out of the Feast online store."""

from pathlib import Path

import pandas as pd

REPO_PATH = Path(__file__).resolve().parent / "feature_repo"

FEATURES = [
    "RSI_14",
    "MACD_12_26",
    "BB_upper_20",
    "BB_lower_20",
    "SMA_5",
    "SMA_20",
    "volume_ratio",
]


def get_features(ticker):
    """Return a one-row DataFrame of the newest features Feast has for `ticker`."""
    from feast import FeatureStore

    store = FeatureStore(repo_path=str(REPO_PATH))
    response = store.get_online_features(
        entity_rows=[{"ticker": ticker.upper()}],
        features=[f"stock_indicators_fv:{name}" for name in FEATURES],
    )

    frame = pd.DataFrame(response.to_dict())
    # A ticker that was never materialised comes back with nulls.
    return frame.reindex(columns=FEATURES).fillna(0.0)
