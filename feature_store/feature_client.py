"""Client wrapper around Feast for online feature retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

_DEFAULT_REPO = Path(__file__).resolve().parent / "feature_repo"

_FEATURES: tuple[str, ...] = (
    "stock_indicators_fv:RSI_14",
    "stock_indicators_fv:MACD_12_26",
    "stock_indicators_fv:BB_upper_20",
    "stock_indicators_fv:BB_lower_20",
    "stock_indicators_fv:SMA_5",
    "stock_indicators_fv:SMA_20",
    "stock_indicators_fv:volume_ratio",
)

_OUTPUT_COLUMNS = (
    "RSI_14", "MACD_12_26", "BB_upper_20", "BB_lower_20",
    "SMA_5", "SMA_20", "volume_ratio",
)


def get_features(
    ticker: str,
    date_range: Iterable[pd.Timestamp] | None = None,
    *,
    repo_path: Path = _DEFAULT_REPO,
) -> pd.DataFrame:
    """Return a DataFrame of indicator features for a ticker.

    Online retrieval returns one row per entity. ``date_range`` is accepted
    for API symmetry with the inline computation; only the most recent
    feature row is returned by the online store.
    """

    try:
        from feast import FeatureStore
    except ImportError as exc:
        raise RuntimeError("feast is not installed: pip install feast") from exc

    store = FeatureStore(repo_path=str(repo_path))
    rows = store.get_online_features(
        entity_rows=[{"ticker": ticker.upper()}],
        features=list(_FEATURES),
    ).to_dict()
    frame = pd.DataFrame(rows)
    for col in _OUTPUT_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[list(_OUTPUT_COLUMNS)]
