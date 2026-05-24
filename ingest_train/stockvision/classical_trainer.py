from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import AppConfig, load_config
from .data import MarketDataRepository
from .evaluation import ForecastMetrics, compute_metrics
from .logging_setup import get_logger
from .models import get_model

log = get_logger(__name__)


@dataclass(frozen=True)
class ClassicalResult:
    ticker: str
    model_name: str
    model_path: Path
    metadata_path: Path
    metrics: ForecastMetrics


def _eval_naive(model, fitted, df) -> ForecastMetrics:
    """Walk the last few rows and compare one-step forecasts against the truth."""
    if len(df) < 6:
        return ForecastMetrics(0.0, 0.0, 0.0, 0.0, 0)
    look_back = min(5, len(df) - 1)
    truths = df["Close"].astype(float).values[-look_back:]
    last_pred = float(model.predict_next(fitted, df))
    preds = np.full_like(truths, fill_value=last_pred, dtype=float)
    return compute_metrics(truths, preds)


def train_classical(
    ticker: str,
    model_name: str,
    config: AppConfig | None = None,
    *,
    repository: MarketDataRepository | None = None,
    force_refresh: bool = False,
) -> ClassicalResult:
    cfg = config or load_config()
    repo = repository or MarketDataRepository(cfg.data)
    model = get_model(model_name)
    backend = model.metadata.backend
    if backend == "keras":
        raise ValueError(f"Model '{model_name}' uses the keras backend; use train_and_export().")

    log.info("Training %s (%s backend) on %s", model_name, backend, ticker)
    df = repo.get(ticker, force_refresh=force_refresh)
    fitted = model.fit(df, horizon=cfg.training.horizon)

    model_dir = cfg.artifacts.model_dir(ticker, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    model.serialise(fitted, model_path)

    metrics = _eval_naive(model, fitted, df)
    metadata_path = cfg.artifacts.metadata_path(ticker, model_name)
    metadata_path.write_text(
        json.dumps(
            {
                "ticker": ticker.upper(),
                "model": model_name,
                "backend": backend,
                "metrics": metrics.to_dict(),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    log.info("Saved %s artefacts to %s", model_name, model_dir)
    return ClassicalResult(
        ticker=ticker.upper(),
        model_name=model_name,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics=metrics,
    )
