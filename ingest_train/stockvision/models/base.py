from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    description: str
    paper: str | None = None
    backend: str = "keras"


class BaseForecastModel(ABC):
    metadata: ModelMetadata

    @abstractmethod
    def build(self, window: int, num_features: int):
        """Return an uncompiled Keras model (keras backend only)."""

    def fit(self, df, *, horizon: int = 1):
        """Train a non-keras backend on raw OHLCV data and return a fitted estimator."""
        raise NotImplementedError("fit() is only required for non-keras backends")

    def serialise(self, fitted, path) -> None:
        """Persist a non-keras fitted estimator. Default: joblib."""
        import joblib

        joblib.dump(fitted, path)

    def predict_next(self, fitted, df) -> float:
        """Produce a one-step-ahead forecast for a fitted non-keras estimator."""
        raise NotImplementedError("predict_next() is only required for non-keras backends")
