from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    description: str
    paper: str | None = None


class BaseForecastModel(ABC):
    metadata: ModelMetadata

    @abstractmethod
    def build(self, window: int, num_features: int):
        """Return an uncompiled Keras model."""
