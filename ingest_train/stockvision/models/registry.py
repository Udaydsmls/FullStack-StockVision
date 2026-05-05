from __future__ import annotations

from typing import Callable, Dict, List

from .base import BaseForecastModel

_REGISTRY: Dict[str, Callable[[], BaseForecastModel]] = {}


def register_model(name: str) -> Callable[[type[BaseForecastModel]], type[BaseForecastModel]]:
    key = name.lower().strip()

    def decorator(cls: type[BaseForecastModel]) -> type[BaseForecastModel]:
        if key in _REGISTRY:
            raise ValueError(f"Model already registered: {key}")
        _REGISTRY[key] = cls  # type: ignore[assignment]
        return cls

    return decorator


def get_model(name: str) -> BaseForecastModel:
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]()


def available_models() -> List[str]:
    return sorted(_REGISTRY.keys())
