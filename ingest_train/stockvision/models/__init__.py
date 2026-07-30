"""Model plugins. Dropping a new module into this folder adds a new architecture."""

import importlib
import pkgutil

from .registry import MODELS, available_models, get_model, register

# Import every other module in this package so each one runs its register() call.
for _module in pkgutil.iter_modules(__path__):
    if _module.name != "registry":
        importlib.import_module(f"{__name__}.{_module.name}")

__all__ = ["MODELS", "available_models", "get_model", "register"]
