from .base import BaseForecastModel, ModelMetadata
from .registry import available_models, get_model, register_model

# Side-effect imports: each module registers itself via @register_model.
from . import lstm  # noqa: F401
from . import bilstm  # noqa: F401
from . import gru  # noqa: F401
from . import cnn_lstm  # noqa: F401
from . import transformer  # noqa: F401
from . import tcn  # noqa: F401
from . import linear  # noqa: F401
from . import prophet_model  # noqa: F401
from . import statsforecast_model  # noqa: F401

__all__ = [
    "BaseForecastModel",
    "ModelMetadata",
    "available_models",
    "get_model",
    "register_model",
]
