"""Serve one-step forecasts from the artifacts that training wrote."""

import json
import logging

from . import config
from .data import get_prices
from .dataset import inverse_transform_target, transform_window
from .features import build_feature_frame
from .models import get_model

log = logging.getLogger(__name__)

# (TICKER, model) -> loaded artifacts. Loading is slow, so keep them around.
_cache = {}


def load_model(ticker, model_name):
    """Load a trained model and its scalers, reusing anything already in memory."""
    import joblib

    key = (ticker.upper(), model_name)
    if key in _cache:
        return _cache[key]

    directory = config.model_dir(ticker, model_name)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No trained model for {ticker.upper()}/{model_name}. "
            f"Run: stockvision train {ticker.upper()} --model {model_name}"
        )
    metadata = json.loads(metadata_path.read_text())

    if metadata["backend"] == "keras":
        import onnxruntime as ort

        scalers = joblib.load(directory / "scaler.joblib")
        loaded = {
            "backend": "keras",
            "session": ort.InferenceSession(
                str(directory / "model.onnx"), providers=["CPUExecutionProvider"]
            ),
            "input_name": metadata["input_name"],
            "output_name": metadata["output_name"],
            "window": metadata["window"],
            "feature_names": scalers["feature_names"],
            "feature_scaler": scalers["feature_scaler"],
            "target_scaler": scalers["target_scaler"],
        }
    else:
        loaded = {"backend": metadata["backend"], "fitted": joblib.load(directory / "model.joblib")}

    _cache[key] = loaded
    log.info("Loaded %s/%s", ticker.upper(), model_name)
    return loaded


def predict(ticker, model_name, days=config.HISTORY_DAYS):
    """Forecast the next close and return it alongside the recent history."""
    model = load_model(ticker, model_name)
    prices = get_prices(ticker)

    if model["backend"] == "keras":
        window = build_window(model, prices, ticker)
        scaled = model["session"].run([model["output_name"]], {model["input_name"]: window})[0]
        prediction = inverse_transform_target(scaled.ravel()[0], model["target_scaler"])
    else:
        prediction = float(get_model(model_name)["predict_next"](model["fitted"], prices))

    response = get_history(ticker, days, prices)
    response["model"] = model_name
    response["prediction"] = prediction
    return response


def build_window(model, prices, ticker):
    """Build the (1, window, features) input tensor a Keras-trained model expects."""
    frame = build_feature_frame(prices, ticker=ticker)
    return transform_window(frame[model["feature_names"]], model["feature_scaler"], model["window"])


def get_history(ticker, days=config.HISTORY_DAYS, prices=None):
    """The last `days` closes, in the shape the frontend charts."""
    if prices is None:
        prices = get_prices(ticker)
    recent = prices.tail(days)
    dates = recent["Date"] if "Date" in recent.columns else recent.index
    return {
        "ticker": ticker.upper(),
        "last_close": float(prices["Close"].iloc[-1]),
        "history": [float(value) for value in recent["Close"]],
        "history_dates": [str(date)[:10] for date in dates],
    }
