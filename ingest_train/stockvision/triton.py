"""Third serving backend: run the same ONNX graph inside Triton Inference Server.

The request and response look identical to /predict; only the place the graph
runs changes.
"""

import logging

from . import config
from .data import get_prices
from .dataset import inverse_transform_target
from .inference import build_window, get_history, load_model

log = logging.getLogger(__name__)


class TritonUnavailable(RuntimeError):
    """Raised when Triton is unreachable or tritonclient is not installed."""


def triton_model_name(ticker, model_name):
    """Triton names models after their folder, e.g. aapl_lstm."""
    return f"{ticker}_{model_name}".lower()


def predict(ticker, model_name, days=config.HISTORY_DAYS):
    try:
        import tritonclient.http as triton_http
        from tritonclient.utils import np_to_triton_dtype
    except ImportError as exc:
        raise TritonUnavailable("tritonclient is not installed") from exc

    # The scalers and tensor names still come from the local artifacts.
    model = load_model(ticker, model_name)
    if model["backend"] != "keras":
        raise TritonUnavailable(f"'{model_name}' is not an ONNX model, so Triton cannot serve it")

    prices = get_prices(ticker)
    window = build_window(model, prices, ticker)

    request = triton_http.InferInput(
        model["input_name"], window.shape, np_to_triton_dtype(window.dtype)
    )
    request.set_data_from_numpy(window)

    try:
        client = triton_http.InferenceServerClient(url=config.TRITON_URL)
        response = client.infer(
            model_name=triton_model_name(ticker, model_name),
            inputs=[request],
            outputs=[triton_http.InferRequestedOutput(model["output_name"])],
        )
    except Exception as exc:
        raise TritonUnavailable(f"Triton request failed: {exc}") from exc

    scaled = response.as_numpy(model["output_name"])
    if scaled is None or scaled.size == 0:
        raise TritonUnavailable("Triton returned an empty output tensor")

    result = get_history(ticker, days, prices)
    result["model"] = model_name
    result["prediction"] = inverse_transform_target(scaled.ravel()[0], model["target_scaler"])
    return result
