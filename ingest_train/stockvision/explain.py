"""SHAP explanations: which features moved a single prediction the most."""

import numpy as np

from .data import get_prices
from .inference import build_window, load_model


def explain(ticker, model_name, top_k=10):
    """Return the top features by absolute SHAP contribution, largest first."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("shap is not installed: pip install shap") from exc

    model = load_model(ticker, model_name)
    if model["backend"] != "keras":
        raise ValueError(f"'{model_name}' is not an ONNX model, so it cannot be explained")

    prices = get_prices(ticker)
    window = build_window(model, prices, ticker)
    flat = window.reshape(1, -1)  # SHAP works on flat rows

    def run(rows):
        batch = rows.reshape(len(rows), model["window"], len(model["feature_names"]))
        scaled = model["session"].run(
            [model["output_name"]], {model["input_name"]: batch.astype(np.float32)}
        )[0]
        scaler = model["target_scaler"]
        return scaled.ravel() * scaler.scale_[0] + scaler.mean_[0]

    # One row of background is enough when we only explain a single prediction.
    explainer = shap.Explainer(run, np.repeat(flat, 8, axis=0))
    contributions = explainer(flat).values[0]
    per_feature = contributions.reshape(model["window"], len(model["feature_names"]))

    # Sum each feature's contribution across every day in the window.
    totals = np.abs(per_feature).sum(axis=0)
    ranked = sorted(zip(model["feature_names"], totals), key=lambda pair: pair[1], reverse=True)

    return {
        "ticker": ticker.upper(),
        "model": model_name,
        "prediction": float(run(flat)[0]),
        "base_value": float(model["target_scaler"].mean_[0]),
        "shap_values": {name: float(value) for name, value in ranked[:top_k]},
    }
