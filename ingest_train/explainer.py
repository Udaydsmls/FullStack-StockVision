"""SHAP-based explanations for StockVision predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stockvision.features import build_feature_frame
from stockvision.inference import PredictionService


@dataclass
class ExplanationResult:
    ticker: str
    model: str
    prediction: float
    shap_values: dict[str, float]
    base_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "model": self.model,
            "prediction": self.prediction,
            "base_value": self.base_value,
            "shap_values": self.shap_values,
        }


class StockVisionExplainer:
    """Wraps shap.Explainer around an ONNX-backed model."""

    def __init__(self, service: PredictionService | None = None, top_k: int = 10) -> None:
        self._service = service or PredictionService()
        self._top_k = top_k

    def _predict_wrapper(self, bundle, flat_inputs: np.ndarray) -> np.ndarray:
        n = flat_inputs.shape[0]
        shaped = flat_inputs.reshape(n, bundle.window, len(bundle.feature_names)).astype(np.float32)
        scaled_pred = bundle.session.run(
            [bundle.output_name], {bundle.input_name: shaped}
        )[0]
        return scaled_pred.reshape(-1) * bundle.target_scaler.scale_[0] + bundle.target_scaler.mean_[0]

    def explain(self, ticker: str, model: str) -> ExplanationResult:
        try:
            import shap
        except ImportError as exc:
            raise RuntimeError("shap is not installed: pip install shap") from exc

        bundle = self._service._load_bundle(ticker, model)  # noqa: SLF001
        df = self._service._repo.get(ticker)
        feature_frame = build_feature_frame(df)[list(bundle.feature_names)]
        raw = feature_frame.values[-bundle.window:].astype(np.float32)
        scaled = bundle.feature_scaler.transform(raw).astype(np.float32)
        flat = scaled.reshape(1, -1)

        background = np.repeat(flat, repeats=8, axis=0)
        explainer = shap.Explainer(lambda x: self._predict_wrapper(bundle, x), background)
        shap_values = explainer(flat).values[0]
        prediction = float(self._predict_wrapper(bundle, flat)[0])

        per_feature = shap_values.reshape(bundle.window, len(bundle.feature_names))
        aggregated = np.abs(per_feature).sum(axis=0)
        ranking = sorted(
            zip(bundle.feature_names, aggregated.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )[: self._top_k]

        return ExplanationResult(
            ticker=ticker.upper(),
            model=model,
            prediction=prediction,
            shap_values={name: float(value) for name, value in ranking},
            base_value=float(bundle.target_scaler.mean_[0]),
        )
