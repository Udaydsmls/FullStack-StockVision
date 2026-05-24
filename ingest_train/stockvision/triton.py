from __future__ import annotations

import os
from typing import Any

import numpy as np

from .features import build_feature_frame
from .inference import PredictionService


class TritonUnavailable(RuntimeError):
    pass


class TritonInferenceClient:
    """Thin wrapper that talks to a running Triton server using tritonclient."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.environ.get("TRITON_URL", "localhost:8000")

    @staticmethod
    def _model_name(ticker: str, model: str) -> str:
        return f"{ticker.upper()}_{model}".lower()

    def predict(
        self,
        service: PredictionService,
        ticker: str,
        model: str,
        *,
        history_size: int = 60,
    ) -> dict[str, Any]:
        try:
            import tritonclient.http as httpclient
            from tritonclient.utils import np_to_triton_dtype
        except ImportError as exc:
            raise TritonUnavailable("tritonclient is not installed") from exc

        bundle = service._load_bundle(ticker, model)  # noqa: SLF001 (reuses metadata + scalers)
        df = service._repo.get(ticker)
        feature_frame = build_feature_frame(df)[list(bundle.feature_names)]
        tensor = feature_frame.values[-bundle.window:].astype(np.float32)
        scaled = bundle.feature_scaler.transform(tensor).astype(np.float32)[None, :, :]

        client = httpclient.InferenceServerClient(url=self.url)
        infer_input = httpclient.InferInput("input", scaled.shape, np_to_triton_dtype(scaled.dtype))
        infer_input.set_data_from_numpy(scaled)
        request_output = httpclient.InferRequestedOutput("output")

        try:
            response = client.infer(
                model_name=self._model_name(ticker, model),
                inputs=[infer_input],
                outputs=[request_output],
            )
        except Exception as exc:
            raise TritonUnavailable(f"Triton request failed: {exc}") from exc

        scaled_pred = response.as_numpy("output")
        if scaled_pred is None or scaled_pred.size == 0:
            raise TritonUnavailable("Triton returned an empty output tensor")
        prediction = float(scaled_pred.ravel()[0] * bundle.target_scaler.scale_[0]
                           + bundle.target_scaler.mean_[0])

        tail = df.tail(history_size)
        return {
            "ticker": ticker.upper(),
            "model": model,
            "prediction": prediction,
            "last_close": float(df["Close"].iloc[-1]),
            "history": [float(v) for v in tail["Close"].tolist()],
            "history_dates": [str(d)[:10] for d in tail.get("Date", tail.index).tolist()],
        }
