from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List

import numpy as np

from .config import AppConfig, load_config
from .data import MarketDataRepository
from .dataset import inverse_transform_target, transform_window
from .features import build_feature_frame
from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    ticker: str
    model: str
    prediction: float
    last_close: float
    history: List[float]
    history_dates: List[str]


class _OnnxBundle:
    __slots__ = ("session", "input_name", "output_name", "feature_names", "feature_scaler", "target_scaler", "window")

    def __init__(self, session, input_name, output_name, feature_names, feature_scaler, target_scaler, window):
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.feature_names = feature_names
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.window = window


class PredictionService:
    def __init__(
        self,
        config: AppConfig | None = None,
        repository: MarketDataRepository | None = None,
    ) -> None:
        self._cfg = config or load_config()
        self._repo = repository or MarketDataRepository(self._cfg.data)
        self._cache: Dict[tuple[str, str], _OnnxBundle] = {}
        self._lock = Lock()

    @staticmethod
    def _read_metadata(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing metadata file: {path}")
        return json.loads(path.read_text())

    def _load_bundle(self, ticker: str, model: str) -> _OnnxBundle:
        import joblib
        import onnxruntime as ort

        ticker = ticker.upper()
        key = (ticker, model)
        if key in self._cache:
            return self._cache[key]

        with self._lock:
            if key in self._cache:
                return self._cache[key]

            onnx_path = self._cfg.artifacts.onnx_path(ticker, model)
            scaler_path = self._cfg.artifacts.scaler_path(ticker, model)
            metadata_path = self._cfg.artifacts.metadata_path(ticker, model)

            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"Model not trained for ({ticker}, {model}). Run: stockvision train {ticker} --model {model}"
                )

            metadata = self._read_metadata(metadata_path)
            scalers = joblib.load(scaler_path)
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            bundle = _OnnxBundle(
                session=session,
                input_name=input_name,
                output_name=output_name,
                feature_names=tuple(scalers["feature_names"]),
                feature_scaler=scalers["feature_scaler"],
                target_scaler=scalers["target_scaler"],
                window=int(metadata["window"]),
            )
            self._cache[key] = bundle
            log.info("Loaded ONNX bundle for %s/%s", ticker, model)
            return bundle

    def predict(self, ticker: str, model: str, *, history_size: int = 60) -> PredictionResult:
        ticker = ticker.upper()
        bundle = self._load_bundle(ticker, model)
        df = self._repo.get(ticker)
        feature_frame = build_feature_frame(df)[list(bundle.feature_names)]
        x = transform_window(feature_frame, bundle.feature_scaler, bundle.window)
        scaled_pred = bundle.session.run(
            [bundle.output_name], {bundle.input_name: x.astype(np.float32)}
        )[0]
        prediction = inverse_transform_target(float(scaled_pred.ravel()[0]), bundle.target_scaler)

        tail = df.tail(history_size)
        history = [float(v) for v in tail["Close"].tolist()]
        history_dates = [str(d)[:10] for d in tail.get("Date", tail.index).tolist()]
        return PredictionResult(
            ticker=ticker,
            model=model,
            prediction=prediction,
            last_close=float(df["Close"].iloc[-1]),
            history=history,
            history_dates=history_dates,
        )

    def history(self, ticker: str, *, history_size: int = 60) -> dict:
        df = self._repo.get(ticker)
        tail = df.tail(history_size)
        return {
            "ticker": ticker.upper(),
            "history": [float(v) for v in tail["Close"].tolist()],
            "history_dates": [str(d)[:10] for d in tail.get("Date", tail.index).tolist()],
        }
