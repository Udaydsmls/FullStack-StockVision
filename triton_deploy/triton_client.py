"""HTTP client for the Triton-served StockVision models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_path() -> None:
    """Make the stockvision package importable without installing it."""
    here = Path(__file__).resolve().parent
    candidate = here.parent / "ingest_train"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _build_input(ticker: str, window: int) -> tuple[np.ndarray, list[str]]:
    _ensure_path()
    from stockvision.data import MarketDataRepository
    from stockvision.dataset import transform_window
    from stockvision.features import build_feature_frame

    repo = MarketDataRepository()
    df = repo.get(ticker)
    feature_frame = build_feature_frame(df)
    feature_names = list(feature_frame.columns)
    raw = feature_frame.values[-window:].astype(np.float32)
    return raw[None, :, :], feature_names


def predict(host: str, model_name: str, tensor: np.ndarray) -> dict[str, Any]:
    try:
        import tritonclient.http as httpclient
        from tritonclient.utils import np_to_triton_dtype
    except ImportError as exc:
        raise SystemExit("tritonclient is not installed: pip install tritonclient[http]") from exc

    client = httpclient.InferenceServerClient(url=host)
    inputs = [httpclient.InferInput("input", tensor.shape, np_to_triton_dtype(tensor.dtype))]
    inputs[0].set_data_from_numpy(tensor)
    outputs = [httpclient.InferRequestedOutput("output")]
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return {"output": response.as_numpy("output").tolist()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost:8000")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--model", required=True, help="The Triton model name, e.g. AAPL_lstm.")
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args(argv)

    tensor, _ = _build_input(args.ticker, args.window)
    result = predict(args.host, args.model.lower(), tensor)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
