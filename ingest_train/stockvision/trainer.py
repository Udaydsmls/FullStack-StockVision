"""Train one model for one ticker and write the artifacts the servers read."""

import json
import logging
import random
from datetime import datetime, timezone

import numpy as np

from . import config
from .data import get_prices
from .dataset import build_dataset
from .evaluation import compute_metrics
from .features import build_feature_frame
from .models import get_model
from .tracking import log_to_mlflow

log = logging.getLogger(__name__)


def train(
    ticker,
    model_name,
    force_refresh=False,
    track=True,
    window=config.WINDOW,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
):
    """Train `model_name` on `ticker` and return a summary dict.

    Keras models are exported to ONNX so all three backends can serve them;
    Prophet and AutoARIMA are pickled and stay on the FastAPI backend.
    """
    model = get_model(model_name)
    prices = get_prices(ticker, force_refresh=force_refresh)

    if model["backend"] == "keras":
        result = _train_keras(ticker, model, prices, window, epochs, batch_size, learning_rate)
    else:
        result = _train_classical(ticker, model, prices)

    if track:
        log_to_mlflow(result)
    return result


def _train_keras(ticker, model, prices, window, epochs, batch_size, learning_rate):
    import joblib
    import tensorflow as tf

    _set_seeds()
    log.info("Training %s on %s", model["name"], ticker.upper())

    frame = build_feature_frame(prices, ticker=ticker)
    data = build_dataset(frame, window=window)

    network = model["build"](window=data.window, num_features=data.num_features)
    network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    network.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True
            )
        ],
    )

    metrics = _evaluate(network, data)
    log.info("%s/%s scored %s", ticker.upper(), model["name"], metrics)

    directory = config.model_dir(ticker, model["name"])
    directory.mkdir(parents=True, exist_ok=True)
    onnx_path = directory / "model.onnx"
    scaler_path = directory / "scaler.joblib"
    metadata_path = directory / "metadata.json"
    params_path = directory / "params.txt"

    _export_onnx(network, onnx_path, data.window, data.num_features)
    input_name, output_name = _onnx_tensor_names(onnx_path)

    joblib.dump(
        {
            "feature_scaler": data.feature_scaler,
            "target_scaler": data.target_scaler,
            "feature_names": data.feature_names,
        },
        scaler_path,
    )
    metadata = {
        "ticker": ticker.upper(),
        "model": model["name"],
        "backend": "keras",
        "window": data.window,
        "horizon": data.horizon,
        "num_features": data.num_features,
        "feature_names": data.feature_names,
        "input_name": input_name,
        "output_name": output_name,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    _write_cpp_params(params_path, data, input_name, output_name)
    log.info("Saved artifacts to %s", directory)

    return {
        "ticker": ticker.upper(),
        "model": model["name"],
        "backend": "keras",
        "metrics": metrics,
        "params": {
            "ticker": ticker.upper(),
            "model": model["name"],
            "window": window,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "onnx_path": onnx_path,
        "artifacts": [onnx_path, scaler_path, metadata_path, params_path],
    }


def _train_classical(ticker, model, prices):
    import joblib

    log.info("Training %s on %s", model["name"], ticker.upper())
    fitted = model["fit"](prices)

    directory = config.model_dir(ticker, model["name"])
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.joblib"
    metadata_path = directory / "metadata.json"
    joblib.dump(fitted, model_path)

    # Sanity check only: compare one forecast against the last few real closes.
    recent = prices["Close"].astype(float).values[-5:]
    forecast = model["predict_next"](fitted, prices)
    metrics = compute_metrics(recent, np.full(len(recent), forecast))

    metadata = {
        "ticker": ticker.upper(),
        "model": model["name"],
        "backend": model["backend"],
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    log.info("Saved artifacts to %s", directory)

    return {
        "ticker": ticker.upper(),
        "model": model["name"],
        "backend": model["backend"],
        "metrics": metrics,
        "params": {"ticker": ticker.upper(), "model": model["name"]},
        "onnx_path": None,
        "artifacts": [model_path, metadata_path],
    }


def _set_seeds():
    """Same seed everywhere so two runs of the same config are comparable."""
    import tensorflow as tf

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)


def _evaluate(network, data):
    """Un-scale the test-set predictions so the metrics are in dollars."""
    if data.X_test.size == 0:
        return compute_metrics([], [])
    scaled = network.predict(data.X_test, verbose=0).reshape(-1, 1)
    predicted = data.target_scaler.inverse_transform(scaled).ravel()
    actual = data.target_scaler.inverse_transform(data.y_test.reshape(-1, 1)).ravel()
    return compute_metrics(actual, predicted)


def _export_onnx(network, path, window, num_features):
    import tensorflow as tf
    import tf2onnx

    # None as the batch dimension lets Triton batch requests together.
    signature = (tf.TensorSpec((None, window, num_features), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(network, input_signature=signature, output_path=str(path), opset=15)


def _onnx_tensor_names(path):
    """Read the tensor names back out of the exported graph.

    Every backend has to address the tensors by the names ONNX actually used,
    so we record them rather than assume them.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.get_inputs()[0].name, session.get_outputs()[0].name


def _write_cpp_params(path, data, input_name, output_name):
    """The C++ server has no joblib, so the scaler is written as plain text."""
    lines = [
        f"WINDOW {data.window}",
        f"NUM_FEATURES {data.num_features}",
        f"INPUT_NAME {input_name}",
        f"OUTPUT_NAME {output_name}",
        "FEATURE_NAMES " + ",".join(data.feature_names),
        "FEATURE_MEAN " + _join_floats(data.feature_scaler.mean_),
        "FEATURE_SCALE " + _join_floats(data.feature_scaler.scale_),
        f"TARGET_MEAN {data.target_scaler.mean_[0]:.10g}",
        f"TARGET_SCALE {data.target_scaler.scale_[0]:.10g}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _join_floats(values):
    return ",".join(f"{value:.10g}" for value in values)
