from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import AppConfig, load_config
from .data import MarketDataRepository
from .dataset import WindowedDataset, build_dataset
from .evaluation import ForecastMetrics, compute_metrics
from .features import build_feature_frame
from .logging_setup import get_logger
from .models import get_model

log = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    ticker: str
    model_name: str
    onnx_path: Path
    scaler_path: Path
    metadata_path: Path
    metrics: ForecastMetrics


def _set_seeds(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _save_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _write_cpp_params(
    path: Path,
    *,
    window: int,
    num_features: int,
    feature_names: list[str],
    feature_mean: list[float],
    feature_scale: list[float],
    target_mean: float,
    target_scale: float,
    input_name: str,
    output_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"WINDOW {window}",
        f"NUM_FEATURES {num_features}",
        f"INPUT_NAME {input_name}",
        f"OUTPUT_NAME {output_name}",
        "FEATURE_NAMES " + ",".join(feature_names),
        "FEATURE_MEAN " + ",".join(f"{v:.10g}" for v in feature_mean),
        "FEATURE_SCALE " + ",".join(f"{v:.10g}" for v in feature_scale),
        f"TARGET_MEAN {target_mean:.10g}",
        f"TARGET_SCALE {target_scale:.10g}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _export_onnx(model, onnx_path: Path, window: int, num_features: int) -> None:
    import tensorflow as tf
    import tf2onnx

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    spec = (tf.TensorSpec((None, window, num_features), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(onnx_path), opset=15)


def _evaluate(model, dataset: WindowedDataset) -> ForecastMetrics:
    if dataset.X_test.size == 0:
        return ForecastMetrics(0.0, 0.0, 0.0, 0.0, 0)
    preds_scaled = model.predict(dataset.X_test, verbose=0).reshape(-1, 1)
    preds = dataset.target_scaler.inverse_transform(preds_scaled).ravel()
    truths = dataset.target_scaler.inverse_transform(dataset.y_test.reshape(-1, 1)).ravel()
    return compute_metrics(truths, preds)


def train_and_export(
    ticker: str,
    model_name: str,
    config: AppConfig | None = None,
    *,
    repository: MarketDataRepository | None = None,
    force_refresh: bool = False,
) -> TrainingResult:
    import joblib
    import tensorflow as tf

    cfg = config or load_config()
    repo = repository or MarketDataRepository(cfg.data)
    _set_seeds(cfg.training.seed)

    log.info("Training %s on %s", model_name, ticker)
    raw = repo.get(ticker, force_refresh=force_refresh)
    features = build_feature_frame(raw)
    dataset = build_dataset(
        features,
        target_column="close",
        window=cfg.training.window,
        horizon=cfg.training.horizon,
        val_split=cfg.training.val_split,
        test_split=cfg.training.test_split,
    )

    builder = get_model(model_name)
    model = builder.build(window=dataset.window, num_features=dataset.num_features)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=cfg.training.early_stopping_patience,
            restore_best_weights=True,
            monitor="val_loss",
        )
    ]
    model.fit(
        dataset.X_train,
        dataset.y_train,
        validation_data=(dataset.X_val, dataset.y_val),
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    metrics = _evaluate(model, dataset)
    log.info("%s/%s metrics: %s", ticker, model_name, metrics.to_dict())

    onnx_path = cfg.artifacts.onnx_path(ticker, model_name)
    scaler_path = cfg.artifacts.scaler_path(ticker, model_name)
    metadata_path = cfg.artifacts.metadata_path(ticker, model_name)
    params_path = cfg.artifacts.params_path(ticker, model_name)

    _export_onnx(model, onnx_path, dataset.window, dataset.num_features)
    joblib.dump(
        {
            "feature_scaler": dataset.feature_scaler,
            "target_scaler": dataset.target_scaler,
            "feature_names": list(dataset.feature_names),
        },
        scaler_path,
    )

    _save_metadata(
        metadata_path,
        {
            "ticker": ticker.upper(),
            "model": model_name,
            "window": dataset.window,
            "horizon": dataset.horizon,
            "num_features": dataset.num_features,
            "feature_names": list(dataset.feature_names),
            "metrics": metrics.to_dict(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "tf_version": tf.__version__,
            "scaler": {
                "feature_mean": dataset.feature_scaler.mean_.astype(float).tolist(),
                "feature_scale": dataset.feature_scaler.scale_.astype(float).tolist(),
                "target_mean": float(dataset.target_scaler.mean_[0]),
                "target_scale": float(dataset.target_scaler.scale_[0]),
            },
        },
    )
    _write_cpp_params(
        params_path,
        window=dataset.window,
        num_features=dataset.num_features,
        feature_names=list(dataset.feature_names),
        feature_mean=dataset.feature_scaler.mean_.astype(float).tolist(),
        feature_scale=dataset.feature_scaler.scale_.astype(float).tolist(),
        target_mean=float(dataset.target_scaler.mean_[0]),
        target_scale=float(dataset.target_scaler.scale_[0]),
        input_name="input",
        output_name=model.output_names[0] if hasattr(model, "output_names") else "output",
    )
    log.info("Saved artifacts to %s", onnx_path.parent)

    return TrainingResult(
        ticker=ticker.upper(),
        model_name=model_name,
        onnx_path=onnx_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        metrics=metrics,
    )
