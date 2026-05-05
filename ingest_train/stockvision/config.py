from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    period: str = "2y"
    interval: str = "1d"
    cache_ttl_seconds: int = 6 * 60 * 60
    cache_dir: Path = field(default_factory=lambda: _project_root() / "ingest_train" / "data")
    feature_columns: Sequence[str] = ("Close", "Open", "High", "Low", "Volume")
    target_column: str = "Close"


@dataclass(frozen=True)
class TrainingConfig:
    window: int = 30
    horizon: int = 1
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    val_split: float = 0.15
    test_split: float = 0.10
    early_stopping_patience: int = 5
    seed: int = 42


@dataclass(frozen=True)
class ArtifactConfig:
    artifacts_dir: Path = field(
        default_factory=lambda: _project_root() / "ingest_train" / "artifacts"
    )

    def model_dir(self, ticker: str, model_name: str) -> Path:
        return self.artifacts_dir / ticker.upper() / model_name

    def onnx_path(self, ticker: str, model_name: str) -> Path:
        return self.model_dir(ticker, model_name) / "model.onnx"

    def scaler_path(self, ticker: str, model_name: str) -> Path:
        return self.model_dir(ticker, model_name) / "scaler.joblib"

    def metadata_path(self, ticker: str, model_name: str) -> Path:
        return self.model_dir(ticker, model_name) / "metadata.json"

    def params_path(self, ticker: str, model_name: str) -> Path:
        return self.model_dir(ticker, model_name) / "params.txt"


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    default_model: str = "lstm"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)


def load_config() -> AppConfig:
    def env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        return int(raw) if raw is not None and raw.strip() else default

    def env_str(name: str, default: str) -> str:
        return os.getenv(name, default)

    data = DataConfig(
        period=env_str("STOCKVISION_PERIOD", DataConfig.period),
        interval=env_str("STOCKVISION_INTERVAL", DataConfig.interval),
    )
    training = TrainingConfig(
        window=env_int("STOCKVISION_WINDOW", TrainingConfig.window),
        epochs=env_int("STOCKVISION_EPOCHS", TrainingConfig.epochs),
        batch_size=env_int("STOCKVISION_BATCH_SIZE", TrainingConfig.batch_size),
    )
    service = ServiceConfig(
        host=env_str("STOCKVISION_HOST", ServiceConfig.host),
        port=env_int("STOCKVISION_PORT", ServiceConfig.port),
        log_level=env_str("STOCKVISION_LOG_LEVEL", ServiceConfig.log_level),
        default_model=env_str("STOCKVISION_DEFAULT_MODEL", ServiceConfig.default_model),
    )
    return AppConfig(data=data, training=training, service=service)
