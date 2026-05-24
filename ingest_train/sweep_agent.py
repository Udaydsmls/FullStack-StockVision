"""W&B sweep agent. Delegates to the unchanged train_and_export function."""

from __future__ import annotations

import dataclasses
import os

from stockvision.config import load_config
from stockvision.trainer import train_and_export


def run() -> None:
    import wandb

    wandb.init()
    cfg = load_config()
    sweep = wandb.config

    training = dataclasses.replace(
        cfg.training,
        window=int(sweep.get("window", cfg.training.window)),
        epochs=int(sweep.get("epochs", cfg.training.epochs)),
        batch_size=int(sweep.get("batch_size", cfg.training.batch_size)),
        learning_rate=float(sweep.get("learning_rate", cfg.training.learning_rate)),
    )
    cfg = dataclasses.replace(cfg, training=training)

    ticker = str(sweep.get("ticker", os.environ.get("STOCKVISION_SWEEP_TICKER", "AAPL")))
    model = str(sweep.get("model", "lstm"))

    result = train_and_export(ticker, model, config=cfg)
    metrics = result.metrics.to_dict()
    wandb.log(
        {
            "val_mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "directional_accuracy": metrics["directional_accuracy"],
        }
    )
    if result.onnx_path.exists():
        artifact = wandb.Artifact(name=f"{ticker}-{model}-onnx", type="model")
        artifact.add_file(str(result.onnx_path))
        wandb.log_artifact(artifact)


if __name__ == "__main__":
    run()
