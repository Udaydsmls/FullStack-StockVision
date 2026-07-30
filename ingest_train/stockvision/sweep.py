"""Weights & Biases sweep: Bayesian search over the seven Keras architectures."""

import logging

from . import config
from .trainer import train

log = logging.getLogger(__name__)


def start(project="stockvision", count=20):
    """Register the sweep defined in sweep_config.yaml and run `count` trials."""
    import wandb
    import yaml

    settings = yaml.safe_load(config.SWEEP_CONFIG.read_text())
    sweep_id = wandb.sweep(settings, project=project)
    log.info("Started sweep %s", sweep_id)
    wandb.agent(sweep_id, function=run_trial, count=count)
    return sweep_id


def run_trial():
    """One trial: W&B picks the hyper-parameters, we train and report the metrics."""
    import wandb

    wandb.init()
    chosen = wandb.config
    result = train(
        chosen.ticker,
        chosen.model,
        track=False,
        window=chosen.window,
        epochs=chosen.epochs,
        batch_size=chosen.batch_size,
        learning_rate=chosen.learning_rate,
    )

    wandb.log(result["metrics"])
    artifact = wandb.Artifact(f"{chosen.ticker}-{chosen.model}-onnx", type="model")
    artifact.add_file(str(result["onnx_path"]))
    wandb.log_artifact(artifact)
