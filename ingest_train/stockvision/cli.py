from __future__ import annotations

import json

import click

from .classical_trainer import train_classical
from .config import load_config
from .data import DataFetchError, MarketDataRepository
from .inference import PredictionService
from .logging_setup import configure_logging
from .models import available_models, get_model
from .trainer import train_and_export


@click.group()
@click.option("--log-level", default=None, help="Override log level (debug, info, warning, error).")
def main(log_level: str | None) -> None:
    cfg = load_config()
    configure_logging(log_level or cfg.service.log_level)


@main.command("models")
def models_cmd() -> None:
    """List the available model architectures."""
    from .models import get_model

    for name in available_models():
        meta = get_model(name).metadata
        click.echo(f"{meta.name:<12} {meta.description}")


@main.command("fetch")
@click.argument("ticker")
@click.option("--force/--no-force", default=False, help="Bypass the local cache.")
def fetch_cmd(ticker: str, force: bool) -> None:
    """Download OHLCV data for a ticker into the local cache."""
    cfg = load_config()
    repo = MarketDataRepository(cfg.data)
    try:
        df = repo.get(ticker, force_refresh=force)
    except DataFetchError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Fetched {len(df)} rows for {ticker.upper()} -> {repo.cache_path(ticker)}")


@main.command("train")
@click.argument("ticker")
@click.option("--model", default="lstm", show_default=True, help="Model architecture name.")
@click.option("--force/--no-force", default=False, help="Force refresh of cached data.")
@click.option("--track/--no-track", default=True, help="Log the run to MLflow if installed.")
def train_cmd(ticker: str, model: str, force: bool, track: bool) -> None:
    """Train a model for a ticker and export it to ONNX."""
    if model not in available_models():
        raise click.ClickException(
            f"Unknown model '{model}'. Available: {available_models()}"
        )
    cfg = load_config()
    backend = get_model(model).metadata.backend

    if backend != "keras":
        result = train_classical(ticker, model, force_refresh=force)
        click.echo(json.dumps({
            "ticker": result.ticker,
            "model": result.model_name,
            "backend": backend,
            "metrics": result.metrics.to_dict(),
            "artifact": str(result.model_path),
        }, indent=2))
        return

    if track:
        from mlflow_wrapper import track_run

        params = {
            "ticker": ticker.upper(),
            "model": model,
            "window": cfg.training.window,
            "horizon": cfg.training.horizon,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
        }
        with track_run(model, ticker, params) as run:
            result = train_and_export(ticker, model, force_refresh=force)
            run.log_metrics(result.metrics.to_dict())
            run.log_artifact(result.onnx_path)
            run.log_artifact(result.scaler_path)
            run.log_artifact(result.metadata_path)
    else:
        result = train_and_export(ticker, model, force_refresh=force)
    click.echo(json.dumps({
        "ticker": result.ticker,
        "model": result.model_name,
        "metrics": result.metrics.to_dict(),
        "onnx": str(result.onnx_path),
    }, indent=2))


@main.command("mlflow-ui")
@click.option("--port", default=5000, show_default=True)
@click.option("--backend-store-uri", default="./mlruns", show_default=True)
def mlflow_ui_cmd(port: int, backend_store_uri: str) -> None:
    """Launch the MLflow UI against the local tracking store."""
    import subprocess

    cmd = ["mlflow", "ui", "--backend-store-uri", backend_store_uri, "--port", str(port)]
    raise SystemExit(subprocess.call(cmd))


@main.command("sweep")
@click.option("--project", default="stockvision", show_default=True)
@click.option("--config", "config_path", default="sweep_config.yaml", show_default=True,
              help="Path to the sweep YAML, relative to the ingest_train directory.")
@click.option("--count", default=20, show_default=True, help="Number of agent runs to execute.")
def sweep_cmd(project: str, config_path: str, count: int) -> None:
    """Initialise and launch a Weights & Biases sweep."""
    import yaml
    try:
        import wandb
    except ImportError as exc:
        raise click.ClickException("wandb is not installed: pip install wandb") from exc

    from pathlib import Path

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise click.ClickException(f"Sweep config not found: {cfg_path.resolve()}")
    sweep_cfg = yaml.safe_load(cfg_path.read_text())

    sweep_id = wandb.sweep(sweep_cfg, project=project)
    click.echo(f"Started sweep {sweep_id}")
    from sweep_agent import run as agent_run

    wandb.agent(sweep_id, function=agent_run, count=count)


@main.command("dvc-push")
def dvc_push_cmd() -> None:
    """Push DVC-tracked data and artefacts to the configured remote."""
    import subprocess

    raise SystemExit(subprocess.call(["dvc", "push"]))


@main.command("dvc-pull")
def dvc_pull_cmd() -> None:
    """Pull DVC-tracked data and artefacts from the configured remote."""
    import subprocess

    raise SystemExit(subprocess.call(["dvc", "pull"]))


@main.command("predict")
@click.argument("ticker")
@click.option("--model", default="lstm", show_default=True)
@click.option("--days", default=60, show_default=True)
def predict_cmd(ticker: str, model: str, days: int) -> None:
    """Run inference for a ticker using a trained ONNX model."""
    service = PredictionService()
    try:
        result = service.predict(ticker, model, history_size=days)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))
    click.echo(json.dumps({
        "ticker": result.ticker,
        "model": result.model,
        "prediction": result.prediction,
        "last_close": result.last_close,
        "history_size": len(result.history),
    }, indent=2))


@main.command("serve")
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--reload/--no-reload", default=False)
def serve_cmd(host: str | None, port: int | None, reload: bool) -> None:
    """Run the FastAPI service with uvicorn."""
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "stockvision.api:app",
        host=host or cfg.service.host,
        port=port or cfg.service.port,
        reload=reload,
        log_level=cfg.service.log_level,
    )


if __name__ == "__main__":
    main()
