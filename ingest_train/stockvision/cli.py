from __future__ import annotations

import json

import click

from .config import load_config
from .data import DataFetchError, MarketDataRepository
from .inference import PredictionService
from .logging_setup import configure_logging
from .models import available_models
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
def train_cmd(ticker: str, model: str, force: bool) -> None:
    """Train a model for a ticker and export it to ONNX."""
    if model not in available_models():
        raise click.ClickException(
            f"Unknown model '{model}'. Available: {available_models()}"
        )
    result = train_and_export(ticker, model, force_refresh=force)
    click.echo(json.dumps({
        "ticker": result.ticker,
        "model": result.model_name,
        "metrics": result.metrics.to_dict(),
        "onnx": str(result.onnx_path),
    }, indent=2))


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
