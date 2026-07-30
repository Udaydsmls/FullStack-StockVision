"""Command line interface: `stockvision <command>`."""

import json
import logging

import click

from . import config, inference, sweep
from .data import get_prices
from .models import available_models, get_model
from .trainer import train


@click.group()
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )


@main.command()
def models():
    """List the registered architectures."""
    for name in available_models():
        model = get_model(name)
        click.echo(f"{name:<12} {model['backend']:<14} {model['description']}")


@main.command()
@click.argument("ticker")
@click.option("--force", is_flag=True, help="Ignore the cached CSV and re-download.")
def fetch(ticker, force):
    """Download OHLCV data for TICKER into the local cache."""
    prices = get_prices(ticker, force_refresh=force)
    click.echo(f"Fetched {len(prices)} rows for {ticker.upper()}")


@main.command("train")
@click.argument("ticker")
@click.option("--model", default=config.DEFAULT_MODEL, show_default=True)
@click.option("--force", is_flag=True, help="Ignore the cached CSV and re-download.")
@click.option("--no-track", is_flag=True, help="Skip the MLflow run.")
def train_command(ticker, model, force, no_track):
    """Train MODEL on TICKER and write its artifacts."""
    _check_model(model)
    result = train(ticker, model, force_refresh=force, track=not no_track)
    click.echo(json.dumps(result, indent=2, default=str))


@main.command("predict")
@click.argument("ticker")
@click.option("--model", default=config.DEFAULT_MODEL, show_default=True)
@click.option("--days", default=config.HISTORY_DAYS, show_default=True)
def predict_command(ticker, model, days):
    """Forecast the next close for TICKER."""
    _check_model(model)
    try:
        result = inference.predict(ticker, model, days)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))
    click.echo(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


@main.command()
@click.option("--host", default=config.HOST, show_default=True)
@click.option("--port", default=config.PORT, show_default=True)
@click.option("--reload", is_flag=True, help="Restart on code changes.")
def serve(host, port, reload):
    """Run the FastAPI backend."""
    import uvicorn

    uvicorn.run("stockvision.api:app", host=host, port=port, reload=reload)


@main.command("sweep")
@click.option("--project", default="stockvision", show_default=True)
@click.option("--count", default=20, show_default=True, help="Number of trials to run.")
def sweep_command(project, count):
    """Launch the Bayesian W&B sweep from sweep_config.yaml."""
    sweep.start(project=project, count=count)


def _check_model(name):
    if name not in available_models():
        raise click.ClickException(f"Unknown model '{name}'. Available: {available_models()}")


if __name__ == "__main__":
    main()
