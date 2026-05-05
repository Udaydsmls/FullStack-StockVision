# StockVision Python Package

Data ingestion, feature engineering, training, ONNX export, and a FastAPI
inference service.

## Installation

```bash
cd ingest_train
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Command-line interface

The package installs a `stockvision` console script with the following
subcommands.

| Command                               | Purpose                                                         |
| ------------------------------------- | --------------------------------------------------------------- |
| `stockvision models`                  | List the registered model architectures.                        |
| `stockvision fetch TICKER`            | Download OHLCV data into the local cache.                       |
| `stockvision train TICKER --model X`  | Train a model and export it to ONNX.                            |
| `stockvision predict TICKER --model X`| Run a one-shot inference against a trained model.               |
| `stockvision serve`                   | Launch the FastAPI service (default: `http://0.0.0.0:8000`).    |

Examples:

```bash
stockvision fetch AAPL
stockvision train AAPL --model transformer
stockvision predict AAPL --model transformer
stockvision serve --port 8000
```

## Available architectures

| Name          | Description                                                            |
| ------------- | ---------------------------------------------------------------------- |
| `lstm`        | Stacked LSTM with dropout.                                             |
| `bilstm`      | Bidirectional LSTM stack.                                              |
| `gru`         | Stacked GRU; faster than LSTM.                                         |
| `cnn_lstm`    | 1-D convolution front-end into an LSTM.                                |
| `transformer` | Encoder-only Transformer with multi-head self-attention.               |
| `tcn`         | Temporal Convolutional Network with dilated causal convolutions.       |
| `linear`      | Flattened linear baseline.                                             |

Add a new architecture by subclassing `BaseForecastModel` and decorating it
with `@register_model("name")` inside `stockvision/models/`. The registry is
imported automatically on package load.

## HTTP endpoints

| Method | Path        | Description                                                        |
| ------ | ----------- | ------------------------------------------------------------------ |
| `GET`  | `/health`   | Returns service status and the registered model names.             |
| `GET`  | `/history`  | `?ticker=AAPL&days=60` — recent close prices.                      |
| `GET`  | `/predict`  | `?ticker=AAPL&model=lstm&days=60` — next-step forecast + history.  |

## Configuration

Configuration is loaded from `stockvision.config.AppConfig` and can be
overridden via environment variables:

| Variable                       | Default | Description                                  |
| ------------------------------ | ------- | -------------------------------------------- |
| `STOCKVISION_PERIOD`           | `2y`    | yfinance lookback period.                    |
| `STOCKVISION_INTERVAL`         | `1d`    | yfinance bar interval.                       |
| `STOCKVISION_WINDOW`           | `30`    | Sliding-window length (rows).                |
| `STOCKVISION_EPOCHS`           | `25`    | Training epochs.                             |
| `STOCKVISION_BATCH_SIZE`       | `32`    | Mini-batch size.                             |
| `STOCKVISION_HOST` / `_PORT`   | `0.0.0.0:8000` | API bind address.                     |
| `STOCKVISION_LOG_LEVEL`        | `info`  | Log level for the CLI and the API service.   |
| `STOCKVISION_DEFAULT_MODEL`    | `lstm`  | Fallback model when none is requested.       |

## Tests

```bash
pytest -q
```
