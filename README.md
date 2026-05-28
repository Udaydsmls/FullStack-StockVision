# StockVision

A full-stack stock-forecasting project: a Python training pipeline, three
serving backends, and a React UI. Works for any ticker that yfinance can
return, ships with nine model architectures, and exposes optional layers
for experiment tracking, data versioning, hyperparameter sweeps, model
serving via Triton, prediction explanations, load testing, and a feature
store.

## Layout

```
FullStack-StockVision/
├── ingest_train/          # Python package (data, features, training, API, CLI)
├── cpp_server/            # C++ HTTP inference server
├── frontend/              # React + Tailwind UI
├── triton_deploy/         # Triton Inference Server scaffolding
├── feature_store/         # Optional Feast feature repo
├── load_tests/            # Locust suite comparing the three backends
└── testing models/        # Exploratory notebooks
```

## Quick start

### 1. Train a model

```bash
cd ingest_train
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .[dev]

stockvision fetch AAPL
stockvision train AAPL --model lstm
stockvision train MSFT --model transformer
```

Each train command writes:

```
ingest_train/artifacts/<TICKER>/<MODEL>/
├── model.onnx
├── scaler.joblib
├── metadata.json
└── params.txt          # consumed by the C++ server
```

### 2. Serve predictions

Pick any of the three backends; they expose the same REST contract.

```bash
# Python FastAPI
stockvision serve --port 8000

# C++ ONNX server
cd cpp_server
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build --config Release
./build/stock_server --port 8080 --artifacts-dir ../ingest_train/artifacts

# Triton (third option)
cd triton_deploy
python setup_triton_repo.py --artifacts ../ingest_train/artifacts
docker compose -f docker-compose.triton.yml up
```

### 3. Frontend

```bash
cd frontend
cp .env.example .env       # set REACT_APP_API_URL
npm install
npm start
```

## REST API

| Method | Path              | Query                          | Description                       |
| ------ | ----------------- | ------------------------------ | --------------------------------- |
| GET    | `/health`         | -                              | Liveness + registered model list. |
| GET    | `/history`        | `ticker`, `days`               | Recent close prices.              |
| GET    | `/predict`        | `ticker`, `model`, `days`      | Next-step forecast.               |
| GET    | `/predict/triton` | `ticker`, `model`, `days`      | Same forecast, served via Triton. |
| GET    | `/explain`        | `ticker`, `model`              | Top SHAP feature contributions.   |

The C++ server implements `/health`, `/history`, `/predict`.

## Models

| Name          | Backend       | Notes                                                      |
| ------------- | ------------- | ---------------------------------------------------------- |
| `lstm`        | keras         | Two-layer LSTM with dropout.                               |
| `bilstm`      | keras         | Bidirectional LSTM stack.                                  |
| `gru`         | keras         | GRU regressor; faster than LSTM.                           |
| `cnn_lstm`    | keras         | 1-D CNN front-end into an LSTM head.                       |
| `transformer` | keras         | Encoder-only Transformer with multi-head self-attention.   |
| `tcn`         | keras         | Dilated causal convolutions with residual connections.     |
| `linear`      | keras         | Flattened linear baseline.                                 |
| `prophet`     | prophet       | Additive trend + seasonality (Facebook Prophet).           |
| `autoarima`   | statsforecast | Automatic ARIMA from Nixtla statsforecast.                 |

Keras models export to ONNX; the prophet and statsforecast backends
serialise via joblib and are not served by the C++ or Triton paths.

To add a new architecture, drop a class under `stockvision/models/` and
decorate it with `@register_model("name")`. The CLI, API, and UI dropdown
pick it up automatically.

## CLI reference

```
stockvision fetch TICKER           # download OHLCV into the cache
stockvision train TICKER --model X # train + export to ONNX (or joblib for non-keras)
stockvision predict TICKER --model X
stockvision serve                  # FastAPI service (default :8000)
stockvision models                 # list registered architectures

stockvision mlflow-ui              # local MLflow UI
stockvision sweep                  # launch a W&B sweep from sweep_config.yaml
stockvision dvc-push / dvc-pull    # sync data + artefacts with the DVC remote
```

## Optional layers

### MLflow tracking

`stockvision train` wraps each run in an MLflow context manager
(`ingest_train/mlflow_wrapper.py`). Params, metrics, and ONNX artefacts are
logged in parallel to the regular `artifacts/` output. Disable with
`--no-track`. Browse with `stockvision mlflow-ui`.

### DVC versioning

`.dvc/config` is preconfigured for an S3 remote. The two pointer files
`ingest_train/data.dvc` and `ingest_train/artifacts.dvc` track the raw CSVs
and the trained ONNX artefacts. Run `dvc add` to refresh them and use
`stockvision dvc-push` / `stockvision dvc-pull` from the CLI.

### Weights & Biases sweeps

`ingest_train/sweep_config.yaml` defines a Bayesian search over all seven
keras architectures. The agent in `sweep_agent.py` re-uses the unchanged
`train_and_export` function and logs metrics + ONNX as a W&B artifact.

```bash
cd ingest_train
stockvision sweep --project stockvision --count 20
```

### Triton Inference Server

`triton_deploy/setup_triton_repo.py` walks `artifacts/` and lays out the
Triton `model_repository`. `triton_deploy/docker-compose.triton.yml` boots
`nvcr.io/nvidia/tritonserver` on ports 8000 (HTTP), 8001 (gRPC), 8002
(metrics). The FastAPI `/predict/triton` endpoint proxies into Triton.

### SHAP explanations

`GET /explain?ticker=AAPL&model=lstm` returns the top feature contributions
computed by `ingest_train/explainer.py`. The frontend has an "Explain" tab
that renders the bar chart in `frontend/src/components/ExplainChart.jsx`.

### Locust load tests

`load_tests/locustfile.py` defines three user classes — FastAPI, C++ server,
and Triton (through the FastAPI proxy). After a run, generate a Markdown
comparison with `python load_tests/compare_report.py`.

### Feast feature store

`feature_store/` contains a Feast project (entities, feature views, a
local-SQLite registry). Set `STOCKVISION_USE_FEAST=1` to overlay the most
recent indicator row from the online store onto the inline feature frame.
The inline pandas computation remains the default.

## Configuration

| Variable                     | Default        | Purpose                            |
| ---------------------------- | -------------- | ---------------------------------- |
| `STOCKVISION_PERIOD`         | `2y`           | yfinance lookback window.          |
| `STOCKVISION_INTERVAL`       | `1d`           | yfinance bar interval.             |
| `STOCKVISION_WINDOW`         | `30`           | Sliding window length.             |
| `STOCKVISION_EPOCHS`         | `25`           | Training epochs.                   |
| `STOCKVISION_BATCH_SIZE`     | `32`           | Mini-batch size.                   |
| `STOCKVISION_HOST` / `_PORT` | `0.0.0.0:8000` | FastAPI bind address.              |
| `STOCKVISION_DEFAULT_MODEL`  | `lstm`         | Fallback model.                    |
| `STOCKVISION_USE_FEAST`      | unset          | Enable the Feast feature overlay.  |
| `MLFLOW_TRACKING_URI`        | `./mlruns`     | MLflow backend store.              |
| `TRITON_URL`                 | `localhost:8000` | Triton server URL used by /predict/triton. |

## Tests

```bash
# Python
cd ingest_train && pytest

# C++
cd cpp_server
cmake -S . -B build -DSTOCKVISION_BUILD_TESTS=ON -DONNXRUNTIME_ROOT=/path
cmake --build build && ctest --test-dir build
```

## Notebooks

`testing models/` holds the original notebooks comparing LSTM, ARIMA,
gradient boosting, and linear regression on sample equity price data.
Production training and inference use `ingest_train/`.

## License

MIT. See `LICENSE`.
