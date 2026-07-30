# StockVision

Stock price forecasting for any ticker yfinance can return, with three
interchangeable serving backends behind one REST contract and a React UI that
switches between them at runtime.

- **Python FastAPI** — the reference backend, serves all nine architectures.
- **C++ / ONNX Runtime** — a hand-written HTTP server with no web framework.
- **Triton Inference Server** — NVIDIA's server, fed by the same ONNX files.

## Layout

```
FullStack-StockVision/
├── ingest_train/          # Python package: data, features, training, API, CLI
├── cpp_server/            # C++ HTTP inference server
├── frontend/              # React + Tailwind UI
├── triton_deploy/         # Triton model repository + docker compose
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
```

Each training run writes:

```
ingest_train/artifacts/<TICKER>/<MODEL>/
├── model.onnx          # served by all three backends
├── scaler.joblib       # scalers for the Python backends
├── metadata.json       # window, feature names, tensor names, metrics
└── params.txt          # the same numbers as plain text, for the C++ server
```

### 2. Start the backends

```bash
# Python FastAPI on :8000
stockvision serve

# C++ server on :8080
cd cpp_server
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build --config Release
./build/stock_server --artifacts-dir ../ingest_train/artifacts \
                     --data-dir ../ingest_train/data

# Triton on :8000 (HTTP), reached through the FastAPI service
cd triton_deploy
python setup_triton_repo.py --artifacts ../ingest_train/artifacts
docker compose -f docker-compose.triton.yml up
```

### 3. Start the frontend

```bash
cd frontend
cp .env.example .env       # one URL per backend
npm install
npm start
```

The backend row at the top of the page switches between FastAPI, C++ and
Triton without a reload.

## REST contract

| Method | Path              | Query                     | FastAPI | C++ | Triton |
| ------ | ----------------- | ------------------------- | :-----: | :-: | :----: |
| GET    | `/health`         | -                         | ✅ | ✅ | ✅ |
| GET    | `/history`        | `ticker`, `days`          | ✅ | ✅ | – |
| GET    | `/predict`        | `ticker`, `model`, `days` | ✅ | ✅ | – |
| GET    | `/predict/triton` | `ticker`, `model`, `days` | ✅ | – | via FastAPI |
| GET    | `/explain`        | `ticker`, `model`         | ✅ | – | – |

Every prediction response looks the same:

```json
{
  "ticker": "AAPL",
  "model": "lstm",
  "prediction": 231.44,
  "last_close": 229.87,
  "history": [/* recent closes */],
  "history_dates": ["2025-06-02", "..."]
}
```

## Models

Nine architectures live behind a plugin registry in
`ingest_train/stockvision/models/`.

| Name          | Backend       | Notes                                            |
| ------------- | ------------- | ------------------------------------------------ |
| `lstm`        | keras         | Two-layer LSTM with dropout.                     |
| `bilstm`      | keras         | Bidirectional LSTM stack.                        |
| `gru`         | keras         | Stacked GRU; faster than LSTM.                   |
| `cnn_lstm`    | keras         | 1-D CNN front-end into an LSTM.                  |
| `transformer` | keras         | Encoder-only multi-head self-attention.          |
| `tcn`         | keras         | Dilated causal convolutions with residuals.      |
| `linear`      | keras         | Flattened linear baseline.                       |
| `prophet`     | prophet       | Additive trend and seasonality.                  |
| `autoarima`   | statsforecast | Automatic ARIMA order selection.                 |

The seven Keras models export to ONNX, so all three backends can serve them.
Prophet and AutoARIMA are pickled with joblib and stay on FastAPI.

**Adding an architecture is a one-file change.** Drop a module into
`stockvision/models/` that calls `register(...)`; the package imports every
module in that folder on load, so the CLI, the API, and the UI dropdown pick it
up with no other edits.

```python
# stockvision/models/my_model.py
from .registry import register


def build(window, num_features):
    import tensorflow as tf
    return tf.keras.Sequential([...], name="my_model")


register("my_model", "What it does.", build=build)
```

## CLI

```
stockvision models                  # list the registered architectures
stockvision fetch TICKER            # download OHLCV into the cache
stockvision train TICKER --model X  # train, evaluate, export, log to MLflow
stockvision predict TICKER --model X
stockvision serve                   # FastAPI backend on :8000
stockvision sweep                   # Bayesian W&B sweep
```

## Experiment tracking, versioning and tuning

**MLflow** — `stockvision train` logs every run (hyper-parameters, metrics and
artifacts) via `stockvision/tracking.py`. Skip it with `--no-track`, browse it
with `mlflow ui`.

**DVC** — `.dvc/config` points at an S3 remote. `ingest_train/data.dvc` tracks
the raw CSVs and `ingest_train/artifacts.dvc` tracks the ONNX artifacts. Use
`dvc add`, then `dvc push` / `dvc pull`.

**Weights & Biases** — `ingest_train/sweep_config.yaml` defines a Bayesian
search over all seven Keras architectures (learning rate, batch size, epochs,
window). `stockvision sweep --count 20` registers the sweep and runs the agent.

**SHAP** — `GET /explain` ranks the features that moved a single prediction the
most; the frontend renders it under the "Explain" tab.

**Feast** — optional. Set `STOCKVISION_USE_FEAST=1` to overlay the newest
indicator row from the online store onto the inline pandas features.

## Load testing

`load_tests/locustfile.py` drives all three backends at once:

```bash
cd load_tests
locust -f locustfile.py --headless -u 100 -r 10 --run-time 60s --csv results/run
python compare_report.py --stats results/run_stats.csv
```

`compare_report.py` prints a side-by-side table of requests/s, median latency,
p95 and failures for each backend.

## Configuration

| Variable                     | Default          | Purpose                          |
| ---------------------------- | ---------------- | -------------------------------- |
| `STOCKVISION_PERIOD`         | `2y`             | yfinance lookback window.        |
| `STOCKVISION_INTERVAL`       | `1d`             | yfinance bar interval.           |
| `STOCKVISION_WINDOW`         | `30`             | Sliding window length.           |
| `STOCKVISION_EPOCHS`         | `25`             | Training epochs.                 |
| `STOCKVISION_BATCH_SIZE`     | `32`             | Mini-batch size.                 |
| `STOCKVISION_LEARNING_RATE`  | `0.001`          | Adam learning rate.              |
| `STOCKVISION_HOST` / `_PORT` | `0.0.0.0:8000`   | FastAPI bind address.            |
| `STOCKVISION_DEFAULT_MODEL`  | `lstm`           | Model used when none is given.   |
| `STOCKVISION_USE_FEAST`      | unset            | Set to `1` for the Feast overlay.|
| `MLFLOW_TRACKING_URI`        | `./mlruns`       | MLflow backend store.            |
| `TRITON_URL`                 | `localhost:8000` | Triton server address.           |

## Tests

```bash
cd ingest_train && pytest

cd cpp_server
cmake -S . -B build -DSTOCKVISION_BUILD_TESTS=ON -DONNXRUNTIME_ROOT=/path
cmake --build build && ctest --test-dir build
```

## Notebooks

`testing models/` holds the original exploratory notebooks comparing LSTM,
ARIMA, gradient boosting and linear regression. Everything that runs in
production lives in `ingest_train/`.

## License

MIT. See `LICENSE`.
