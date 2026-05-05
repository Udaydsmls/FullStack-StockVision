# StockVision

A full-stack stock-forecasting project: a Python training pipeline, a C++
inference server, and a React UI. Works for any ticker that yfinance can
return, and ships with seven model architectures.

## Layout

```
FullStack-StockVision/
├── ingest_train/          # Python package (data, features, training, API, CLI)
├── cpp_server/            # C++ HTTP inference server
├── frontend/              # React + Tailwind UI
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

Python service:

```bash
stockvision serve --port 8000
```

C++ server (after `stockvision train`):

```bash
cd cpp_server
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build --config Release
./build/stock_server --port 8080 --artifacts-dir ../ingest_train/artifacts
```

### 3. Frontend

```bash
cd frontend
cp .env.example .env       # set REACT_APP_API_URL
npm install
npm start
```

## REST API

Both servers expose the same endpoints.

| Method | Path        | Query                       | Description             |
| ------ | ----------- | --------------------------- | ----------------------- |
| GET    | `/health`   | -                           | Liveness + model list.  |
| GET    | `/history`  | `ticker`, `days`            | Recent close prices.    |
| GET    | `/predict`  | `ticker`, `model`, `days`   | Next-step forecast.     |

Sample response:

```json
{
  "ticker": "AAPL",
  "model": "transformer",
  "prediction": 195.42,
  "last_close": 193.18,
  "history": [188.10, 189.05],
  "history_dates": ["2025-04-01", "2025-04-02"]
}
```

## Models

| Name          | Notes                                                      |
| ------------- | ---------------------------------------------------------- |
| `lstm`        | Two-layer LSTM with dropout.                               |
| `bilstm`      | Bidirectional LSTM stack.                                  |
| `gru`         | GRU regressor; faster than LSTM.                           |
| `cnn_lstm`    | 1-D CNN front-end into an LSTM head.                       |
| `transformer` | Encoder-only Transformer with multi-head self-attention.   |
| `tcn`         | Dilated causal convolutions with residual connections.     |
| `linear`      | Flattened linear baseline.                                 |

To add a new architecture, drop a class under `stockvision/models/` and
decorate it with `@register_model("name")`. The CLI, API, and UI dropdown
pick it up automatically.

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

The C++ server takes the same options as CLI flags; see
`cpp_server/README.md`.

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
gradient boosting, and linear regression on TATA MOTORS data. Production
training and inference use `ingest_train/`.

## License

MIT. See `LICENSE`.
