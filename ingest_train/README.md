# StockVision Python package

Data ingestion, feature engineering, training, ONNX export, and the FastAPI
serving backend.

## Install

```bash
cd ingest_train
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Modules

| File            | Purpose                                                        |
| --------------- | -------------------------------------------------------------- |
| `config.py`     | Every setting, with environment-variable overrides.            |
| `data.py`       | yfinance download plus a CSV cache.                            |
| `features.py`   | Technical indicators; the C++ server mirrors these.            |
| `dataset.py`    | Chronological split, scaling, and sliding windows.             |
| `models/`       | The plugin registry and the nine architectures.                |
| `trainer.py`    | Train, evaluate, export to ONNX, write `params.txt`.           |
| `tracking.py`   | MLflow logging for each run.                                   |
| `inference.py`  | Load artifacts and serve forecasts.                            |
| `triton.py`     | Same forecast, executed inside Triton.                         |
| `explain.py`    | SHAP feature attributions for one prediction.                  |
| `api.py`        | FastAPI routes.                                                |
| `sweep.py`      | Weights & Biases Bayesian sweep.                               |
| `cli.py`        | The `stockvision` command.                                     |

## Commands

```bash
stockvision models
stockvision fetch AAPL
stockvision train AAPL --model transformer
stockvision predict AAPL --model transformer
stockvision serve --port 8000
stockvision sweep --count 20
```

## Adding an architecture

Create one file under `stockvision/models/` and call `register(...)`. The
package imports every module in that folder on load, so nothing else needs to
change:

```python
from .registry import register


def build(window, num_features):
    import tensorflow as tf
    return tf.keras.Sequential([...], name="my_model")


register("my_model", "What it does.", build=build)
```

Classical models pass `fit=` and `predict_next=` instead of `build=`; see
`prophet_model.py`.

## Tests

```bash
pytest
```
