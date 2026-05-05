# StockVision C++ Inference Server

HTTP server that loads ONNX models exported by the Python pipeline and serves
predictions for any ticker. Features are recomputed in C++ so there's no
Python dependency at runtime.

## Layout

```
cpp_server/
├── include/             # Public headers (csv_loader, feature_engineer, predictor, ...)
├── src/                 # Implementation files
├── tests/               # Compile-only unit tests
└── CMakeLists.txt
```

## Build

```bash
cmake -S . -B build \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.17.0 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To compile the unit tests, add `-DSTOCKVISION_BUILD_TESTS=ON` and run
`ctest --test-dir build`.

## Run

```bash
./build/stock_server \
    --host 0.0.0.0 \
    --port 8080 \
    --artifacts-dir ../ingest_train/artifacts \
    --default-model lstm
```

Set `STOCKVISION_DATA_DIR` to point to the directory of cached CSV files
(default: sibling of `--artifacts-dir`, named `data/`).

## Endpoints

| Method | Path        | Query                                    | Description                         |
| ------ | ----------- | ---------------------------------------- | ----------------------------------- |
| GET    | `/health`   | -                                        | Liveness probe.                     |
| GET    | `/history`  | `ticker`, `days`                         | Returns recent close prices.        |
| GET    | `/predict`  | `ticker`, `model`, `days`                | Returns the next-step prediction.   |

The server expects the artefacts to live under
`<artifacts-dir>/<TICKER>/<MODEL>/{model.onnx, params.txt}`. Both files are
produced by `stockvision train`.
