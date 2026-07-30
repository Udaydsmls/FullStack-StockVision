# StockVision C++ inference server

An HTTP server built directly on ONNX Runtime and a single-header HTTP library,
with no web framework. It recomputes the technical indicators in C++, so there
is no Python at runtime.

## Layout

```
cpp_server/
├── include/
│   ├── csv_loader.h    # read the cached OHLCV CSVs
│   ├── features.h      # the same 17 columns as stockvision/features.py
│   ├── predictor.h     # ONNX Runtime sessions + the saved scaler
│   ├── server.h        # settings + run_server()
│   └── httplib.h       # third-party single-header HTTP library
├── src/                # one .cpp per header, plus main.cpp for the CLI flags
└── tests/
```

## Build

```bash
cmake -S . -B build \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.17.0 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Add `-DSTOCKVISION_BUILD_TESTS=ON` and run `ctest --test-dir build` for the
CSV and feature tests. They do not need ONNX Runtime.

## Run

```bash
./build/stock_server \
    --port 8080 \
    --artifacts-dir ../ingest_train/artifacts \
    --data-dir ../ingest_train/data
```

| Flag              | Default     | Purpose                                   |
| ----------------- | ----------- | ----------------------------------------- |
| `--host`          | `0.0.0.0`   | Bind address.                             |
| `--port`          | `8080`      | Bind port.                                |
| `--artifacts-dir` | `artifacts` | Where `stockvision train` wrote the ONNX. |
| `--data-dir`      | `data`      | Where the cached CSVs live.               |
| `--default-model` | `lstm`      | Used when a request omits `model`.        |
| `--history-days`  | `60`        | How many closes to return.                |

## Endpoints

`GET /health`, `GET /history?ticker=&days=`, and
`GET /predict?ticker=&model=&days=` — the same paths and the same JSON as the
FastAPI backend.

The server reads `<artifacts-dir>/<TICKER>/<MODEL>/model.onnx` and the matching
`params.txt`, both written by `stockvision train`.
