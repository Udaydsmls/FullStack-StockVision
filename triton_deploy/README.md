# Triton Inference Server Deployment

A third serving option alongside the FastAPI service and the C++ server.
Triton consumes the same ONNX files produced by `stockvision train`.

## Workflow

```bash
# 1. Train at least one model
cd ingest_train
stockvision train AAPL --model lstm

# 2. Lay out the Triton model repository
cd ../triton_deploy
python setup_triton_repo.py \
    --artifacts ../ingest_train/artifacts \
    --output model_repository

# 3. Boot the Triton container
docker compose -f docker-compose.triton.yml up

# 4. Send a request
python triton_client.py --ticker AAPL --model lstm
```

The FastAPI service also exposes `GET /predict/triton?ticker=...&model=...`
which calls Triton on `localhost:8001`.

## Layout

```
triton_deploy/
├── model_repository/
│   └── <model_name>/
│       ├── 1/model.onnx
│       └── config.pbtxt
├── setup_triton_repo.py
├── triton_client.py
└── docker-compose.triton.yml
```
