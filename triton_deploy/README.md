# Triton Inference Server

The third serving backend. Triton runs the same ONNX files that
`stockvision train` produced for FastAPI and the C++ server.

## Workflow

```bash
# 1. Train at least one model
cd ingest_train
stockvision train AAPL --model lstm

# 2. Lay out the Triton model repository
cd ../triton_deploy
python setup_triton_repo.py --artifacts ../ingest_train/artifacts

# 3. Start Triton
docker compose -f docker-compose.triton.yml up
```

`setup_triton_repo.py` reads each `metadata.json` so the generated
`config.pbtxt` uses the tensor names and shapes the exported graph actually
has.

Requests go through the FastAPI service, which scales the features locally and
forwards the tensor:

```
GET http://localhost:8000/predict/triton?ticker=AAPL&model=lstm
```

Set `TRITON_URL` if Triton is not on `localhost:8000`.

## Layout

```
triton_deploy/
├── model_repository/          # generated
│   └── <ticker>_<model>/
│       ├── 1/model.onnx
│       └── config.pbtxt
├── setup_triton_repo.py
└── docker-compose.triton.yml
```
