"""FastAPI backend.

The C++ server and Triton answer the same paths with the same JSON, so the
frontend can switch between all three without changing anything else.
"""

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config, inference, triton
from .data import DataFetchError
from .models import available_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="StockVision API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

TICKER = Query(..., min_length=1, max_length=10)
MODEL = Query(config.DEFAULT_MODEL)
DAYS = Query(config.HISTORY_DAYS, ge=5, le=720)


@app.get("/health")
def health():
    return {"status": "ok", "models": available_models()}


@app.get("/history")
def history(ticker: str = TICKER, days: int = DAYS):
    return inference.get_history(ticker, days)


@app.get("/predict")
def predict(ticker: str = TICKER, model: str = MODEL, days: int = DAYS):
    check_model(model)
    return inference.predict(ticker, model, days)


@app.get("/predict/triton")
def predict_via_triton(ticker: str = TICKER, model: str = MODEL, days: int = DAYS):
    check_model(model)
    return triton.predict(ticker, model, days)


@app.get("/explain")
def explain(ticker: str = TICKER, model: str = MODEL):
    check_model(model)
    from .explain import explain as run_explain

    return run_explain(ticker, model)


def check_model(name):
    if name not in available_models():
        raise HTTPException(400, f"Unknown model '{name}'. Available: {available_models()}")


# One handler per failure mode, instead of the same try/except in every route.
def _error(status, exc):
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
def handle_untrained_model(request, exc):
    return _error(404, exc)


@app.exception_handler(DataFetchError)
def handle_bad_ticker(request, exc):
    return _error(502, exc)


@app.exception_handler(triton.TritonUnavailable)
def handle_triton_down(request, exc):
    return _error(503, exc)
