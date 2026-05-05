from __future__ import annotations

from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import AppConfig, load_config
from .data import DataFetchError, MarketDataRepository
from .inference import PredictionService
from .logging_setup import configure_logging, get_logger
from .models import available_models

log = get_logger(__name__)


class PredictionResponse(BaseModel):
    ticker: str
    model: str
    prediction: float
    last_close: float
    history: List[float] = Field(default_factory=list)
    history_dates: List[str] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    ticker: str
    history: List[float]
    history_dates: List[str]


class HealthResponse(BaseModel):
    status: str
    models: List[str]


@lru_cache(maxsize=1)
def _service_singleton(_cache_key: int = 0) -> PredictionService:
    cfg = load_config()
    return PredictionService(cfg, MarketDataRepository(cfg.data))


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    configure_logging(cfg.service.log_level)
    app = FastAPI(title="StockVision API", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )

    service = _service_singleton()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", models=available_models())

    @app.get("/history", response_model=HistoryResponse)
    def history(
        ticker: str = Query(..., min_length=1, max_length=10),
        days: int = Query(60, ge=5, le=720),
    ) -> HistoryResponse:
        try:
            payload = service.history(ticker, history_size=days)
        except DataFetchError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return HistoryResponse(**payload)

    @app.get("/predict", response_model=PredictionResponse)
    def predict(
        ticker: str = Query(..., min_length=1, max_length=10),
        model: str = Query(default=cfg.service.default_model),
        days: int = Query(60, ge=5, le=720),
    ) -> PredictionResponse:
        if model not in available_models():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model}'. Available: {available_models()}",
            )
        try:
            result = service.predict(ticker, model, history_size=days)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DataFetchError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return PredictionResponse(
            ticker=result.ticker,
            model=result.model,
            prediction=result.prediction,
            last_close=result.last_close,
            history=result.history,
            history_dates=result.history_dates,
        )

    return app


app = create_app()
