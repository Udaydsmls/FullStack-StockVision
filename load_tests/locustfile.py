"""Locust users hammering each of the three serving backends in parallel."""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

TICKERS = os.environ.get("LOADTEST_TICKERS", "AAPL,MSFT,TSLA,NVDA,GOOG").split(",")
MODELS = os.environ.get("LOADTEST_MODELS", "lstm,gru,transformer").split(",")


def _params() -> dict[str, str]:
    return {
        "ticker": random.choice(TICKERS),
        "model": random.choice(MODELS),
        "days": "60",
    }


class FastAPIUser(HttpUser):
    """Targets the Python FastAPI service on :8000."""

    host = os.environ.get("LOADTEST_FASTAPI", "http://localhost:8000")
    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        self.client.get("/predict", params=_params(), name="fastapi:/predict")


class CppServerUser(HttpUser):
    """Targets the hand-rolled C++ server on :8080."""

    host = os.environ.get("LOADTEST_CPP", "http://localhost:8080")
    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        self.client.get("/predict", params=_params(), name="cpp:/predict")


class TritonUser(HttpUser):
    """Targets the FastAPI shim that proxies to Triton; tests Triton's HTTP path indirectly."""

    host = os.environ.get("LOADTEST_TRITON_PROXY", "http://localhost:8000")
    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        self.client.get("/predict/triton", params=_params(), name="triton:/predict")
