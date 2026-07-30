"""Load test that hits all three serving backends at the same time."""

import os
import random

from locust import HttpUser, between, task

TICKERS = os.getenv("LOADTEST_TICKERS", "AAPL,MSFT,TSLA,NVDA,GOOG").split(",")
MODELS = os.getenv("LOADTEST_MODELS", "lstm,gru,transformer").split(",")


class BackendUser(HttpUser):
    """Shared behaviour. Each subclass below points at one backend."""

    abstract = True
    wait_time = between(0.1, 0.5)
    path = "/predict"
    backend = "backend"

    @task
    def predict(self):
        params = {"ticker": random.choice(TICKERS), "model": random.choice(MODELS), "days": 60}
        # The name is what shows up as a row in the Locust stats CSV.
        self.client.get(self.path, params=params, name=f"{self.backend}:/predict")


class FastApiUser(BackendUser):
    host = os.getenv("LOADTEST_FASTAPI", "http://localhost:8000")
    backend = "fastapi"


class CppUser(BackendUser):
    host = os.getenv("LOADTEST_CPP", "http://localhost:8080")
    backend = "cpp"


class TritonUser(BackendUser):
    host = os.getenv("LOADTEST_TRITON", "http://localhost:8000")
    path = "/predict/triton"
    backend = "triton"
