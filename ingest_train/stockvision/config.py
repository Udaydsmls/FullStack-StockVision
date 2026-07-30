"""All tunable settings in one place. Every value can be overridden with an env var."""

import os
from pathlib import Path

# This file lives in ingest_train/stockvision/, so the package root is two levels up.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SWEEP_CONFIG = BASE_DIR / "sweep_config.yaml"

# Data download
PERIOD = os.getenv("STOCKVISION_PERIOD", "2y")
INTERVAL = os.getenv("STOCKVISION_INTERVAL", "1d")
CACHE_TTL_SECONDS = 6 * 60 * 60

# Training
WINDOW = int(os.getenv("STOCKVISION_WINDOW", "30"))
HORIZON = 1
EPOCHS = int(os.getenv("STOCKVISION_EPOCHS", "25"))
BATCH_SIZE = int(os.getenv("STOCKVISION_BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("STOCKVISION_LEARNING_RATE", "0.001"))
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
PATIENCE = 5
SEED = 42

# Serving
HOST = os.getenv("STOCKVISION_HOST", "0.0.0.0")
PORT = int(os.getenv("STOCKVISION_PORT", "8000"))
DEFAULT_MODEL = os.getenv("STOCKVISION_DEFAULT_MODEL", "lstm")
HISTORY_DAYS = 60

# Optional extras
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
USE_FEAST = os.getenv("STOCKVISION_USE_FEAST", "") == "1"


def model_dir(ticker, model_name):
    """Folder holding the artifacts for one (ticker, model) pair."""
    return ARTIFACTS_DIR / ticker.upper() / model_name
