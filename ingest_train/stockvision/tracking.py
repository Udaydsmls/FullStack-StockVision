"""MLflow tracking. Every training run is logged if mlflow is installed."""

import logging

from . import config

log = logging.getLogger(__name__)


def log_to_mlflow(result):
    """Log one run's hyper-parameters, metrics and artifacts."""
    try:
        import mlflow
    except ImportError:
        log.info("mlflow is not installed, skipping tracking")
        return

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"stockvision/{result['ticker']}")
    with mlflow.start_run(run_name=f"{result['ticker']}-{result['model']}"):
        mlflow.log_params(result["params"])
        mlflow.log_metrics(result["metrics"])
        for path in result["artifacts"]:
            mlflow.log_artifact(str(path))
    log.info("Logged run to MLflow at %s", config.MLFLOW_TRACKING_URI)
