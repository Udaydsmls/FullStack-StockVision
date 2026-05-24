from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Iterator, Mapping


@contextlib.contextmanager
def track_run(model_name: str, ticker: str, params: Mapping[str, Any]) -> Iterator["_RunHandle"]:
    """Wrap a training call so it logs to MLflow without touching the loop.

    Usage:
        with track_run("lstm", "AAPL", {"window": 30}) as run:
            result = train_and_export(...)
            run.log_metrics(result.metrics.to_dict())
            run.log_artifact(result.onnx_path)
    """
    try:
        import mlflow
    except ImportError:
        yield _NullRun()
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"stockvision/{ticker.upper()}")
    with mlflow.start_run(run_name=f"{ticker.upper()}-{model_name}") as run:
        mlflow.set_tags({"ticker": ticker.upper(), "model": model_name})
        mlflow.log_params(dict(params))
        handle = _MlflowRun(run, mlflow)
        try:
            yield handle
        finally:
            handle.flush()


class _RunHandle:
    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...
    def log_artifact(self, path: Path | str) -> None: ...


class _NullRun(_RunHandle):
    """Fallback used when mlflow isn't installed."""

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        return

    def log_artifact(self, path: Path | str) -> None:
        return


class _MlflowRun(_RunHandle):
    def __init__(self, run, mlflow_module) -> None:
        self._run = run
        self._mlflow = mlflow_module
        self._pending_metrics: dict[str, float] = {}
        self._pending_artifacts: list[Path] = []

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self._pending_metrics.update(clean)

    def log_artifact(self, path: Path | str) -> None:
        p = Path(path)
        if p.exists():
            self._pending_artifacts.append(p)

    def flush(self) -> None:
        if self._pending_metrics:
            self._mlflow.log_metrics(self._pending_metrics)
        for artifact in self._pending_artifacts:
            try:
                self._mlflow.log_artifact(str(artifact))
            except Exception:
                pass
