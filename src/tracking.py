"""A thin layer over MLflow, so the training code stays easy to read."""

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow

from src.config import EXPERIMENT_NAME


def git_state() -> str:
    """The short commit hash, with '-dirty' added if tracked files have uncommitted changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if changes else commit


def require_tracking_uri() -> str:
    """Return the MLflow server address, or stop with a clear message if it is not set."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit(
            "MLFLOW_TRACKING_URI is not set, so we would not know where to log.\n"
            "  DagsHub:  set -a; source .env; set +a\n"
            "  Local:    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000\n"
            "  Or run without tracking: add --no-mlflow"
        )
    return uri


@contextmanager
def tracked_run(run_name: str, params: dict[str, Any]) -> Iterator[None]:
    """Open an MLflow run, record its settings and the code version, and close it afterwards."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("git_commit", git_state())
        mlflow.log_params({key: str(value) for key, value in params.items()})
        yield


def log_metric(name: str, value: float, step: int | None = None) -> None:
    """Log one number. Does nothing when no run is open, so training also works without MLflow."""
    if mlflow.active_run() is not None:
        mlflow.log_metric(name, value, step=step)


def log_text(text: str, artifact_file: str) -> None:
    """Save a piece of text as a file attached to the current run (if there is one)."""
    if mlflow.active_run() is not None:
        mlflow.log_text(text, artifact_file)
