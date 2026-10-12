import mlflow
import pytest
from mlflow import MlflowClient

from src.config import EXPERIMENT_NAME
from src.tracking import git_state, log_metric, require_tracking_uri, tracked_run


def test_tracked_run_records_params_metrics_and_git_tag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep MLflow's files inside the temporary folder
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)

    with tracked_run("unit-test", {"lr": 2e-5, "model_name": "some-model"}):
        log_metric("f1_macro", 0.5)

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.data.params["model_name"] == "some-model"
    assert run.data.params["lr"] == "2e-05"
    assert run.data.metrics["f1_macro"] == 0.5
    assert "git_commit" in run.data.tags


def test_log_metric_without_an_open_run_does_nothing():
    log_metric("anything", 1.0)  # must not raise
    assert mlflow.active_run() is None


def test_git_state_returns_text():
    assert isinstance(git_state(), str)


def test_missing_tracking_address_stops_with_a_message(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(SystemExit):
        require_tracking_uri()
