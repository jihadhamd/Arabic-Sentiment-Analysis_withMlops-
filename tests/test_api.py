from typing import get_args

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.config import LABELS
from src.schemas import Label, Prediction


class FakePredictor:
    """Stands in for the real model, so API tests run instantly and need no GPU."""

    def predict(self, texts):
        return [Prediction(label="positive", confidence=0.94) for _ in texts]


@pytest.fixture()
def client():
    with TestClient(create_app(FakePredictor())) as test_client:  # `with` runs startup
        yield test_client


def test_health_returns_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_label_confidence_and_version(client):
    response = client.post("/predict", json={"text": "الفندق ممتاز ونظيف"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert body["confidence"] == 0.94
    assert body["model_version"]


def test_predict_accepts_a_one_word_review(client):
    assert client.post("/predict", json={"text": "ممتاز"}).status_code == 200


@pytest.mark.parametrize("bad_body", [{"text": ""}, {"text": "   "}, {"text": None}, {}])
def test_predict_rejects_empty_or_missing_text(client, bad_body):
    assert client.post("/predict", json=bad_body).status_code == 422


def test_predict_rejects_absurdly_long_text(client):
    assert client.post("/predict", json={"text": "ا" * 10_001}).status_code == 422


def test_label_type_matches_config():
    assert get_args(Label) == LABELS
