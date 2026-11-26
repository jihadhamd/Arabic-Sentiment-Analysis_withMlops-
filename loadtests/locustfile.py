# loadtests/locustfile.py
import csv
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

# TARGET=fastapi (default) or TARGET=bento
TARGET = os.getenv("TARGET", "fastapi")
HEALTH_PATH = "/healthz" if TARGET == "bento" else "/health"

DATA_PATH = Path("data/processed/test.csv")
TEXT_COLUMN = "text"
EXPECTED_STATUS = 200

random.seed(42)


def load_texts(path: Path, column: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: fix DATA_PATH")
    with path.open(encoding="utf-8", newline="") as f:
        texts = [row[column].strip() for row in csv.DictReader(f) if row.get(column)]
    if not texts:
        raise ValueError(f"No texts read from column '{column}'")
    print(f"Loaded {len(texts)} texts for load testing (target: {TARGET})")
    return texts


TEXTS = load_texts(DATA_PATH, TEXT_COLUMN)


def build_payload(text: str) -> dict:
    # FastAPI takes one text; BentoML's batchable endpoint takes a list
    return {"texts": [text]} if TARGET == "bento" else {"text": text}


def get_label(body):
    return body[0].get("label") if TARGET == "bento" else body.get("label")


class SentimentUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(weight=10)
    def predict(self):
        payload = build_payload(random.choice(TEXTS))
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code != EXPECTED_STATUS:
                resp.failure(f"Expected {EXPECTED_STATUS}, got {resp.status_code}")
            elif get_label(resp.json()) is None:
                resp.failure("No label in response")

    @task(weight=1)
    def health(self):
        self.client.get(HEALTH_PATH)
