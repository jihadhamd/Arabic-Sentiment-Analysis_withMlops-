# loadtests/locustfile.py
import csv
import random
from pathlib import Path

from locust import HttpUser, between, task

# --- Verify these against your project ---
DATA_PATH = Path("data/processed/test.csv")
TEXT_COLUMN = "text"  # <-- column that holds the review text
PAYLOAD_KEY = "text"  # request body field (see /docs)
RESPONSE_KEY = "label"  # response field (see /docs)
EXPECTED_STATUS = 200
# -----------------------------------------

random.seed(42)  # same random sequence every run, so before/after see the same input mix


def load_texts(path: Path, column: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: fix DATA_PATH")
    with path.open(encoding="utf-8", newline="") as f:
        texts = [row[column].strip() for row in csv.DictReader(f) if row.get(column)]
    if not texts:
        raise ValueError(f"No texts read from column '{column}'")
    print(f"Loaded {len(texts)} texts for load testing")
    return texts


TEXTS = load_texts(DATA_PATH, TEXT_COLUMN)  # loaded once, not on every request


class SentimentUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(weight=10)
    def predict(self):
        payload = {PAYLOAD_KEY: random.choice(TEXTS)}
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code != EXPECTED_STATUS:
                resp.failure(f"Expected {EXPECTED_STATUS}, got {resp.status_code}")
            elif RESPONSE_KEY not in resp.json():
                resp.failure(f"Missing '{RESPONSE_KEY}' in response")

    @task(weight=1)
    def health(self):
        self.client.get("/health")
