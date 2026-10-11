"""Central settings. Every other module imports from here."""

import os
from pathlib import Path

# --- Labels -------------------------------------------------------------
# The ORDER matters: the trained model stores class numbers (0, 1, 2), not
# names. Changing this order after training would silently swap the labels.
LABELS: tuple[str, ...] = ("negative", "mixed", "positive")
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

# --- Model --------------------------------------------------------------
BASE_MODEL: str = os.getenv("BASE_MODEL", "aubmindlab/bert-base-arabertv02")
MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "128"))
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "0.1.0")

# --- Paths (relative to the project root: run commands from there) -------
RAW_DATA_PATH = Path("data/raw/ar_reviews_100k.tsv")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/arabert-sentiment"))
