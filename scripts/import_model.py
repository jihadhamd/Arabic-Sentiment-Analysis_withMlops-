"""Import the DVC-trained model into the local BentoML model store."""

import json
import shutil
from pathlib import Path

import bentoml

MODEL_DIR = Path("models/dvc-training")
MODEL_NAME = "arabic_sentiment"


def main() -> None:
    if not (MODEL_DIR / "model.safetensors").exists():
        raise SystemExit(f"{MODEL_DIR} is missing the weights: run `dvc pull` first")

    metrics_file = MODEL_DIR / "metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}

    with bentoml.models.create(
        name=MODEL_NAME,
        labels={"task": "arabic-sentiment"},
        metadata=metrics,
    ) as bento_model:
        for item in MODEL_DIR.iterdir():
            if item.is_file():
                shutil.copy2(item, Path(bento_model.path) / item.name)
        print(f"Saved {bento_model.tag}")


if __name__ == "__main__":
    main()
