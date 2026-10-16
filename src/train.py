"""Fine-tune AraBERT on the review splits, logging every run to MLflow.

Run from the project root, with MLFLOW_TRACKING_URI set (see README), or add --no-mlflow:
    python -m src.train --run-name quickcheck --max-train-samples 2000 --epochs 1
    python -m src.train --run-name baseline --output-dir models/baseline
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import (
    BASE_MODEL,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MAX_LENGTH,
    MODEL_DIR,
    PROCESSED_DIR,
    SEED,
)
from src.text import clean_text
from src.tracking import (
    log_metric,
    log_text,
    require_tracking_uri,
    tracked_run,
)


@dataclass
class TrainConfig:
    """Every setting of one training run. MLflow logs exactly these."""

    data_dir: Path = PROCESSED_DIR
    output_dir: Path = MODEL_DIR
    base_model: str = BASE_MODEL
    epochs: int = 2
    batch_size: int = 16
    lr: float = 5e-5
    max_length: int = MAX_LENGTH
    max_train_samples: int | None = None  # None = use all training rows
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = SEED
    evaluate_test: bool = False  # the test set is opened only when we ask for it


def set_seed(seed: int) -> None:
    """Make the random parts of training repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReviewDataset(Dataset):
    """Cleaned review texts with their label numbers."""

    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def load_split(path: Path, limit: int | None = None, seed: int = SEED) -> ReviewDataset:
    """Read one split CSV, clean every review, and turn label names into numbers."""
    df = pd.read_csv(path)
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed)
    texts = [clean_text(text) for text in df["text"].astype(str)]
    labels = [LABEL2ID[label] for label in df["label"]]
    return ReviewDataset(texts, labels)


def make_collate(tokenizer: Any, max_length: int):
    """Build the function that turns a list of reviews into one padded batch."""

    def collate(batch: list[tuple[str, int]]):
        texts, labels = zip(*batch, strict=True)
        encoded = tokenizer(
            list(texts),
            padding=True,  # pad only up to the longest review in THIS batch
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels)
        return encoded

    return collate


@torch.inference_mode()
def evaluate(model: Any, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Score the model on a split: accuracy, macro-F1 and the raw predictions."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(**batch).logits
        y_pred.extend(logits.argmax(dim=-1).tolist())
        y_true.extend(labels.tolist())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def format_report(result: dict[str, Any]) -> str:
    """The per-class precision / recall / F1 table, as text."""
    return classification_report(
        result["y_true"],
        result["y_pred"],
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        digits=4,
    )


def train(cfg: TrainConfig) -> dict[str, float]:
    """Fine-tune, keep the best epoch on disk, and return its metrics."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("WARNING: no GPU found, so training will be very slow.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    print("Cleaning the text (this takes a minute)...")
    train_ds = load_split(cfg.data_dir / "train.csv", cfg.max_train_samples, cfg.seed)
    val_ds = load_split(cfg.data_dir / "val.csv", None, cfg.seed)
    print(f"Training on {len(train_ds)} reviews, validating on {len(val_ds)}")

    collate = make_collate(tokenizer, cfg.max_length)
    pin = device.type == "cuda"
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, pin_memory=pin
    )
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, collate_fn=collate, pin_memory=pin)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_dl) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )
    use_amp = device.type == "cuda"  # fp16 mixed precision: faster and lighter on memory
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_epoch = -1.0, 0
    best_val: dict[str, Any] = {}
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_dl, start=1):
            global_step += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stops exploding updates
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            if step % 200 == 0 or step == len(train_dl):
                avg = running_loss / step
                print(f"epoch {epoch}/{cfg.epochs}  step {step}/{len(train_dl)}  loss {avg:.4f}")
                log_metric("train_loss", avg, step=global_step)

        val = evaluate(model, val_dl, device)
        print(
            f"epoch {epoch}: validation accuracy {val['accuracy']:.4f}, "
            f"macro-F1 {val['f1_macro']:.4f}"
        )
        log_metric("val_accuracy", val["accuracy"], step=epoch)
        log_metric("val_f1_macro", val["f1_macro"], step=epoch)
        if val["f1_macro"] > best_f1:  # keep only the best epoch
            best_f1, best_epoch, best_val = val["f1_macro"], epoch, val
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(cfg.output_dir)
            tokenizer.save_pretrained(cfg.output_dir)

    train_seconds = time.perf_counter() - started
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if use_amp else 0.0
    print(f"Best epoch: {best_epoch}. Training took {train_seconds / 60:.1f} minutes.")
    print(f"Peak GPU memory: {peak_vram_mb:.0f} MB")
    report = format_report(best_val)
    print(report)
    log_text(report, "val_report.txt")

    metrics: dict[str, float] = {
        "best_epoch": float(best_epoch),
        "val_accuracy": best_val["accuracy"],
        "val_f1_macro": best_val["f1_macro"],
        "train_seconds": train_seconds,
        "peak_vram_mb": peak_vram_mb,
    }

    if cfg.evaluate_test:  # the final exam: run it once, after all decisions are made
        del model, optimizer
        if use_amp:
            torch.cuda.empty_cache()
        best_model = AutoModelForSequenceClassification.from_pretrained(cfg.output_dir).to(device)
        test_ds = load_split(cfg.data_dir / "test.csv", None, cfg.seed)
        test_dl = DataLoader(
            test_ds, batch_size=cfg.batch_size * 2, collate_fn=collate, pin_memory=pin
        )
        test = evaluate(best_model, test_dl, device)
        print(f"TEST accuracy {test['accuracy']:.4f}, macro-F1 {test['f1_macro']:.4f}")
        test_report = format_report(test)
        print(test_report)
        log_text(test_report, "test_report.txt")
        metrics["test_accuracy"] = test["accuracy"]
        metrics["test_f1_macro"] = test["f1_macro"]

    record = {"config": {k: str(v) for k, v in asdict(cfg).items()}, "metrics": metrics}
    (cfg.output_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return metrics


def train_tracked(cfg: TrainConfig, run_name: str) -> dict[str, float]:
    """Train, log a Transformers model, and record experiment metrics."""
    params = {
        "model_name": cfg.base_model,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "max_length": cfg.max_length,
        "max_train_samples": cfg.max_train_samples or "all",
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
    }

    with tracked_run(run_name, params):
        metrics = train(cfg)

        # Log metrics before model packaging, so they are recorded even
        # if MLflow fails to package the model.
        log_metric("accuracy", metrics["val_accuracy"])
        log_metric("f1_macro", metrics["val_f1_macro"])

        for name in ("best_epoch", "train_seconds", "peak_vram_mb"):
            log_metric(name, metrics[name])

        for name in ("test_accuracy", "test_f1_macro"):
            if name in metrics:
                log_metric(name, metrics[name])

        # Reload the BEST saved epoch, not the final in-memory epoch.
        saved_model = AutoModelForSequenceClassification.from_pretrained(cfg.output_dir)
        saved_tokenizer = AutoTokenizer.from_pretrained(cfg.output_dir)

        # Log the Transformers model with explicit dependencies.
        model_info = mlflow.transformers.log_model(
            transformers_model={
                "model": saved_model,
                "tokenizer": saved_tokenizer,
            },
            name="model",
            task="text-classification",
            pip_requirements=[
                "mlflow==3.17.1",
                "torch==2.14.1",
                "transformers==5.19.0",
                "safetensors",
                "huggingface-hub",
                "tokenizers",
            ],
        )

        # Keep a reference to the logged model for later registration.
        mlflow.set_tag("logged_model_uri", model_info.model_uri)

    return metrics


def parse_args() -> tuple[TrainConfig, str, bool]:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Fine-tune AraBERT for Arabic review sentiment.")
    parser.add_argument("--run-name", default="run", help="the name shown in the MLflow table")
    parser.add_argument("--no-mlflow", action="store_true", help="train without logging to MLflow")
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--base-model", default=defaults.base_model)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=defaults.weight_decay,
    )
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--evaluate-test", action="store_true", help="also score the held-out test split"
    )
    args = vars(parser.parse_args())
    run_name = args.pop("run_name")
    no_mlflow = args.pop("no_mlflow")
    return TrainConfig(**args), run_name, no_mlflow


if __name__ == "__main__":
    config, name, skip_tracking = parse_args()
    if skip_tracking:
        train(config)
    else:
        print(f"Logging to MLflow at {require_tracking_uri()}")
        train_tracked(config, name)
