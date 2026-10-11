"""Turn the raw TSV into clean train / validation / test CSV files.

Run from the project root:  python -m src.data
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABELS, PROCESSED_DIR, RAW_DATA_PATH, SEED


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the tab-separated file and check it has the columns we expect."""
    df = pd.read_csv(path, sep="\t")
    missing = {"label", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}. Found {list(df.columns)}")
    return df[["text", "label"]]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise labels, then drop empty and duplicate reviews (and say how many)."""
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels {unknown}. Expected {list(LABELS)}")

    df["text"] = df["text"].astype(str).str.strip()
    start = len(df)
    df = df[df["text"] != ""]
    empty_dropped = start - len(df)
    df = df.drop_duplicates(subset="text")
    duplicates_dropped = start - empty_dropped - len(df)
    print(f"Kept {len(df)} rows (dropped {empty_dropped} empty, {duplicates_dropped} duplicates)")
    return df.reset_index(drop=True)


def split_frame(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """80% train, 10% validation, 10% test, with the same class mix in each."""
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
    return {"train": train, "val": val, "test": test}


def prepare(
    raw_path: Path = RAW_DATA_PATH, out_dir: Path = PROCESSED_DIR, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Load, clean, split, and write one CSV per split."""
    splits = split_frame(clean_frame(load_raw(raw_path)), seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        counts = part["label"].value_counts().reindex(LABELS)
        mix = ", ".join(f"{label}={counts[label]}" for label in LABELS)
        print(f"{name:5s} {len(part):6d} rows | {mix}")
    return splits


if __name__ == "__main__":
    prepare()
