import pandas as pd
import pytest

from src.data import clean_frame, split_frame


def make_df(n_per_class: int = 20) -> pd.DataFrame:
    rows = [
        {"text": f"review {label} {i}", "label": label}
        for label in ["Negative", "Mixed", "Positive"]
        for i in range(n_per_class)
    ]
    return pd.DataFrame(rows)


def test_clean_frame_lowercases_labels():
    out = clean_frame(make_df())
    assert set(out["label"]) == {"negative", "mixed", "positive"}


def test_clean_frame_drops_empty_and_duplicate_text():
    df = pd.DataFrame(
        {
            "text": ["good", "  ", "good", "bad"],
            "label": ["Positive", "Mixed", "Positive", "Negative"],
        }
    )
    out = clean_frame(df)
    assert list(out["text"]) == ["good", "bad"]


def test_clean_frame_rejects_unknown_label():
    df = pd.DataFrame({"text": ["x"], "label": ["Neutral"]})
    with pytest.raises(ValueError):
        clean_frame(df)


def test_split_keeps_every_class_and_never_overlaps():
    splits = split_frame(clean_frame(make_df(50)))
    train, val, test = splits["train"], splits["val"], splits["test"]
    assert len(train) + len(val) + len(test) == 150
    assert not set(train["text"]) & set(val["text"])
    assert not set(train["text"]) & set(test["text"])
    assert not set(val["text"]) & set(test["text"])
    for part in splits.values():
        assert set(part["label"]) == {"negative", "mixed", "positive"}
