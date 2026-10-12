import pandas as pd
import torch

from src.config import LABEL2ID
from src.train import load_split, set_seed


def write_split(path, n):
    labels = ["negative", "mixed", "positive"]
    rows = [{"text": f"المنتج   رائع {i}", "label": labels[i % 3]} for i in range(n)]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def test_load_split_cleans_text_and_maps_labels(tmp_path):
    path = tmp_path / "train.csv"
    write_split(path, 6)
    dataset = load_split(path)
    assert len(dataset) == 6
    text, label = dataset[0]
    assert "  " not in text  # the cleaning step collapsed the extra spaces
    assert label == LABEL2ID["negative"]


def test_load_split_respects_the_sample_limit(tmp_path):
    path = tmp_path / "train.csv"
    write_split(path, 10)
    assert len(load_split(path, limit=4)) == 4


def test_set_seed_makes_random_numbers_repeat():
    set_seed(7)
    first = torch.rand(3)
    set_seed(7)
    second = torch.rand(3)
    assert torch.equal(first, second)
