"""Model tests build a tiny random model, so they need no download and no GPU."""

import pytest
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from src.config import LABELS
from src.model import SentimentClassifier


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-model")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "ممتاز", "سيء", "جيد"]
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(path / "vocab.txt"))
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=len(LABELS),
    )
    BertForSequenceClassification(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def classifier(tiny_model_dir):
    return SentimentClassifier(model_dir=tiny_model_dir, device="cpu").load()


def test_predict_returns_one_valid_prediction_per_text(classifier):
    results = classifier.predict(["ممتاز", "سيء جدا", "جيد"])
    assert len(results) == 3
    for result in results:
        assert result.label in LABELS
        assert 0.0 <= result.confidence <= 1.0


def test_batching_does_not_change_the_answer(classifier):
    alone = classifier.predict(["ممتاز"])[0]
    in_batch = classifier.predict(["ممتاز", "سيء سيء سيء جيد جيد"])[0]
    assert alone.confidence == pytest.approx(in_batch.confidence, abs=1e-3)


def test_empty_input_gives_empty_output(classifier):
    assert classifier.predict([]) == []


def test_predict_before_load_raises(tiny_model_dir):
    unloaded = SentimentClassifier(model_dir=tiny_model_dir, device="cpu")
    with pytest.raises(RuntimeError):
        unloaded.predict(["ممتاز"])


def test_missing_model_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SentimentClassifier(model_dir=tmp_path / "does-not-exist").load()
