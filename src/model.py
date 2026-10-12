"""The AraBERT classifier: load it once, then predict for many reviews."""

from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config import ID2LABEL, MAX_LENGTH, MODEL_DIR
from src.schemas import Prediction
from src.text import clean_text


class SentimentClassifier:
    """Fine-tuned AraBERT that labels a review as negative, mixed or positive.

    Create it once when the service starts and reuse it for every request.
    Loading takes seconds. Predicting takes milliseconds.
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        max_length: int = MAX_LENGTH,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> "SentimentClassifier":
        """Read the tokenizer and weights from disk and move them to the device."""
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"No model found at '{self.model_dir}'. Train one first (python -m src.train)."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        if model.config.num_labels != len(ID2LABEL):
            raise ValueError(
                f"The model has {model.config.num_labels} classes, "
                f"but config.py defines {len(ID2LABEL)}."
            )
        self._model = model.to(self.device).eval()  # eval(): switches dropout off
        return self

    @torch.inference_mode()  # no gradients needed: faster and uses less memory
    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Return one Prediction per text, in the same order."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        if len(texts) == 0:
            return []

        cleaned = [clean_text(text) for text in texts]
        batch = self._tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        probabilities = torch.softmax(self._model(**batch).logits, dim=-1)
        confidence, index = probabilities.max(dim=-1)
        return [
            Prediction(label=ID2LABEL[i], confidence=round(c, 4))
            for i, c in zip(index.tolist(), confidence.tolist(), strict=True)
        ]
