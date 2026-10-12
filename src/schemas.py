"""Data shapes shared across the project."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    """One model answer: the winning class and how sure the model is."""

    label: str
    confidence: float
