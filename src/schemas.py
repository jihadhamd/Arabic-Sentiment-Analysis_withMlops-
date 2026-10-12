"""Data shapes shared across the project."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Label = Literal["negative", "mixed", "positive"]


@dataclass(frozen=True)
class Prediction:
    """One model answer: the winning class and how sure the model is."""

    label: str
    confidence: float


class PredictRequest(BaseModel):
    """What a client sends to /predict."""

    # The longest review in our data has 8,163 characters, so 10,000 never rejects a real one.
    text: str = Field(..., min_length=1, max_length=10_000, examples=["الفندق ممتاز ونظيف"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value.strip()


class PredictResponse(BaseModel):
    """What /predict sends back."""

    label: Label
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["healthy"]
