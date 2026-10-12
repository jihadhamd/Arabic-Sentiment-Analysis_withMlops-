"""The FastAPI service. `create_app` is a factory so tests can pass in a fake model."""

from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool

from src.config import MODEL_VERSION
from src.schemas import HealthResponse, Prediction, PredictRequest, PredictResponse


class Predictor(Protocol):
    """Anything with this method can serve predictions: the real model or a test fake."""

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...


def create_app(predictor: Predictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is not None:
            app.state.predictor = predictor
        else:
            from src.model import SentimentClassifier  # imported here because torch is heavy

            app.state.predictor = SentimentClassifier().load()  # once, at startup
        yield

    app = FastAPI(title="Arabic Sentiment API", version=MODEL_VERSION, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="healthy")

    @app.post("/predict", response_model=PredictResponse)
    async def predict(body: PredictRequest, request: Request) -> PredictResponse:
        # Model inference blocks. Running it directly inside `async def` would freeze the
        # server for every other client, so it goes to a worker thread.
        results = await run_in_threadpool(request.app.state.predictor.predict, [body.text])
        result = results[0]
        return PredictResponse(
            label=result.label,
            confidence=result.confidence,
            model_version=MODEL_VERSION,
        )

    return app


app = create_app()  # uvicorn src.api:app
