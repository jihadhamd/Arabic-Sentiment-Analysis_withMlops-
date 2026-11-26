import bentoml

from src.model import SentimentClassifier


@bentoml.service(workers=1)
class Sentiment:
    # Resolves the stored model version and gives us its folder on disk
    bento_model = bentoml.models.BentoModel("arabic_sentiment:latest")

    def __init__(self) -> None:
        # Runs once per worker: load the model into the GPU a single time
        self.classifier = SentimentClassifier(model_dir=self.bento_model.path).load()

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=32, max_latency_ms=5000)
    def predict(self, texts: list[str]) -> list[dict]:
        print("batch size:", len(texts))  # temporary: delete after you've watched it
        results = self.classifier.predict(texts)
        return [{"label": r.label, "confidence": r.confidence} for r in results]
