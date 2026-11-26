"""BentoML service using the FP16 TensorRT engine instead of PyTorch."""

from pathlib import Path

import bentoml
import tensorrt as trt
import torch
from transformers import AutoTokenizer

from src.config import ID2LABEL, MAX_LENGTH
from src.text import clean_text

ENGINE_PATH = Path("models/trt/arabic_sentiment_fp16.engine")
TOKENIZER_DIR = Path("models/dvc-training")  # tokenizer is unchanged by TensorRT


class TRTEngine:
    def __init__(self, path: Path, device: torch.device) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        with path.open("rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.device = device

    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.context.set_input_shape("input_ids", tuple(input_ids.shape))
        self.context.set_input_shape("attention_mask", tuple(attention_mask.shape))
        out_shape = tuple(self.context.get_tensor_shape("logits"))
        logits = torch.empty(out_shape, dtype=torch.float32, device=self.device)

        self.context.set_tensor_address("input_ids", input_ids.data_ptr())
        self.context.set_tensor_address("attention_mask", attention_mask.data_ptr())
        self.context.set_tensor_address("logits", logits.data_ptr())

        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return logits


@bentoml.service(workers=1)
class Sentiment:
    def __init__(self) -> None:
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        self.engine = TRTEngine(ENGINE_PATH, self.device)

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=32, max_latency_ms=5000)
    def predict(self, texts: list[str]) -> list[dict]:
        cleaned = [clean_text(t) for t in texts]
        enc = self.tokenizer(
            cleaned, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device).contiguous()
        attention_mask = enc["attention_mask"].to(self.device).contiguous()

        logits = self.engine.run(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        confidence, index = probs.max(dim=-1)
        return [
            {"label": ID2LABEL[i], "confidence": round(c, 4)}
            for i, c in zip(index.tolist(), confidence.tolist(), strict=True)
        ]
