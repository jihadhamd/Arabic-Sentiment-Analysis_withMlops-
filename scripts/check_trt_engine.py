"""Compare the FP16 TensorRT engine with PyTorch on the whole test set."""

import csv
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("models/dvc-training")
ENGINE_PATH = Path("models/trt/arabic_sentiment_fp16.engine")
MAX_LENGTH = 256
BATCH = 32  # matches the engine's optimization profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "TensorRT needs a GPU; stop any BentoML server holding it first"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
torch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()


class TRTEngine:
    """Loads the engine once; run() takes token tensors, returns logits."""

    def __init__(self, path: Path) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        with path.open("rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.context.set_input_shape("input_ids", tuple(input_ids.shape))
        self.context.set_input_shape("attention_mask", tuple(attention_mask.shape))
        out_shape = tuple(self.context.get_tensor_shape("logits"))
        logits = torch.empty(out_shape, dtype=torch.float32, device=device)

        self.context.set_tensor_address("input_ids", input_ids.data_ptr())
        self.context.set_tensor_address("attention_mask", attention_mask.data_ptr())
        self.context.set_tensor_address("logits", logits.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        return logits


engine = TRTEngine(ENGINE_PATH)

with open("data/processed/test.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
texts = [r["text"] for r in rows]
label_names = sorted({r["label"] for r in rows})  # just for display, not used in comparisons

trt_preds, torch_preds, true_labels = [], [], []
for i in range(0, len(rows), BATCH):
    chunk = rows[i : i + BATCH]
    enc = tokenizer(
        [r["text"] for r in chunk],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device).contiguous()
    attention_mask = enc["attention_mask"].to(device).contiguous()

    with torch.no_grad():
        ref_logits = torch_model(input_ids=input_ids, attention_mask=attention_mask).logits
    trt_logits = engine.run(input_ids, attention_mask)

    torch_preds.extend(ref_logits.argmax(1).cpu().tolist())
    trt_preds.extend(trt_logits.argmax(1).cpu().tolist())
    true_labels.extend(int(torch_model.config.label2id[r["label"]]) for r in chunk)

    if i % (BATCH * 20) == 0:
        print(f"{i + len(chunk)}/{len(rows)} reviews done")

trt_preds, torch_preds, true_labels = map(np.array, (trt_preds, torch_preds, true_labels))
print()
print("total reviews:", len(true_labels))
print("TensorRT vs PyTorch agreement:", f"{(trt_preds == torch_preds).mean():.2%}")
print("PyTorch accuracy:", f"{(torch_preds == true_labels).mean():.2%}")
print("TensorRT accuracy:", f"{(trt_preds == true_labels).mean():.2%}")
