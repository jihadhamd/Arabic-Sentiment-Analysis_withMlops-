"""Export the fine-tuned model to ONNX and check it matches PyTorch."""

import csv
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("models/dvc-training")
OUT = Path("models/onnx/arabic_sentiment.onnx")
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, attn_implementation="eager"  # plain attention exports most reliably
).eval()


class Wrapper(torch.nn.Module):
    """Takes only input_ids and attention_mask, returns only the logits."""

    def __init__(self, m: torch.nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


# a small batch of 2 (not 1), only used to trace the graph
sample = tokenizer(
    ["الخدمة ممتازة", "المنتج سيء جدا"],
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
args = (sample["input_ids"], sample["attention_mask"])

# batch size and sequence length stay symbolic; both inputs share the same dims
batch = torch.export.Dim("batch", min=1, max=64)
seq = torch.export.Dim("seq", min=2, max=MAX_LENGTH)

OUT.parent.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    torch.onnx.export(
        Wrapper(model),
        args,
        str(OUT),
        dynamo=True,
        dynamic_shapes={
            "input_ids": {0: batch, 1: seq},
            "attention_mask": {0: batch, 1: seq},
        },
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=18,
    )
print(f"exported {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")

# --- check 1: the file is structurally valid
onnx.checker.check_model(str(OUT))
print("onnx.checker: file is structurally valid")

# --- check 2: the interface is what BentoML and TensorRT will expect
session = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
for i in session.get_inputs():
    print("input :", i.name, i.shape, i.type)
for o in session.get_outputs():
    print("output:", o.name, o.shape, o.type)
assert [i.name for i in session.get_inputs()] == ["input_ids", "attention_mask"]
assert [o.name for o in session.get_outputs()] == ["logits"]

# --- check 3: same logits and labels as PyTorch, on several batch sizes and lengths
random.seed(0)
with open("data/processed/test.csv", encoding="utf-8", newline="") as f:
    texts = [r["text"] for r in csv.DictReader(f)]


def compare(n: int) -> None:
    enc = tokenizer(
        random.sample(texts, n),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        ref = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.numpy()
    out = session.run(
        ["logits"],
        {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
        },
    )[0]
    assert out.shape == (n, model.config.num_labels), out.shape
    diff = float(np.abs(ref - out).max())
    same = int((ref.argmax(1) == out.argmax(1)).sum())
    print(
        f"batch {n:3d} x seq {enc['input_ids'].shape[1]:3d}: "
        f"max diff {diff:.2e} | same label {same}/{n}"
    )


for n in (1, 4, 16, 64):
    compare(n)
