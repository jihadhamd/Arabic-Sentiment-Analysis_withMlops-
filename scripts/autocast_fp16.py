"""Convert the FP32 ONNX file to FP16 using NVIDIA's ModelOpt AutoCast."""

from pathlib import Path

import onnx
from modelopt.onnx.autocast import convert_to_mixed_precision

IN_PATH = Path("models/onnx/arabic_sentiment.onnx")
OUT_PATH = Path("models/onnx/arabic_sentiment_fp16.onnx")

converted = convert_to_mixed_precision(
    onnx_path=str(IN_PATH),
    low_precision_type="fp16",
    keep_io_types=True,  # inputs/outputs keep their original types (int64 ids, float logits)
)
onnx.save(converted, str(OUT_PATH))
print(f"saved {OUT_PATH}")
