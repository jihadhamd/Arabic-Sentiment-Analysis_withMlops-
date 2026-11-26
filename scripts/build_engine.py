"""Build an FP16 TensorRT engine from the converted ONNX file."""

import time
from pathlib import Path

import tensorrt as trt

ONNX_PATH = Path("models/onnx/arabic_sentiment_fp16.onnx")
ENGINE_PATH = Path("models/trt/arabic_sentiment_fp16.engine")

BATCH_MIN, BATCH_OPT, BATCH_MAX = 1, 8, 32
SEQ_MIN, SEQ_OPT, SEQ_MAX = 2, 128, 256

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

try:
    network = builder.create_network()
except TypeError:
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser = trt.OnnxParser(network, logger)
print(f"Parsing {ONNX_PATH} ...")
if not parser.parse_from_file(str(ONNX_PATH)):
    for i in range(parser.num_errors):
        print("PARSE ERROR:", parser.get_error(i))
    raise SystemExit("ONNX parsing failed, see errors above")
print("Parsed OK:", network.num_layers, "layers")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024**3)

profile = builder.create_optimization_profile()
for name in ("input_ids", "attention_mask"):
    profile.set_shape(
        name,
        min=(BATCH_MIN, SEQ_MIN),
        opt=(BATCH_OPT, SEQ_OPT),
        max=(BATCH_MAX, SEQ_MAX),
    )
config.add_optimization_profile(profile)

print("Building the engine, this can take several minutes ...")
t0 = time.time()
engine_bytes = builder.build_serialized_network(network, config)
if engine_bytes is None:
    raise SystemExit("Build failed, see the log above")
print(f"Build finished in {time.time() - t0:.0f} s")

ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)
ENGINE_PATH.write_bytes(engine_bytes)
print(f"Saved {ENGINE_PATH} ({ENGINE_PATH.stat().st_size / 1e6:.0f} MB)")
