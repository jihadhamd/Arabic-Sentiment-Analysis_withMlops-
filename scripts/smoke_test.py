import json
import os
import sys
import time
import urllib.error
import urllib.request

base = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
expected_version = os.environ.get("EXPECTED_MODEL_VERSION", "2")
deadline = time.monotonic() + 240


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


while True:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            if r.status != 200:
                raise RuntimeError(f"/health returned {r.status}")
            if json.load(r).get("status") != "healthy":
                raise RuntimeError("unhealthy payload")
        print("Health check passed.")
        break
    except (OSError, json.JSONDecodeError, RuntimeError) as e:
        if time.monotonic() >= deadline:
            fail(f"API not healthy within 4 min: {e}")
        time.sleep(5)

payload = json.dumps({"text": "المنتج ممتاز"}).encode("utf-8")
req = urllib.request.Request(
    f"{base}/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as r:
    if r.status != 200:
        fail(f"/predict returned {r.status}")
    pred = json.load(r)

if pred.get("label") not in {"negative", "mixed", "positive"}:
    fail(f"bad label: {pred!r}")
if not isinstance(pred.get("confidence"), (int, float)):
    fail(f"bad confidence: {pred!r}")
if not (0 <= pred["confidence"] <= 1):
    fail(f"confidence out of range: {pred!r}")
if pred.get("model_version") != expected_version:
    fail(f"model_version {pred.get('model_version')!r} != {expected_version!r}")

print("Prediction smoke test passed.")
print(json.dumps(pred, ensure_ascii=False, indent=2))
