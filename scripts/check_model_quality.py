import argparse
import json
import math
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a saved model's validation macro-F1 against baseline."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("models/dvc-training/metrics.json"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("reports/model_quality_baseline.json"),
    )
    args = parser.parse_args()

    try:
        baseline_data = load_json(args.baseline)
        candidate_data = load_json(args.metrics)

        if baseline_data.get("metric") != "val_f1_macro":
            raise ValueError("Baseline metric must be 'val_f1_macro'")

        baseline = float(baseline_data["baseline"])
        max_drop = float(baseline_data["max_allowed_drop"])
        candidate = float(candidate_data["metrics"]["val_f1_macro"])

        if not all(math.isfinite(value) for value in (baseline, max_drop, candidate)):
            raise ValueError("Metric values must be finite numbers")

        if not 0.0 <= baseline <= 1.0:
            raise ValueError("Baseline must be between 0 and 1")

        if not 0.0 <= candidate <= 1.0:
            raise ValueError("Candidate macro-F1 must be between 0 and 1")

        if not 0.0 <= max_drop <= 1.0:
            raise ValueError("Allowed drop must be between 0 and 1")

        threshold = baseline - max_drop

        print(f"Reference experiment: {baseline_data['reference_experiment']}")
        print(f"Baseline macro-F1:     {baseline:.6f}")
        print(f"Allowed drop:          {max_drop:.6f}")
        print(f"Minimum macro-F1:      {threshold:.6f}")
        print(f"Artifact macro-F1:     {candidate:.6f}")

        if candidate < threshold:
            print("QUALITY GATE FAILED: macro-F1 is below the minimum.")
            return 1

        print("QUALITY GATE PASSED.")
        return 0

    except (ValueError, KeyError, TypeError) as exc:
        print(f"QUALITY GATE ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
