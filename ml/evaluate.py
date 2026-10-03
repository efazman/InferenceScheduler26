"""Metrics + evaluation of saved artifacts on the held-out test split.

    python -m ml.evaluate [--artifacts artifacts/distilbert_length_predictor]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ml.baseline import PromptLengthBaseline
from ml.config import Config
from ml.data import load_jsonl


def compute_metrics(predicted, actual, severe_threshold: float = 0.5) -> dict:
    p = np.asarray(predicted, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    abs_err = np.abs(p - a)
    return {
        "n": int(len(a)),
        "mae": float(abs_err.mean()),
        "p90_abs_error": float(np.percentile(abs_err, 90)),
        "underprediction_rate": float((p < a).mean()),
        "severe_underprediction_rate": float((p < severe_threshold * a).mean()),
        "severe_underprediction_threshold": severe_threshold,
        "mean_signed_error": float((p - a).mean()),
    }


def measure_latency(predictor, prompts: list[str], warmup: int = 3) -> dict:
    """Per-prompt (batch size 1) wall-clock latency of predictor.predict, in milliseconds."""
    for p in prompts[:warmup]:
        predictor.predict(p)
    times = []
    for p in prompts:
        t0 = time.perf_counter()
        predictor.predict(p)  # returns python floats, so device work is synchronized
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {"n": len(t), "mean_ms": float(t.mean()), "median_ms": float(np.median(t)),
            "p95_ms": float(np.percentile(t, 95)), "device": str(predictor.device)}


def evaluate_split(predictor, baseline: PromptLengthBaseline, examples: list[dict], config: Config) -> dict:
    prompts = [ex["prompt"] for ex in examples]
    actual = [ex["target_output_tokens"] for ex in examples]
    preds = [r["expected_output_tokens"] for r in predictor.predict_batch(prompts)]
    thr = config.severe_underprediction_threshold
    return {
        "distilbert": compute_metrics(preds, actual, thr),
        "baseline_prompt_length_linreg": compute_metrics(baseline.predict(prompts), actual, thr),
        "distilbert_latency": measure_latency(predictor, prompts, config.latency_warmup),
    }


def load_split(artifacts: Path, config: Config, split: str) -> list[dict]:
    ids = json.loads((artifacts / "splits.json").read_text())[split]
    by_id = {ex["id"]: ex for ex in load_jsonl(config.data_path)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} {split} ids not found in {config.data_path}; "
                         "was the dataset regenerated after training?")
    return [by_id[i] for i in ids]


def main(argv=None) -> dict:
    from ml.predictor import LengthPredictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=Config().output_dir)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args(argv)
    artifacts = Path(args.artifacts)

    predictor = LengthPredictor.load(artifacts)
    baseline = PromptLengthBaseline.load(artifacts / "baseline.json", predictor.tokenizer)
    results = evaluate_split(predictor, baseline, load_split(artifacts, predictor.config, args.split),
                             predictor.config)
    results["split"] = args.split
    results["warning"] = "SYNTHETIC labels: numbers validate the code path only."  # REAL-DATA: remove
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
