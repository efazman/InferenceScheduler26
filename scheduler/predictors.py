"""Predictor interface + implementations. The scheduler depends only on ``Predictor``.

Every predictor returns at least::

    {"expected_output_tokens": float, "uncertainty": float | None}

Extra keys (e.g. ``bin_probabilities`` from the DistilBERT predictor) are allowed and ignored.
``ml.predictor.LengthPredictor`` already satisfies this protocol unchanged.

    load_predictor("mock")                                        # deterministic, tests/sim
    load_predictor("distilbert:artifacts/distilbert_length_predictor_llama31")
    load_predictor("baseline:artifacts/distilbert_length_predictor_llama31")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Predictor(Protocol):
    def predict(self, prompt: str) -> dict: ...


def normalize_prediction(raw: dict) -> tuple[float, float | None]:
    """Validate a predictor's output -> (expected_output_tokens, uncertainty)."""
    try:
        expected = float(raw["expected_output_tokens"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"predictor output missing a numeric 'expected_output_tokens': {raw!r}") from e
    if expected != expected or expected < 0:  # NaN or negative
        raise ValueError(f"invalid expected_output_tokens={expected!r}")
    unc = raw.get("uncertainty")
    return expected, (None if unc is None else float(unc))


_LONG_FORM = re.compile(r"\b(write|essay|article|story|explain|detailed|summari[sz]e|report|code|implement|"
                        r"step by step|compare|plan|draft)\b", re.IGNORECASE)


class MockPredictor:
    """Deterministic stand-in predictor. SYNTHETIC: its numbers are not model predictions.

    - ``table`` maps exact prompt text -> expected tokens (workloads use this to inject the mock
      prediction they generated, including deliberate prediction error).
    - Otherwise a transparent heuristic: 40 + 3 tokens per prompt word, +350 for long-form cues.
    """

    name = "mock"

    def __init__(self, table: dict[str, float] | None = None, uncertainty: float | None = None):
        self.table = dict(table or {})
        self.uncertainty = uncertainty

    def predict(self, prompt: str) -> dict:
        if prompt in self.table:
            expected = float(self.table[prompt])
        else:
            expected = 40.0 + 3.0 * len(prompt.split()) + (350.0 if _LONG_FORM.search(prompt) else 0.0)
        return {"expected_output_tokens": expected, "uncertainty": self.uncertainty}


class DistilBertPredictor:
    """Adapter over the trained ml.predictor.LengthPredictor (loaded lazily; needs torch)."""

    name = "distilbert"

    def __init__(self, artifacts_dir: str | Path, device: str | None = None):
        from ml.predictor import LengthPredictor

        self._inner = LengthPredictor.load(artifacts_dir, device=device)

    def predict(self, prompt: str) -> dict:
        return self._inner.predict(prompt)  # includes bin_probabilities; extra keys are ignored


class LinearBaselinePredictor:
    """Adapter over ml.baseline.PromptLengthBaseline (prompt token count -> linear regression)."""

    name = "baseline"

    def __init__(self, artifacts_dir: str | Path):
        from ml.baseline import PromptLengthBaseline

        artifacts_dir = Path(artifacts_dir)
        tokenizer = None
        if (artifacts_dir / "tokenizer").exists():  # needed only for the "distilbert" counter spec
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(artifacts_dir / "tokenizer")
        self._inner = PromptLengthBaseline.load(artifacts_dir / "baseline.json", tokenizer)

    def predict(self, prompt: str) -> dict:
        return {"expected_output_tokens": float(self._inner.predict([prompt])[0]), "uncertainty": None}


def load_predictor(spec: str) -> Predictor:
    """'mock' | 'distilbert:<artifacts dir>' | 'baseline:<artifacts dir>'."""
    kind, _, arg = spec.partition(":")
    if kind == "mock":
        return MockPredictor()
    if kind == "distilbert" and arg:
        return DistilBertPredictor(arg)
    if kind == "baseline" and arg:
        return LinearBaselinePredictor(arg)
    raise ValueError(f"unknown predictor spec {spec!r} (mock | distilbert:<dir> | baseline:<dir>)")


def predictor_name(p) -> str:
    return getattr(p, "name", type(p).__name__)
