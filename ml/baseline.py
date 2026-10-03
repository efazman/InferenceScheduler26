"""Naive baseline: prompt token count -> linear regression -> output token count.

Answers "does the learned predictor add value beyond prompt length?"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from ml.prompt_tokens import TokenCounter, make_token_counter


class PromptLengthBaseline:
    def __init__(self, count_tokens: TokenCounter, counter_spec: str = "distilbert",
                 coef: float = 0.0, intercept: float = 0.0):
        # counter_spec is saved with the model so a reload counts tokens the same way
        # (see ml/prompt_tokens.py; real Llama runs use "llamacpp:<url>").
        self.count_tokens = count_tokens
        self.counter_spec = counter_spec
        self.coef = coef
        self.intercept = intercept

    @classmethod
    def from_spec(cls, counter_spec: str, fallback_tokenizer=None) -> "PromptLengthBaseline":
        return cls(make_token_counter(counter_spec, fallback_tokenizer), counter_spec)

    def prompt_token_counts(self, prompts: list[str]) -> np.ndarray:
        return np.array([self.count_tokens(p) for p in prompts], dtype=np.float64)

    def fit(self, prompts: list[str], targets) -> "PromptLengthBaseline":
        reg = LinearRegression().fit(self.prompt_token_counts(prompts).reshape(-1, 1),
                                     np.asarray(targets, dtype=np.float64))
        self.coef, self.intercept = float(reg.coef_[0]), float(reg.intercept_)
        return self

    def predict(self, prompts: list[str]) -> np.ndarray:
        # Clamp at 0: a negative output length is meaningless.
        return np.maximum(self.coef * self.prompt_token_counts(prompts) + self.intercept, 0.0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "type": "linear_regression_on_prompt_token_count",
            "prompt_token_counter": self.counter_spec,
            "coef": self.coef, "intercept": self.intercept,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path, fallback_tokenizer=None) -> "PromptLengthBaseline":
        raw = json.loads(Path(path).read_text())
        spec = raw.get("prompt_token_counter", "distilbert")  # artifacts from before this field existed
        return cls(make_token_counter(spec, fallback_tokenizer), spec, raw["coef"], raw["intercept"])
