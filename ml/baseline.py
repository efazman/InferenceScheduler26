"""Naive baseline: prompt token count -> linear regression -> output token count.

Answers "does the learned predictor add value beyond prompt length?"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression


class PromptLengthBaseline:
    def __init__(self, tokenizer, coef: float = 0.0, intercept: float = 0.0):
        # REAL-DATA: prompt length is counted with the predictor's (DistilBERT) tokenizer. Once the
        # Llama model is chosen, consider counting with the Llama tokenizer instead, since that is
        # what the serving backend actually sees.
        self.tokenizer = tokenizer
        self.coef = coef
        self.intercept = intercept

    def prompt_token_counts(self, prompts: list[str]) -> np.ndarray:
        return np.array([len(self.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts],
                        dtype=np.float64)

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
            "coef": self.coef, "intercept": self.intercept,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path, tokenizer) -> "PromptLengthBaseline":
        raw = json.loads(Path(path).read_text())
        return cls(tokenizer, raw["coef"], raw["intercept"])
