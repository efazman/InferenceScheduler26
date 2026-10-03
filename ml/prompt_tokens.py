"""Prompt token counters for the prompt-length baseline, chosen by a spec string.

    "distilbert"      count with the predictor's own tokenizer (development fallback)
    "llamacpp:<url>"  POST <url>/tokenize on llama-server; the real Llama 3.1 tokenizer from the GGUF,
                      no model files needed on the training machine
    "hf:<name|path>"  any Hugging Face tokenizer, e.g. hf:meta-llama/Llama-3.1-8B-Instruct
                      (tokenizer files only, no weights)

Counts are of the raw prompt text without special tokens, chat template or system prompt.
Those add a near-constant offset, which the linear regression's intercept absorbs.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable

TokenCounter = Callable[[str], int]


def _hf_counter(tokenizer) -> TokenCounter:
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _llamacpp_counter(url: str, timeout: float = 30.0) -> TokenCounter:
    endpoint = url.rstrip("/") + "/tokenize"

    def count(text: str) -> int:
        body = json.dumps({"content": text, "add_special": False}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return len(json.loads(resp.read().decode("utf-8"))["tokens"])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"llama.cpp tokenizer at {endpoint} failed ({e}); start llama-server "
                               "or use --prompt-token-counter distilbert") from e

    return count


def make_token_counter(spec: str, fallback_tokenizer=None) -> TokenCounter:
    if spec == "distilbert":
        if fallback_tokenizer is None:
            raise ValueError("'distilbert' token counter needs the predictor tokenizer")
        return _hf_counter(fallback_tokenizer)
    if spec.startswith("llamacpp:"):
        return _llamacpp_counter(spec.removeprefix("llamacpp:"))
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        return _hf_counter(AutoTokenizer.from_pretrained(spec.removeprefix("hf:")))
    raise ValueError(f"unknown prompt token counter spec {spec!r}")
