"""Inference backend interface + implementations. The engine depends only on ``InferenceBackend``.

    generate(prompt, seed=None) -> {"text": str, "output_tokens": int, "latency_ms": float, ...}

Raise ``BackendError`` (re-exported from datagen.backends) for a failed generation.
"""

from __future__ import annotations

import hashlib
import time
from typing import Protocol, runtime_checkable

from datagen.backends import BackendError  # noqa: F401  (re-exported: one error type project-wide)
from scheduler import config


@runtime_checkable
class InferenceBackend(Protocol):
    name: str

    def generate(self, prompt: str, seed: int | None = None) -> dict: ...


class MockInferenceBackend:
    """Deterministic fake backend. SYNTHETIC: never report its numbers as real inference results.

    Output length comes from ``table`` (exact prompt -> tokens) or a hash of (prompt, seed).
    latency_ms = base_overhead_ms + tokens * ms_per_token. With ``realtime=True`` it also sleeps
    for latency_ms / speedup, so wall-clock runs (and the live UI) can be exercised without a GPU.
    """

    name = "mock"

    def __init__(self, table: dict[str, int] | None = None, base_overhead_ms: float = config.SIM_BASE_OVERHEAD_MS,
                 ms_per_token: float = config.SIM_MS_PER_TOKEN, realtime: bool = False, speedup: float = 1.0,
                 fail_prompts: set[str] | None = None):
        self.table = dict(table or {})
        self.base_overhead_ms = base_overhead_ms
        self.ms_per_token = ms_per_token
        self.realtime = realtime
        self.speedup = speedup
        self.fail_prompts = set(fail_prompts or ())
        self.calls: list[str] = []

    def service_time_ms(self, tokens: int) -> float:
        return self.base_overhead_ms + tokens * self.ms_per_token

    def generate(self, prompt: str, seed: int | None = None) -> dict:
        self.calls.append(prompt)
        if prompt in self.fail_prompts:
            raise BackendError("mock backend failure", retryable=False)
        if prompt in self.table:
            tokens = int(self.table[prompt])
        else:
            h = hashlib.sha256(f"{prompt}\x1f{seed}".encode()).digest()
            tokens = 20 + int.from_bytes(h[:4], "big") % 600
        latency = self.service_time_ms(tokens)
        if self.realtime:
            time.sleep(latency / 1000.0 / self.speedup)
        return {"text": f"[MOCK OUTPUT - synthetic] {tokens} tokens", "output_tokens": tokens,
                "latency_ms": latency, "finish_reason": "stop"}


class LlamaCppBackend:
    """Client for llama.cpp `llama-server` (OpenAI-compatible /v1/chat/completions).

    Thin wrapper over datagen.backends.OpenAICompatBackend, the same client that produced the
    training labels, with the same model / system prompt / temperature / top_p / max_new_tokens
    from datagen.config, so measured lengths are comparable with what the predictor learned.
    Start the server with `-np 1` (one slot) so our queue order is the execution order.
    """

    name = "llamacpp"

    def __init__(self, base_url: str | None = None, timeout_s: float | None = None, max_new_tokens: int | None = None):
        from datagen import config as gen
        from datagen.backends import OpenAICompatBackend

        self.base_url = base_url or gen.BACKEND_URL
        self._client = OpenAICompatBackend(
            self.base_url, gen.MODEL_NAME, gen.SYSTEM_PROMPT, gen.TEMPERATURE, gen.TOP_P,
            max_new_tokens or gen.MAX_NEW_TOKENS, timeout_s or gen.REQUEST_TIMEOUT)

    def check(self) -> dict:
        return self._client.check()

    def generate(self, prompt: str, seed: int | None = None) -> dict:
        r = self._client.generate(prompt, seed if seed is not None else 0)
        return {"text": r.text, "output_tokens": r.output_tokens, "latency_ms": r.latency_ms,
                "finish_reason": r.finish_reason, "prompt_tokens": r.prompt_tokens,
                "server_timings": r.server_timings}


def make_backend(name: str, url: str | None = None, **mock_kwargs) -> InferenceBackend:
    if name == "mock":
        return MockInferenceBackend(**mock_kwargs)
    if name == "llamacpp":
        return LlamaCppBackend(url)
    raise ValueError(f"unknown backend {name!r} (mock | llamacpp)")
