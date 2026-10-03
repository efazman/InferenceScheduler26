"""Thin generation backends:  generate(prompt, seed) -> GenerationResult.

- OpenAICompatBackend: any server exposing POST /v1/chat/completions with a `usage` block.
  The target is llama.cpp's `llama-server`; llama-cpp-python's server, Ollama and vLLM speak the
  same protocol, so the label generator is not tied to one implementation.
- MockBackend: deterministic fake results for testing the pipeline without a GPU.
  MOCK OUTPUT IS SYNTHETIC AND TEST-ONLY.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    output_tokens: int
    latency_ms: float
    finish_reason: str | None = None  # "stop" = natural end, "length" = hit max_new_tokens
    prompt_tokens: int | None = None  # as counted by the server, chat template + system prompt included
    server_timings: dict | None = None  # llama.cpp per-request timings, when provided


class BackendError(Exception):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class OpenAICompatBackend:
    name = "openai-compat"

    def __init__(self, base_url: str, model: str, system_prompt: str, temperature: float, top_p: float,
                 max_new_tokens: int, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            # 4xx = the request itself is bad (e.g. prompt exceeds context); retrying won't help.
            raise BackendError(f"HTTP {e.code}: {detail}", retryable=e.code >= 500 or e.code == 429) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise BackendError(f"{type(e).__name__}: {e}", retryable=True) from e
        except json.JSONDecodeError as e:
            raise BackendError(f"invalid JSON response: {e}", retryable=True) from e

    def check(self) -> dict:
        """Fail fast before a long run if the server is unreachable."""
        req = urllib.request.Request(self.base_url + "/v1/models")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - surfaced to the user verbatim
            raise BackendError(f"backend not reachable at {self.base_url}: {e}", retryable=False) from e

    def generate(self, prompt: str, seed: int) -> GenerationResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
            "seed": seed,
            "stream": False,
        }
        t0 = time.perf_counter()
        resp = self._post("/v1/chat/completions", body)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        try:
            choice = resp["choices"][0]
            output_tokens = int(resp["usage"]["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as e:
            # Never guess a token count: a missing usage block is a failed measurement.
            raise BackendError(f"response missing choices/usage.completion_tokens: {str(resp)[:300]}",
                               retryable=False) from e
        return GenerationResult(
            text=(choice.get("message") or {}).get("content") or "",
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=resp["usage"].get("prompt_tokens"),
            server_timings=resp.get("timings"),
        )


class MockBackend:
    """Deterministic fake backend. SYNTHETIC / TEST-ONLY: never use its labels for ML evaluation.

    Token counts and latencies are pure functions of (prompt, seed). Failure injection:
      transient_failures  every (prompt, seed) selected by ``fail_rate`` fails this many times
                          before succeeding (exercises retries)
      fail_rate           fraction of (prompt, seed) pairs that get transient failures
      permanent_failures  prompts (exact text) that always fail (exercises failure logging)
    """

    name = "mock"

    def __init__(self, max_new_tokens: int, transient_failures: int = 0, fail_rate: float = 0.0,
                 permanent_failures: set[str] | None = None):
        self.max_new_tokens = max_new_tokens
        self.transient_failures = transient_failures
        self.fail_rate = fail_rate
        self.permanent_failures = set(permanent_failures or ())
        self.attempts: dict[tuple[str, int], int] = {}
        self.calls = 0

    @staticmethod
    def _h(*parts) -> int:
        return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "big")

    def check(self) -> dict:
        return {"backend": "mock"}

    def generate(self, prompt: str, seed: int) -> GenerationResult:
        self.calls += 1
        key = (prompt, seed)
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if prompt in self.permanent_failures:
            raise BackendError("mock permanent failure", retryable=True)
        selected = (self._h("fail", prompt, seed) % 10_000) < self.fail_rate * 10_000
        if selected and self.attempts[key] <= self.transient_failures:
            raise BackendError(f"mock transient failure (attempt {self.attempts[key]})", retryable=True)
        base = 20 + self._h("len", prompt) % 480  # per-prompt "typical" length
        jitter = 0.75 + (self._h("seed", prompt, seed) % 5001) / 10_000  # 0.75 .. 1.25
        raw = int(base * jitter)
        tokens = min(raw, self.max_new_tokens)
        return GenerationResult(
            text=f"[MOCK OUTPUT - synthetic, test only] {tokens} tokens",
            output_tokens=tokens,
            latency_ms=round(30.0 + 20.0 * tokens, 3),  # fake ~50 tok/s decode, no real sleep
            finish_reason="length" if raw >= self.max_new_tokens else "stop",
            prompt_tokens=len(prompt.split()),
        )
