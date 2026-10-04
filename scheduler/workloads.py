"""Deterministic SYNTHETIC workloads for validating scheduler behaviour.

Nothing here is a measurement. Prompts are placeholders, and the "actual" token counts and the
"predicted" ones (actual x lognormal noise, mimicking an imperfect predictor) are drawn from fixed
seeds. Use them to check policy behaviour, never to report model or Llama performance.

    mostly_short   80% short / 20% long, Poisson arrivals
    balanced       50% / 50%
    mostly_long    20% / 80%
    head_of_line   a job is running; one long request queues, then several short ones behind it
    bursty         steady mixed background traffic, then a burst of short requests

Poisson arrival rates are set from a target offered ``load`` (mean service time / mean
inter-arrival time), so queues actually form. Without queueing, every policy behaves identically.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

from scheduler import config
from scheduler.backends import MockInferenceBackend
from scheduler.models import Request
from scheduler.predictors import MockPredictor

SHORT_TOKENS = (20, 150)  # uniform range of synthetic "actual" output lengths
LONG_TOKENS = (400, 1000)

_SHORT_PROMPTS = ["What is the default port for PostgreSQL?", "Who owns the on-call rotation this week?",
                  "Convert 3 PM EST to UTC.", "What does HTTP 429 mean?", "Is the VPN required for the wiki?"]
_LONG_PROMPTS = ["Write a detailed incident report for yesterday's outage.",
                 "Explain step by step how to migrate our billing service to Kubernetes.",
                 "Draft a design document for the new onboarding flow.",
                 "Summarize these meeting notes and list every action item with an owner."]


@dataclass
class WorkloadRequest:
    request_id: str
    prompt: str
    arrival_ms: float
    size_class: str | None
    mock_actual_tokens: int
    mock_predicted_tokens: float
    seed: int


@dataclass
class Workload:
    name: str
    description: str
    seed: int
    requests: list[WorkloadRequest]
    params: dict = field(default_factory=dict)
    synthetic: bool = True

    def to_requests(self) -> list[Request]:
        return [Request(w.request_id, w.prompt, w.arrival_ms, seed=w.seed, size_class=w.size_class)
                for w in self.requests]

    def mock_predictor(self) -> MockPredictor:
        return MockPredictor({w.prompt: w.mock_predicted_tokens for w in self.requests})

    def mock_backend(self, **kwargs) -> MockInferenceBackend:
        return MockInferenceBackend({w.prompt: w.mock_actual_tokens for w in self.requests}, **kwargs)

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "seed": self.seed, "synthetic": True,
                "params": self.params, "requests": [asdict(w) for w in self.requests]}


class _Builder:
    def __init__(self, name: str, seed: int, prediction_noise: float):
        self.rng = random.Random(f"{seed}:{name}")  # str seeds are deterministic across runs/platforms
        self.noise = prediction_noise
        self.items: list[WorkloadRequest] = []

    def add(self, size_class: str, arrival_ms: float) -> None:
        lo, hi = SHORT_TOKENS if size_class == "short" else LONG_TOKENS
        actual = self.rng.randint(lo, hi)
        predicted = actual * math.exp(self.rng.gauss(0.0, self.noise)) if self.noise > 0 else float(actual)
        idx = len(self.items)
        rid = f"r{idx:04d}"
        base = self.rng.choice(_SHORT_PROMPTS if size_class == "short" else _LONG_PROMPTS)
        self.items.append(WorkloadRequest(rid, f"[sim {rid}] {base}", round(arrival_ms, 3), size_class, actual,
                                          round(predicted, 1), seed=idx))


def _mean_service_ms(p_long: float, sim) -> float:
    mean_tokens = (1 - p_long) * sum(SHORT_TOKENS) / 2 + p_long * sum(LONG_TOKENS) / 2
    return sim["base_overhead_ms"] + mean_tokens * sim["ms_per_token"]


def _poisson(name, p_long, n, seed, load, noise, sim):
    b = _Builder(name, seed, noise)
    mean_gap = _mean_service_ms(p_long, sim) / load
    t = 0.0
    for _ in range(n):
        b.add("long" if b.rng.random() < p_long else "short", t)
        t += b.rng.expovariate(1.0 / mean_gap)
    return b.items, {"p_long": p_long, "n": n, "load": load, "mean_interarrival_ms": round(mean_gap, 1)}


def _head_of_line(name, n_short, seed, noise):
    b = _Builder(name, seed, noise)
    b.add("long", 0.0)  # occupies the backend first (no policy can preempt it)
    b.add("long", 50.0)  # the head-of-line blocker: first in the queue
    for i in range(n_short):
        b.add("short", 100.0 + 100.0 * i)
    return b.items, {"n_short": n_short}


def _bursty(name, n_background, n_burst, seed, load, noise, sim):
    b = _Builder(name, seed, noise)
    p_long = 0.4
    mean_gap = _mean_service_ms(p_long, sim) / load
    times, t = [], 0.0
    for _ in range(n_background):
        times.append(t)
        t += b.rng.expovariate(1.0 / mean_gap)
    burst_at = times[min(len(times) - 1, n_background // 3)]
    arrivals = [(at, "long" if b.rng.random() < p_long else "short") for at in times]
    arrivals += [(burst_at + b.rng.uniform(0, 1000), "short") for _ in range(n_burst)]
    for at, cls in sorted(arrivals):
        b.add(cls, at)
    return b.items, {"n_background": n_background, "n_burst": n_burst, "burst_at_ms": round(burst_at, 1),
                     "load": load}


DESCRIPTIONS = {
    "mostly_short": "80% short / 20% long requests, Poisson arrivals",
    "balanced": "50% short / 50% long requests, Poisson arrivals",
    "mostly_long": "20% short / 80% long requests, Poisson arrivals",
    "head_of_line": "a long job is running; another long request queues first, short ones arrive behind it",
    "bursty": "mixed background traffic, then a burst of short requests within one second",
}
WORKLOAD_NAMES = tuple(DESCRIPTIONS)


def make_workload(name: str, n: int = 60, seed: int = 42, load: float = 0.95, prediction_noise: float = 0.25,
                  base_overhead_ms: float = config.SIM_BASE_OVERHEAD_MS,
                  ms_per_token: float = config.SIM_MS_PER_TOKEN) -> Workload:
    """Build a named workload. ``n`` sizes the Poisson workloads (bursty uses n as background
    count plus n//3 burst requests; head_of_line uses max(n//6, 8) short requests)."""
    sim = {"base_overhead_ms": base_overhead_ms, "ms_per_token": ms_per_token}
    if name == "mostly_short":
        items, params = _poisson(name, 0.2, n, seed, load, prediction_noise, sim)
    elif name == "balanced":
        items, params = _poisson(name, 0.5, n, seed, load, prediction_noise, sim)
    elif name == "mostly_long":
        items, params = _poisson(name, 0.8, n, seed, load, prediction_noise, sim)
    elif name == "head_of_line":
        items, params = _head_of_line(name, max(n // 6, 8), seed, prediction_noise)
    elif name == "bursty":
        items, params = _bursty(name, n, max(n // 3, 5), seed, load, prediction_noise, sim)
    else:
        raise ValueError(f"unknown workload {name!r}; choose from {WORKLOAD_NAMES}")
    params.update(prediction_noise=prediction_noise, **sim)
    return Workload(name, DESCRIPTIONS[name], seed, items, params)


def workload_from_prompts(prompts: list[str], name: str = "real_prompts", mean_interarrival_ms: float = 5000.0,
                          seed: int = 42, ids: list[str] | None = None) -> Workload:
    """Real prompts with seeded Poisson arrival times, for wall-clock runs against llama.cpp.
    Lengths are unknown until generated, so the mock fields are left at -1. ``ids`` (e.g. the
    dataset prompt_id) become request IDs so results can be joined back to the labels."""
    rng = random.Random(f"{seed}:{name}")
    t, items = 0.0, []
    for i, p in enumerate(prompts):
        rid = ids[i] if ids else f"r{i:04d}"
        # size_class None: metrics derive short/long from the measured output length instead.
        items.append(WorkloadRequest(rid, p, round(t, 3), None, -1, -1.0, seed=i))
        t += rng.expovariate(1.0 / mean_interarrival_ms)
    return Workload(name, "real prompts, seeded Poisson arrivals", seed, items,
                    {"mean_interarrival_ms": mean_interarrival_ms}, synthetic=False)
