"""Adaptive MAX_WAIT resolution.

    explicit:  MAX_WAIT = the value given (simulation / controlled experiments)
    derived:   MAX_WAIT = multiplier x median service time (real runs; multiplier defaults to 3.0)

The median service time can be given directly or measured from a file:
  - a label-generation runs.jsonl (datagen; one line per real generation, field latency_ms), or
  - a scheduler events.jsonl (field service_time_ms on inference_completed events).
Mock or simulated measurements are rejected, so a real threshold is never derived from fake data.
The resolved value and where it came from are recorded in run_started events and summary.json.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from scheduler import config


@dataclass
class MaxWaitConfig:
    max_wait_ms: float
    mode: str  # "explicit" | "derived"
    multiplier: float | None = None
    median_service_ms: float | None = None
    source: str | None = None  # file the median was measured from, or "given"
    n_samples: int | None = None

    def describe(self) -> str:
        if self.mode == "explicit":
            return f"{self.max_wait_ms / 1000:.1f} s (explicit)"
        return (f"{self.max_wait_ms / 1000:.1f} s = {self.multiplier:g} x median service "
                f"{self.median_service_ms / 1000:.2f} s ({self.source}"
                f"{f', n={self.n_samples}' if self.n_samples else ''})")

    def to_dict(self) -> dict:
        return {**asdict(self), "description": self.describe()}


def service_times_from_file(path: str | Path) -> list[float]:
    """Real service times (ms) from a label runs.jsonl or a scheduler events.jsonl."""
    out, mock = [], 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial last line of a live file
            if rec.get("is_mock") or rec.get("simulated") or rec.get("backend") == "mock":
                mock += 1
                continue
            if rec.get("event_type") == "inference_completed" and rec.get("service_time_ms") is not None:
                out.append(float(rec["service_time_ms"]))
            elif "event_type" not in rec and rec.get("latency_ms") is not None:
                out.append(float(rec["latency_ms"]))  # datagen runs.jsonl line
    if not out:
        raise ValueError(f"no real service-time measurements in {path}"
                         f"{f' ({mock} mock/simulated lines ignored)' if mock else ''}")
    return out


def resolve_max_wait(explicit_ms: float | None = None, multiplier: float = config.DEFAULT_MAX_WAIT_MULTIPLIER,
                     median_service_ms: float | None = None, median_from: str | Path | None = None,
                     default_ms: float | None = None) -> MaxWaitConfig:
    """Pick the threshold: explicit > derived (given median) > derived (measured) > default_ms.
    With no source and no default, raises: real runs must state where MAX_WAIT came from."""
    if explicit_ms is not None:
        if explicit_ms <= 0:
            raise ValueError("max wait must be positive")
        return MaxWaitConfig(float(explicit_ms), "explicit")
    if multiplier <= 0:
        raise ValueError("max-wait multiplier must be positive")
    if median_service_ms is not None:
        if median_service_ms <= 0:
            raise ValueError("median service time must be positive")
        return MaxWaitConfig(multiplier * median_service_ms, "derived", multiplier, float(median_service_ms), "given")
    if median_from is not None:
        times = service_times_from_file(median_from)
        med = statistics.median(times)
        return MaxWaitConfig(multiplier * med, "derived", multiplier, med, str(median_from), len(times))
    if default_ms is not None:
        return MaxWaitConfig(float(default_ms), "explicit")
    raise ValueError("no MAX_WAIT source: pass --max-wait-ms, --median-service-ms or --median-service-from "
                     "(e.g. the label run's runs.jsonl)")
