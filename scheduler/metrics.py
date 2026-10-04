"""Per-request records and aggregate scheduler metrics (one implementation for sim and real runs).

Percentiles use linear interpolation (numpy's default). Tail percentiles of small runs are noisy:
``p99_reliable`` is False below P99_MIN_SAMPLES completed requests, and reports should say so.
"""

from __future__ import annotations

import math

from scheduler import config
from scheduler.models import Request, RequestState

P99_MIN_SAMPLES = 100


def percentile(values, q: float) -> float | None:
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def size_class(r: Request, long_threshold_tokens: int = config.LONG_TOKEN_THRESHOLD) -> str | None:
    if r.size_class:
        return r.size_class
    if r.actual_output_tokens is None:
        return None
    return "long" if r.actual_output_tokens >= long_threshold_tokens else "short"


def request_record(r: Request) -> dict:
    return {
        "request_id": r.request_id, "state": r.state.value, "size_class": r.size_class,
        "arrival_time": r.arrival_time, "start_time": r.start_time, "end_time": r.end_time,
        "predicted_tokens": r.predicted_tokens, "actual_output_tokens": r.actual_output_tokens,
        "queue_wait_ms": r.queue_wait_ms, "service_time_ms": r.service_time_ms,
        "end_to_end_latency_ms": r.end_to_end_latency_ms, "selected_by": r.metadata.get("selected_by"),
    }


def summarize(requests: list[Request], starvation_threshold_ms: float = config.DEFAULT_STARVATION_THRESHOLD_MS,
              long_threshold_tokens: int = config.LONG_TOKEN_THRESHOLD) -> dict:
    done = [r for r in requests if r.state == RequestState.COMPLETED]
    failed = [r for r in requests if r.state == RequestState.FAILED]
    lat = [r.end_to_end_latency_ms for r in done]
    wait = [r.queue_wait_ms for r in done]
    svc = [r.service_time_ms for r in done]
    ended = [r for r in requests if r.end_time is not None]
    makespan = (max(r.end_time for r in ended) - min(r.arrival_time for r in ended)) if ended else 0.0
    secs = makespan / 1000.0
    by_class = {c: [r for r in done if size_class(r, long_threshold_tokens) == c] for c in ("short", "long")}
    return {
        "n_completed": len(done),
        "n_failed": len(failed),
        "mean_latency_ms": _mean(lat),
        "p50_latency_ms": percentile(lat, 50),
        "p95_latency_ms": percentile(lat, 95),
        "p99_latency_ms": percentile(lat, 99),
        "p99_reliable": len(done) >= P99_MIN_SAMPLES,
        "mean_queue_wait_ms": _mean(wait),
        "p95_queue_wait_ms": percentile(wait, 95),
        "max_queue_wait_ms": max(wait) if wait else None,
        "mean_service_ms": _mean(svc),
        "makespan_ms": makespan,
        "throughput_rps": len(done) / secs if secs > 0 else None,
        "throughput_tokens_per_s": sum(r.actual_output_tokens for r in done) / secs if secs > 0 else None,
        "starvation_threshold_ms": starvation_threshold_ms,
        "starvation_count": sum(w > starvation_threshold_ms for w in wait),
        "short_n": len(by_class["short"]),
        "long_n": len(by_class["long"]),
        "short_mean_latency_ms": _mean(r.end_to_end_latency_ms for r in by_class["short"]),
        "long_mean_latency_ms": _mean(r.end_to_end_latency_ms for r in by_class["long"]),
        "short_p95_latency_ms": percentile([r.end_to_end_latency_ms for r in by_class["short"]], 95),
        "long_max_queue_wait_ms": max((r.queue_wait_ms for r in by_class["long"]), default=None),
    }
