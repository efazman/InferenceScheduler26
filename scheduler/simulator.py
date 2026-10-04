"""K=1 non-preemptive simulation: the real engine + VirtualClock + mock predictor/backend.

service_time_ms = base_overhead_ms + actual_tokens * ms_per_token (see scheduler.config).
This validates scheduler behaviour only. It is NOT a benchmark of Llama or of the predictor.
"""

from __future__ import annotations

from dataclasses import dataclass

from scheduler import config
from scheduler.clock import VirtualClock
from scheduler.engine import SchedulerEngine
from scheduler.events import InMemoryEventSink, NullSink
from scheduler.metrics import summarize
from scheduler.models import Request
from scheduler.policies import POLICY_NAMES, make_policy
from scheduler.workloads import Workload


@dataclass
class SimResult:
    workload: str
    policy: str
    requests: list[Request]
    summary: dict
    events: list | None = None


def simulate(workload: Workload, policy_name: str, max_wait_ms: float = config.DEFAULT_MAX_WAIT_MS,
             starvation_threshold_ms: float | None = None, sink=None, keep_events: bool = False,
             emit_queue_snapshots: bool = True,
             predictor_latency_ms: float = config.SIM_PREDICTOR_LATENCY_MS) -> SimResult:
    """Run one policy on one workload. The mock backend's service model comes from the workload's
    own params, so the arrival rate and service times stay consistent."""
    threshold = max_wait_ms if starvation_threshold_ms is None else starvation_threshold_ms
    mem = InMemoryEventSink() if keep_events else None
    out_sink = mem if sink is None else sink
    engine = SchedulerEngine(
        workload.mock_predictor(),
        workload.mock_backend(base_overhead_ms=workload.params["base_overhead_ms"],
                              ms_per_token=workload.params["ms_per_token"]),
        make_policy(policy_name, max_wait_ms), out_sink or NullSink(), VirtualClock(),
        workload_name=workload.name, sim_predictor_latency_ms=predictor_latency_ms,
        emit_queue_snapshots=emit_queue_snapshots, starvation_threshold_ms=threshold)
    done = engine.run(workload.to_requests())
    return SimResult(workload.name, policy_name, done, summarize(done, threshold),
                     mem.events if mem is not None else None)


def compare(workload: Workload, max_wait_ms: float = config.DEFAULT_MAX_WAIT_MS,
            policies=POLICY_NAMES) -> dict[str, dict]:
    """Same workload, every policy -> {policy: summary}."""
    return {p: simulate(workload, p, max_wait_ms, emit_queue_snapshots=False).summary for p in policies}
