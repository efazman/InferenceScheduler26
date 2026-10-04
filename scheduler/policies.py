"""Scheduling policies. A policy only orders the queue; it never calls the backend or predictor.

``order(queue, now)`` returns the whole queue in service order (the dashboard draws this), and
``choose_next`` is its first element. Every ordering ends with (enqueue_time, request_id), so the
result is fully deterministic, including ties.

FIFO      earliest enqueued first.
SEJF      Shortest Estimated Job First: lowest predicted_tokens first.
Adaptive  SEJF with max-wait starvation protection (exact rule in AdaptivePolicy).
"""

from __future__ import annotations

from scheduler import config
from scheduler.models import Request


def _fifo_key(r: Request):
    return (r.enqueue_time, r.request_id)


def _sejf_key(r: Request):
    return (r.predicted_tokens, r.enqueue_time, r.request_id)


class SchedulerPolicy:
    name = "base"

    def order(self, queue: list[Request], now: float) -> list[Request]:
        raise NotImplementedError

    def choose_next(self, queue: list[Request], now: float) -> Request | None:
        ordered = self.order(queue, now)
        return ordered[0] if ordered else None

    def reason(self, request: Request, now: float) -> str:
        """Short label explaining why ``request`` was chosen (recorded on request_selected)."""
        return self.name

    def describe(self) -> dict:
        return {"name": self.name}


class FIFOPolicy(SchedulerPolicy):
    name = "fifo"

    def order(self, queue, now):
        return sorted(queue, key=_fifo_key)


class SEJFPolicy(SchedulerPolicy):
    name = "sejf"

    def order(self, queue, now):
        return sorted(queue, key=_sejf_key)


class AdaptivePolicy(SchedulerPolicy):
    """SEJF + max-wait protection.

    At each decision point (the backend is idle and the queue is non-empty), with
    wait(r) = now - r.arrival_time:

      1. overdue = { r in queue : wait(r) >= max_wait_ms }
      2. if overdue is non-empty: run the OLDEST overdue request
         (min by arrival_time, then enqueue_time, then request_id)
      3. otherwise: run the shortest estimated job (min by predicted_tokens, enqueue_time, request_id)

    Why it bounds starvation: overdue requests are served in arrival order and ahead of everything
    else, so once a request becomes overdue only (a) the job already running and (b) requests that
    became overdue before it can still delay it. No amount of newer short traffic can. Its wait is
    therefore at most max_wait_ms + (that finite backlog). It is not a hard cap of max_wait_ms:
    with K=1 and no preemption a request can't be started while another job is running.
    """

    name = "adaptive"

    def __init__(self, max_wait_ms: float = config.DEFAULT_MAX_WAIT_MS):
        if max_wait_ms <= 0:
            raise ValueError("max_wait_ms must be positive")
        self.max_wait_ms = float(max_wait_ms)

    def is_overdue(self, r: Request, now: float) -> bool:
        return r.waited_ms(now) >= self.max_wait_ms

    def order(self, queue, now):
        overdue = sorted((r for r in queue if self.is_overdue(r, now)),
                         key=lambda r: (r.arrival_time, r.enqueue_time, r.request_id))
        rest = sorted((r for r in queue if not self.is_overdue(r, now)), key=_sejf_key)
        return overdue + rest

    def reason(self, request, now):
        return "overdue" if self.is_overdue(request, now) else "shortest_estimate"

    def describe(self):
        return {"name": self.name, "max_wait_ms": self.max_wait_ms}


POLICY_NAMES = ("fifo", "sejf", "adaptive")


def make_policy(name: str, max_wait_ms: float = config.DEFAULT_MAX_WAIT_MS) -> SchedulerPolicy:
    name = name.lower()
    if name == "fifo":
        return FIFOPolicy()
    if name == "sejf":
        return SEJFPolicy()
    if name == "adaptive":
        return AdaptivePolicy(max_wait_ms)
    raise ValueError(f"unknown policy {name!r}; choose from {POLICY_NAMES}")
