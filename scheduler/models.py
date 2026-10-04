"""Core request model. Times are milliseconds on the engine's clock (virtual in simulation,
wall-clock-since-run-start in real runs)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RequestState(str, Enum):
    PENDING = "pending"  # not yet arrived
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Request:
    request_id: str
    prompt: str
    arrival_time: float
    seed: int | None = None
    # Workload labels used ONLY for reporting (e.g. "short"/"long"); policies never read them.
    size_class: str | None = None
    category: str | None = None

    state: RequestState = RequestState.PENDING
    enqueue_time: float | None = None
    start_time: float | None = None
    end_time: float | None = None

    predicted_tokens: float | None = None
    uncertainty: float | None = None
    predictor_latency_ms: float | None = None

    actual_output_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    # Free-form per-request data (e.g. which rule chose it: {"selected_by": "overdue"}).
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- derived (computed, not stored) ------------------------------------------
    # queue_wait is measured from ARRIVAL, i.e. what the user experiences before generation starts
    # (it includes the few ms spent predicting).
    @property
    def queue_wait_ms(self) -> float | None:
        return None if self.start_time is None else self.start_time - self.arrival_time

    @property
    def service_time_ms(self) -> float | None:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def end_to_end_latency_ms(self) -> float | None:
        return None if self.end_time is None else self.end_time - self.arrival_time

    def waited_ms(self, now: float) -> float:
        return now - self.arrival_time

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        d["queue_wait_ms"] = self.queue_wait_ms
        d["service_time_ms"] = self.service_time_ms
        d["end_to_end_latency_ms"] = self.end_to_end_latency_ms
        return d
