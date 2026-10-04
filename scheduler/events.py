"""Scheduler events + sinks.

One flat event schema is shared by the JSONL log, the dashboard and (later) Tiger Data, so the
same file format works for simulated and real runs. ``timestamp_ms`` is engine-clock time
(virtual in simulation, ms since run start in real runs); ``wall_time`` is when it was emitted.

Event types (in a request's lifetime order):
    request_arrived -> cost_predicted -> request_enqueued -> request_selected
    -> inference_started -> inference_completed | request_failed
Run-level: run_started, queue_snapshot (policy order after every queue change), run_completed.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

RUN_STARTED = "run_started"
REQUEST_ARRIVED = "request_arrived"
COST_PREDICTED = "cost_predicted"
REQUEST_ENQUEUED = "request_enqueued"
REQUEST_SELECTED = "request_selected"
INFERENCE_STARTED = "inference_started"
INFERENCE_COMPLETED = "inference_completed"
REQUEST_FAILED = "request_failed"
QUEUE_SNAPSHOT = "queue_snapshot"
RUN_COMPLETED = "run_completed"

EVENT_TYPES = (RUN_STARTED, REQUEST_ARRIVED, COST_PREDICTED, REQUEST_ENQUEUED, REQUEST_SELECTED,
               INFERENCE_STARTED, INFERENCE_COMPLETED, REQUEST_FAILED, QUEUE_SNAPSHOT, RUN_COMPLETED)


@dataclass
class Event:
    event_type: str
    timestamp_ms: float
    seq: int  # emission order within the run; tie-breaker for equal timestamps
    run_id: str
    workload_name: str
    scheduler_policy: str
    backend: str
    predictor: str
    simulated: bool
    request_id: str | None = None
    queue_depth: int | None = None
    predicted_output_tokens: float | None = None
    uncertainty: float | None = None
    actual_output_tokens: int | None = None
    queue_wait_ms: float | None = None
    service_time_ms: float | None = None
    end_to_end_latency_ms: float | None = None
    success: bool | None = None
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    wall_time: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"))

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...
    def close(self) -> None: ...


class NullSink:
    def emit(self, event: Event) -> None:
        pass

    def close(self) -> None:
        pass


class InMemoryEventSink:
    def __init__(self):
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def of_type(self, event_type: str) -> list[Event]:
        return [e for e in self.events if e.event_type == event_type]


class LocalJsonlEventSink:
    """Append-only JSONL, flushed per event so a live dashboard can tail it.
    ``fsync=True`` additionally survives power loss (slower; use for long real runs)."""

    def __init__(self, path: str | Path, fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")
        self.fsync = fsync

    def emit(self, event: Event) -> None:
        self._f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


class FanoutSink:
    """Send every event to several sinks (e.g. local JSONL + Tiger Data). A failing secondary sink
    must never break a run, so errors from sinks after the first are counted, not raised."""

    def __init__(self, primary: EventSink, *secondary: EventSink):
        self.primary = primary
        self.secondary = list(secondary)
        self.secondary_errors = 0

    def emit(self, event: Event) -> None:
        self.primary.emit(event)
        for s in self.secondary:
            try:
                s.emit(event)
            except Exception:  # noqa: BLE001
                self.secondary_errors += 1

    def close(self) -> None:
        for s in [self.primary, *self.secondary]:
            s.close()


TIGER_COLUMNS = ("timestamp", "engine_time_ms", "seq", "run_id", "manifest_id", "event_type", "request_id",
                 "policy", "queue_depth", "predicted_output_tokens", "uncertainty", "actual_output_tokens",
                 "queue_wait_ms", "service_time_ms", "end_to_end_latency_ms", "workload_name", "backend",
                 "predictor", "simulated", "success", "failure_reason", "data")


def tiger_row(event: dict, manifest_id: str | None = None) -> dict:
    """Map one event (Event.to_dict() or a JSONL line) to a scheduler_events row
    (scheduler/tigerdata_schema.sql). ``manifest_id`` comes from the run's run_started event."""
    return {
        "timestamp": event.get("wall_time"), "engine_time_ms": event["timestamp_ms"], "seq": event["seq"],
        "run_id": event.get("run_id"), "manifest_id": manifest_id, "event_type": event["event_type"],
        "request_id": event.get("request_id"), "policy": event.get("scheduler_policy"),
        "queue_depth": event.get("queue_depth"), "predicted_output_tokens": event.get("predicted_output_tokens"),
        "uncertainty": event.get("uncertainty"), "actual_output_tokens": event.get("actual_output_tokens"),
        "queue_wait_ms": event.get("queue_wait_ms"), "service_time_ms": event.get("service_time_ms"),
        "end_to_end_latency_ms": event.get("end_to_end_latency_ms"), "workload_name": event.get("workload_name"),
        "backend": event.get("backend"), "predictor": event.get("predictor"), "simulated": event.get("simulated"),
        "success": event.get("success"), "failure_reason": event.get("error"), "data": event.get("data") or {},
    }


class TigerDataEventSink:
    """INTEGRATION POINT - not implemented (no credentials yet). Local JSONL never depends on it.

    Tiger Data is PostgreSQL + TimescaleDB. The table is in scheduler/tigerdata_schema.sql, and
    tiger_row() already maps events to its columns, so the remaining work is a psycopg connection
    from TIGER_DATA_DSN that buffers tiger_row(event.to_dict(), manifest_id) rows and inserts them
    in batches (flushing on close). Wire it as
    FanoutSink(LocalJsonlEventSink(...), TigerDataEventSink(...)) in scheduler/__main__.py
    cmd_run, so the local log stays the source of truth and a database outage can't break a run.
    Never store credentials in the repo.
    """

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get("TIGER_DATA_DSN")
        raise NotImplementedError(
            "TigerDataEventSink is a placeholder; use LocalJsonlEventSink until Tiger Data is wired up "
            "(see the docstring for the planned schema).")

    def emit(self, event: Event) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        pass


def read_events(path: str | Path) -> list[dict]:
    """Load a JSONL event log, ignoring a partially written last line (live logs)."""
    events = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return sorted(events, key=lambda e: (e["timestamp_ms"], e["seq"]))
