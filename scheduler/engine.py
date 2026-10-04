"""K=1, non-preemptive scheduler engine.

The engine owns the request lifecycle and the clock; everything else is injected:

    SchedulerEngine(predictor, backend, policy, sink, clock)

- K = 1: at most one request is running at any time.
- Non-preemptive: once started, a request runs to completion; the policy is consulted only when
  the backend is idle and the queue is non-empty.
- The same loop serves simulation (VirtualClock: time jumps, the backend's reported latency
  advances the clock) and real runs (WallClock: arrivals are replayed in real time and the
  blocking generate() call takes as long as it takes).

Arrivals that happen while a job runs are admitted (predicted and enqueued) as soon as the engine
regains control, with their original arrival_time. With K=1 and no preemption this changes no
decision, because the next decision can only happen when the running job ends anyway.
"""

from __future__ import annotations

import time
import uuid

from scheduler import config, events as ev
from scheduler.clock import VirtualClock
from scheduler.events import Event, NullSink
from scheduler.metrics import summarize
from scheduler.models import Request, RequestState
from scheduler.predictors import normalize_prediction, predictor_name


class SchedulerEngine:
    def __init__(self, predictor, backend, policy, sink=None, clock=None, *, workload_name: str = "adhoc",
                 run_id: str | None = None, sim_predictor_latency_ms: float = config.SIM_PREDICTOR_LATENCY_MS,
                 emit_queue_snapshots: bool = True,
                 starvation_threshold_ms: float = config.DEFAULT_STARVATION_THRESHOLD_MS):
        self.predictor = predictor
        self.backend = backend
        self.policy = policy
        self.sink = sink or NullSink()
        self.clock = clock or VirtualClock()
        self.workload_name = workload_name
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.sim_predictor_latency_ms = sim_predictor_latency_ms
        self.emit_queue_snapshots = emit_queue_snapshots
        self.starvation_threshold_ms = starvation_threshold_ms

        self.queue: list[Request] = []
        self.active: Request | None = None
        self.finished: list[Request] = []
        self._seq = 0

    # ------------------------------------------------------------------ events

    def _emit(self, event_type: str, t: float, r: Request | None = None, **fields) -> None:
        self._seq += 1
        base = dict(event_type=event_type, timestamp_ms=t, seq=self._seq, run_id=self.run_id,
                    workload_name=self.workload_name, scheduler_policy=self.policy.name,
                    backend=getattr(self.backend, "name", type(self.backend).__name__),
                    predictor=predictor_name(self.predictor), simulated=self.clock.simulated,
                    queue_depth=len(self.queue))
        if r is not None:
            base.update(request_id=r.request_id, predicted_output_tokens=r.predicted_tokens,
                        uncertainty=r.uncertainty, actual_output_tokens=r.actual_output_tokens,
                        queue_wait_ms=r.queue_wait_ms, service_time_ms=r.service_time_ms,
                        end_to_end_latency_ms=r.end_to_end_latency_ms)
        base.update(fields)
        self.sink.emit(Event(**base))

    def _snapshot(self, t: float) -> None:
        if not self.emit_queue_snapshots:
            return
        order = self.policy.order(self.queue, t)
        data = {"order": [r.request_id for r in order],
                "running": self.active.request_id if self.active else None}
        if hasattr(self.policy, "is_overdue"):
            data["overdue"] = [r.request_id for r in order if self.policy.is_overdue(r, t)]
        self._emit(ev.QUEUE_SNAPSHOT, t, data=data)

    # ------------------------------------------------------------------ lifecycle

    def _admit(self, r: Request) -> None:
        self._emit(ev.REQUEST_ARRIVED, r.arrival_time, r, data={"size_class": r.size_class, "category": r.category,
                                                               "prompt_chars": len(r.prompt)})
        t0 = time.perf_counter()
        try:
            expected, unc = normalize_prediction(self.predictor.predict(r.prompt))
        except Exception as e:  # noqa: BLE001 - a bad prediction fails that request, not the run
            r.state, r.error = RequestState.FAILED, f"prediction failed: {e}"
            r.end_time = max(self.clock.now(), r.arrival_time)
            self.finished.append(r)
            self._emit(ev.REQUEST_FAILED, r.end_time, r, success=False, error=r.error)
            return
        r.predictor_latency_ms = (time.perf_counter() - t0) * 1000.0
        r.predicted_tokens, r.uncertainty = expected, unc
        if self.clock.simulated:
            r.enqueue_time = r.arrival_time + self.sim_predictor_latency_ms
            self.clock.advance_to(r.enqueue_time)
        else:
            r.enqueue_time = self.clock.now()
        self._emit(ev.COST_PREDICTED, r.enqueue_time, r, data={"predictor_latency_ms": r.predictor_latency_ms})
        r.state = RequestState.QUEUED
        self.queue.append(r)
        self._emit(ev.REQUEST_ENQUEUED, r.enqueue_time, r)
        self._snapshot(r.enqueue_time)

    def _run_one(self, r: Request) -> tuple[bool, str | None]:
        now = self.clock.now()
        self.queue.remove(r)
        r.metadata["selected_by"] = self.policy.reason(r, now)
        self.active = r
        r.state = RequestState.RUNNING
        r.start_time = now
        self._emit(ev.REQUEST_SELECTED, now, r, data={"reason": r.metadata["selected_by"]})
        self._snapshot(now)
        self._emit(ev.INFERENCE_STARTED, now, r)
        try:
            out = self.backend.generate(r.prompt, seed=r.seed)
            self.clock.after_service(float(out["latency_ms"]))
            r.actual_output_tokens = int(out["output_tokens"])
            r.finish_reason = out.get("finish_reason")
            ok, err = True, None
        except Exception as e:  # noqa: BLE001 - one failed generation must not stop the run
            ok, err = False, f"{type(e).__name__}: {e}"
        return ok, err

    def _finish(self, r: Request, ok: bool, err: str | None) -> None:
        r.end_time = self.clock.now()
        self.active = None
        self.finished.append(r)
        if ok:
            r.state = RequestState.COMPLETED
            self._emit(ev.INFERENCE_COMPLETED, r.end_time, r, success=True,
                       data={"finish_reason": r.finish_reason, "selected_by": r.metadata.get("selected_by")})
        else:
            r.state, r.error = RequestState.FAILED, err
            self._emit(ev.REQUEST_FAILED, r.end_time, r, success=False, error=err)

    # ------------------------------------------------------------------ main loop

    def run(self, requests: list[Request]) -> list[Request]:
        pending = sorted(requests, key=lambda r: (r.arrival_time, r.request_id))
        ids = [r.request_id for r in pending]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate request_id in workload")
        self._emit(ev.RUN_STARTED, self.clock.now(),
                   data={"policy": self.policy.describe(), "n_requests": len(pending), "k": 1, "preemptive": False})
        i = 0

        def admit_until(t: float) -> None:
            nonlocal i
            while i < len(pending) and pending[i].arrival_time <= t:
                self._admit(pending[i])
                i += 1

        while i < len(pending) or self.queue:
            admit_until(self.clock.now())
            if not self.queue:
                if i < len(pending):
                    self.clock.advance_to(pending[i].arrival_time)
                continue
            r = self.policy.choose_next(self.queue, self.clock.now())
            ok, err = self._run_one(r)
            admit_until(self.clock.now())  # arrivals during the job, stamped with their own times
            self._finish(r, ok, err)

        summary = summarize(self.finished, self.starvation_threshold_ms)
        self._emit(ev.RUN_COMPLETED, self.clock.now(), data={"summary": summary})
        return self.finished
