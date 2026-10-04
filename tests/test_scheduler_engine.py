"""Engine behaviour: K=1, non-preemption, arrivals during service, starvation, failures, events."""

import math

import numpy as np
import pytest

from scheduler import events as ev
from scheduler.backends import LlamaCppBackend, MockInferenceBackend
from scheduler.clock import VirtualClock, WallClock
from scheduler.engine import SchedulerEngine
from scheduler.events import (FanoutSink, InMemoryEventSink, LocalJsonlEventSink, NullSink, TigerDataEventSink,
                              read_events)
from scheduler.metrics import percentile, summarize
from scheduler.models import Request, RequestState
from scheduler.policies import AdaptivePolicy, FIFOPolicy, SEJFPolicy
from scheduler.predictors import MockPredictor, normalize_prediction


def build(spec):
    """spec: [(request_id, arrival_ms, tokens)] -> requests + oracle predictor + 1 ms/token backend."""
    reqs = [Request(rid, f"prompt-{rid}", float(t)) for rid, t, _ in spec]
    table = {f"prompt-{rid}": n for rid, _, n in spec}
    return reqs, MockPredictor(table), MockInferenceBackend(table, base_overhead_ms=0, ms_per_token=1.0)


def run(spec, policy, sink=None, **kw):
    reqs, pred, backend = build(spec)
    eng = SchedulerEngine(pred, backend, policy, sink or InMemoryEventSink(), VirtualClock(),
                          sim_predictor_latency_ms=0, **kw)
    done = eng.run(reqs)
    return {r.request_id: r for r in done}, eng


def execution_order(done):
    return [r.request_id for r in sorted(done.values(), key=lambda r: r.start_time)]


# Head-of-line: w0 occupies the backend while a long job and three short jobs queue up behind it.
HOL = [("w0", 0, 100), ("long", 10, 1000), ("s1", 20, 50), ("s2", 30, 50), ("s3", 40, 50)]


def test_fifo_preserves_arrival_order():
    done, _ = run(HOL, FIFOPolicy())
    assert execution_order(done) == ["w0", "long", "s1", "s2", "s3"]


def test_sejf_reorders_shortest_first():
    done, _ = run(HOL, SEJFPolicy())
    assert execution_order(done) == ["w0", "s1", "s2", "s3", "long"]
    fifo, _ = run(HOL, FIFOPolicy())
    assert np.mean([done[s].end_to_end_latency_ms for s in ("s1", "s2", "s3")]) < \
        np.mean([fifo[s].end_to_end_latency_ms for s in ("s1", "s2", "s3")])


@pytest.mark.parametrize("policy", [FIFOPolicy(), SEJFPolicy(), AdaptivePolicy(500)], ids=lambda p: p.name)
def test_k1_non_preemptive_no_overlap(policy):
    spec = [("big", 0, 2000)] + [(f"s{i}", 10 + i * 5, 10) for i in range(10)]
    done, _ = run(spec, policy)
    assert done["big"].start_time == 0 and done["big"].end_time == 2000  # never interrupted
    intervals = sorted((r.start_time, r.end_time) for r in done.values())
    for (s0, e0), (s1, _) in zip(intervals, intervals[1:]):
        assert s1 >= e0  # at most one request running at any time
    assert all(r.start_time >= 2000 for k, r in done.items() if k != "big")


def test_arrivals_during_service_keep_their_arrival_time():
    done, eng = run([("a", 0, 1000), ("b", 400, 10)], SEJFPolicy())
    assert done["b"].arrival_time == 400 and done["b"].start_time == 1000
    assert done["b"].queue_wait_ms == 600
    types = [(e.event_type, e.request_id) for e in eng.sink.events if e.request_id == "b"]
    assert [t for t, _ in types] == [ev.REQUEST_ARRIVED, ev.COST_PREDICTED, ev.REQUEST_ENQUEUED,
                                     ev.REQUEST_SELECTED, ev.INFERENCE_STARTED, ev.INFERENCE_COMPLETED]
    stamps = [e.timestamp_ms for e in eng.sink.events]
    assert stamps == sorted(stamps)  # simulated logs are chronological


def test_idle_gap_jumps_clock_to_next_arrival():
    done, _ = run([("a", 0, 10), ("b", 5000, 10)], FIFOPolicy())
    assert done["b"].start_time == 5000 and done["b"].queue_wait_ms == 0


def test_equal_cost_requests_run_in_arrival_order():
    spec = [("w0", 0, 100)] + [(f"r{i}", 10 + i, 50) for i in range(5)]
    for policy in (FIFOPolicy(), SEJFPolicy(), AdaptivePolicy(10_000)):
        done, _ = run(spec, policy)
        assert execution_order(done) == ["w0", "r0", "r1", "r2", "r3", "r4"]


STARVE = [("w0", 0, 100), ("long", 1, 5000)] + [(f"s{i:03d}", 2 + 150 * i, 200) for i in range(100)]


def test_sejf_starves_long_job_under_steady_short_traffic():
    done, _ = run(STARVE, SEJFPolicy())
    assert done["long"].queue_wait_ms > 15_000  # waits behind the entire short stream


def test_adaptive_bounds_starvation():
    max_wait = 2000
    done, _ = run(STARVE, AdaptivePolicy(max_wait))
    # Overdue at t=2001; at most one in-flight 200 ms short job delays it after that.
    assert done["long"].queue_wait_ms <= max_wait + 200 + 1
    assert done["long"].metadata["selected_by"] == "overdue"
    sejf, _ = run(STARVE, SEJFPolicy())
    shorts = [k for k in done if k.startswith("s")]
    assert np.mean([done[k].end_to_end_latency_ms for k in shorts]) < \
        np.mean([run(STARVE, FIFOPolicy())[0][k].end_to_end_latency_ms for k in shorts])


def test_simulation_is_deterministic():
    def strip(events):  # drop wall-clock fields; predictor_latency_ms is a real timing even in simulation
        out = []
        for e in events:
            d = {k: v for k, v in e.to_dict().items() if k not in ("wall_time", "run_id")}
            d["data"] = {k: v for k, v in d["data"].items() if k != "predictor_latency_ms"}
            out.append(d)
        return out

    _, a = run(HOL, AdaptivePolicy(50))
    _, b = run(HOL, AdaptivePolicy(50))
    assert strip(a.sink.events) == strip(b.sink.events)


def test_backend_failure_is_recorded_and_run_continues():
    reqs, pred, _ = build(HOL)
    backend = MockInferenceBackend({f"prompt-{r}": n for r, _, n in HOL}, base_overhead_ms=0, ms_per_token=1,
                                   fail_prompts={"prompt-long"})
    sink = InMemoryEventSink()
    done = {r.request_id: r for r in SchedulerEngine(pred, backend, FIFOPolicy(), sink, VirtualClock()).run(reqs)}
    assert done["long"].state == RequestState.FAILED and "mock backend failure" in done["long"].error
    assert all(done[k].state == RequestState.COMPLETED for k in ("w0", "s1", "s2", "s3"))
    assert len(sink.of_type(ev.REQUEST_FAILED)) == 1
    summary = sink.of_type(ev.RUN_COMPLETED)[0].data["summary"]
    assert summary["n_failed"] == 1 and summary["n_completed"] == 4


def test_predictor_failure_fails_only_that_request():
    class Flaky(MockPredictor):
        def predict(self, prompt):
            if prompt == "prompt-s2":
                return {"oops": 1}
            return super().predict(prompt)

    reqs, pred, backend = build(HOL)
    done = {r.request_id: r for r in SchedulerEngine(Flaky(pred.table), backend, SEJFPolicy()).run(reqs)}
    assert done["s2"].state == RequestState.FAILED and done["s2"].start_time is None
    assert sum(r.state == RequestState.COMPLETED for r in done.values()) == 4


def test_duplicate_request_ids_rejected():
    reqs, pred, backend = build([("a", 0, 1), ("a", 1, 1)])
    with pytest.raises(ValueError):
        SchedulerEngine(pred, backend, FIFOPolicy()).run(reqs)


def test_queue_snapshots_follow_policy_order():
    _, eng = run(HOL, SEJFPolicy())
    snaps = eng.sink.of_type(ev.QUEUE_SNAPSHOT)
    full = max(snaps, key=lambda e: len(e.data["order"]))
    assert full.data["order"] == ["s1", "s2", "s3", "long"] and full.data["running"] == "w0"


def test_wall_clock_run_with_realtime_mock():
    reqs, pred, _ = build([("a", 0, 10), ("b", 5, 10), ("c", 6, 1)])
    backend = MockInferenceBackend(pred.table, base_overhead_ms=0, ms_per_token=1.0, realtime=True, speedup=1.0)
    done = {r.request_id: r for r in SchedulerEngine(pred, backend, SEJFPolicy(), clock=WallClock()).run(reqs)}
    assert all(r.state == RequestState.COMPLETED for r in done.values())
    assert done["c"].start_time < done["b"].start_time  # shorter prediction wins in real time too
    assert done["a"].service_time_ms >= 9  # it really slept


# --------------------------------------------------------------------------- metrics + sinks + adapters

def test_percentile_matches_numpy():
    xs = [5, 1, 9, 3, 7, 2, 8]
    for q in (0, 50, 90, 95, 99, 100):
        assert percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)))
    assert percentile([], 50) is None


def test_summarize_known_values():
    done, _ = run(HOL, FIFOPolicy())
    s = summarize(list(done.values()), starvation_threshold_ms=1000)
    lat = [r.end_to_end_latency_ms for r in done.values()]
    assert s["n_completed"] == 5 and s["mean_latency_ms"] == pytest.approx(np.mean(lat))
    assert s["max_queue_wait_ms"] == max(r.queue_wait_ms for r in done.values())
    assert s["makespan_ms"] == 1250 and s["throughput_rps"] == pytest.approx(5 / 1.25)
    assert s["starvation_count"] == 3  # s1, s2, s3 waited > 1 s behind the long job
    assert s["p99_reliable"] is False


def test_jsonl_sink_roundtrip_and_partial_line(tmp_path):
    path = tmp_path / "events.jsonl"
    _, _ = run(HOL, SEJFPolicy(), sink=LocalJsonlEventSink(path))
    with path.open("a") as f:
        f.write('{"event_type": "half')  # live log mid-write
    events = read_events(path)
    assert events[0]["event_type"] == ev.RUN_STARTED and events[-1]["event_type"] == ev.RUN_COMPLETED
    assert {e["event_type"] for e in events} >= {ev.REQUEST_ARRIVED, ev.COST_PREDICTED, ev.REQUEST_ENQUEUED,
                                                 ev.REQUEST_SELECTED, ev.INFERENCE_STARTED,
                                                 ev.INFERENCE_COMPLETED}


def test_fanout_tolerates_failing_secondary_sink():
    class Broken(NullSink):
        def emit(self, event):
            raise ConnectionError("db down")

    mem = InMemoryEventSink()
    fan = FanoutSink(mem, Broken())
    run(HOL, FIFOPolicy(), sink=fan)
    assert mem.events and fan.secondary_errors == len(mem.events)


def test_tiger_data_sink_is_an_explicit_placeholder():
    with pytest.raises(NotImplementedError):
        TigerDataEventSink(dsn="postgres://example")


def test_normalize_prediction_accepts_extra_keys_and_rejects_bad_output():
    assert normalize_prediction({"expected_output_tokens": 12.5, "uncertainty": 2.9,
                                 "bin_probabilities": [0.05] * 20}) == (12.5, 2.9)
    assert normalize_prediction({"expected_output_tokens": 3}) == (3.0, None)
    for bad in ({}, {"expected_output_tokens": "x"}, {"expected_output_tokens": -1},
                {"expected_output_tokens": math.nan}):
        with pytest.raises(ValueError):
            normalize_prediction(bad)


def test_llamacpp_backend_drives_engine_against_fake_server(fake_llama_url):
    backend = LlamaCppBackend(fake_llama_url, timeout_s=5)
    backend.check()
    reqs = [Request("a", "hello", 0, seed=1), Request("b", "write an essay", 0, seed=2)]
    done = {r.request_id: r for r in SchedulerEngine(MockPredictor(), backend, SEJFPolicy(),
                                                      clock=WallClock()).run(reqs)}
    assert done["a"].actual_output_tokens == 8 and done["b"].actual_output_tokens == 9
