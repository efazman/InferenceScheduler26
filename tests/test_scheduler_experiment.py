"""Real-experiment readiness: manifests, derived MAX_WAIT, run provenance, the report's
like-for-like checks, and the llama.cpp run path (against a fake server)."""

import json
import statistics

import pytest

from scheduler.__main__ import main as cli
from scheduler.clock import VirtualClock
from scheduler.engine import SchedulerEngine
from scheduler.events import read_events
from scheduler.manifest import Manifest, manifest_from_prompt_rows, manifest_from_workload
from scheduler.max_wait import resolve_max_wait, service_times_from_file
from scheduler.policies import make_policy
from scheduler.workloads import make_workload

ROWS = [{"prompt_id": f"lmsys-{i:02d}", "prompt": f"question number {i}", "category": "knowledge_factual",
         "label_target_tokens": 50.0 + i} for i in range(6)]


# --------------------------------------------------------------------------- manifest

def test_manifest_is_deterministic_and_order_free():
    a = manifest_from_workload(make_workload("head_of_line", n=60))
    b = manifest_from_workload(make_workload("head_of_line", n=60))
    assert a.manifest_id == b.manifest_id and a.requests == b.requests
    keys = set().union(*(r.keys() for r in a.requests))
    assert not keys & {"start_time", "end_time", "position", "policy", "order"}  # no schedule baked in
    assert {"request_id", "prompt", "arrival_ms", "seed"} <= keys


def test_manifest_roundtrip(tmp_path):
    m = manifest_from_prompt_rows(ROWS, "real_test", mean_interarrival_ms=7000, seed=42,
                                  source={"type": "prompts_file"})
    path = m.save(tmp_path / "m.json")
    back = Manifest.load(path)
    assert back.manifest_id == m.manifest_id and not back.synthetic
    assert [r["request_id"] for r in back.requests] == [r["prompt_id"] for r in ROWS]  # dataset ids kept
    assert back.requests[0]["arrival_ms"] == 0 and back.requests[0]["mock_actual_tokens"] is None
    wl = back.to_workload()
    assert [w.arrival_ms for w in wl.requests] == [r["arrival_ms"] for r in back.requests]
    assert [w.seed for w in wl.requests] == list(range(6)) and wl.requests[0].category == "knowledge_factual"


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["requests"].append(dict(d["requests"][0])), "duplicate"),
    (lambda d: d["requests"].reverse(), "sorted"),
    (lambda d: d["requests"][1].pop("prompt"), "missing"),
    (lambda d: d["requests"][1].update(arrival_ms=-5), "negative"),
    (lambda d: d["requests"][1].update(prompt="edited"), "manifest_id"),
    (lambda d: d.update(manifest_version=99), "manifest_version"),
])
def test_manifest_validation(tmp_path, mutate, msg):
    m = manifest_from_prompt_rows(ROWS, "real_test", mean_interarrival_ms=7000, seed=42)
    d = m.to_dict()
    mutate(d)
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match=msg):
        Manifest.load(tmp_path / "bad.json")


def test_all_policies_replay_the_same_trace_in_different_orders():
    wl = manifest_from_workload(make_workload("head_of_line", n=60)).to_workload()
    orders, arrivals = {}, {}
    for policy in ("fifo", "sejf", "adaptive"):
        done = SchedulerEngine(wl.mock_predictor(), wl.mock_backend(), make_policy(policy), clock=VirtualClock()).run(
            wl.to_requests())
        arrivals[policy] = {r.request_id: r.arrival_time for r in done}
        orders[policy] = [r.request_id for r in sorted(done, key=lambda r: r.start_time)]
    assert arrivals["fifo"] == arrivals["sejf"] == arrivals["adaptive"]
    assert orders["fifo"] != orders["sejf"]


# --------------------------------------------------------------------------- derived MAX_WAIT

def _label_runs(tmp_path, latencies, mock_lines=0):
    path = tmp_path / "runs.jsonl"
    lines = [{"prompt_id": f"p{i}", "seed": 42, "latency_ms": v, "output_tokens": 100, "backend": "openai-compat",
              "is_mock": False} for i, v in enumerate(latencies)]
    lines += [{"prompt_id": "m", "seed": 1, "latency_ms": 1.0, "backend": "mock", "is_mock": True}] * mock_lines
    path.write_text("".join(json.dumps(x) + "\n" for x in lines) + '{"partial')
    return path


def test_max_wait_explicit_and_given_median():
    assert resolve_max_wait(explicit_ms=9000).max_wait_ms == 9000
    mw = resolve_max_wait(median_service_ms=6000)
    assert mw.mode == "derived" and mw.multiplier == 3.0 and mw.max_wait_ms == 18000
    assert resolve_max_wait(multiplier=5, median_service_ms=6000).max_wait_ms == 30000
    assert "3 x median service 6.00 s" in mw.describe()


def test_max_wait_from_label_runs_ignores_mock_and_partial_lines(tmp_path):
    lat = [4000.0, 6000.0, 6500.0, 9000.0, 30000.0]
    path = _label_runs(tmp_path, lat, mock_lines=50)
    assert service_times_from_file(path) == lat
    mw = resolve_max_wait(median_from=path)
    assert mw.median_service_ms == statistics.median(lat) and mw.max_wait_ms == 3 * 6500 and mw.n_samples == 5


def test_max_wait_from_scheduler_events(tmp_path):
    path = tmp_path / "events.jsonl"
    evs = [{"event_type": "inference_completed", "service_time_ms": v, "backend": "llamacpp", "simulated": False}
           for v in (100.0, 300.0, 200.0)] + [{"event_type": "inference_started", "service_time_ms": None}]
    path.write_text("".join(json.dumps(e) + "\n" for e in evs))
    assert resolve_max_wait(multiplier=2, median_from=path).max_wait_ms == 400


def test_max_wait_refuses_fake_or_missing_sources(tmp_path):
    with pytest.raises(ValueError, match="no real service-time"):
        resolve_max_wait(median_from=_label_runs(tmp_path, [], mock_lines=10))
    with pytest.raises(ValueError, match="no MAX_WAIT source"):
        resolve_max_wait()
    for bad in (dict(explicit_ms=0), dict(multiplier=0, median_service_ms=10), dict(median_service_ms=-1)):
        with pytest.raises(ValueError):
            resolve_max_wait(**bad)


def test_measure_service_cli(tmp_path, capsys):
    cli(["measure-service", "--from", str(_label_runs(tmp_path, [1000.0, 2000.0, 3000.0]))])
    out = json.loads(capsys.readouterr().out)
    assert out["median_service_ms"] == 2000 and out["derived_max_wait_ms"] == 6000 and out["n"] == 3


# --------------------------------------------------------------------------- run provenance + report

def _mock_runs(tmp_path, manifest_path, policies=("fifo", "sejf", "adaptive"), prefix="r"):
    for p in policies:
        cli(["run", "--backend", "mock", "--manifest", str(manifest_path), "--policy", p, "--speedup", "500",
             "--runs-dir", str(tmp_path / "runs"), "--run-name", f"{prefix}-{p}"])
    return [str(tmp_path / "runs" / f"{prefix}-{p}") for p in policies]


def test_report_passes_for_same_manifest_and_fails_otherwise(tmp_path, capsys):
    cli(["make-manifest", "--workload", "head_of_line", "--n", "60", "--out", str(tmp_path / "a.json")])
    cli(["make-manifest", "--workload", "head_of_line", "--n", "60", "--seed", "7", "--out", str(tmp_path / "b.json")])
    runs = _mock_runs(tmp_path, tmp_path / "a.json")
    capsys.readouterr()
    cli(["report", *runs, "--out", str(tmp_path / "report.md")])
    report = (tmp_path / "report.md").read_text()
    assert "FAIL" not in report and "MOCK / SIMULATED" in report and "sanity check" in report
    other = _mock_runs(tmp_path, tmp_path / "b.json", policies=("sejf",), prefix="other")
    with pytest.raises(SystemExit, match="FAILED"):
        cli(["report", runs[0], other[0], runs[2]])


def test_run_refuses_to_append_to_an_existing_run(tmp_path):
    cli(["make-manifest", "--workload", "head_of_line", "--n", "60", "--out", str(tmp_path / "a.json")])
    _mock_runs(tmp_path, tmp_path / "a.json", policies=("fifo",))
    with pytest.raises(SystemExit, match="already has events"):
        _mock_runs(tmp_path, tmp_path / "a.json", policies=("fifo",))


def test_real_run_requires_a_max_wait_source(tmp_path, fake_llama_url):
    m = manifest_from_prompt_rows(ROWS[:2], "real_test", 5, 42).save(tmp_path / "m.json")
    with pytest.raises(SystemExit, match="no MAX_WAIT source"):
        cli(["run", "--backend", "llamacpp", "--url", fake_llama_url, "--manifest", str(m), "--policy", "adaptive",
             "--runs-dir", str(tmp_path / "runs")])


def test_real_run_records_derived_max_wait_and_provenance(tmp_path, fake_llama_url, capsys):
    m = manifest_from_prompt_rows(ROWS[:3], "real_test", 5, 42)
    mpath = m.save(tmp_path / "m.json")
    runs = _label_runs(tmp_path, [1000.0, 2000.0, 9000.0])
    cli(["run", "--backend", "llamacpp", "--url", fake_llama_url, "--timeout-s", "12", "--manifest", str(mpath),
         "--policy", "adaptive", "--median-service-from", str(runs), "--runs-dir", str(tmp_path / "runs"),
         "--run-name", "real-adaptive"])
    run_dir = tmp_path / "runs" / "real-adaptive"
    meta = json.loads((run_dir / "summary.json").read_text())
    assert meta["measurement"] == "real" and meta["manifest_id"] == m.manifest_id
    assert meta["max_wait"]["mode"] == "derived" and meta["max_wait_ms"] == 6000.0
    assert meta["generation"]["max_new_tokens"] > 0 and "system_prompt_sha256" in meta["generation"]
    assert Manifest.load(run_dir / "manifest.json").manifest_id == m.manifest_id
    started = read_events(run_dir / "events.jsonl")[0]
    assert started["event_type"] == "run_started" and started["data"]["policy"]["max_wait_ms"] == 6000.0
    assert started["data"]["max_wait"]["median_service_ms"] == 2000.0
    done = [e for e in read_events(run_dir / "events.jsonl") if e["event_type"] == "inference_completed"]
    assert len(done) == 3 and all(e["data"]["prompt_tokens"] == 5 for e in done)


def test_llamacpp_backend_timeout_and_url_are_configurable():
    from scheduler.backends import LlamaCppBackend

    b = LlamaCppBackend("http://10.0.0.5:9999/", timeout_s=42)
    assert b.base_url == "http://10.0.0.5:9999/" and b._client.timeout == 42
    assert b._client.base_url == "http://10.0.0.5:9999"
