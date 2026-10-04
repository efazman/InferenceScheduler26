"""Workloads, simulator, comparison and CLI (all synthetic; no real model or backend)."""

import json
from pathlib import Path

import pytest

from scheduler.__main__ import main as cli
from scheduler.events import read_events
from scheduler.policies import POLICY_NAMES
from scheduler.simulator import compare, simulate
from scheduler.workloads import WORKLOAD_NAMES, make_workload, workload_from_prompts


@pytest.mark.parametrize("name", WORKLOAD_NAMES)
def test_workloads_are_deterministic(name):
    a, b = make_workload(name, seed=42), make_workload(name, seed=42)
    assert a.to_dict() == b.to_dict()
    assert make_workload(name, seed=7).to_dict() != a.to_dict()
    assert a.synthetic and all(r.prompt.startswith("[sim ") for r in a.requests)
    ids = [r.request_id for r in a.requests]
    assert len(set(ids)) == len(ids) and len({r.prompt for r in a.requests}) == len(ids)
    assert [r.arrival_ms for r in a.requests] == sorted(r.arrival_ms for r in a.requests)


@pytest.mark.parametrize("name,p_long", [("mostly_short", 0.2), ("balanced", 0.5), ("mostly_long", 0.8)])
def test_poisson_mix_proportions(name, p_long):
    wl = make_workload(name, n=2000)
    share = sum(r.size_class == "long" for r in wl.requests) / len(wl.requests)
    assert abs(share - p_long) < 0.04


def test_head_of_line_shape():
    wl = make_workload("head_of_line", n=60)
    classes = [r.size_class for r in wl.requests]
    assert classes[:2] == ["long", "long"] and set(classes[2:]) == {"short"} and len(classes) == 12


def test_bursty_has_a_one_second_burst():
    wl = make_workload("bursty", n=30)
    at = wl.params["burst_at_ms"]
    in_burst = [r for r in wl.requests if at <= r.arrival_ms <= at + 1000 and r.size_class == "short"]
    assert len(in_burst) >= wl.params["n_burst"]


def test_zero_noise_gives_oracle_predictions():
    wl = make_workload("balanced", prediction_noise=0.0)
    assert all(r.mock_predicted_tokens == r.mock_actual_tokens for r in wl.requests)


def test_simulated_service_time_model():
    wl = make_workload("balanced", n=20)
    res = simulate(wl, "fifo")
    for r in res.requests:
        assert r.service_time_ms == pytest.approx(60.0 + 13.3 * r.actual_output_tokens)


def test_qualitative_policy_behaviour_on_head_of_line():
    wl = make_workload("head_of_line", n=60)
    res = compare(wl, max_wait_ms=15_000)
    assert res["sejf"]["short_mean_latency_ms"] < res["fifo"]["short_mean_latency_ms"]  # HOL blocking removed
    assert res["sejf"]["long_max_queue_wait_ms"] > res["fifo"]["long_max_queue_wait_ms"]  # cost: long waits
    for s in res.values():
        assert s["n_completed"] == 12 and s["throughput_rps"] == pytest.approx(res["fifo"]["throughput_rps"])


def test_adaptive_bounds_max_wait_relative_to_sejf_under_load():
    wl = make_workload("mostly_short", n=400)
    res = compare(wl, max_wait_ms=15_000)
    assert res["adaptive"]["max_queue_wait_ms"] < res["sejf"]["max_queue_wait_ms"]
    assert res["adaptive"]["mean_latency_ms"] < res["fifo"]["mean_latency_ms"]


def test_workload_from_prompts_is_real_and_unlabelled():
    wl = workload_from_prompts(["a", "b", "c"], mean_interarrival_ms=100)
    assert not wl.synthetic and [r.size_class for r in wl.requests] == [None, None, None]
    assert wl.requests[0].arrival_ms == 0 and wl.requests[2].arrival_ms > wl.requests[1].arrival_ms


def test_cli_simulate_writes_events(tmp_path, capsys):
    out = tmp_path / "ev.jsonl"
    cli(["simulate", "--workload", "head_of_line", "--policy", "adaptive", "--events-out", str(out)])
    summary = json.loads(capsys.readouterr().out)
    assert "SIMULATED" in summary["warning"] and summary["n_completed"] == 12
    assert read_events(out)[-1]["event_type"] == "run_completed"


def test_cli_compare_and_export(tmp_path, capsys):
    cli(["compare", "--workloads", "head_of_line", "--out", str(tmp_path / "cmp.md")])
    assert "| Metric | FIFO | SEJF | Adaptive |" in (tmp_path / "cmp.md").read_text()
    cli(["export-ui", "--out-dir", str(tmp_path / "ui")])
    index = json.loads((tmp_path / "ui" / "index.json").read_text())
    assert [w["name"] for w in index["workloads"]] == list(WORKLOAD_NAMES)
    bundle = json.loads((tmp_path / "ui" / "head_of_line.json").read_text())
    assert set(bundle["runs"]) == set(POLICY_NAMES) and "SIMULATED" in bundle["warning"]
    assert any(e["event_type"] == "queue_snapshot" for e in bundle["runs"]["sejf"]["events"])


def test_cli_run_mock_realtime(tmp_path, capsys):
    cli(["run", "--backend", "mock", "--workload", "head_of_line", "--n", "60", "--policy", "sejf", "--speedup", "400",
         "--runs-dir", str(tmp_path), "--run-name", "demo"])
    meta = json.loads((tmp_path / "demo" / "summary.json").read_text())
    assert meta["summary"]["n_completed"] == 12 and "MOCK" in meta["warning"]
    events = read_events(tmp_path / "demo" / "events.jsonl")
    assert events[0]["simulated"] is False  # wall clock, not the virtual one


def test_cli_sweep(tmp_path, capsys):
    cli(["sweep", "--n", "40", "--workloads", "balanced", "--loads", "0.8", "--max-waits-s", "15,60",
         "--out", str(tmp_path / "sweep.md")])
    rows = [l for l in (tmp_path / "sweep.md").read_text().splitlines() if l.startswith("| balanced")]
    assert [r.split("|")[3].strip() for r in rows] == ["FIFO", "SEJF", "Adaptive", "Adaptive"]


def test_prompts_from_split_uses_held_out_ids(tmp_path, capsys):
    from ml.config import Config

    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"prompt_id": f"p{i}", "prompt": f"prompt {i}", "category": "x",
                                          "target_p90_output_tokens": 10.0 * i}) + "\n" for i in range(6)))
    art = tmp_path / "art"
    art.mkdir()
    Config(data_path=str(labels), target_field="target_p90_output_tokens").save(art / "config.json")
    (art / "splits.json").write_text(json.dumps({"train": ["p0", "p1", "p2", "p3"], "val": ["p4"], "test": ["p5"]}))
    out = tmp_path / "test_prompts.jsonl"
    cli(["prompts-from-split", "--artifacts", str(art), "--out", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows == [{"prompt_id": "p5", "prompt": "prompt 5", "category": "x", "label_target_tokens": 50.0}]


def test_cli_run_real_backend_path_with_prompt_file(tmp_path, capsys, fake_llama_url):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(json.dumps({"prompt_id": f"lmsys-{i}", "prompt": f"question {i}"}) + "\n"
                               for i in range(3)))
    cli(["run", "--backend", "llamacpp", "--url", fake_llama_url, "--policy", "adaptive",
         "--prompts-file", str(prompts), "--mean-interarrival-ms", "5", "--median-service-ms", "6000",
         "--runs-dir", str(tmp_path),
         "--run-name", "real"])
    meta = json.loads((tmp_path / "real" / "summary.json").read_text())
    assert meta["summary"]["n_completed"] == 3 and "warning" not in meta
    done = [e for e in read_events(tmp_path / "real" / "events.jsonl") if e["event_type"] == "inference_completed"]
    assert {e["request_id"] for e in done} == {"lmsys-0", "lmsys-1", "lmsys-2"}  # dataset ids kept


ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "distilbert_length_predictor"


@pytest.mark.skipif(not (ARTIFACTS / "head.pt").exists(), reason="no local predictor artifacts")
def test_real_predictor_adapters_load_local_artifacts():
    from scheduler.predictors import load_predictor, normalize_prediction

    for spec in (f"distilbert:{ARTIFACTS}", f"baseline:{ARTIFACTS}"):
        expected, _ = normalize_prediction(load_predictor(spec).predict("How do I reset my VPN password?"))
        assert expected >= 0
