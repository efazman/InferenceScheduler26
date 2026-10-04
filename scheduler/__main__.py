"""Scheduler command line.

    python -m scheduler simulate  --workload head_of_line --policy sejf [--events-out f.jsonl]
    python -m scheduler compare   [--n 1000] [--out docs/sim_results.md]
    python -m scheduler export-ui                      # bundle simulated runs for the dashboard
    python -m scheduler run --backend mock --workload balanced --policy adaptive --speedup 10
    python -m scheduler run --backend llamacpp --predictor distilbert:<artifacts> \\
        --prompts-file <prompts.jsonl> --policy adaptive         # real run on the RTX 3060 Ti
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from scheduler import config
from scheduler.clock import WallClock
from scheduler.engine import SchedulerEngine
from scheduler.events import LocalJsonlEventSink
from scheduler.metrics import P99_MIN_SAMPLES
from scheduler.policies import POLICY_NAMES, make_policy
from scheduler.predictors import MockPredictor, load_predictor
from scheduler.simulator import compare, simulate
from scheduler.workloads import WORKLOAD_NAMES, make_workload, workload_from_prompts

SIM_WARNING = ("SIMULATED: synthetic workloads, mock predictor, mock backend "
               "(service = base overhead + tokens x ms/token). Validates policy behaviour only; "
               "not a measurement of Llama or of the predictor.")

ROWS = [  # (label, key, scale) - ms values shown in seconds
    ("Mean latency (s)", "mean_latency_ms", 1e-3), ("p50 latency (s)", "p50_latency_ms", 1e-3),
    ("p95 latency (s)", "p95_latency_ms", 1e-3), ("p99 latency (s)", "p99_latency_ms", 1e-3),
    ("Mean queue wait (s)", "mean_queue_wait_ms", 1e-3), ("Max queue wait (s)", "max_queue_wait_ms", 1e-3),
    ("Throughput (req/min)", "throughput_rps", 60.0), ("Starved (wait > threshold)", "starvation_count", 1),
    ("Short-request mean latency (s)", "short_mean_latency_ms", 1e-3),
    ("Long-request mean latency (s)", "long_mean_latency_ms", 1e-3),
    ("Long-request max wait (s)", "long_max_queue_wait_ms", 1e-3),
]


def _fmt(v, scale):
    if v is None:
        return "-"
    return f"{v:.0f}" if scale == 1 else f"{v * scale:.2f}"


def comparison_table(name: str, desc: str, results: dict, n: int) -> str:
    any_s = next(iter(results.values()))
    lines = [f"### {name} ({n} requests: {any_s['short_n']} short / {any_s['long_n']} long)", "", desc, "",
             "| Metric | FIFO | SEJF | Adaptive |", "| --- | --- | --- | --- |"]
    for label, key, scale in ROWS:
        if key == "p99_latency_ms" and not any_s["p99_reliable"]:
            label += f" (unstable, n<{P99_MIN_SAMPLES})"
        lines.append(f"| {label} | " + " | ".join(_fmt(results[p][key], scale) for p in POLICY_NAMES) + " |")
    return "\n".join(lines)


def cmd_simulate(a):
    wl = make_workload(a.workload, n=a.n, seed=a.seed, load=a.load, prediction_noise=a.noise)
    sink = LocalJsonlEventSink(a.events_out) if a.events_out else None
    res = simulate(wl, a.policy, a.max_wait_ms, sink=sink)
    if sink:
        sink.close()
    print(json.dumps({"warning": SIM_WARNING, "workload": wl.name, "policy": a.policy, **res.summary}, indent=2))


def cmd_compare(a):
    names = WORKLOAD_NAMES if a.workloads == "all" else a.workloads.split(",")
    out = [f"# Simulated policy comparison", "", f"> {SIM_WARNING}", "",
           f"Settings: seed {a.seed}, offered load {a.load}, prediction noise sigma {a.noise} (lognormal), "
           f"adaptive max wait {a.max_wait_ms / 1000:.1f} s (also the starvation threshold), "
           f"service = {config.SIM_BASE_OVERHEAD_MS:.0f} ms + {config.SIM_MS_PER_TOKEN} ms/token. "
           f"Short = {20}-{150} tokens, long = {400}-{1000} tokens.", ""]
    raw = {}
    for name in names:
        wl = make_workload(name, n=a.n, seed=a.seed, load=a.load, prediction_noise=a.noise)
        res = compare(wl, a.max_wait_ms)
        raw[name] = res
        out += [comparison_table(name, wl.description, res, len(wl.requests)), ""]
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
        Path(a.out).with_suffix(".json").write_text(json.dumps(raw, indent=2))


_COMPACT_KEYS = ("event_type", "timestamp_ms", "seq", "request_id", "queue_depth", "predicted_output_tokens",
                 "actual_output_tokens", "queue_wait_ms", "service_time_ms", "end_to_end_latency_ms", "success",
                 "error", "data")


def cmd_export_ui(a):
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for name in WORKLOAD_NAMES:
        wl = make_workload(name, n=a.n, seed=a.seed, load=a.load, prediction_noise=a.noise)
        runs = {}
        for p in POLICY_NAMES:
            res = simulate(wl, p, a.max_wait_ms, keep_events=True)
            runs[p] = {"summary": res.summary,
                       "events": [{k: e.to_dict()[k] for k in _COMPACT_KEYS} for e in res.events]}
        bundle = {"warning": SIM_WARNING, "workload": {k: v for k, v in wl.to_dict().items() if k != "requests"},
                  "max_wait_ms": a.max_wait_ms,
                  "requests": [{"request_id": w.request_id, "size_class": w.size_class, "arrival_ms": w.arrival_ms,
                                "predicted_tokens": w.mock_predicted_tokens, "actual_tokens": w.mock_actual_tokens,
                                "prompt": w.prompt} for w in wl.requests],
                  "runs": runs}
        (out_dir / f"{name}.json").write_text(json.dumps(bundle))
        index.append({"name": name, "description": wl.description, "n": len(wl.requests)})
    (out_dir / "index.json").write_text(json.dumps({"workloads": index, "max_wait_ms": a.max_wait_ms,
                                                    "warning": SIM_WARNING}, indent=2))
    print(f"wrote {len(index)} simulated workloads to {out_dir}")


def cmd_run(a):
    """Wall-clock run: replays arrivals in real time against a real (or realtime-mock) backend."""
    if a.prompts_file:
        prompts = [json.loads(l)["prompt"] for l in Path(a.prompts_file).read_text(encoding="utf-8").splitlines()
                   if l.strip()][: a.limit]
        wl = workload_from_prompts(prompts, mean_interarrival_ms=a.mean_interarrival_ms, seed=a.seed)
    else:
        wl = make_workload(a.workload, n=a.n, seed=a.seed, load=a.load, prediction_noise=a.noise)

    if a.backend == "mock":
        if not wl.synthetic:
            sys.exit("--backend mock needs a synthetic --workload (real prompts have no mock lengths)")
        backend = wl.mock_backend(realtime=True, speedup=a.speedup)
        for w in wl.requests:  # compress arrivals by the same factor as the mock's sleeps
            w.arrival_ms /= a.speedup
    else:
        from scheduler.backends import LlamaCppBackend

        if a.speedup != 1.0:
            sys.exit("--speedup only applies to the mock backend")
        backend = LlamaCppBackend(a.url)
        try:
            backend.check()
        except Exception as e:  # noqa: BLE001
            sys.exit(f"error: {e}")
    if a.predictor == "mock":
        predictor = wl.mock_predictor() if wl.synthetic else MockPredictor()
    else:
        predictor = load_predictor(a.predictor)

    run_name = a.run_name or f"{dt.datetime.now():%Y%m%d-%H%M%S}-{wl.name}-{a.policy}-{a.backend}"
    run_dir = Path(a.runs_dir) / run_name
    sink = LocalJsonlEventSink(run_dir / "events.jsonl", fsync=a.backend != "mock")
    engine = SchedulerEngine(predictor, backend, make_policy(a.policy, a.max_wait_ms), sink, WallClock(),
                             workload_name=wl.name, run_id=run_name, starvation_threshold_ms=a.max_wait_ms)
    print(f"running {len(wl.requests)} requests -> {run_dir / 'events.jsonl'}")
    try:
        done = engine.run(wl.to_requests())
    finally:
        sink.close()
    from scheduler.metrics import summarize

    summary = summarize(done, a.max_wait_ms)
    meta = {"run_name": run_name, "backend": a.backend, "predictor": a.predictor, "policy": a.policy,
            "max_wait_ms": a.max_wait_ms, "workload": wl.name, "synthetic_workload": wl.synthetic,
            "time_scale": 1.0 / a.speedup, "summary": summary}
    if a.backend == "mock":
        meta["warning"] = f"MOCK backend: synthetic lengths; times compressed {a.speedup}x."
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scheduler")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, n_default=60):
        p.add_argument("--n", type=int, default=n_default, help="requests per Poisson workload")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--load", type=float, default=0.95, help="offered load (mean service / mean gap)")
        p.add_argument("--noise", type=float, default=0.25, help="mock prediction error, lognormal sigma")
        p.add_argument("--max-wait-ms", type=float, default=config.DEFAULT_MAX_WAIT_MS)

    p = sub.add_parser("simulate", help="one policy on one synthetic workload")
    common(p)
    p.add_argument("--workload", choices=WORKLOAD_NAMES, required=True)
    p.add_argument("--policy", choices=POLICY_NAMES, required=True)
    p.add_argument("--events-out", default=None)
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("compare", help="FIFO vs SEJF vs adaptive on the same workloads")
    common(p)
    p.add_argument("--workloads", default="all", help="comma list or 'all'")
    p.add_argument("--out", default=None, help="write the markdown table here (+ .json)")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("export-ui", help="write simulated runs for the dashboard")
    common(p, n_default=30)
    p.add_argument("--out-dir", default=str(config.UI_SIM_DIR))
    p.set_defaults(fn=cmd_export_ui)

    p = sub.add_parser("run", help="wall-clock run against a backend, events to scheduler_runs/")
    common(p, n_default=30)
    p.add_argument("--backend", choices=["mock", "llamacpp"], required=True)
    p.add_argument("--url", default=None, help="llama-server URL (default: LLM_BACKEND_URL or :8080)")
    p.add_argument("--predictor", default="mock", help="mock | distilbert:<dir> | baseline:<dir>")
    p.add_argument("--policy", choices=POLICY_NAMES, required=True)
    p.add_argument("--workload", choices=WORKLOAD_NAMES, default="balanced")
    p.add_argument("--prompts-file", default=None, help="JSONL with a 'prompt' field (real runs)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mean-interarrival-ms", type=float, default=5000.0)
    p.add_argument("--speedup", type=float, default=1.0, help="mock only: compress time by this factor")
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default=str(config.RUNS_DIR))
    p.set_defaults(fn=cmd_run)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
