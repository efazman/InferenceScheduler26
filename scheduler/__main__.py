"""Scheduler command line.

Simulation (synthetic, no GPU):
    python -m scheduler simulate  --workload head_of_line --policy sejf [--events-out f.jsonl]
    python -m scheduler compare   [--n 1000] [--out docs/sim_results.md]
    python -m scheduler sweep     # adaptive max-wait threshold x load
    python -m scheduler export-ui # bundle simulated runs for the dashboard

Real experiment (see docs/RUNBOOK_REAL_EXPERIMENT.md):
    python -m scheduler prompts-from-split --artifacts <predictor dir> --out <prompts.jsonl>
    python -m scheduler make-manifest --prompts-file <prompts.jsonl> --mean-interarrival-ms 7000 --out <m.json>
    python -m scheduler measure-service --from data/labels/llama31_8b_q4km/runs.jsonl
    python -m scheduler run --backend llamacpp --manifest <m.json> --policy adaptive \\
        --predictor distilbert:<dir> --median-service-from data/labels/llama31_8b_q4km/runs.jsonl
    python -m scheduler report scheduler_runs/<fifo run> scheduler_runs/<sejf run> scheduler_runs/<adaptive run>
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
from scheduler.events import LocalJsonlEventSink, read_events
from scheduler.manifest import Manifest, manifest_from_prompt_rows, manifest_from_workload
from scheduler.max_wait import resolve_max_wait, service_times_from_file
from scheduler.metrics import P99_MIN_SAMPLES, percentile, summarize
from scheduler.policies import POLICY_NAMES, make_policy
from scheduler.predictors import MockPredictor, load_predictor
from scheduler.simulator import compare, simulate
from scheduler.workloads import WORKLOAD_NAMES, make_workload

SIM_WARNING = ("SIMULATED: synthetic workloads, mock predictor, mock backend "
               "(service = base overhead + tokens x ms/token). Validates policy behaviour only; "
               "not a measurement of Llama or of the predictor.")

# Primary metrics for K=1 non-preemptive scheduling are latency and fairness. Throughput is kept
# as a sanity check only: reordering the same work on one GPU should not change it.
ROWS = [  # (label, key, scale) - ms values shown in seconds
    ("Mean latency (s)", "mean_latency_ms", 1e-3), ("p50 latency (s)", "p50_latency_ms", 1e-3),
    ("p95 latency (s)", "p95_latency_ms", 1e-3), ("p99 latency (s)", "p99_latency_ms", 1e-3),
    ("Mean queue wait (s)", "mean_queue_wait_ms", 1e-3), ("Max queue wait (s)", "max_queue_wait_ms", 1e-3),
    ("Short-request mean latency (s)", "short_mean_latency_ms", 1e-3),
    ("Long-request mean latency (s)", "long_mean_latency_ms", 1e-3),
    ("Long-request max wait (s)", "long_max_queue_wait_ms", 1e-3),
    ("Starved (wait > threshold)", "starvation_count", 1),
    ("Throughput, sanity check (req/min)", "throughput_rps", 60.0),
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


def cmd_sweep(a):
    """Adaptive max-wait threshold x offered load, against FIFO and SEJF on the same workload."""
    sec = lambda v: "-" if v is None else f"{v / 1000:.1f}"  # noqa: E731
    lines = [f"> {SIM_WARNING}", "", f"{a.n} requests per run, seed {a.seed}, prediction noise sigma {a.noise}.", "",
             "| Workload | Load | Policy | Max wait param (s) | Mean (s) | p50 (s) | p95 (s) | p99 (s) | "
             "Max wait (s) | Short mean (s) |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in a.workloads.split(","):
        for load in (float(x) for x in a.loads.split(",")):
            wl = make_workload(name, n=a.n, seed=a.seed, load=load, prediction_noise=a.noise)
            rows = [("FIFO", None, simulate(wl, "fifo", emit_queue_snapshots=False).summary),
                    ("SEJF", None, simulate(wl, "sejf", emit_queue_snapshots=False).summary)]
            rows += [("Adaptive", mw, simulate(wl, "adaptive", mw, emit_queue_snapshots=False).summary)
                     for mw in (float(x) * 1000 for x in a.max_waits_s.split(","))]
            for pol, mw, r in rows:
                lines.append(f"| {name} | {load} | {pol} | {'-' if mw is None else f'{mw / 1000:g}'} | "
                             f"{sec(r['mean_latency_ms'])} | {sec(r['p50_latency_ms'])} | {sec(r['p95_latency_ms'])} | "
                             f"{sec(r['p99_latency_ms'])} | {sec(r['max_queue_wait_ms'])} | "
                             f"{sec(r['short_mean_latency_ms'])} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")


def cmd_prompts_from_split(a):
    """Held-out prompts for real scheduler runs: the given split of a trained predictor's data.

    Uses the predictor's own artifacts (config.json -> label file, splits.json -> ids), so the
    scheduler is evaluated on prompts the predictor never trained on."""
    from ml.config import Config
    from ml.data import load_jsonl

    art = Path(a.artifacts)
    cfg = Config.load(art / "config.json")
    ids = json.loads((art / "splits.json").read_text(encoding="utf-8"))[a.split]
    by_id = {ex["id"]: ex for ex in load_jsonl(a.labels or cfg.data_path, cfg.target_field)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        sys.exit(f"{len(missing)} {a.split} ids not in the label file; pass --labels <file used for training>")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in ids:
            ex = by_id[i]
            f.write(json.dumps({"prompt_id": i, "prompt": ex["prompt"], "category": ex.get("category"),
                                "label_target_tokens": ex["target_output_tokens"]}, ensure_ascii=False) + "\n")
    print(f"wrote {len(ids)} {a.split}-split prompts to {out}")


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


def _read_prompt_rows(path: str, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit]


def _build_manifest(a) -> Manifest:
    if getattr(a, "manifest", None):
        return Manifest.load(a.manifest)
    if a.prompts_file:
        return manifest_from_prompt_rows(_read_prompt_rows(a.prompts_file, a.limit), Path(a.prompts_file).stem,
                                         a.mean_interarrival_ms, a.seed, {"type": "prompts_file",
                                                                          "path": str(a.prompts_file)})
    return manifest_from_workload(make_workload(a.workload, n=a.n, seed=a.seed, load=a.load,
                                                prediction_noise=a.noise))


def cmd_make_manifest(a):
    m = _build_manifest(a)
    m.save(a.out)
    print(f"manifest {m.manifest_id}: {len(m.requests)} requests ({'synthetic' if m.synthetic else 'real prompts'}) "
          f"-> {a.out}")


def cmd_measure_service(a):
    times = service_times_from_file(a.from_file)
    mw = resolve_max_wait(multiplier=a.multiplier, median_from=a.from_file)
    print(json.dumps({"source": a.from_file, "n": len(times), "median_service_ms": mw.median_service_ms,
                      "p10_service_ms": percentile(times, 10), "p90_service_ms": percentile(times, 90),
                      "multiplier": a.multiplier, "derived_max_wait_ms": mw.max_wait_ms,
                      "description": mw.describe()}, indent=2))


def _generation_settings(backend_name: str) -> dict | None:
    if backend_name != "llamacpp":
        return None
    from datagen import config as gen

    keys = ("MODEL_NAME", "QUANTIZATION", "RUNTIME", "RUNTIME_VERSION", "TEMPERATURE", "TOP_P",
            "MAX_NEW_TOKENS", "SYSTEM_PROMPT_SHA256")
    return {k.lower(): getattr(gen, k) for k in keys if hasattr(gen, k)}


def cmd_run(a):
    """Wall-clock run: replays one manifest in real time against a real (or realtime-mock) backend."""
    manifest = _build_manifest(a)
    wl = manifest.to_workload()

    # MAX_WAIT: explicit, or multiplier x median service time. Real runs must state a source.
    try:
        mw = resolve_max_wait(a.max_wait_ms, a.max_wait_multiplier, a.median_service_ms, a.median_service_from,
                              default_ms=config.DEFAULT_MAX_WAIT_MS if a.backend == "mock" else None)
    except (ValueError, OSError) as e:
        sys.exit(f"error: {e}")

    if a.backend == "mock":
        if not wl.synthetic:
            sys.exit("--backend mock needs a synthetic manifest/workload (real prompts have no mock lengths)")
        backend = wl.mock_backend(realtime=True, speedup=a.speedup)
        for w in wl.requests:  # compress arrivals by the same factor as the mock's sleeps
            w.arrival_ms /= a.speedup
        effective_max_wait = mw.max_wait_ms / a.speedup  # the threshold lives on the compressed clock too
    else:
        from scheduler.backends import LlamaCppBackend

        if a.speedup != 1.0:
            sys.exit("--speedup only applies to the mock backend")
        backend = LlamaCppBackend(a.url, timeout_s=a.timeout_s)
        try:
            backend.check()
        except Exception as e:  # noqa: BLE001
            sys.exit(f"error: {e}")
        effective_max_wait = mw.max_wait_ms
    if a.predictor == "mock":
        predictor = wl.mock_predictor() if wl.synthetic else MockPredictor()
    else:
        predictor = load_predictor(a.predictor)

    run_name = a.run_name or f"{dt.datetime.now():%Y%m%d-%H%M%S}-{manifest.name}-{a.policy}-{a.backend}"
    run_dir = Path(a.runs_dir) / run_name
    if (run_dir / "events.jsonl").exists():
        sys.exit(f"{run_dir} already has events.jsonl; pick a new --run-name (runs are never appended to)")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.save(run_dir / "manifest.json")  # the exact trace this run replayed
    measurement = "mock" if a.backend == "mock" else "real"
    mw_info = {**mw.to_dict(), "effective_max_wait_ms": effective_max_wait}
    if a.speedup != 1.0:
        mw_info["description"] += f"; {effective_max_wait / 1000:.2f} s on the {a.speedup:g}x-compressed mock clock"
    run_meta = {"manifest_id": manifest.manifest_id, "manifest_name": manifest.name, "measurement": measurement,
                "max_wait": mw_info,
                "predictor_spec": a.predictor, "generation": _generation_settings(a.backend),
                "time_scale": 1.0 / a.speedup}
    sink = LocalJsonlEventSink(run_dir / "events.jsonl", fsync=a.backend != "mock")
    engine = SchedulerEngine(predictor, backend, make_policy(a.policy, effective_max_wait), sink, WallClock(),
                             workload_name=manifest.name, run_id=run_name, starvation_threshold_ms=effective_max_wait,
                             run_metadata=run_meta)
    print(f"running {len(wl.requests)} requests, policy {a.policy}, max wait {mw.describe()} "
          f"-> {run_dir / 'events.jsonl'}")
    try:
        done = engine.run(wl.to_requests())
    finally:
        sink.close()
    meta = {"run_name": run_name, "backend": a.backend, "predictor": a.predictor, "policy": a.policy,
            "workload": manifest.name, "synthetic_workload": manifest.synthetic, **run_meta,
            "max_wait_ms": effective_max_wait, "summary": summarize(done, effective_max_wait)}
    if a.backend == "mock":
        meta["warning"] = f"MOCK backend: synthetic lengths; times compressed {a.speedup}x."
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta["summary"], indent=2))


def _run_facts(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = read_events(run_dir / "events.jsonl")
    arrivals = {e["request_id"]: e["timestamp_ms"] for e in events if e["event_type"] == "request_arrived"}
    preds = {e["request_id"]: e["predicted_output_tokens"] for e in events if e["event_type"] == "cost_predicted"}
    return {"dir": run_dir, "meta": meta, "arrivals": arrivals, "preds": preds}


def consistency_checks(runs: list[dict]) -> list[tuple[str, bool, str]]:
    """Did these runs replay the same trace under the same conditions? (name, ok, detail)."""
    first = runs[0]
    same = lambda key: all(r["meta"].get(key) == first["meta"].get(key) for r in runs)  # noqa: E731
    checks = [("same manifest_id", same("manifest_id"), str(first["meta"].get("manifest_id"))),
              ("same backend", same("backend"), str(first["meta"].get("backend"))),
              ("same generation settings", same("generation"), "datagen.config"),
              ("same max wait / starvation threshold", same("max_wait_ms"), f"{first['meta'].get('max_wait_ms')} ms"),
              ("one run per policy", len({r["meta"]["policy"] for r in runs}) == len(runs),
               ", ".join(r["meta"]["policy"] for r in runs)),
              ("same request set", all(r["arrivals"].keys() == first["arrivals"].keys() for r in runs),
               f"{len(first['arrivals'])} requests")]
    arr_ok = all(abs(r["arrivals"][k] - v) < 1e-6 for r in runs for k, v in first["arrivals"].items()
                 if k in r["arrivals"])
    checks.append(("same arrival timestamps", arr_ok, "from the manifest"))
    worst = max((abs((r["preds"].get(k) or 0) - (v or 0)) for r in runs for k, v in first["preds"].items()),
                default=0.0)
    checks.append(("same predictor output per prompt", worst <= 0.01, f"max difference {worst:.4f} tokens"))
    return checks


def cmd_report(a):
    runs = [_run_facts(Path(d)) for d in a.run_dirs]
    order = {p: i for i, p in enumerate(POLICY_NAMES)}
    runs.sort(key=lambda r: order.get(r["meta"]["policy"], 99))
    checks = consistency_checks(runs)
    kind = {r["meta"].get("measurement", "mock" if r["meta"]["backend"] == "mock" else "real") for r in runs}
    label = "REAL MEASUREMENTS (llama.cpp)" if kind == {"real"} else "MOCK / SIMULATED - not real measurements"
    s0 = runs[0]["meta"]["summary"]
    lines = [f"# Policy comparison: {runs[0]['meta'].get('workload')}", "", f"**{label}**", "",
             f"Manifest `{runs[0]['meta'].get('manifest_id')}`, {s0['n_completed'] + s0['n_failed']} requests. "
             f"Adaptive max wait: {runs[0]['meta'].get('max_wait', {}).get('description', '?')}. Short/long split: "
             f"workload labels, else actual output >= {config.LONG_TOKEN_THRESHOLD} tokens.", "",
             "| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in checks]
    lines += ["", "| Metric | " + " | ".join(r["meta"]["policy"].upper() for r in runs) + " |",
              "| --- |" + " --- |" * len(runs)]
    for lab, key, scale in ROWS:
        if key == "p99_latency_ms" and not s0["p99_reliable"]:
            lab += f" (unstable, n<{P99_MIN_SAMPLES})"
        lines.append(f"| {lab} | " + " | ".join(_fmt(r["meta"]["summary"][key], scale) for r in runs) + " |")
    lines.append(f"| Failed requests | " + " | ".join(str(r["meta"]["summary"]["n_failed"]) for r in runs) + " |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    if not all(ok for _, ok, _ in checks):
        sys.exit("consistency check FAILED: these runs are not a like-for-like comparison")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scheduler")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, n_default=60, max_wait_default=config.DEFAULT_MAX_WAIT_MS):
        p.add_argument("--n", type=int, default=n_default, help="requests per Poisson workload")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--load", type=float, default=0.95, help="offered load (mean service / mean gap)")
        p.add_argument("--noise", type=float, default=0.25, help="mock prediction error, lognormal sigma")
        p.add_argument("--max-wait-ms", type=float, default=max_wait_default,
                       help="explicit adaptive threshold (simulation default: 15000)")

    def trace_source(p):
        p.add_argument("--workload", choices=WORKLOAD_NAMES, default="balanced")
        p.add_argument("--prompts-file", default=None, help="JSONL with a 'prompt' field (real runs)")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--mean-interarrival-ms", type=float, default=5000.0)

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

    p = sub.add_parser("sweep", help="adaptive max-wait threshold x load sweep")
    common(p, n_default=1000)
    p.add_argument("--workloads", default="mostly_short,balanced")
    p.add_argument("--loads", default="0.7,0.85,0.95")
    p.add_argument("--max-waits-s", default="15,30,60,120")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("prompts-from-split", help="held-out prompts for real runs, from predictor artifacts")
    p.add_argument("--artifacts", required=True, help="trained predictor dir (has config.json, splits.json)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--labels", default=None, help="override the label file recorded in config.json")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prompts_from_split)

    p = sub.add_parser("export-ui", help="write simulated runs for the dashboard")
    common(p, n_default=30)
    p.add_argument("--out-dir", default=str(config.UI_SIM_DIR))
    p.set_defaults(fn=cmd_export_ui)

    p = sub.add_parser("make-manifest", help="freeze an arrival trace (requests, prompts, arrivals, seeds)")
    common(p, n_default=30)
    trace_source(p)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_make_manifest)

    p = sub.add_parser("measure-service", help="median service time -> derived adaptive max wait")
    p.add_argument("--from", dest="from_file", required=True, help="label runs.jsonl or scheduler events.jsonl")
    p.add_argument("--multiplier", type=float, default=config.DEFAULT_MAX_WAIT_MULTIPLIER)
    p.set_defaults(fn=cmd_measure_service)

    p = sub.add_parser("run", help="wall-clock run against a backend, events to scheduler_runs/")
    common(p, n_default=30, max_wait_default=None)
    trace_source(p)
    p.add_argument("--manifest", default=None, help="replay this manifest (preferred for experiments)")
    p.add_argument("--backend", choices=["mock", "llamacpp"], required=True)
    p.add_argument("--url", default=None, help="llama-server URL (default: LLM_BACKEND_URL or :8080)")
    p.add_argument("--timeout-s", type=float, default=None, help="per-generation timeout (default: datagen)")
    p.add_argument("--predictor", default="mock", help="mock | distilbert:<dir> | baseline:<dir>")
    p.add_argument("--policy", choices=POLICY_NAMES, required=True)
    p.add_argument("--max-wait-multiplier", type=float, default=config.DEFAULT_MAX_WAIT_MULTIPLIER)
    p.add_argument("--median-service-ms", type=float, default=None)
    p.add_argument("--median-service-from", default=None, help="label runs.jsonl or scheduler events.jsonl")
    p.add_argument("--speedup", type=float, default=1.0, help="mock only: compress time by this factor")
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default=str(config.RUNS_DIR))
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="compare runs of one manifest + check they are like-for-like")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
