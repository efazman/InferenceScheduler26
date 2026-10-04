"""Read-only progress report for a label-generation run. Safe to call while generation is running.

    python -m datagen.status                      # the real run
    python -m datagen.status --output-dir <dir>   # any other run

Only reads, never writes, and tolerates a half-written final line, so it cannot disturb or
corrupt an in-flight run. Throughput and ETA come from the timestamps already recorded in
runs.jsonl, so they survive restarts and reflect the whole run rather than one session.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from datagen import config

# A gap longer than this counts as the pipeline being idle rather than slow, and is excluded
# from the rate used for the ETA.
IDLE_GAP_S = 120.0
RECENT_WINDOW = 200  # generations used for the "recent" rate


def _iter_tolerant(path: Path):
    """Yield records, silently ignoring a trailing partial line from an in-flight write."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n") or not line.strip():
                continue  # last line still being written
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse(ts: str | None):
    try:
        return dt.datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def status(out_dir: str | Path, subset_path: str | Path, n_generations: int | None = None) -> dict:
    out_dir, subset_path = Path(out_dir), Path(subset_path)
    gen_cfg_path = out_dir / config.GENERATION_CONFIG_FILE
    gen_cfg = json.loads(gen_cfg_path.read_text(encoding="utf-8")) if gen_cfg_path.exists() else {}
    per_prompt = n_generations or gen_cfg.get("num_generations") or config.NUM_GENERATIONS

    total_prompts = sum(1 for _ in _iter_tolerant(subset_path)) if subset_path.exists() else None
    labels = list(_iter_tolerant(out_dir / config.LABELS_FILE))
    runs = list(_iter_tolerant(out_dir / config.RUNS_FILE))
    failures = list(_iter_tolerant(out_dir / config.FAILURES_FILE))

    done_prompts = len({r["prompt_id"] for r in labels})
    done_runs = len({(r["prompt_id"], r["seed"]) for r in runs})
    target_runs = total_prompts * per_prompt if total_prompts else None

    times = sorted(t for t in (_parse(r.get("timestamp")) for r in runs) if t)
    wall_s = (times[-1] - times[0]).total_seconds() if len(times) > 1 else 0.0
    rate = done_runs / wall_s if wall_s > 0 else 0.0  # generations/sec, whole-run average

    # A pause (machine asleep, operator running something heavy, server restarted) permanently
    # depresses the whole-run average and makes the ETA far too pessimistic. Report idle time
    # separately and base the ETA on a recent window, which reflects the rate actually achievable.
    gaps = [(times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))]
    idle_s = sum(g for g in gaps if g > IDLE_GAP_S)
    active_s = wall_s - idle_s
    active_rate = done_runs / active_s if active_s > 0 else 0.0

    window = times[-min(len(times), RECENT_WINDOW):]
    window_s = (window[-1] - window[0]).total_seconds() if len(window) > 1 else 0.0
    window_gaps = [(window[i] - window[i - 1]).total_seconds() for i in range(1, len(window))]
    window_active_s = window_s - sum(g for g in window_gaps if g > IDLE_GAP_S)
    recent_rate = (len(window) - 1) / window_active_s if window_active_s > 0 else 0.0

    eta_rate = recent_rate or active_rate or rate
    remaining = (target_runs - done_runs) if target_runs else None
    eta_s = remaining / eta_rate if (remaining and eta_rate > 0) else None

    truncated = sum(r.get("finish_reason") == "length" for r in runs)
    out_tokens = sorted(r["output_tokens"] for r in runs if isinstance(r.get("output_tokens"), int))
    latencies = sorted(r["latency_ms"] for r in runs if isinstance(r.get("latency_ms"), (int, float)))

    def pct(xs, q):
        return float(xs[min(len(xs) - 1, int(q / 100 * len(xs)))]) if xs else None

    final_failures = [f for f in failures if f.get("final")]
    rep = {
        "output_dir": str(out_dir),
        "running_since": times[0].isoformat() if times else None,
        "last_generation_at": times[-1].isoformat() if times else None,
        "prompts_complete": done_prompts,
        "prompts_total": total_prompts,
        "prompts_pct": round(100 * done_prompts / total_prompts, 2) if total_prompts else None,
        "generations_done": done_runs,
        "generations_target": target_runs,
        "generations_per_min": round(rate * 60, 2),              # whole-run average, incl. idle
        "generations_per_min_active": round(active_rate * 60, 2),  # excluding idle gaps
        "generations_per_min_recent": round(recent_rate * 60, 2),  # last RECENT_WINDOW gens
        "elapsed_hours": round(wall_s / 3600, 2),
        "idle_hours": round(idle_s / 3600, 2),                   # time in gaps > IDLE_GAP_S
        "longest_idle_gap_min": round(max(gaps) / 60, 1) if gaps else 0.0,
        "eta_basis": "recent" if recent_rate else ("active" if active_rate else "overall"),
        "eta_hours": round(eta_s / 3600, 2) if eta_s else None,
        "eta_finish_utc": (times[-1] + dt.timedelta(seconds=eta_s)).isoformat(timespec="seconds")
                          if eta_s and times else None,
        "truncated_runs": truncated,
        "truncation_rate": round(truncated / done_runs, 4) if done_runs else None,
        "output_tokens_median": pct(out_tokens, 50),
        "output_tokens_p90": pct(out_tokens, 90),
        "latency_ms_median": pct(latencies, 50),
        "latency_ms_p90": pct(latencies, 90),
        "failed_attempts": len(failures),
        "prompts_failed_permanently": len({f["prompt_id"] for f in final_failures}),
        "failure_reasons": dict(Counter(str(f.get("error"))[:80] for f in final_failures).most_common(5)),
        "checkpoints_ready": [n for n in config.CHECKPOINT_SIZES if done_prompts >= n],
        "generation_config": {k: gen_cfg.get(k) for k in
                              ("model", "quantization", "runtime", "runtime_version", "temperature",
                               "top_p", "max_new_tokens", "num_generations", "seeds", "backend")},
    }
    return rep


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default=str(config.LABELS_DIR_REAL))
    parser.add_argument("--subset", default=str(config.SUBSET_PATH))
    args = parser.parse_args(argv)
    rep = status(args.output_dir, args.subset)
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    main()
