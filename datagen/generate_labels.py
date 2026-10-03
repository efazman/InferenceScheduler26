"""Resumable label generation: run every subset prompt NUM_GENERATIONS times and record lengths.

    # now (no GPU): deterministic mock backend -> data/labels/mock/   (SYNTHETIC, TEST-ONLY)
    python -m datagen.generate_labels --backend mock

    # RTX 3060 Ti: llama-server running on BACKEND_URL -> data/labels/llama31_8b_q4km/
    python -m datagen.generate_labels --backend llamacpp

Output directory layout:
    generation_config.json  settings this label set was produced with; a resume must match exactly
    runs.jsonl              one line per successful generation (raw measurement, includes text)
    labels.jsonl            one line per prompt with all runs done + its p90 target (training input)
    failures.jsonl          one line per failed attempt (error, attempt number, final or not)

Resume semantics: rerun the same command. Prompts already in labels.jsonl are skipped, and
generations already in runs.jsonl are reused, so an interruption loses at most the single
generation that was in flight. Every line is fsynced as it is written, and a partial final line
from a hard crash is repaired on start-up. A prompt whose generation still fails after retries
is logged and left incomplete; the next run retries only its missing generations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from datagen import config
from datagen.backends import BackendError, MockBackend, OpenAICompatBackend
from datagen.jsonl import DurableAppender, iter_jsonl, repair_jsonl_tail

MOCK_WARNING = "MOCK labels: synthetic test data from MockBackend. Never use for ML training or evaluation."


class ConfigMismatch(RuntimeError):
    pass


def percentile_target(values, q: float = config.TARGET_PERCENTILE,
                      method: str = config.PERCENTILE_METHOD) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method=method))


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _check_generation_config(out_dir: Path, gen_cfg: dict) -> None:
    path = out_dir / config.GENERATION_CONFIG_FILE
    if path.exists():
        existing = json.loads(path.read_text())
        existing.pop("created_at", None)
        diff = {k: (existing.get(k), gen_cfg.get(k)) for k in set(existing) | set(gen_cfg)
                if existing.get(k) != gen_cfg.get(k)}
        if diff:
            raise ConfigMismatch(
                f"{out_dir} was produced with different settings {diff} (existing, requested). "
                "Mixing settings in one label set would corrupt it; use a new --output-dir.")
    else:
        path.write_text(json.dumps({**gen_cfg, "created_at": _now()}, indent=2))


def _load_state(out_dir: Path):
    for name in (config.RUNS_FILE, config.LABELS_FILE, config.FAILURES_FILE):
        removed = repair_jsonl_tail(out_dir / name)
        if removed:
            print(f"repaired {name}: dropped {removed} bytes of a partially written final line")
    completed = set()
    if (out_dir / config.LABELS_FILE).exists():
        completed = {r["prompt_id"] for r in iter_jsonl(out_dir / config.LABELS_FILE)}
    runs: dict[tuple[str, int], dict] = {}
    if (out_dir / config.RUNS_FILE).exists():
        for r in iter_jsonl(out_dir / config.RUNS_FILE):
            runs.setdefault((r["prompt_id"], r["seed"]), r)
    return completed, runs


def build_label(prompt_rec: dict, runs: list[dict], gen_cfg: dict, is_mock: bool) -> dict:
    tokens = [r["output_tokens"] for r in runs]
    label = {
        "prompt_id": prompt_rec["prompt_id"],
        "prompt": prompt_rec["prompt"],
        "category": prompt_rec.get("category"),
        "category_source": prompt_rec.get("category_source"),
        "source": prompt_rec.get("source"),
        "runs": [{k: r.get(k) for k in ("seed", "output_tokens", "latency_ms", "finish_reason", "prompt_tokens")}
                 for r in runs],
        config.TARGET_FIELD: percentile_target(tokens, gen_cfg["target_percentile"], gen_cfg["percentile_method"]),
        # Runs that hit max_new_tokens are right-censored: their true length is >= the recorded count.
        "n_truncated_runs": sum(r.get("finish_reason") == "length" for r in runs),
        "model": gen_cfg["model"],
        "quantization": gen_cfg["quantization"],
        "runtime": gen_cfg["runtime"],
        "temperature": gen_cfg["temperature"],
        "top_p": gen_cfg["top_p"],
        "max_new_tokens": gen_cfg["max_new_tokens"],
        "system_prompt_sha256": gen_cfg["system_prompt_sha256"],
        "target_percentile": gen_cfg["target_percentile"],
        "percentile_method": gen_cfg["percentile_method"],
        "backend": gen_cfg["backend"],
        "is_mock": is_mock,
        "completed_at": _now(),
    }
    if is_mock:
        label["warning"] = MOCK_WARNING
    return label


def run_label_generation(prompts: list[dict], backend, out_dir: str | Path,
                         seeds: list[int] = config.GENERATION_SEEDS, max_retries: int = config.MAX_RETRIES,
                         backoff_s: float = config.RETRY_BACKOFF_S, limit: int | None = None,
                         log=print, sleep=time.sleep) -> dict:
    out_dir = Path(out_dir)
    is_mock = backend.name == "mock"
    if is_mock and out_dir.resolve() == config.LABELS_DIR_REAL.resolve():
        raise ValueError("refusing to write MOCK results into the real label directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [p["prompt_id"] for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt_id in input subset")

    gen_cfg = {**config.generation_config(backend.name), "seeds": list(seeds), "num_generations": len(seeds)}
    _check_generation_config(out_dir, gen_cfg)
    completed, done_runs = _load_state(out_dir)

    stats = {"total": len(prompts), "already_complete": 0, "completed_now": 0, "failed_prompts": 0,
             "generations_run": 0, "generations_reused": 0, "failed_attempts": 0}
    todo = [p for p in prompts if p["prompt_id"] not in completed]
    stats["already_complete"] = len(prompts) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    log(f"{stats['already_complete']}/{len(prompts)} prompts already complete; processing {len(todo)}")

    t_start = time.perf_counter()
    with DurableAppender(out_dir / config.RUNS_FILE) as runs_out, \
            DurableAppender(out_dir / config.LABELS_FILE) as labels_out, \
            DurableAppender(out_dir / config.FAILURES_FILE) as fail_out:
        for i, prec in enumerate(todo, 1):
            pid, runs, failed = prec["prompt_id"], [], False
            for run_index, seed in enumerate(seeds):
                if (pid, seed) in done_runs:
                    runs.append(done_runs[(pid, seed)])
                    stats["generations_reused"] += 1
                    continue
                for attempt in range(1, max_retries + 2):
                    try:
                        res = backend.generate(prec["prompt"], seed)
                    except BackendError as e:
                        stats["failed_attempts"] += 1
                        final = (not e.retryable) or attempt > max_retries
                        fail_out.append({"prompt_id": pid, "seed": seed, "attempt": attempt,
                                         "error": str(e), "retryable": e.retryable, "final": final,
                                         "backend": backend.name, "timestamp": _now()})
                        if final:
                            failed = True
                            break
                        sleep(backoff_s * 2 ** (attempt - 1))
                        continue
                    run = {"prompt_id": pid, "seed": seed, "run_index": run_index,
                           "output_tokens": res.output_tokens, "latency_ms": res.latency_ms,
                           "finish_reason": res.finish_reason, "prompt_tokens": res.prompt_tokens,
                           "server_timings": res.server_timings, "attempt": attempt,
                           "backend": backend.name, "is_mock": is_mock, "timestamp": _now(),
                           "text": res.text}
                    runs_out.append(run)
                    done_runs[(pid, seed)] = run
                    runs.append(run)
                    stats["generations_run"] += 1
                    break
                if failed:
                    break
            if failed:
                stats["failed_prompts"] += 1
                log(f"[{i}/{len(todo)}] {pid} FAILED after retries (see {config.FAILURES_FILE}); "
                    f"{len(runs)}/{len(seeds)} generations kept for next run")
                continue
            label = build_label(prec, runs, gen_cfg, is_mock)
            labels_out.append(label)
            stats["completed_now"] += 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (len(todo) - i)
            log(f"[{i}/{len(todo)}] {pid} tokens={[r['output_tokens'] for r in runs]} "
                f"p{gen_cfg['target_percentile']}={label[config.TARGET_FIELD]:.1f} eta={eta / 60:.1f}min")

    stats["complete_total"] = stats["already_complete"] + stats["completed_now"]
    stats["output_dir"] = str(out_dir)
    return stats


def make_backend(args):
    if args.backend == "mock":
        return MockBackend(config.MAX_NEW_TOKENS, transient_failures=args.mock_transient_failures,
                           fail_rate=args.mock_fail_rate)
    return OpenAICompatBackend(args.url, config.MODEL_NAME, config.SYSTEM_PROMPT, config.TEMPERATURE,
                               config.TOP_P, config.MAX_NEW_TOKENS, args.timeout)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Resumable output-length label generation.")
    parser.add_argument("--backend", choices=["mock", "llamacpp"], required=True,
                        help="llamacpp = any OpenAI-compatible server (llama.cpp llama-server)")
    parser.add_argument("--input", default=str(config.SUBSET_PATH))
    parser.add_argument("--output-dir", default=None,
                        help=f"default: {config.LABELS_DIR_REAL} (llamacpp) or {config.LABELS_DIR_MOCK} (mock)")
    parser.add_argument("--url", default=config.BACKEND_URL)
    parser.add_argument("--timeout", type=float, default=config.REQUEST_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=config.MAX_RETRIES)
    parser.add_argument("--backoff", type=float, default=config.RETRY_BACKOFF_S)
    parser.add_argument("--limit", type=int, default=None, help="process at most N incomplete prompts")
    parser.add_argument("--mock-fail-rate", type=float, default=0.0)
    parser.add_argument("--mock-transient-failures", type=int, default=1)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir or (config.LABELS_DIR_MOCK if args.backend == "mock" else config.LABELS_DIR_REAL))
    backend = make_backend(args)
    try:
        backend.check()
    except BackendError as e:
        sys.exit(f"error: {e}\nStart llama-server first (see README) or pass --url.")
    prompts = list(iter_jsonl(args.input))
    try:
        stats = run_label_generation(prompts, backend, out_dir, max_retries=args.max_retries,
                                     backoff_s=args.backoff, limit=args.limit)
    except (ConfigMismatch, ValueError) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted: all finished generations are saved. Rerun the same command to resume.")
    if args.backend == "mock":
        stats["warning"] = MOCK_WARNING
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
