"""Re-run only the capped (right-censored) base generations with a larger token cap.

    python -m datagen.extend_censored --backend llamacpp            # the real pass
    python -m datagen.extend_censored --backend llamacpp --dry-run   # count work, generate nothing

A base run that stopped at ``max_new_tokens`` has ``finish_reason == "length"``. Its recorded
length is a **lower bound**, not a measurement. This pass re-runs exactly those (prompt_id, seed)
pairs with ``EXTENDED_MAX_NEW_TOKENS`` and *nothing else changed* - same prompt text, same seed,
same temperature, top_p and system prompt - and writes the results to a SEPARATE directory.

The base files are opened read-only and never modified. The two caps are never mixed in one file;
`datagen.assemble_final_labels` combines them explicitly and records which measurement it used.

Prompt text comes from the subset file keyed by prompt_id, not from the base label records, so a
capped run belonging to a not-yet-complete prompt can still be extended, and prompt identity is
guaranteed to match what the base run was given.

Resume semantics: rerun the same command. Every (prompt_id, seed) already in extended_runs.jsonl
is skipped, so the pass is idempotent and never produces a duplicate extended run. Each line is
fsynced as written and a half-written final line is repaired on start-up, so a crash loses at
most the one generation in flight. A resume whose settings differ is refused.

An extended run that *also* hits the new cap stays censored. Its length is still recorded, still
flagged, and counted in the summary - never replaced by a guess.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from datagen import config
from datagen.backends import BackendError, MockBackend, OpenAICompatBackend
from datagen.jsonl import DurableAppender, iter_jsonl, repair_jsonl_tail

CENSORED_FINISH_REASON = "length"


class ConfigMismatch(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def is_censored(run: dict, cap: int | None = None) -> bool:
    """True if this generation stopped because it ran out of token budget.

    finish_reason is authoritative. The token-count comparison is a fallback for a backend that
    omits it; it is only trusted when the count actually reaches the cap.
    """
    if run.get("finish_reason") == CENSORED_FINISH_REASON:
        return True
    if run.get("finish_reason") is None and cap is not None:
        tokens = run.get("output_tokens")
        return isinstance(tokens, int) and tokens >= cap
    return False


def find_censored_runs(base_dir: str | Path, cap: int = config.MAX_NEW_TOKENS) -> list[dict]:
    """Capped base generations, de-duplicated on (prompt_id, seed), in a deterministic order."""
    runs_path = Path(base_dir) / config.RUNS_FILE
    if not runs_path.exists():
        raise FileNotFoundError(f"no base runs at {runs_path}")
    seen: dict[tuple[str, int], dict] = {}
    for run in iter_jsonl(runs_path):
        if is_censored(run, cap):
            seen.setdefault((run["prompt_id"], run["seed"]), run)
    return [seen[k] for k in sorted(seen)]


def _check_extension_config(out_dir: Path, ext_cfg: dict) -> None:
    path = out_dir / config.EXTENSION_CONFIG_FILE
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.pop("created_at", None)
        diff = {k: (existing.get(k), ext_cfg.get(k)) for k in set(existing) | set(ext_cfg)
                if existing.get(k) != ext_cfg.get(k)}
        if diff:
            raise ConfigMismatch(
                f"{out_dir} was produced with different settings {diff} (existing, requested). "
                "Mixing settings in one extension pass would corrupt it; use a new --output-dir.")
    else:
        path.write_text(json.dumps({**ext_cfg, "created_at": _now()}, indent=2), encoding="utf-8")


def _load_done(out_dir: Path) -> set[tuple[str, int]]:
    for name in (config.EXTENDED_RUNS_FILE, config.EXTENSION_FAILURES_FILE):
        removed = repair_jsonl_tail(out_dir / name)
        if removed:
            print(f"repaired {name}: dropped {removed} bytes of a partially written final line")
    path = out_dir / config.EXTENDED_RUNS_FILE
    if not path.exists():
        return set()
    return {(r["prompt_id"], r["seed"]) for r in iter_jsonl(path)}


def extend_censored(censored: list[dict], prompts: dict[str, dict], backend, out_dir: str | Path,
                    extended_cap: int = config.EXTENDED_MAX_NEW_TOKENS,
                    base_dir: str | Path = config.LABELS_DIR_REAL,
                    max_retries: int = config.MAX_RETRIES, backoff_s: float = config.RETRY_BACKOFF_S,
                    limit: int | None = None, log=print, sleep=time.sleep) -> dict:
    out_dir = Path(out_dir)
    if Path(out_dir).resolve() == Path(base_dir).resolve():
        raise ValueError("refusing to write extension results into the base label directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    ext_cfg = config.extension_config(backend.name, extended_cap, str(base_dir))
    _check_extension_config(out_dir, ext_cfg)
    done = _load_done(out_dir)

    todo = [r for r in censored if (r["prompt_id"], r["seed"]) not in done]
    missing = sorted({r["prompt_id"] for r in todo if r["prompt_id"] not in prompts})
    if missing:
        raise ValueError(f"{len(missing)} censored prompt_id(s) are not in the subset file, "
                         f"e.g. {missing[:3]}; wrong --subset?")
    if limit is not None:
        todo = todo[:limit]

    stats = {"censored_total": len(censored), "already_extended": len(censored) - len(todo) if limit is None
             else len([r for r in censored if (r["prompt_id"], r["seed"]) in done]),
             "attempted": len(todo), "extended_ok": 0, "still_censored": 0, "resolved": 0,
             "failed": 0, "failed_attempts": 0, "shorter_than_base": 0}
    log(f"{stats['already_extended']} already extended; attempting {len(todo)} "
        f"of {len(censored)} censored runs at cap {extended_cap}")

    t0 = time.perf_counter()
    with DurableAppender(out_dir / config.EXTENDED_RUNS_FILE) as out, \
            DurableAppender(out_dir / config.EXTENSION_FAILURES_FILE) as fail_out:
        for i, base in enumerate(todo, 1):
            pid, seed = base["prompt_id"], base["seed"]
            prompt = prompts[pid]["prompt"]
            for attempt in range(1, max_retries + 2):
                try:
                    res = backend.generate(prompt, seed)
                except BackendError as e:
                    stats["failed_attempts"] += 1
                    final = (not e.retryable) or attempt > max_retries
                    fail_out.append({"prompt_id": pid, "seed": seed, "attempt": attempt,
                                     "error": str(e), "retryable": e.retryable, "final": final,
                                     "backend": backend.name, "timestamp": _now()})
                    if final:
                        stats["failed"] += 1
                        log(f"[{i}/{len(todo)}] {pid} seed={seed} FAILED: {e}")
                        break
                    sleep(backoff_s * 2 ** (attempt - 1))
                    continue

                still = is_censored({"finish_reason": res.finish_reason,
                                     "output_tokens": res.output_tokens}, extended_cap)
                rec = {
                    "prompt_id": pid,
                    "seed": seed,
                    # Which of the prompt's NUM_GENERATIONS runs this is, carried over from the base run.
                    "run_index": base.get("run_index"),
                    "original_output_tokens": base.get("output_tokens"),
                    "original_finish_reason": base.get("finish_reason"),
                    "original_max_new_tokens": config.MAX_NEW_TOKENS,
                    "extended_output_tokens": res.output_tokens,
                    "extended_finish_reason": res.finish_reason,
                    "extended_max_new_tokens": extended_cap,
                    "still_censored": still,
                    "latency_ms": res.latency_ms,
                    "prompt_tokens": res.prompt_tokens,
                    "server_timings": res.server_timings,
                    "attempt": attempt,
                    # provenance: these labels must be traceable to one configuration
                    "model": ext_cfg["model"],
                    "quantization": ext_cfg["quantization"],
                    "runtime": ext_cfg["runtime"],
                    "runtime_version": ext_cfg["runtime_version"],
                    "temperature": ext_cfg["temperature"],
                    "top_p": ext_cfg["top_p"],
                    "system_prompt_sha256": ext_cfg["system_prompt_sha256"],
                    "backend": backend.name,
                    "base_labels_dir": str(base_dir),
                    "is_mock": backend.name == "mock",
                    "timestamp": _now(),
                    "text": res.text,
                }
                out.append(rec)
                stats["extended_ok"] += 1
                stats["still_censored" if still else "resolved"] += 1
                # Same prompt + same seed should regenerate the same prefix, so the extended length
                # should be >= the base cap. A shorter result means the backend was not
                # deterministic; record it rather than hide it.
                if isinstance(base.get("output_tokens"), int) and res.output_tokens < base["output_tokens"]:
                    stats["shorter_than_base"] += 1
                elapsed = time.perf_counter() - t0
                eta = elapsed / i * (len(todo) - i)
                log(f"[{i}/{len(todo)}] {pid} seed={seed} {base.get('output_tokens')}"
                    f"->{res.output_tokens} {'STILL CENSORED' if still else 'resolved'} "
                    f"eta={eta / 60:.1f}min")
                break

    stats["output_dir"] = str(out_dir)
    stats["extended_cap"] = extended_cap
    return stats


def base_generator_running(pid_path: Path = Path("logs/generation.pid")) -> int | None:
    """PID of a live base-run driver, or None. Used to refuse GPU contention by default."""
    try:
        pid = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001 - if we cannot tell, do not block the operator
        return None
    return pid if str(pid) in out else None


def make_backend(args):
    if args.backend == "mock":
        return MockBackend(args.extended_cap, transient_failures=args.mock_transient_failures,
                           fail_rate=args.mock_fail_rate)
    return OpenAICompatBackend(args.url, config.MODEL_NAME, config.SYSTEM_PROMPT, config.TEMPERATURE,
                               config.TOP_P, args.extended_cap, args.timeout)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=["mock", "llamacpp"], required=True)
    parser.add_argument("--base-dir", default=str(config.LABELS_DIR_REAL))
    parser.add_argument("--subset", default=str(config.SUBSET_PATH))
    parser.add_argument("--output-dir", default=str(config.LABELS_DIR_EXT))
    parser.add_argument("--extended-cap", type=int, default=config.EXTENDED_MAX_NEW_TOKENS)
    parser.add_argument("--base-cap", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--url", default=config.BACKEND_URL)
    parser.add_argument("--timeout", type=float, default=config.REQUEST_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=config.MAX_RETRIES)
    parser.add_argument("--backoff", type=float, default=config.RETRY_BACKOFF_S)
    parser.add_argument("--limit", type=int, default=None, help="extend at most N runs")
    parser.add_argument("--dry-run", action="store_true", help="report what would run, generate nothing")
    parser.add_argument("--allow-concurrent-base", action="store_true",
                        help="run even though the base generator is live (GPU contention; not advised)")
    parser.add_argument("--mock-fail-rate", type=float, default=0.0)
    parser.add_argument("--mock-transient-failures", type=int, default=1)
    args = parser.parse_args(argv)

    if args.extended_cap <= args.base_cap:
        sys.exit(f"error: --extended-cap ({args.extended_cap}) must exceed the base cap "
                 f"({args.base_cap}); a smaller or equal cap cannot uncensor anything.")

    censored = find_censored_runs(args.base_dir, args.base_cap)
    prompts = {r["prompt_id"]: r for r in iter_jsonl(args.subset)}

    if args.dry_run:
        done = _load_done(Path(args.output_dir)) if Path(args.output_dir).exists() else set()
        remaining = [r for r in censored if (r["prompt_id"], r["seed"]) not in done]
        rep = {"base_dir": args.base_dir, "censored_runs": len(censored),
               "distinct_prompts_affected": len({r["prompt_id"] for r in censored}),
               "already_extended": len(censored) - len(remaining), "would_attempt": len(remaining),
               "base_cap": args.base_cap, "extended_cap": args.extended_cap, "dry_run": True}
        print(json.dumps(rep, indent=2))
        return rep

    live = None if args.allow_concurrent_base else base_generator_running()
    if live:
        sys.exit(f"error: the base generator looks alive (PID {live} from logs/generation.pid).\n"
                 "The extension pass would contend for the same GPU and distort the latency it "
                 "records, and the base run is still producing new censored runs.\n"
                 "Wait for it to finish, or pass --allow-concurrent-base if you accept that.")

    backend = make_backend(args)
    try:
        backend.check()
    except BackendError as e:
        sys.exit(f"error: {e}\nStart llama-server first (see datagen/README.md) or pass --url.")

    try:
        stats = extend_censored(censored, prompts, backend, args.output_dir, args.extended_cap,
                                args.base_dir, args.max_retries, args.backoff, args.limit)
    except (ConfigMismatch, ValueError) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted: all finished extended runs are saved. Rerun to resume.")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
