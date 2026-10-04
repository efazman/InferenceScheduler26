"""Assemble final training labels from base runs + the censored-output extension pass.

    python -m datagen.assemble_final_labels          # -> data/labels/llama31_8b_q4km_final/labels_final.jsonl
    python -m datagen.assemble_final_labels --no-extension   # base only (reports censoring honestly)

Per-run precedence, applied independently to each of a prompt's generations:

    base finished naturally (finish_reason != "length")
        -> use the base length. Exact.
    base was capped AND a successful extended rerun exists
        -> use the EXTENDED length. Exact if the extended run also finished naturally,
           still censored if it hit the extended cap too.
    base was capped and no extended run exists
        -> keep the base length, flagged as censored. Never silently treated as exact.

The prompt-level target is then recomputed with the same p90-over-4-runs rule and the same
numpy call the base pipeline uses, so the arithmetic is identical - only the inputs improve.

``has_censored_target`` is **not** "any run was censored". It is true only when the p90 value
actually depends on a censored measurement. With 4 runs and linear interpolation, p90 sits at
sorted index 2.7, so it is a blend of the two largest values; a censored run among the two
smallest cannot move it. Ties are resolved conservatively: if any run sharing a contributing
value is censored, the target is flagged.

Neither the base files nor the extension files are modified. Output is a new directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from datagen import config
from datagen.extend_censored import is_censored
from datagen.generate_labels import percentile_target
from datagen.jsonl import iter_jsonl, repair_jsonl_tail, write_jsonl


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def percentile_contributors(n: int, q: float = config.TARGET_PERCENTILE,
                            method: str = config.PERCENTILE_METHOD) -> list[int] | None:
    """Sorted-array indices the percentile actually reads, or None if we cannot be sure.

    Only "linear" is modelled. For any other method, None means "fall back to the conservative
    rule" rather than quietly assuming this interpolation.
    """
    if method != "linear" or n < 1:
        return None
    pos = (q / 100.0) * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    frac = pos - lo
    idx = set()
    if frac < 1.0:
        idx.add(lo)
    if frac > 0.0:
        idx.add(hi)
    return sorted(idx)


def target_is_censored(effective: list[float], censored: list[bool],
                       q: float = config.TARGET_PERCENTILE,
                       method: str = config.PERCENTILE_METHOD) -> bool:
    """True if the percentile depends on a right-censored value."""
    contributors = percentile_contributors(len(effective), q, method)
    if contributors is None:
        return any(censored)  # unknown interpolation -> assume the worst
    order = sorted(range(len(effective)), key=lambda i: effective[i])
    values_sorted = [effective[i] for i in order]
    for pos in contributors:
        value_at_pos = values_sorted[pos]
        # Conservative tie handling: any censored run sharing this value taints the target.
        if any(censored[i] for i in range(len(effective)) if effective[i] == value_at_pos):
            return True
    return False


def load_extended(ext_dir: str | Path, allow_mock: bool = False) -> dict[tuple[str, int], dict]:
    """(prompt_id, seed) -> extended run. Last write wins, matching a resumed pass."""
    path = Path(ext_dir) / config.EXTENDED_RUNS_FILE
    if not path.exists():
        return {}
    repair_jsonl_tail(path)
    out: dict[tuple[str, int], dict] = {}
    for rec in iter_jsonl(path):
        if rec.get("is_mock") and not allow_mock:
            raise ValueError(f"{path} contains MOCK extended runs; they are test-only and must "
                             "not reach a training label set")
        out[(rec["prompt_id"], rec["seed"])] = rec
    return out


def resolve_runs(label: dict, extended: dict[tuple[str, int], dict],
                 base_cap: int = config.MAX_NEW_TOKENS) -> list[dict]:
    """One resolved record per run, with the effective length, its source and censoring."""
    pid = label["prompt_id"]
    resolved = []
    for run in label["runs"]:
        seed = run["seed"]
        base_tokens = run["output_tokens"]
        base_censored = is_censored(run, base_cap)
        ext = extended.get((pid, seed))

        if not base_censored:
            eff, src, cens = base_tokens, "base", False
            ext_tokens = ext_reason = None
        elif ext is not None:
            eff = ext["extended_output_tokens"]
            src = "extended"
            cens = bool(ext.get("still_censored"))
            ext_tokens, ext_reason = ext["extended_output_tokens"], ext.get("extended_finish_reason")
        else:
            eff, src, cens = base_tokens, "base_censored_unextended", True
            ext_tokens = ext_reason = None

        resolved.append({
            "seed": seed,
            "run_index": run.get("run_index"),
            "base_output_tokens": base_tokens,
            "base_finish_reason": run.get("finish_reason"),
            "base_censored": base_censored,
            "extended_output_tokens": ext_tokens,
            "extended_finish_reason": ext_reason,
            "effective_output_tokens": eff,
            "length_source": src,
            "is_censored": cens,
            "latency_ms": run.get("latency_ms"),
            "extended_latency_ms": ext.get("latency_ms") if ext else None,
            "prompt_tokens": run.get("prompt_tokens"),
        })
    return resolved


def build_final_label(label: dict, extended: dict[tuple[str, int], dict],
                      base_cap: int = config.MAX_NEW_TOKENS, ext_cap: int | None = None) -> dict:
    runs = resolve_runs(label, extended, base_cap)
    eff = [r["effective_output_tokens"] for r in runs]
    cens = [r["is_censored"] for r in runs]
    target = percentile_target(eff, label.get("target_percentile", config.TARGET_PERCENTILE),
                               label.get("percentile_method", config.PERCENTILE_METHOD))
    out = {
        # --- training-loader fields (ml.data.load_jsonl reads these) ---
        "prompt_id": label["prompt_id"],
        "prompt": label["prompt"],
        "category": label.get("category"),
        config.TARGET_FIELD: target,
        # --- audit ---
        "runs": runs,
        "n_runs": len(runs),
        "n_base_truncated_runs": sum(r["base_censored"] for r in runs),
        "n_extended_runs": sum(r["length_source"] == "extended" for r in runs),
        "n_still_censored_runs": sum(cens),
        "n_unextended_censored_runs": sum(r["length_source"] == "base_censored_unextended" for r in runs),
        "has_censored_target": target_is_censored(
            eff, cens, label.get("target_percentile", config.TARGET_PERCENTILE),
            label.get("percentile_method", config.PERCENTILE_METHOD)),
        "base_target_p90_output_tokens": label.get(config.TARGET_FIELD),
        # --- provenance ---
        "category_source": label.get("category_source"),
        "source": label.get("source"),
        "model": label.get("model"),
        "quantization": label.get("quantization"),
        "runtime": label.get("runtime"),
        "runtime_version": label.get("runtime_version"),
        "temperature": label.get("temperature"),
        "top_p": label.get("top_p"),
        "base_max_new_tokens": label.get("max_new_tokens"),
        "extended_max_new_tokens": ext_cap,
        "system_prompt_sha256": label.get("system_prompt_sha256"),
        "target_percentile": label.get("target_percentile"),
        "percentile_method": label.get("percentile_method"),
        "is_mock": bool(label.get("is_mock")),
        "assembled_at": _now(),
    }
    return out


def assemble(base_dir: str | Path = config.LABELS_DIR_REAL, ext_dir: str | Path | None = config.LABELS_DIR_EXT,
             out_dir: str | Path = config.LABELS_DIR_FINAL, base_cap: int = config.MAX_NEW_TOKENS,
             allow_mock: bool = False) -> dict:
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    labels_path = base_dir / config.LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(f"no base labels at {labels_path}")
    if out_dir.resolve() in (base_dir.resolve(), Path(ext_dir).resolve() if ext_dir else None):
        raise ValueError("refusing to write final labels into the base or extension directory")
    repair_jsonl_tail(labels_path)  # never read a half-written final line from a live run

    extended = load_extended(ext_dir, allow_mock) if ext_dir else {}
    ext_cap = None
    if ext_dir:
        cfg_path = Path(ext_dir) / config.EXTENSION_CONFIG_FILE
        if cfg_path.exists():
            ext_cap = json.loads(cfg_path.read_text(encoding="utf-8")).get("extended_max_new_tokens")

    base_labels = list(iter_jsonl(labels_path))
    mock = [l for l in base_labels if l.get("is_mock")]
    if mock and not allow_mock:
        raise ValueError(f"{labels_path} contains {len(mock)} MOCK label(s); refusing to build a "
                         "training label set from them")

    final = [build_final_label(l, extended, base_cap, ext_cap) for l in base_labels]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config.FINAL_LABELS_FILE
    write_jsonl(final, out_path)

    n_runs = sum(f["n_runs"] for f in final)
    base_trunc = sum(f["n_base_truncated_runs"] for f in final)
    still = sum(f["n_still_censored_runs"] for f in final)
    unext = sum(f["n_unextended_censored_runs"] for f in final)
    moved = [f for f in final if f["n_extended_runs"] and f["base_target_p90_output_tokens"] is not None
             and abs(f[config.TARGET_FIELD] - f["base_target_p90_output_tokens"]) > 1e-9]
    summary = {
        "base_dir": str(base_dir),
        "extension_dir": str(ext_dir) if ext_dir else None,
        "output": str(out_path),
        "base_cap": base_cap,
        "extended_cap": ext_cap,
        "prompts": len(final),
        "runs": n_runs,
        "base_truncated_runs": base_trunc,
        "base_truncation_rate": round(base_trunc / n_runs, 6) if n_runs else 0.0,
        "extended_runs_applied": sum(f["n_extended_runs"] for f in final),
        "censored_runs_without_extension": unext,
        "still_censored_runs": still,
        "still_censored_run_rate": round(still / n_runs, 6) if n_runs else 0.0,
        "prompts_with_any_base_truncation": sum(1 for f in final if f["n_base_truncated_runs"]),
        "prompts_with_any_still_censored_run": sum(1 for f in final if f["n_still_censored_runs"]),
        "prompts_with_censored_target": sum(1 for f in final if f["has_censored_target"]),
        "prompts_whose_target_changed_vs_base": len(moved),
        "categories": dict(sorted(Counter(f.get("category") for f in final).items(),
                                  key=lambda kv: (kv[0] is None, kv[0]))),
        "target_field": config.TARGET_FIELD,
        "assembled_at": _now(),
    }
    if final:
        t = np.asarray([f[config.TARGET_FIELD] for f in final], dtype=np.float64)
        summary.update(target_mean=float(t.mean()), target_median=float(np.median(t)),
                       target_min=float(t.min()), target_max=float(t.max()),
                       target_p90=float(np.percentile(t, 90)))
    (out_dir / "assembly.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-dir", default=str(config.LABELS_DIR_REAL))
    parser.add_argument("--extension-dir", default=str(config.LABELS_DIR_EXT))
    parser.add_argument("--no-extension", action="store_true",
                        help="assemble from base runs only; censoring is reported, not resolved")
    parser.add_argument("--output-dir", default=str(config.LABELS_DIR_FINAL))
    parser.add_argument("--base-cap", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--allow-mock", action="store_true", help="tests only")
    args = parser.parse_args(argv)
    try:
        summary = assemble(args.base_dir, None if args.no_extension else args.extension_dir,
                           args.output_dir, args.base_cap, args.allow_mock)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
