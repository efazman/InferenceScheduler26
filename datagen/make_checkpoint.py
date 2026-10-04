"""Cut standalone, independently trainable label snapshots from an in-progress label run.

    python -m datagen.make_checkpoint                 # cut every size in config.CHECKPOINT_SIZES that is ready
    python -m datagen.make_checkpoint --sizes 250 500  # only these
    python -m datagen.make_checkpoint --list           # show what exists / is ready, write nothing

Checkpoints are **nested**: labels are ranked by their position in the subset file, not by
completion time, so labels_250 is always a subset of labels_500, which is a subset of
labels_1000. Retries and resumes therefore cannot change what a checkpoint contains, and the
250 -> 500 -> 1000 -> 2000 learning curve measures dataset size alone.

An existing checkpoint is never overwritten (``--force`` overrides). Each checkpoint directory
holds a complete, self-sufficient training dataset:

    labels_<N>.jsonl         N label records, same schema as labels.jsonl (ml.data.load_jsonl reads it)
    generation_config.json   copied verbatim from the source run
    checkpoint.summary.json  size, category counts, target stats, truncation counts, source digest
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from datagen import config
from datagen.jsonl import iter_jsonl, repair_jsonl_tail, write_jsonl


def subset_rank(subset_path: str | Path) -> dict[str, int]:
    """prompt_id -> line number in the subset file. Defines the nesting order."""
    return {rec["prompt_id"]: i for i, rec in enumerate(iter_jsonl(subset_path))}


def ordered_labels(labels_path: str | Path, ranks: dict[str, int]) -> tuple[list[dict], list[str]]:
    """Completed labels sorted by subset position. Unknown prompt_ids sort last (and are reported)."""
    labels = list(iter_jsonl(labels_path))
    seen: dict[str, dict] = {}
    for rec in labels:
        seen.setdefault(rec["prompt_id"], rec)  # first write wins; labels.jsonl should not repeat
    unknown = sorted(pid for pid in seen if pid not in ranks)
    ordered = sorted(seen.values(), key=lambda r: (ranks.get(r["prompt_id"], len(ranks)), r["prompt_id"]))
    return ordered, unknown


def _truncated_runs(rec: dict) -> int:
    """Censored-run count, for both raw base labels and assembled final labels.

    Base labels (generate_labels) use "n_truncated_runs". Assembled labels
    (assemble_final_labels) split that into "n_base_truncated_runs" and
    "n_still_censored_runs"; without this fallback a checkpoint cut from assembled labels would
    report zero censoring, which is worse than reporting nothing.
    """
    if "n_truncated_runs" in rec:
        return rec["n_truncated_runs"]
    if "n_still_censored_runs" in rec:
        return rec["n_still_censored_runs"]
    return sum(bool(r.get("is_censored")) for r in (rec.get("runs") or ()))


def summarize(records: list[dict], target_field: str = config.TARGET_FIELD) -> dict:
    targets = np.asarray([r[target_field] for r in records], dtype=np.float64)
    truncated_runs = sum(_truncated_runs(r) for r in records)
    total_runs = sum(len(r.get("runs") or ()) for r in records)
    return {
        "n_prompts": len(records),
        "categories": dict(sorted(Counter(r.get("category") for r in records).items(),
                                  key=lambda kv: (kv[0] is None, kv[0]))),
        "target_field": target_field,
        "target_mean": float(targets.mean()),
        "target_median": float(np.median(targets)),
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "target_p90": float(np.percentile(targets, 90)),
        "total_runs": total_runs,
        "truncated_runs": truncated_runs,
        "truncated_run_fraction": round(truncated_runs / total_runs, 6) if total_runs else 0.0,
        "prompts_with_any_truncated_run": sum(bool(_truncated_runs(r)) for r in records),
    }


def cut_checkpoint(records: list[dict], n: int, out_root: Path, gen_cfg_path: Path,
                   source_labels: Path, force: bool = False) -> dict:
    """Write the first ``n`` ordered records as a standalone dataset. Returns its summary."""
    if len(records) < n:
        raise ValueError(f"only {len(records)} completed prompts, need {n}")
    out_dir = out_root / f"checkpoint_{n:05d}"
    labels_out = out_dir / f"labels_{n}.jsonl"
    if labels_out.exists() and not force:
        raise FileExistsError(f"{labels_out} already exists; checkpoints are never overwritten "
                              "(pass --force only if you know the source run was reset)")
    chosen = records[:n]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(chosen, labels_out)
    if gen_cfg_path.exists():
        (out_dir / config.GENERATION_CONFIG_FILE).write_text(
            gen_cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    digest = hashlib.sha256(labels_out.read_bytes()).hexdigest()
    summary = {
        "checkpoint_size": n,
        "labels_path": str(labels_out),
        "source_labels": str(source_labels),
        "ordering": "subset file position (nested: smaller checkpoints are subsets of larger ones)",
        "labels_sha256": digest,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **summarize(chosen),
    }
    (out_dir / "checkpoint.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels", default=str(config.REAL_LABELS_PATH))
    parser.add_argument("--subset", default=str(config.SUBSET_PATH))
    parser.add_argument("--out-root", default=None, help="default: <labels dir>/checkpoints")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(config.CHECKPOINT_SIZES))
    parser.add_argument("--list", action="store_true", help="report status only, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite an existing checkpoint")
    parser.add_argument("--allow-mock", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    labels_path = Path(args.labels)
    if not labels_path.exists():
        sys.exit(f"error: {labels_path} does not exist yet (no prompt has completed)")
    repair_jsonl_tail(labels_path)  # a checkpoint must never read a half-written final line
    out_root = Path(args.out_root) if args.out_root else labels_path.parent / "checkpoints"

    ranks = subset_rank(args.subset)
    records, unknown = ordered_labels(labels_path, ranks)
    mock = [r["prompt_id"] for r in records if r.get("is_mock")]
    if mock and not args.allow_mock:
        sys.exit(f"error: {len(mock)} label(s) are MOCK/synthetic; refusing to build a training "
                 "checkpoint from them. Use --allow-mock for tests only.")

    report = {"labels": str(labels_path), "subset": str(args.subset), "out_root": str(out_root),
              "completed_prompts": len(records), "subset_size": len(ranks),
              "sizes_requested": args.sizes, "written": {}, "skipped": {}}
    if unknown:
        report["prompt_ids_not_in_subset"] = unknown[:20]

    for n in sorted(args.sizes):
        existing = out_root / f"checkpoint_{n:05d}" / f"labels_{n}.jsonl"
        if existing.exists() and not args.force:
            report["skipped"][str(n)] = "already exists"
            continue
        if len(records) < n:
            report["skipped"][str(n)] = f"not ready ({len(records)}/{n} prompts complete)"
            continue
        if args.list:
            report["skipped"][str(n)] = "ready (not written: --list)"
            continue
        report["written"][str(n)] = cut_checkpoint(
            records, n, out_root, labels_path.parent / config.GENERATION_CONFIG_FILE,
            labels_path, force=args.force)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
