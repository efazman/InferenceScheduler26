"""LMSYS-Chat-1M -> clean single-turn English prompt JSONL (streaming, deterministic).

    python -m datagen.preprocess_lmsys [--input data/raw/lmsys-chat-1m] [--output data/lmsys/clean_prompts.jsonl]

Input: a directory of the dataset's parquet shards (as downloaded by `hf download`), a single
.parquet file, or a .jsonl file with the same columns. Rows are streamed in batches, so the 1M
conversations are never loaded at once; only a set of 32-byte prompt hashes is kept for dedup.

Filters, in order (each rejection is counted in the summary):
  malformed     row missing / non-list conversation, or non-string user content
  language      `language` != config.LANGUAGE
  multi_turn    not exactly one user message, or the first message is not from the user
  empty         prompt shorter than MIN_PROMPT_CHARS after stripping
  too_long      prompt longer than MAX_PROMPT_CHARS
  moderation    flagged by the OpenAI moderation pass (if EXCLUDE_MODERATION_FLAGGED)
  duplicate     exact duplicate of an earlier prompt (first occurrence in file order is kept)

The prompt text itself is never modified. prompt_id = "lmsys-" + first 16 hex chars of the
SHA-256 of the exact prompt text, so it is stable across reruns and independent of row order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from datagen import config
from datagen.jsonl import iter_jsonl

COLUMNS = ["conversation_id", "model", "conversation", "turn", "language", "openai_moderation", "redacted"]


def prompt_id_for(prompt: str) -> str:
    return "lmsys-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def iter_raw_rows(path: str | Path, batch_size: int = 2048) -> Iterator[dict]:
    """Yield raw LMSYS rows from parquet shard(s) or JSONL, in a deterministic order."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no .parquet files under {path}")
    else:
        files = [path]
    for f in files:
        if f.suffix == ".jsonl":
            yield from iter_jsonl(f)
            continue
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(f)
        cols = [c for c in COLUMNS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            yield from batch.to_pylist()


def _moderation_flagged(row: dict) -> bool:
    mods = row.get("openai_moderation") or []
    return bool(mods) and bool(mods[0] and mods[0].get("flagged"))


def clean_row(row: dict, language: str = config.LANGUAGE, min_chars: int = config.MIN_PROMPT_CHARS,
              max_chars: int = config.MAX_PROMPT_CHARS,
              exclude_flagged: bool = config.EXCLUDE_MODERATION_FLAGGED) -> tuple[dict | None, str]:
    """Return (record, "ok") or (None, rejection_reason). Does not handle duplicates."""
    conv = row.get("conversation") if isinstance(row, dict) else None
    if not isinstance(conv, list) or not conv or not all(isinstance(m, dict) for m in conv):
        return None, "malformed"
    if row.get("language") != language:
        return None, "language"
    user_msgs = [m for m in conv if m.get("role") == "user"]
    if len(user_msgs) != 1 or conv[0].get("role") != "user":
        return None, "multi_turn"
    prompt = user_msgs[0].get("content")
    if not isinstance(prompt, str):
        return None, "malformed"
    if len(prompt.strip()) < max(min_chars, 1):
        return None, "empty"
    if len(prompt) > max_chars:
        return None, "too_long"
    if exclude_flagged and _moderation_flagged(row):
        return None, "moderation"
    return {
        "prompt_id": prompt_id_for(prompt),
        "prompt": prompt,
        "category": None,
        "source": config.DATASET_NAME,
        "source_metadata": {
            "conversation_id": row.get("conversation_id"),
            "model": row.get("model"),
            "language": row.get("language"),
            "turn": row.get("turn"),
            "redacted": row.get("redacted"),
            "moderation_flagged": _moderation_flagged(row),
        },
    }, "ok"


def preprocess(input_path: str | Path, output_path: str | Path, **filters) -> dict:
    """Stream input -> output JSONL. Returns a summary with per-reason rejection counts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    seen: set[bytes] = set()
    counts: Counter = Counter()
    with tmp.open("w", encoding="utf-8") as out:
        for row in iter_raw_rows(input_path):
            counts["rows_read"] += 1
            rec, reason = clean_row(row, **filters)
            if rec is None:
                counts[f"rejected_{reason}"] += 1
                continue
            digest = hashlib.sha256(rec["prompt"].encode("utf-8")).digest()
            if digest in seen:
                counts["rejected_duplicate"] += 1
                continue
            seen.add(digest)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["kept"] += 1
    tmp.replace(output_path)  # never leave a half-written clean file behind
    summary = {"input": str(input_path), "output": str(output_path), **dict(sorted(counts.items()))}
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default=str(config.LMSYS_RAW_DIR))
    parser.add_argument("--output", default=str(config.CLEAN_PROMPTS_PATH))
    parser.add_argument("--max-prompt-chars", type=int, default=config.MAX_PROMPT_CHARS)
    parser.add_argument("--keep-flagged", action="store_true", help="keep moderation-flagged prompts")
    args = parser.parse_args(argv)
    summary = preprocess(args.input, args.output, max_chars=args.max_prompt_chars,
                         exclude_flagged=not args.keep_flagged)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
