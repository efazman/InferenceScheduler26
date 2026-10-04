"""Fail if the committed measurement archive contains prompt or generated text.

    python scripts/check_archive_safe.py

The scheduler's own run directories are gitignored because `manifest.json` embeds LMSYS prompt
text. `docs/measurements/` deliberately archives only `events.jsonl` and `summary.json`, which
carry counts (`prompt_chars`, `prompt_tokens`) rather than text. This script is the guard for that
claim: it walks every value in the archive and flags any long string or text-shaped field name, so
a future change that starts logging prompt text cannot slip into the repo unnoticed.
"""

from __future__ import annotations

import glob
import json
import sys

MAX_STRING = 200  # no legitimate field in these files is this long
TEXT_FIELDS = {"prompt", "text", "content", "output", "response", "message", "completion"}
SKIP_LONG = {"description"}  # human-readable provenance strings are fine


def walk(node, path=""):
    """Yield (path, value) for every leaf in a nested JSON structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, node


def main() -> int:
    files = sorted(glob.glob("docs/measurements/*/events.jsonl") +
                   glob.glob("docs/measurements/*/summary.json"))
    if not files:
        print("no archive files found under docs/measurements/")
        return 1

    long_strings, text_named, n_values = [], [], 0
    for f in files:
        records = ([json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
                   if f.endswith(".jsonl") else [json.load(open(f, encoding="utf-8"))])
        for rec in records:
            for path, value in walk(rec):
                n_values += 1
                leaf = path.split(".")[-1].split("[")[0]
                if isinstance(value, str) and len(value) > MAX_STRING and leaf not in SKIP_LONG:
                    long_strings.append((f, path, len(value), value[:70]))
                if leaf in TEXT_FIELDS:
                    text_named.append((f, path))

    stray = sorted(glob.glob("docs/measurements/*/manifest.json"))

    print(f"files checked      : {len(files)}")
    print(f"values walked      : {n_values}")
    print(f"long strings       : {len(long_strings)}")
    print(f"text-named fields  : {len(text_named)}")
    print(f"manifest.json found: {len(stray)}  (must be 0 - manifests embed prompt text)")

    for f, path, n, head in long_strings[:10]:
        print(f"  LONG  {f} :: {path} ({n} chars) {head!r}")
    for f, path in text_named[:10]:
        print(f"  TEXT  {f} :: {path}")
    for f in stray:
        print(f"  STRAY {f}")

    if long_strings or text_named or stray:
        print("\nFAIL: the archive may contain prompt or generated text.")
        return 1
    print("\nOK: archive contains measurements only, no prompt or generated text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
