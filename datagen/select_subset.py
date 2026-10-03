"""Pick the initial category-balanced prompt subset (default 500) from the clean prompt file.

    python -m datagen.select_subset [--input data/lmsys/clean_prompts.jsonl] [--n 500]

1. Each prompt gets a heuristic category (datagen.categories).
2. Per-category quotas come from config.CATEGORY_PROPORTIONS (largest-remainder rounding, so they
   sum to exactly n).
3. One seeded reservoir sample per category is kept while streaming, so memory stays at
   O(n x categories) no matter how large the clean file is.
4. If a category has fewer prompts than its quota, the shortfall is filled from the other
   categories' reservoirs in a fixed order, and the summary records it.
5. The result is shuffled with the seed. Same input + seed -> byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from datagen import config
from datagen.categories import CATEGORIES, CATEGORY_SOURCE, assign_category
from datagen.jsonl import iter_jsonl, write_jsonl


def compute_quotas(n: int, proportions: dict[str, float]) -> dict[str, int]:
    total = sum(proportions.values())
    raw = {c: n * p / total for c, p in proportions.items()}
    quotas = {c: math.floor(v) for c, v in raw.items()}
    leftover = n - sum(quotas.values())
    for c in sorted(raw, key=lambda c: (-(raw[c] - quotas[c]), c))[:leftover]:
        quotas[c] += 1
    return quotas


def select_subset(records, n: int = config.NUM_PROMPTS, seed: int = config.SEED,
                  proportions: dict[str, float] = config.CATEGORY_PROPORTIONS):
    """Return (subset, summary). ``records`` is any iterable of clean prompt dicts."""
    unknown = set(proportions) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown categories in proportions: {sorted(unknown)}")
    quotas = compute_quotas(n, proportions)
    rng = random.Random(seed)
    cap = n  # reservoir size per category; n is enough to cover any redistributed shortfall
    reservoirs: dict[str, list[dict]] = {c: [] for c in proportions}
    seen = {c: 0 for c in proportions}
    available = {c: 0 for c in CATEGORIES}

    for rec in records:
        cat = assign_category(rec["prompt"])
        available[cat] += 1
        if cat not in reservoirs:
            continue  # category not requested
        rec = {**rec, "category": cat, "category_source": CATEGORY_SOURCE}
        seen[cat] += 1
        res = reservoirs[cat]
        if len(res) < cap:
            res.append(rec)
        else:
            j = rng.randrange(seen[cat])
            if j < cap:
                res[j] = rec

    for res in reservoirs.values():  # reservoir order depends on replacement history; fix it
        random.Random(seed).shuffle(res)
    chosen = {c: reservoirs[c][:quotas[c]] for c in proportions}
    shortfall = n - sum(len(v) for v in chosen.values())
    for c in proportions:  # fill gaps in config order, deterministically
        if shortfall <= 0:
            break
        extra = reservoirs[c][len(chosen[c]):len(chosen[c]) + shortfall]
        chosen[c] += extra
        shortfall -= len(extra)

    subset = [r for c in proportions for r in chosen[c]]
    random.Random(seed).shuffle(subset)
    summary = {
        "n_requested": n, "n_selected": len(subset), "seed": seed, "category_source": CATEGORY_SOURCE,
        "quotas": quotas, "selected": {c: len(chosen[c]) for c in proportions},
        "available_in_input": available,
    }
    if len(subset) < n:
        summary["warning"] = f"input only had {len(subset)} usable prompts"
    return subset, summary


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default=str(config.CLEAN_PROMPTS_PATH))
    parser.add_argument("--output", default=None, help="default: data/lmsys/subset_<n>.jsonl")
    parser.add_argument("--n", type=int, default=config.NUM_PROMPTS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)
    output = Path(args.output or config.DATA_DIR / "lmsys" / f"subset_{args.n}.jsonl")
    subset, summary = select_subset(iter_jsonl(args.input), args.n, args.seed)
    write_jsonl(subset, output)
    summary["output"] = str(output)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
