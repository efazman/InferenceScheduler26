"""Dataset loading, splitting, quantile binning, and soft labels.

The synthetic generator in this file exists ONLY to validate the pipeline.
Its labels are invented, not measured; do not draw conclusions from them.

Run ``python -m ml.data`` to (re)write the synthetic JSONL file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Synthetic dataset (SYNTHETIC: delete/replace this whole section for real data)
# ---------------------------------------------------------------------------

# REAL-DATA: these per-category ranges are invented placeholders. Real labels will be the
# p90 output-token count over N (default 4) generations of the chosen Llama model.
_SYNTHETIC_CATEGORIES = {
    "factual": {
        "range": (20, 80),
        "templates": [
            "What is {x}?",
            "Who maintains {x}?",
            "When was {x} introduced?",
            "What port does {x} use by default?",
            "Is {x} open source?",
        ],
        "fillers": ["PostgreSQL", "Kubernetes", "the HTTP/2 protocol", "Redis", "OAuth 2.0",
                    "gRPC", "TLS 1.3", "the SMTP protocol", "Kafka", "SSH"],
    },
    "internal_knowledge": {
        "range": (80, 150),
        "templates": [
            "What is our company policy on {x}?",
            "Briefly explain how {x} works at our company.",
            "Who should I contact about {x}?",
            "Where can I find the internal documentation for {x}?",
        ],
        "fillers": ["expense reimbursement", "remote work", "VPN access", "the on-call rotation",
                    "parental leave", "laptop upgrades", "code review approvals", "PTO carryover",
                    "security incident reporting", "the quarterly planning process"],
    },
    "troubleshooting": {
        "range": (100, 300),
        "templates": [
            "My {x} keeps failing with a timeout error. How do I fix it?",
            "I can't connect to {x} after the latest update. What should I check?",
            "{x} is running very slowly today. How can I diagnose the problem?",
            "I get a permission denied error when using {x}. What could cause this?",
        ],
        "fillers": ["CI pipeline", "VPN client", "staging database", "Docker build",
                    "SSO login", "internal wiki", "deployment script", "shared drive mount"],
    },
    "coding": {
        "range": (100, 300),
        "templates": [
            "Write a Python function that {x}.",
            "How do I {x} in JavaScript?",
            "Explain what this does and suggest improvements: a loop that {x}.",
            "Write a unit test for a function that {x}.",
        ],
        "fillers": ["parses a CSV file into a list of dicts", "retries an HTTP request with backoff",
                    "deduplicates a list while preserving order", "validates an email address",
                    "merges two sorted arrays", "reads a JSON config with defaults",
                    "computes a rolling average", "flattens a nested dictionary"],
    },
    "reasoning": {
        "range": (200, 500),
        "templates": [
            "Compare {x} and explain which is better for a mid-sized team, step by step.",
            "Walk through the trade-offs of {x} and give a recommendation.",
            "We are deciding on {x}. Reason through the risks and benefits in detail.",
        ],
        "fillers": ["monolith vs microservices", "SQL vs NoSQL for our order system",
                    "building vs buying an internal ticketing tool",
                    "moving our CI from Jenkins to GitHub Actions",
                    "self-hosting an LLM vs using a hosted API",
                    "a four-day work week for the support team"],
    },
    "summarization": {
        "range": (200, 500),
        "templates": [
            "Summarize the following meeting notes about {x}: The team discussed timelines, "
            "owners, open risks, budget constraints, and next steps for the coming quarter.",
            "Summarize this incident report on {x}, including root cause, impact, timeline, "
            "and remediation items.",
            "Give a detailed summary of the attached design document for {x}.",
        ],
        "fillers": ["the billing migration", "the Q3 outage", "the new onboarding flow",
                    "the data retention project", "the search reindexing effort",
                    "the mobile app redesign"],
    },
    "structured_generation": {
        "range": (100, 300),
        "templates": [
            "Generate a JSON object describing {x} with fields for name, owner, status, and tags.",
            "Create a markdown table listing {x} with columns for item, priority, and due date.",
            "Draft a YAML config for {x}.",
        ],
        "fillers": ["our microservices", "the release checklist", "the team's OKRs",
                    "a nightly backup job", "the support escalation matrix",
                    "a feature flag rollout"],
    },
}


def generate_synthetic_dataset(n_per_category: int = 17, seed: int = 42) -> list[dict]:
    """Return ~n_per_category * 7 synthetic examples. SYNTHETIC labels, pipeline validation only."""
    rng = random.Random(seed)
    examples: list[dict] = []
    for category, spec in _SYNTHETIC_CATEGORIES.items():
        combos = [(t, f) for t in spec["templates"] for f in spec["fillers"]]
        rng.shuffle(combos)
        lo, hi = spec["range"]
        for template, filler in combos[:n_per_category]:
            examples.append({
                "id": f"syn-{len(examples):04d}",
                "category": category,
                "prompt": template.format(x=filler),
                "target_output_tokens": rng.randint(lo, hi),
                "synthetic": True,
            })
    return examples


# ---------------------------------------------------------------------------
# I/O and splitting
# ---------------------------------------------------------------------------

def save_jsonl(examples: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def load_jsonl(path: str | Path, target_field: str = "target_output_tokens",
               allow_mock: bool = False) -> list[dict]:
    """Load labelled prompts and normalize them to {"id", "prompt", "target_output_tokens", ...}.

    Accepts both the synthetic file ("id", "target_output_tokens") and real label files from
    datagen.generate_labels ("prompt_id", "target_p90_output_tokens"). ``target_field`` names the
    label column; its value is copied to "target_output_tokens", which the rest of the pipeline
    uses. "category" and all other fields are kept as-is. MockBackend labels ("is_mock": true)
    are refused unless ``allow_mock`` is set, so they can never end up in a real training run.
    """
    with Path(path).open() as f:
        examples = [json.loads(line) for line in f if line.strip()]
    for i, ex in enumerate(examples):
        if ex.get("is_mock") and not allow_mock:
            raise ValueError(f"{path} contains MOCK labels (example {i}); they are test-only and "
                             "must not be used for training or evaluation")
        if not isinstance(ex.get("prompt"), str) or target_field not in ex:
            raise ValueError(f"example {i} in {path} missing 'prompt' or '{target_field}'")
        target = ex[target_field]
        if not isinstance(target, (int, float)) or not np.isfinite(target) or target < 0:
            raise ValueError(f"example {i} in {path} has invalid {target_field}={target!r}")
        ex["target_output_tokens"] = target
        ex.setdefault("id", ex.get("prompt_id") or f"ex-{i:05d}")
    ids = [ex["id"] for ex in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} has duplicate ids")
    return examples


def split_dataset(examples: list[dict], train_frac: float, val_frac: float, seed: int,
                  strategy: str = "random"):
    """Split into (train, val, test) with ``seed``; test gets the remainder.

    "random": one shuffle of all examples. "stratified": the same fractions applied within each
    "category" (missing category = its own group), so every workload type appears in every split.
    """
    if strategy == "random":
        groups = {None: list(range(len(examples)))}
    elif strategy == "stratified":
        groups = {}
        for i, ex in enumerate(examples):
            groups.setdefault(str(ex.get("category")), []).append(i)
    else:
        raise ValueError(f"unknown split strategy {strategy!r}")
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for key in sorted(groups, key=str):
        idx = groups[key]
        rng.shuffle(idx)
        n_train = int(round(train_frac * len(idx)))
        n_val = int(round(val_frac * len(idx)))
        tr += idx[:n_train]
        va += idx[n_train:n_train + n_val]
        te += idx[n_train + n_val:]
    pick = lambda ids: [examples[i] for i in ids]  # noqa: E731
    return pick(tr), pick(va), pick(te)


def assert_disjoint_splits(*splits: list[dict]) -> None:
    """Leakage guard: no prompt (exact text, whitespace/case-normalized) may appear in two splits."""
    # REAL-DATA: near-duplicate LMSYS prompts (same template, different names) are not caught here.
    seen: dict[str, int] = {}
    for s_idx, split in enumerate(splits):
        for ex in split:
            key = " ".join(ex["prompt"].lower().split())
            if seen.setdefault(key, s_idx) != s_idx:
                raise ValueError(f"prompt appears in splits {seen[key]} and {s_idx}: {ex['prompt'][:80]!r}")


# ---------------------------------------------------------------------------
# Binning and soft labels
# ---------------------------------------------------------------------------

def compute_quantile_bins(train_targets, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (boundaries[n_bins + 1], centers[n_bins]) from TRAINING targets only.

    Boundaries are the 0..100% quantiles. Integer targets can produce tied quantiles; ties are
    nudged apart by a tiny epsilon so boundaries are strictly increasing (the affected bin is
    then simply empty). Each center is the median training target inside the bin, falling back
    to the bin midpoint for empty bins.
    """
    t = np.asarray(train_targets, dtype=np.float64)
    boundaries = np.quantile(t, np.linspace(0.0, 1.0, n_bins + 1))
    for k in range(1, len(boundaries)):
        if boundaries[k] <= boundaries[k - 1]:
            boundaries[k] = boundaries[k - 1] + 1e-6
    bins = assign_bins(t, boundaries)
    centers = np.empty(n_bins)
    for b in range(n_bins):
        members = t[bins == b]
        centers[b] = np.median(members) if len(members) else 0.5 * (boundaries[b] + boundaries[b + 1])
    return boundaries, centers


def assign_bins(targets, boundaries: np.ndarray) -> np.ndarray:
    """Map targets to bin indices; values outside the training range clamp to the edge bins."""
    n_bins = len(boundaries) - 1
    idx = np.searchsorted(boundaries[1:-1], np.asarray(targets, dtype=np.float64), side="right")
    return np.clip(idx, 0, n_bins - 1)


def soft_label(bin_idx: int, n_bins: int, scale: float = 1.0) -> np.ndarray:
    """p(j) proportional to exp(-|j - bin_idx| / scale), normalized to sum to 1."""
    w = np.exp(-np.abs(np.arange(n_bins) - bin_idx) / scale)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------

class LengthDataset(Dataset):
    def __init__(self, examples: list[dict], boundaries: np.ndarray, soft_label_scale: float = 1.0):
        self.prompts = [ex["prompt"] for ex in examples]
        self.targets = np.array([ex["target_output_tokens"] for ex in examples], dtype=np.float32)
        n_bins = len(boundaries) - 1
        bins = assign_bins(self.targets, boundaries)
        self.soft = np.stack([soft_label(b, n_bins, soft_label_scale) for b in bins]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, i):
        return self.prompts[i], self.soft[i], self.targets[i]


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        prompts, soft, targets = zip(*batch)
        enc = tokenizer(list(prompts), padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "soft_targets": torch.tensor(np.stack(soft)),
            "target_tokens": torch.tensor(np.array(targets)),
        }
    return collate


if __name__ == "__main__":
    import argparse

    from ml.config import Config

    parser = argparse.ArgumentParser(description="Write the SYNTHETIC pipeline-validation dataset.")
    parser.add_argument("--out", default=Config().data_path)
    parser.add_argument("--n-per-category", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = generate_synthetic_dataset(args.n_per_category, args.seed)
    save_jsonl(data, args.out)
    print(f"wrote {len(data)} SYNTHETIC examples to {args.out}")
