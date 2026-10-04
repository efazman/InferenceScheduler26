"""Experiment manifest: the fixed arrival trace every policy replays.

A manifest pins everything that must be identical across FIFO / SEJF / adaptive runs: the request
set, prompt text, arrival times, per-request generation seeds and metadata. It contains no
scheduling order; each policy consumes it independently. ``manifest_id`` is a hash of the
requests, so runs can prove they used the same trace (``python -m scheduler report`` checks it).

File format (JSON):
    {
      "manifest_version": 1,
      "manifest_id": "<sha256 of the canonical requests list, first 16 hex>",
      "name": "...", "synthetic": false, "created_at": "...",
      "arrival": {"process": "poisson", "mean_interarrival_ms": 7000, "seed": 42},
      "source": {...},                       # where the prompts came from
      "requests": [
        {"request_id": "lmsys-...", "prompt": "...", "arrival_ms": 0.0, "seed": 0,
         "category": "coding_debugging", "size_class": null, "label_target_tokens": 412.0,
         "mock_actual_tokens": null, "mock_predicted_tokens": null},
        ...
      ]
    }
Synthetic manifests (built from a named workload) also carry mock_* lengths, so the mock backend
and mock predictor can replay them. Real manifests leave those null.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from scheduler.workloads import Workload, WorkloadRequest, workload_from_prompts

MANIFEST_VERSION = 1
REQUIRED = ("request_id", "prompt", "arrival_ms", "seed")


def _manifest_id(requests: list[dict]) -> str:
    canon = json.dumps(requests, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


@dataclass
class Manifest:
    name: str
    requests: list[dict]
    synthetic: bool
    arrival: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))

    @property
    def manifest_id(self) -> str:
        return _manifest_id(self.requests)

    def validate(self) -> "Manifest":
        if not self.requests:
            raise ValueError("manifest has no requests")
        for i, r in enumerate(self.requests):
            missing = [k for k in REQUIRED if k not in r or r[k] is None]
            if missing:
                raise ValueError(f"request {i} missing {missing}")
            if not isinstance(r["prompt"], str) or not r["prompt"]:
                raise ValueError(f"request {i} has an empty prompt")
            if float(r["arrival_ms"]) < 0:
                raise ValueError(f"request {i} has a negative arrival time")
        ids = [r["request_id"] for r in self.requests]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate request_id in manifest")
        arrivals = [float(r["arrival_ms"]) for r in self.requests]
        if arrivals != sorted(arrivals):
            raise ValueError("requests must be sorted by arrival_ms")
        if self.synthetic and any(r.get("mock_actual_tokens") is None for r in self.requests):
            raise ValueError("synthetic manifest needs mock_actual_tokens on every request")
        return self

    def to_dict(self) -> dict:
        return {"manifest_version": MANIFEST_VERSION, "manifest_id": self.manifest_id, "name": self.name,
                "synthetic": self.synthetic, "created_at": self.created_at, "arrival": self.arrival,
                "source": self.source, "requests": self.requests}

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest_version {raw.get('manifest_version')!r}")
        m = cls(raw["name"], raw["requests"], bool(raw["synthetic"]), raw.get("arrival", {}), raw.get("source", {}),
                raw.get("created_at", ""))
        m.validate()
        if raw.get("manifest_id") and raw["manifest_id"] != m.manifest_id:
            raise ValueError("manifest_id does not match its requests (file edited by hand?)")
        return m

    def to_workload(self) -> Workload:
        items = [WorkloadRequest(r["request_id"], r["prompt"], float(r["arrival_ms"]), r.get("size_class"),
                                 int(r["mock_actual_tokens"]) if r.get("mock_actual_tokens") is not None else -1,
                                 float(r["mock_predicted_tokens"]) if r.get("mock_predicted_tokens") is not None else -1.0,
                                 seed=int(r["seed"]), category=r.get("category"))
                 for r in self.requests]
        return Workload(self.name, f"manifest {self.manifest_id}", int(self.arrival.get("seed", 0)), items,
                        dict(self.arrival), synthetic=self.synthetic)


def manifest_from_workload(wl: Workload) -> Manifest:
    """Synthetic manifest from a named workload (keeps its mock lengths for the mock backend)."""
    reqs = [{"request_id": w.request_id, "prompt": w.prompt, "arrival_ms": w.arrival_ms, "seed": w.seed,
             "category": None, "size_class": w.size_class, "label_target_tokens": None,
             "mock_actual_tokens": w.mock_actual_tokens, "mock_predicted_tokens": w.mock_predicted_tokens}
            for w in wl.requests]
    return Manifest(wl.name, reqs, True, {"process": "workload", "seed": wl.seed, **wl.params},
                    {"type": "synthetic_workload", "workload": wl.name})


def manifest_from_prompt_rows(rows: list[dict], name: str, mean_interarrival_ms: float, seed: int,
                              source: dict | None = None) -> Manifest:
    """Real manifest from prompt rows ({"prompt", optional "prompt_id", "category",
    "label_target_tokens"}), with seeded Poisson arrivals."""
    ids = [r.get("prompt_id") for r in rows]
    use_ids = all(ids) and len(set(ids)) == len(ids)
    wl = workload_from_prompts([r["prompt"] for r in rows], name=name, mean_interarrival_ms=mean_interarrival_ms,
                               seed=seed, ids=ids if use_ids else None)
    reqs = [{"request_id": w.request_id, "prompt": w.prompt, "arrival_ms": w.arrival_ms, "seed": w.seed,
             "category": r.get("category"), "size_class": None, "label_target_tokens": r.get("label_target_tokens"),
             "mock_actual_tokens": None, "mock_predicted_tokens": None}
            for w, r in zip(wl.requests, rows)]
    return Manifest(name, reqs, False, {"process": "poisson", "mean_interarrival_ms": mean_interarrival_ms,
                                        "seed": seed}, source or {})
