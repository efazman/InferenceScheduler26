"""Tests for censored-run detection, the 2048 extension pass and final-label assembly.

No GPU and no server: a scripted backend returns exact, chosen token counts so every precedence
branch and audit field can be asserted rather than inferred.
"""

import json

import pytest

from datagen import config
from datagen.assemble_final_labels import (assemble, build_final_label, percentile_contributors,
                                           target_is_censored)
from datagen.backends import BackendError, GenerationResult
from datagen.extend_censored import (ConfigMismatch, extend_censored, find_censored_runs,
                                     is_censored)
from datagen.generate_labels import run_label_generation
from datagen.jsonl import iter_jsonl, write_jsonl
from ml.data import load_jsonl, split_dataset

NO_SLEEP = dict(sleep=lambda s: None, log=lambda *a: None)
BASE_CAP = 100
EXT_CAP = 200


class ScriptedBackend:
    """Returns exactly the token count the test asks for, clipped to `cap` like a real server.

    name is "openai-compat" so records are not flagged is_mock and can reach a label set.
    """

    name = "openai-compat"

    def __init__(self, lengths, cap, fail_keys=(), transient_keys=()):
        self.lengths = lengths            # (prompt, seed) -> uncapped "true" length
        self.cap = cap
        self.fail_keys = set(fail_keys)        # always fail
        self.transient_keys = dict.fromkeys(transient_keys, 1)  # fail once, then succeed
        self.calls = []

    def check(self):
        return {"backend": self.name}

    def generate(self, prompt, seed):
        key = (prompt, seed)
        self.calls.append(key)
        if key in self.fail_keys:
            raise BackendError("scripted permanent failure", retryable=False)
        if self.transient_keys.get(key):
            self.transient_keys[key] -= 1
            raise BackendError("scripted transient failure", retryable=True)
        true_len = self.lengths.get(key, 10)
        tokens = min(true_len, self.cap)
        return GenerationResult(text="x" * tokens, output_tokens=tokens, latency_ms=1.0,
                                finish_reason="length" if tokens >= self.cap else "stop",
                                prompt_tokens=7)


def make_subset(tmp_path, n=4):
    recs = [{"prompt_id": f"p{i:02d}", "prompt": f"prompt {i} ümlaut", "category": "coding_debugging",
             "category_source": "test", "source": "test"} for i in range(n)]
    path = tmp_path / "subset.jsonl"
    write_jsonl(recs, path)
    return recs, path


def base_run(tmp_path, lengths, n=4, cap=BASE_CAP):
    """Produce a base label run whose per-(prompt,seed) lengths are exactly as scripted."""
    recs, subset = make_subset(tmp_path, n)
    keyed = {(r["prompt"], seed): lengths.get((r["prompt_id"], seed), 10)
             for r in recs for seed in config.GENERATION_SEEDS}
    backend = ScriptedBackend(keyed, cap)
    out = tmp_path / "base"
    run_label_generation(recs, backend, out, **NO_SLEEP)
    return recs, subset, out


# --------------------------------------------------------------------- detection

def test_is_censored_uses_finish_reason_then_falls_back_to_count():
    assert is_censored({"finish_reason": "length", "output_tokens": 50}, 100)
    assert not is_censored({"finish_reason": "stop", "output_tokens": 100}, 100)
    # finish_reason absent -> trust the count only when it actually reaches the cap
    assert is_censored({"output_tokens": 100}, 100)
    assert is_censored({"output_tokens": 101}, 100)
    assert not is_censored({"output_tokens": 99}, 100)
    assert not is_censored({"output_tokens": 99})          # no cap, no evidence
    assert not is_censored({"finish_reason": None, "output_tokens": None}, 100)


def test_find_censored_selects_only_capped_runs_and_preserves_identity(tmp_path):
    lengths = {("p00", 42): 500, ("p00", 43): 30, ("p01", 44): 400}  # 2 capped, rest short
    recs, subset, out = base_run(tmp_path, lengths)
    censored = find_censored_runs(out, BASE_CAP)
    assert {(r["prompt_id"], r["seed"]) for r in censored} == {("p00", 42), ("p01", 44)}
    # identity and the cap value itself are preserved from the base measurement
    assert all(r["output_tokens"] == BASE_CAP and r["finish_reason"] == "length" for r in censored)
    assert all(r["seed"] in config.GENERATION_SEEDS for r in censored)
    # deterministic order
    assert [(r["prompt_id"], r["seed"]) for r in censored] == sorted(
        (r["prompt_id"], r["seed"]) for r in censored)


def test_find_censored_deduplicates_repeated_runs(tmp_path):
    _, _, out = base_run(tmp_path, {("p00", 42): 500})
    runs_path = out / config.RUNS_FILE
    original = list(iter_jsonl(runs_path))
    write_jsonl(original + original, runs_path)  # simulate a resume that re-logged runs
    censored = find_censored_runs(out, BASE_CAP)
    assert len(censored) == 1 and censored[0]["prompt_id"] == "p00"


# --------------------------------------------------------------------- extension pass

def test_extension_reruns_same_prompt_and_seed_only(tmp_path):
    lengths = {("p00", 42): 500, ("p01", 44): 150}
    recs, subset, out = base_run(tmp_path, lengths)
    censored = find_censored_runs(out, BASE_CAP)
    prompts = {r["prompt_id"]: r for r in iter_jsonl(subset)}
    keyed = {(r["prompt"], s): lengths.get((r["prompt_id"], s), 10)
             for r in recs for s in config.GENERATION_SEEDS}
    backend = ScriptedBackend(keyed, EXT_CAP)
    stats = extend_censored(censored, prompts, backend, tmp_path / "ext", EXT_CAP, out, **NO_SLEEP)

    assert stats["attempted"] == 2 and stats["extended_ok"] == 2
    # exactly the capped pairs were re-run, with the SAME prompt text and SAME seed
    assert sorted(backend.calls) == sorted([(prompts["p00"]["prompt"], 42),
                                            (prompts["p01"]["prompt"], 44)])
    ext = {(r["prompt_id"], r["seed"]): r for r in iter_jsonl(tmp_path / "ext" / config.EXTENDED_RUNS_FILE)}
    assert set(ext) == {("p00", 42), ("p01", 44)}
    # p01 resolves (150 < 200); p00 is still censored (500 -> clipped to 200)
    assert ext[("p01", 44)]["extended_output_tokens"] == 150
    assert ext[("p01", 44)]["still_censored"] is False
    assert ext[("p00", 42)]["extended_output_tokens"] == EXT_CAP
    assert ext[("p00", 42)]["still_censored"] is True
    assert stats["resolved"] == 1 and stats["still_censored"] == 1
    # required audit fields on every record
    for r in ext.values():
        for field in ("prompt_id", "seed", "run_index", "original_output_tokens",
                      "original_finish_reason", "extended_output_tokens", "extended_finish_reason",
                      "latency_ms", "model", "quantization", "runtime", "runtime_version",
                      "temperature", "top_p", "system_prompt_sha256", "base_labels_dir"):
            assert field in r, field
        assert r["original_output_tokens"] == BASE_CAP
        assert r["original_finish_reason"] == "length"
        assert r["extended_max_new_tokens"] == EXT_CAP


def test_extension_is_idempotent_and_resumable(tmp_path):
    lengths = {("p00", 42): 500, ("p01", 44): 150}
    recs, subset, out = base_run(tmp_path, lengths)
    censored = find_censored_runs(out, BASE_CAP)
    prompts = {r["prompt_id"]: r for r in iter_jsonl(subset)}
    keyed = {(r["prompt"], s): lengths.get((r["prompt_id"], s), 10)
             for r in recs for s in config.GENERATION_SEEDS}
    ext_dir = tmp_path / "ext"

    b1 = ScriptedBackend(keyed, EXT_CAP)
    extend_censored(censored, prompts, b1, ext_dir, EXT_CAP, out, limit=1, **NO_SLEEP)
    assert len(b1.calls) == 1
    first = (ext_dir / config.EXTENDED_RUNS_FILE).read_bytes()

    b2 = ScriptedBackend(keyed, EXT_CAP)           # resume: does the remaining one only
    s2 = extend_censored(censored, prompts, b2, ext_dir, EXT_CAP, out, **NO_SLEEP)
    assert len(b2.calls) == 1 and s2["already_extended"] == 1
    assert (ext_dir / config.EXTENDED_RUNS_FILE).read_bytes().startswith(first)  # append-only

    b3 = ScriptedBackend(keyed, EXT_CAP)           # fully done: generates nothing
    s3 = extend_censored(censored, prompts, b3, ext_dir, EXT_CAP, out, **NO_SLEEP)
    assert b3.calls == [] and s3["attempted"] == 0

    pairs = [(r["prompt_id"], r["seed"]) for r in iter_jsonl(ext_dir / config.EXTENDED_RUNS_FILE)]
    assert len(pairs) == len(set(pairs)) == 2        # no duplicate extended runs


def test_extension_repairs_crash_tail(tmp_path):
    lengths = {("p00", 42): 500, ("p01", 44): 150}
    recs, subset, out = base_run(tmp_path, lengths)
    censored = find_censored_runs(out, BASE_CAP)
    prompts = {r["prompt_id"]: r for r in iter_jsonl(subset)}
    keyed = {(r["prompt"], s): lengths.get((r["prompt_id"], s), 10)
             for r in recs for s in config.GENERATION_SEEDS}
    ext_dir = tmp_path / "ext"
    extend_censored(censored, prompts, ScriptedBackend(keyed, EXT_CAP), ext_dir, EXT_CAP, out,
                    limit=1, **NO_SLEEP)
    with (ext_dir / config.EXTENDED_RUNS_FILE).open("a", encoding="utf-8") as f:
        f.write('{"prompt_id": "p01", "seed": 44, "extended_out')   # crash mid-write
    s = extend_censored(censored, prompts, ScriptedBackend(keyed, EXT_CAP), ext_dir, EXT_CAP, out,
                        **NO_SLEEP)
    assert s["extended_ok"] == 1
    recs_out = list(iter_jsonl(ext_dir / config.EXTENDED_RUNS_FILE))   # all lines parse
    assert len(recs_out) == 2


def test_extension_refuses_setting_change_and_base_dir_overwrite(tmp_path):
    lengths = {("p00", 42): 500}
    recs, subset, out = base_run(tmp_path, lengths)
    censored = find_censored_runs(out, BASE_CAP)
    prompts = {r["prompt_id"]: r for r in iter_jsonl(subset)}
    keyed = {(r["prompt"], s): 500 for r in recs for s in config.GENERATION_SEEDS}
    ext_dir = tmp_path / "ext"
    extend_censored(censored, prompts, ScriptedBackend(keyed, EXT_CAP), ext_dir, EXT_CAP, out, **NO_SLEEP)
    with pytest.raises(ConfigMismatch):   # different extended cap into the same dir
        extend_censored(censored, prompts, ScriptedBackend(keyed, 999), ext_dir, 999, out, **NO_SLEEP)
    with pytest.raises(ValueError):       # never write into the base directory
        extend_censored(censored, prompts, ScriptedBackend(keyed, EXT_CAP), out, EXT_CAP, out, **NO_SLEEP)


def test_extension_logs_failures_and_retries(tmp_path):
    lengths = {("p00", 42): 500, ("p01", 44): 500}
    recs, subset, out = base_run(tmp_path, lengths)
    censored = find_censored_runs(out, BASE_CAP)
    prompts = {r["prompt_id"]: r for r in iter_jsonl(subset)}
    keyed = {(r["prompt"], s): 500 for r in recs for s in config.GENERATION_SEEDS}
    p0, p1 = prompts["p00"]["prompt"], prompts["p01"]["prompt"]
    backend = ScriptedBackend(keyed, EXT_CAP, fail_keys={(p0, 42)}, transient_keys={(p1, 44)})
    ext_dir = tmp_path / "ext"
    s = extend_censored(censored, prompts, backend, ext_dir, EXT_CAP, out, **NO_SLEEP)
    assert s["failed"] == 1 and s["extended_ok"] == 1           # transient one recovered
    fails = list(iter_jsonl(ext_dir / config.EXTENSION_FAILURES_FILE))
    assert any(f["final"] and f["prompt_id"] == "p00" for f in fails)
    assert any(not f["final"] and f["prompt_id"] == "p01" for f in fails)
    # the permanently failed pair is retried on the next pass, not silently dropped
    s2 = extend_censored(censored, prompts, ScriptedBackend(keyed, EXT_CAP), ext_dir, EXT_CAP, out,
                         **NO_SLEEP)
    assert s2["attempted"] == 1


def test_extension_rejects_prompt_id_missing_from_subset(tmp_path):
    _, subset, out = base_run(tmp_path, {("p00", 42): 500})
    censored = find_censored_runs(out, BASE_CAP)
    with pytest.raises(ValueError, match="not in the subset"):
        extend_censored(censored, {}, ScriptedBackend({}, EXT_CAP), tmp_path / "ext", EXT_CAP, out,
                        **NO_SLEEP)


# --------------------------------------------------------------------- p90 censoring logic

def test_percentile_contributors_for_four_runs():
    # p90 of 4 values sits at sorted index 2.7 -> blends indices 2 and 3
    assert percentile_contributors(4, 90, "linear") == [2, 3]
    assert percentile_contributors(4, 100, "linear") == [3]
    assert percentile_contributors(4, 0, "linear") == [0]
    assert percentile_contributors(4, 90, "nearest") is None     # unmodelled -> caller is conservative


def test_target_censored_only_when_p90_depends_on_a_censored_value():
    # censored run is the LARGEST -> p90 is tainted
    assert target_is_censored([10, 20, 30, 200], [False, False, False, True])
    # censored run is the SMALLEST -> p90 (blend of 30 and 200) cannot depend on it
    assert not target_is_censored([10, 20, 30, 200], [True, False, False, False])
    # second-largest is censored and index 2 contributes -> tainted
    assert target_is_censored([10, 20, 30, 200], [False, False, True, False])
    assert not target_is_censored([10, 20, 30, 40], [False] * 4)
    # tie handling is conservative: a censored run sharing a contributing value taints it
    assert target_is_censored([10, 20, 200, 200], [False, False, True, False])


# --------------------------------------------------------------------- final assembly

def test_final_label_precedence_extended_beats_capped_base():
    label = {
        "prompt_id": "p00", "prompt": "q", "category": "coding_debugging",
        "target_percentile": 90, "percentile_method": "linear",
        config.TARGET_FIELD: 100.0,
        "runs": [
            {"seed": 42, "run_index": 0, "output_tokens": 100, "finish_reason": "length"},   # capped, extended
            {"seed": 43, "run_index": 1, "output_tokens": 40, "finish_reason": "stop"},      # exact
            {"seed": 44, "run_index": 2, "output_tokens": 100, "finish_reason": "length"},   # capped, no ext
            {"seed": 45, "run_index": 3, "output_tokens": 20, "finish_reason": "stop"},      # exact
        ],
    }
    extended = {("p00", 42): {"prompt_id": "p00", "seed": 42, "extended_output_tokens": 150,
                              "extended_finish_reason": "stop", "still_censored": False,
                              "latency_ms": 5.0}}
    out = build_final_label(label, extended, BASE_CAP, EXT_CAP)
    by_seed = {r["seed"]: r for r in out["runs"]}
    assert by_seed[42]["effective_output_tokens"] == 150 and by_seed[42]["length_source"] == "extended"
    assert by_seed[42]["is_censored"] is False
    assert by_seed[43]["effective_output_tokens"] == 40 and by_seed[43]["length_source"] == "base"
    assert by_seed[44]["length_source"] == "base_censored_unextended"
    assert by_seed[44]["effective_output_tokens"] == 100 and by_seed[44]["is_censored"] is True

    # p90 recomputed from effective lengths [150, 40, 100, 20] -> sorted [20,40,100,150]
    assert out[config.TARGET_FIELD] == pytest.approx(100 + 0.7 * (150 - 100))
    assert out[config.TARGET_FIELD] != out["base_target_p90_output_tokens"]
    assert out["n_base_truncated_runs"] == 2
    assert out["n_extended_runs"] == 1
    assert out["n_still_censored_runs"] == 1
    assert out["n_unextended_censored_runs"] == 1
    # the unextended capped run (100) sits at sorted index 2, which p90 reads -> tainted
    assert out["has_censored_target"] is True


def test_final_label_still_censored_at_extended_cap_is_preserved():
    label = {"prompt_id": "p0", "prompt": "q", "category": "c",
             "target_percentile": 90, "percentile_method": "linear", config.TARGET_FIELD: 100.0,
             "runs": [{"seed": s, "run_index": i, "output_tokens": 100, "finish_reason": "length"}
                      for i, s in enumerate([42, 43, 44, 45])]}
    extended = {("p0", s): {"prompt_id": "p0", "seed": s, "extended_output_tokens": EXT_CAP,
                            "extended_finish_reason": "length", "still_censored": True,
                            "latency_ms": 1.0} for s in [42, 43, 44, 45]}
    out = build_final_label(label, extended, BASE_CAP, EXT_CAP)
    assert out[config.TARGET_FIELD] == EXT_CAP          # recorded, not invented
    assert out["n_still_censored_runs"] == 4
    assert out["n_extended_runs"] == 4
    assert out["has_censored_target"] is True
    assert all(r["is_censored"] and r["length_source"] == "extended" for r in out["runs"])


def test_final_label_clean_run_has_no_censoring_flags():
    label = {"prompt_id": "p0", "prompt": "q", "category": "c",
             "target_percentile": 90, "percentile_method": "linear", config.TARGET_FIELD: 37.0,
             "runs": [{"seed": s, "run_index": i, "output_tokens": v, "finish_reason": "stop"}
                      for i, (s, v) in enumerate(zip([42, 43, 44, 45], [10, 20, 30, 40]))]}
    out = build_final_label(label, {}, BASE_CAP, EXT_CAP)
    assert out["n_base_truncated_runs"] == 0 and out["n_still_censored_runs"] == 0
    assert out["has_censored_target"] is False
    assert out[config.TARGET_FIELD] == pytest.approx(30 + 0.7 * 10)
    assert out[config.TARGET_FIELD] == out["base_target_p90_output_tokens"]


def test_assemble_end_to_end_and_loads_in_training_pipeline(tmp_path):
    lengths = {("p00", 42): 500, ("p01", 44): 150, ("p02", 43): 300}
    recs, subset, out = base_run(tmp_path, lengths, n=4)
    censored = find_censored_runs(out, BASE_CAP)
    prompts = {r["prompt_id"]: r for r in iter_jsonl(subset)}
    keyed = {(r["prompt"], s): lengths.get((r["prompt_id"], s), 10)
             for r in recs for s in config.GENERATION_SEEDS}
    ext_dir = tmp_path / "ext"
    extend_censored(censored, prompts, ScriptedBackend(keyed, EXT_CAP), ext_dir, EXT_CAP, out, **NO_SLEEP)

    final_dir = tmp_path / "final"
    summary = assemble(out, ext_dir, final_dir, BASE_CAP)
    assert summary["prompts"] == 4 and summary["runs"] == 16
    assert summary["base_truncated_runs"] == 3
    assert summary["extended_runs_applied"] == 3
    assert summary["censored_runs_without_extension"] == 0
    # p00/42 (true 500) and p02/43 (true 300) both exceed the 200 extended cap;
    # p01/44 (true 150) resolves.
    assert summary["still_censored_runs"] == 2
    assert summary["extended_cap"] == EXT_CAP and summary["base_cap"] == BASE_CAP
    assert summary["prompts_whose_target_changed_vs_base"] >= 1

    # base files untouched
    assert len(list(iter_jsonl(out / config.LABELS_FILE))) == 4
    assert all(l["max_new_tokens"] == config.MAX_NEW_TOKENS
               for l in iter_jsonl(out / config.LABELS_FILE))

    # the real training loader accepts the assembled file (non-ASCII prompt text included)
    path = final_dir / config.FINAL_LABELS_FILE
    ex = load_jsonl(path, config.TARGET_FIELD)
    assert len(ex) == 4
    assert all(e["target_output_tokens"] > 0 and e["prompt"] and e["id"] for e in ex)
    assert any("ü" in e["prompt"] for e in ex)
    tr, va, te = split_dataset(ex, 0.70, 0.15, 42, "stratified")
    assert len(tr) + len(va) + len(te) == 4
    assert json.loads((final_dir / "assembly.summary.json").read_text(encoding="utf-8"))["prompts"] == 4


def test_assemble_without_extension_reports_censoring_honestly(tmp_path):
    recs, subset, out = base_run(tmp_path, {("p00", 42): 500})
    summary = assemble(out, None, tmp_path / "final", BASE_CAP)
    assert summary["base_truncated_runs"] == 1
    assert summary["extended_runs_applied"] == 0
    assert summary["censored_runs_without_extension"] == 1
    assert summary["still_censored_runs"] == 1
    assert summary["prompts_with_censored_target"] == 1
    assert summary["prompts_whose_target_changed_vs_base"] == 0   # nothing was improved
    assert summary["extended_cap"] is None


def test_assemble_refuses_mock_sources(tmp_path):
    from datagen.backends import MockBackend
    recs, subset = make_subset(tmp_path, 2)
    out = tmp_path / "base_mock"
    run_label_generation(recs, MockBackend(BASE_CAP), out, **NO_SLEEP)
    with pytest.raises(ValueError, match="MOCK"):
        assemble(out, None, tmp_path / "final", BASE_CAP)


def test_assemble_refuses_writing_into_source_dirs(tmp_path):
    recs, subset, out = base_run(tmp_path, {("p00", 42): 500})
    with pytest.raises(ValueError):
        assemble(out, None, out, BASE_CAP)
