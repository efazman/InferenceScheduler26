"""Tests for LMSYS preprocessing, subset selection and resumable label generation (no GPU needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from datagen import config
from datagen.backends import BackendError, MockBackend, OpenAICompatBackend
from datagen.categories import CATEGORIES, assign_category
from datagen.generate_labels import ConfigMismatch, percentile_target, run_label_generation
from datagen.jsonl import iter_jsonl, repair_jsonl_tail, write_jsonl
from datagen.preprocess_lmsys import preprocess, prompt_id_for
from datagen.select_subset import compute_quotas, select_subset
from ml.data import LengthDataset, assert_disjoint_splits, compute_quantile_bins, load_jsonl, split_dataset
from ml.prompt_tokens import make_token_counter

NO_SLEEP = dict(sleep=lambda s: None, log=lambda *a: None)


# --------------------------------------------------------------------------- fixtures

def lmsys_row(cid, prompt, language="English", extra_turns=0, flagged=False, first_role="user"):
    conv = [{"role": first_role, "content": prompt}, {"role": "assistant", "content": "ok"}]
    for _ in range(extra_turns):
        conv += [{"role": "user", "content": "more"}, {"role": "assistant", "content": "ok"}]
    return {"conversation_id": cid, "model": "vicuna-13b", "conversation": conv, "turn": 1 + extra_turns,
            "language": language, "openai_moderation": [{"flagged": flagged}], "redacted": False}


RAW_ROWS = [
    lmsys_row("c0", "What is the capital of France?"),
    lmsys_row("c1", "What is the capital of France?"),            # exact duplicate
    lmsys_row("c2", "  Summarize this text: the cat sat.  "),     # kept verbatim (whitespace too)
    lmsys_row("c3", "Quelle est la capitale ?", language="French"),
    lmsys_row("c4", "Explain recursion", extra_turns=1),            # multi-turn
    lmsys_row("c5", "   "),                                          # empty
    lmsys_row("c6", "something hateful", flagged=True),
    lmsys_row("c7", "x" * (config.MAX_PROMPT_CHARS + 1)),            # too long
    {"conversation_id": "c8", "conversation": None, "language": "English"},  # malformed
    lmsys_row("c9", "Hello", first_role="assistant"),
    lmsys_row("c10", "Write a python function to sort a list"),
]


def make_prompts(n):
    texts = ["What is {i}?", "Explain why {i} matters", "Write a python function for task {i}",
             "Summarize document {i}", "Write a poem about {i}", "My laptop {i} is not working, help me"]
    return [{"prompt_id": f"p{i:03d}", "prompt": texts[i % 6].format(i=i), "category": None,
             "source": "test"} for i in range(n)]


class Interrupting(MockBackend):
    """Mock that simulates a crash (KeyboardInterrupt) after `stop_after` successful calls."""

    def __init__(self, stop_after, **kw):
        super().__init__(config.MAX_NEW_TOKENS, **kw)
        self.stop_after = stop_after

    def generate(self, prompt, seed):
        if self.calls >= self.stop_after:
            raise KeyboardInterrupt
        return super().generate(prompt, seed)


# --------------------------------------------------------------------------- preprocessing

def test_preprocess_filters_dedupes_and_preserves_text(tmp_path):
    write_jsonl(RAW_ROWS, tmp_path / "raw.jsonl")
    summary = preprocess(tmp_path / "raw.jsonl", tmp_path / "clean.jsonl")
    out = list(iter_jsonl(tmp_path / "clean.jsonl"))
    prompts = [r["prompt"] for r in out]
    assert prompts == ["What is the capital of France?", "  Summarize this text: the cat sat.  ",
                       "Write a python function to sort a list"]
    for key in ("duplicate", "language", "multi_turn", "empty", "moderation", "too_long", "malformed"):
        assert summary[f"rejected_{key}"] >= 1, key
    assert summary["kept"] == 3 and summary["rows_read"] == len(RAW_ROWS)
    r = out[0]
    assert r["prompt_id"] == prompt_id_for(r["prompt"]) and r["category"] is None
    assert r["source"] == "lmsys-chat-1m" and r["source_metadata"]["conversation_id"] == "c0"


def test_preprocess_is_deterministic_and_reads_parquet(tmp_path):
    pa, pq = pytest.importorskip("pyarrow"), pytest.importorskip("pyarrow.parquet")
    rows = [r for r in RAW_ROWS if r.get("conversation")]  # parquet needs a consistent schema
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    pq.write_table(pa.Table.from_pylist(rows[:5]), shard_dir / "train-00000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[5:]), shard_dir / "train-00001.parquet")
    preprocess(shard_dir, tmp_path / "a.jsonl")
    preprocess(shard_dir, tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert len(list(iter_jsonl(tmp_path / "a.jsonl"))) == 3


def test_keep_flagged_option(tmp_path):
    write_jsonl([lmsys_row("c6", "something", flagged=True)], tmp_path / "raw.jsonl")
    assert preprocess(tmp_path / "raw.jsonl", tmp_path / "o.jsonl", exclude_flagged=False)["kept"] == 1


# --------------------------------------------------------------------------- categories + subset

@pytest.mark.parametrize("prompt,expected", [
    ("What is the capital of France?", "knowledge_factual"),
    ("Explain why the sky is blue", "explanation_reasoning"),
    ("Write a Python function that reverses a string", "coding_debugging"),
    ("I get a traceback in my script", "coding_debugging"),
    ("Summarize the following article", "summarization_transformation"),
    ("Rewrite this paragraph to be more formal", "summarization_transformation"),
    ("Write a short story about a dragon", "writing_structured"),
    ("My wifi keeps dropping, how can I fix it?", "troubleshooting_advice"),
])
def test_category_examples(prompt, expected):
    assert assign_category(prompt) == expected


def test_category_always_valid():
    for p in ["", "hi", "```x```", "?" * 50, "WRITE", "why", "日本語のテキスト"] + [r["prompt"] for r in make_prompts(30)]:
        assert assign_category(p) in CATEGORIES


def test_quotas_sum_exactly():
    q = compute_quotas(500, config.CATEGORY_PROPORTIONS)
    assert sum(q.values()) == 500 and q["knowledge_factual"] == 100 and q["coding_debugging"] == 75
    assert sum(compute_quotas(37, config.CATEGORY_PROPORTIONS).values()) == 37


def test_subset_deterministic_and_balanced():
    pool = make_prompts(1200)
    a, sa = select_subset(pool, n=60, seed=42)
    b, _ = select_subset(pool, n=60, seed=42)
    c, _ = select_subset(pool, n=60, seed=7)
    assert [r["prompt_id"] for r in a] == [r["prompt_id"] for r in b]
    assert [r["prompt_id"] for r in a] != [r["prompt_id"] for r in c]
    assert len(a) == 60 and len({r["prompt_id"] for r in a}) == 60
    assert sa["selected"] == sa["quotas"]  # pool has plenty of every category
    assert all(r["category"] in CATEGORIES and r["category_source"] for r in a)


def test_subset_fills_shortfall_from_other_categories():
    pool = [p for p in make_prompts(600) if assign_category(p["prompt"]) != "coding_debugging"]
    subset, summary = select_subset(pool, n=60, seed=42)
    assert len(subset) == 60 and summary["selected"]["coding_debugging"] == 0


# --------------------------------------------------------------------------- p90 + mock

def test_p90():
    assert percentile_target([100, 200, 300, 400]) == pytest.approx(370.0)
    assert percentile_target([400, 100, 300, 200]) == pytest.approx(370.0)
    assert percentile_target([50, 50, 50, 50]) == 50.0


def test_mock_backend_deterministic():
    a, b = MockBackend(512), MockBackend(512)
    outs = [(a.generate("hello", s).output_tokens, b.generate("hello", s).output_tokens) for s in range(42, 46)]
    assert all(x == y for x, y in outs)
    assert len({x for x, _ in outs}) > 1  # different seeds -> different lengths
    r = a.generate("hello", 42)
    assert "MOCK" in r.text and 0 < r.output_tokens <= 512 and r.finish_reason in ("stop", "length")


# --------------------------------------------------------------------------- label generation

def test_label_schema(tmp_path):
    prompts = make_prompts(3)
    stats = run_label_generation(prompts, MockBackend(512), tmp_path, **NO_SLEEP)
    assert stats["completed_now"] == 3
    labels = list(iter_jsonl(tmp_path / config.LABELS_FILE))
    lab = labels[0]
    for key in ("prompt_id", "prompt", "category", "runs", config.TARGET_FIELD, "model", "quantization",
                "temperature", "top_p", "max_new_tokens", "n_truncated_runs", "is_mock", "warning"):
        assert key in lab, key
    assert lab["model"] == config.MODEL_NAME and lab["quantization"] == config.QUANTIZATION
    assert lab["is_mock"] is True and "MOCK" in lab["warning"]
    assert [r["seed"] for r in lab["runs"]] == config.GENERATION_SEEDS
    assert lab[config.TARGET_FIELD] == pytest.approx(percentile_target([r["output_tokens"] for r in lab["runs"]]))
    assert len(list(iter_jsonl(tmp_path / config.RUNS_FILE))) == 3 * config.NUM_GENERATIONS
    assert json.loads((tmp_path / config.GENERATION_CONFIG_FILE).read_text())["backend"] == "mock"


def test_skip_already_completed(tmp_path):
    prompts = make_prompts(4)
    run_label_generation(prompts, MockBackend(512), tmp_path, **NO_SLEEP)
    again = MockBackend(512)
    stats = run_label_generation(prompts, again, tmp_path, **NO_SLEEP)
    assert again.calls == 0 and stats["already_complete"] == 4 and stats["completed_now"] == 0
    assert len(list(iter_jsonl(tmp_path / config.LABELS_FILE))) == 4


def test_resume_after_crash_reuses_finished_generations(tmp_path):
    prompts = make_prompts(5)
    with pytest.raises(KeyboardInterrupt):
        run_label_generation(prompts, Interrupting(stop_after=10), tmp_path, **NO_SLEEP)
    assert len(list(iter_jsonl(tmp_path / config.LABELS_FILE))) == 2  # 10 calls = 2 full prompts + 2 runs
    assert len(list(iter_jsonl(tmp_path / config.RUNS_FILE))) == 10
    resumed = MockBackend(512)
    stats = run_label_generation(prompts, resumed, tmp_path, **NO_SLEEP)
    assert resumed.calls == 5 * 4 - 10  # only the missing generations are run
    assert stats["generations_reused"] == 2 and stats["complete_total"] == 5
    labels = list(iter_jsonl(tmp_path / config.LABELS_FILE))
    assert sorted(r["prompt_id"] for r in labels) == [p["prompt_id"] for p in prompts]
    # resumed labels are identical to an uninterrupted run
    fresh_dir = tmp_path / "fresh"
    run_label_generation(prompts, MockBackend(512), fresh_dir, **NO_SLEEP)
    fresh = {r["prompt_id"]: r[config.TARGET_FIELD] for r in iter_jsonl(fresh_dir / config.LABELS_FILE)}
    assert {r["prompt_id"]: r[config.TARGET_FIELD] for r in labels} == fresh


def test_partial_final_line_is_repaired(tmp_path):
    prompts = make_prompts(3)
    run_label_generation(prompts[:2], MockBackend(512), tmp_path, **NO_SLEEP)
    with (tmp_path / config.LABELS_FILE).open("a") as f:
        f.write('{"prompt_id": "p002", "prom')  # crash mid-write
    stats = run_label_generation(prompts, MockBackend(512), tmp_path, **NO_SLEEP)
    assert stats["completed_now"] == 1
    assert [r["prompt_id"] for r in iter_jsonl(tmp_path / config.LABELS_FILE)] == ["p000", "p001", "p002"]


def test_repair_jsonl_tail_cases(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n')
    assert repair_jsonl_tail(p) == 0
    p.write_text('{"a": 1}\n{"b": ')
    assert repair_jsonl_tail(p) > 0 and p.read_text() == '{"a": 1}\n'
    p.write_text('{"a": 1}\n{bad json}\n')
    repair_jsonl_tail(p)
    assert p.read_text() == '{"a": 1}\n'


def test_retries_then_success_and_failure_log(tmp_path):
    backend = MockBackend(512, transient_failures=2, fail_rate=1.0)
    stats = run_label_generation(make_prompts(2), backend, tmp_path, max_retries=3, **NO_SLEEP)
    assert stats["completed_now"] == 2
    fails = list(iter_jsonl(tmp_path / config.FAILURES_FILE))
    assert len(fails) == 2 * 4 * 2 and not any(f["final"] for f in fails)
    assert {f["attempt"] for f in fails} == {1, 2}
    assert all(r["attempt"] == 3 for r in iter_jsonl(tmp_path / config.RUNS_FILE))


def test_permanent_failure_logged_and_retried_on_next_run(tmp_path):
    prompts = make_prompts(3)
    bad = MockBackend(512, permanent_failures={prompts[1]["prompt"]})
    stats = run_label_generation(prompts, bad, tmp_path, max_retries=2, **NO_SLEEP)
    assert stats["completed_now"] == 2 and stats["failed_prompts"] == 1
    fails = list(iter_jsonl(tmp_path / config.FAILURES_FILE))
    assert len(fails) == 3 and fails[-1]["final"] and fails[-1]["prompt_id"] == "p001"
    stats = run_label_generation(prompts, MockBackend(512), tmp_path, **NO_SLEEP)
    assert stats["completed_now"] == 1 and stats["complete_total"] == 3


def test_non_retryable_error_is_not_retried(tmp_path):
    class Rejecting(MockBackend):
        def generate(self, prompt, seed):
            self.calls += 1
            raise BackendError("HTTP 400: prompt exceeds context", retryable=False)

    backend = Rejecting(512)
    run_label_generation(make_prompts(1), backend, tmp_path, max_retries=5, **NO_SLEEP)
    assert backend.calls == 1
    assert list(iter_jsonl(tmp_path / config.FAILURES_FILE))[0]["final"] is True


def test_resume_with_different_settings_is_refused(tmp_path):
    run_label_generation(make_prompts(1), MockBackend(512), tmp_path, **NO_SLEEP)
    with pytest.raises(ConfigMismatch):
        run_label_generation(make_prompts(1), MockBackend(512), tmp_path, seeds=[1, 2, 3, 4], **NO_SLEEP)


def test_mock_cannot_write_to_real_label_dir():
    existed = config.LABELS_DIR_REAL.exists()
    with pytest.raises(ValueError, match="MOCK"):
        run_label_generation(make_prompts(1), MockBackend(512), config.LABELS_DIR_REAL, **NO_SLEEP)
    assert config.LABELS_DIR_REAL.exists() == existed  # refused before touching the filesystem


# --------------------------------------------------------------------------- ML handoff

@pytest.fixture
def mock_labels(tmp_path):
    prompts, _ = select_subset(make_prompts(600), n=120, seed=42)
    run_label_generation(prompts, MockBackend(512), tmp_path, **NO_SLEEP)
    return tmp_path / config.LABELS_FILE


def test_mock_labels_load_into_ml_pipeline(mock_labels):
    with pytest.raises(ValueError, match="MOCK"):
        load_jsonl(mock_labels, config.TARGET_FIELD)  # refused by default
    examples = load_jsonl(mock_labels, config.TARGET_FIELD, allow_mock=True)
    assert len(examples) == 120
    ex = examples[0]
    assert ex["id"] == ex["prompt_id"] and ex["category"] in CATEGORIES
    assert ex["target_output_tokens"] == ex[config.TARGET_FIELD]
    tr, va, te = split_dataset(examples, 0.7, 0.15, seed=42, strategy="stratified")
    assert len(tr) + len(va) + len(te) == 120
    for split in (tr, va, te):
        assert {e["category"] for e in split} == {e["category"] for e in examples}
    assert_disjoint_splits(tr, va, te)
    boundaries, _ = compute_quantile_bins([e["target_output_tokens"] for e in tr], 20)
    ds = LengthDataset(tr, boundaries)
    assert len(ds) == len(tr) and np.allclose(ds.soft.sum(1), 1.0)


def test_disjoint_guard_catches_leak():
    a = [{"prompt": "Hello  World"}]
    with pytest.raises(ValueError):
        assert_disjoint_splits(a, [{"prompt": "hello world"}], [])


def test_real_config_points_at_real_labels():
    from ml.config import Config

    cfg = Config.for_real_data()
    assert cfg.data_path == str(config.REAL_LABELS_PATH) and cfg.target_field == "target_p90_output_tokens"
    assert cfg.split_strategy == "stratified" and cfg.prompt_token_counter.startswith("llamacpp:")
    assert not cfg.allow_mock_labels


# --------------------------------------------------------------------------- llama.cpp client (fake server)

class FakeLlamaServer(BaseHTTPRequestHandler):
    fail_next = 0
    requests: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"data": [{"id": "fake-llama"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlamaServer.requests.append((self.path, body))
        if self.path == "/tokenize":
            return self._send(200, {"tokens": list(range(len(body["content"].split())))})
        if FakeLlamaServer.fail_next:
            FakeLlamaServer.fail_next -= 1
            return self._send(503, {"error": "busy"})
        user = body["messages"][-1]["content"]
        if user == "too long":
            return self._send(400, {"error": "context exceeded"})
        n = 10 + body["seed"]
        self._send(200, {"choices": [{"message": {"content": "x " * n}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 25, "completion_tokens": n},
                         "timings": {"predicted_n": n, "predicted_ms": 100.0}})


@pytest.fixture
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeLlamaServer.requests, FakeLlamaServer.fail_next = [], 0
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def make_real_backend(url):
    return OpenAICompatBackend(url, config.MODEL_NAME, config.SYSTEM_PROMPT, config.TEMPERATURE,
                               config.TOP_P, config.MAX_NEW_TOKENS, timeout=5)


def test_openai_compat_backend_request_and_parse(fake_server):
    backend = make_real_backend(fake_server)
    backend.check()
    res = backend.generate("hi", seed=42)
    assert res.output_tokens == 52 and res.prompt_tokens == 25 and res.finish_reason == "stop"
    assert res.server_timings["predicted_n"] == 52 and res.latency_ms > 0
    _, body = FakeLlamaServer.requests[-1]
    assert body["seed"] == 42 and body["temperature"] == 0.7 and body["top_p"] == 0.9
    assert body["max_tokens"] == 512 and body["messages"][0] == {"role": "system", "content": config.SYSTEM_PROMPT}


def test_openai_compat_errors(fake_server):
    backend = make_real_backend(fake_server)
    with pytest.raises(BackendError) as e:
        backend.generate("too long", 42)
    assert not e.value.retryable
    with pytest.raises(BackendError) as e:
        make_real_backend("http://127.0.0.1:9").check()
    assert not e.value.retryable


def test_real_backend_label_run_with_server_retry(fake_server, tmp_path):
    FakeLlamaServer.fail_next = 2  # two 503s, then healthy
    stats = run_label_generation(make_prompts(2), make_real_backend(fake_server), tmp_path, **NO_SLEEP)
    assert stats["completed_now"] == 2 and stats["failed_attempts"] == 2
    lab = next(iter_jsonl(tmp_path / config.LABELS_FILE))
    assert lab["is_mock"] is False and "warning" not in lab
    assert [r["output_tokens"] for r in lab["runs"]] == [10 + s for s in config.GENERATION_SEEDS]


def test_llamacpp_token_counter(fake_server):
    count = make_token_counter(f"llamacpp:{fake_server}")
    assert count("one two three") == 3
    path, body = FakeLlamaServer.requests[-1]
    assert path == "/tokenize" and body["add_special"] is False


def test_distilbert_counter_requires_tokenizer():
    with pytest.raises(ValueError):
        make_token_counter("distilbert")
    with pytest.raises(ValueError):
        make_token_counter("nope")
