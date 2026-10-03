"""Sanity checks for the length-predictor pipeline.  Run: python -m pytest tests -q"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from ml.baseline import PromptLengthBaseline
from ml.config import Config
from ml.data import assign_bins, compute_quantile_bins, generate_synthetic_dataset, soft_label, split_dataset
from ml.evaluate import compute_metrics
from ml.model import LengthPredictorModel, joint_loss
from ml.predictor import LengthPredictor

N_BINS = 20
PROMPTS = ["What is Redis?", "Summarize the Q3 outage incident report in detail, including root cause."]


# --------------------------------------------------------------------------- pure functions

@pytest.mark.parametrize("i", range(N_BINS))
def test_soft_label_sums_to_one_and_peaks_at_true_bin(i):
    p = soft_label(i, N_BINS)
    assert len(p) == N_BINS
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == i
    assert p[i] / p[i - 1 if i else i + 1] == pytest.approx(math.e)


def test_quantile_bins_monotonic_even_with_ties():
    rng = np.random.default_rng(0)
    for targets in (rng.integers(20, 500, size=83), np.array([50] * 40 + [60] * 40 + [400] * 3)):
        boundaries, centers = compute_quantile_bins(targets, N_BINS)
        assert len(boundaries) == N_BINS + 1 and len(centers) == N_BINS
        assert np.all(np.diff(boundaries) > 0)
        assert np.all(np.isfinite(centers))
        assert np.all(np.diff(centers) >= 0)


def test_assign_bins_clamps_out_of_range():
    boundaries, _ = compute_quantile_bins(np.arange(100, 200), N_BINS)
    assert assign_bins([0, 10_000], boundaries).tolist() == [0, N_BINS - 1]


def test_split_sizes_and_disjoint():
    data = generate_synthetic_dataset(seed=42)
    tr, va, te = split_dataset(data, 0.7, 0.15, seed=42)
    ids = [ex["id"] for ex in tr + va + te]
    assert len(ids) == len(set(ids)) == len(data)
    assert abs(len(tr) / len(data) - 0.7) < 0.02


def test_metrics_known_values():
    m = compute_metrics(predicted=[10, 100, 40], actual=[100, 100, 100], severe_threshold=0.5)
    assert m["mae"] == pytest.approx(50.0)
    assert m["underprediction_rate"] == pytest.approx(2 / 3)
    assert m["severe_underprediction_rate"] == pytest.approx(2 / 3)
    m = compute_metrics([10, 100, 40], [100, 100, 100], severe_threshold=0.3)
    assert m["severe_underprediction_rate"] == pytest.approx(1 / 3)


def test_joint_loss_lambda_one_is_pure_ce():
    logits = torch.randn(4, N_BINS)
    soft = torch.tensor(np.stack([soft_label(i, N_BINS) for i in (0, 5, 10, 19)]), dtype=torch.float32)
    expected, target = torch.tensor([10.0, 20, 30, 40]), torch.tensor([15.0, 25, 35, 45])
    total, ce, mse = joint_loss(logits, soft, expected, target, 1.0, 10.0)
    assert torch.isfinite(total) and total == pytest.approx(ce.item())
    assert mse.item() == pytest.approx(0.25)


# --------------------------------------------------------------------------- model / predictor

@pytest.fixture(scope="module")
def untrained_predictor():
    """Randomly-initialized DistilBERT (no weight download needed, only config + tokenizer)."""
    config = Config(device="cpu")
    torch.manual_seed(0)
    boundaries, centers = compute_quantile_bins(np.random.default_rng(0).integers(20, 500, 83), N_BINS)
    model = LengthPredictorModel(config.backbone, N_BINS, centers, config.pooling,
                                 config.head_hidden_dim, config.dropout, pretrained=False)
    tok = AutoTokenizer.from_pretrained(config.backbone)
    return LengthPredictor(model, tok, config, boundaries, centers, mse_scale=100.0,
                           device=torch.device("cpu"))


def assert_valid_prediction(pred, centers):
    assert set(pred) == {"expected_output_tokens", "bin_probabilities", "uncertainty"}
    assert len(pred["bin_probabilities"]) == N_BINS
    assert sum(pred["bin_probabilities"]) == pytest.approx(1.0, abs=1e-4)
    assert all(p >= 0 for p in pred["bin_probabilities"])
    e = pred["expected_output_tokens"]
    assert math.isfinite(e) and min(centers) - 1e-3 <= e <= max(centers) + 1e-3
    assert 0.0 <= pred["uncertainty"] <= math.log(N_BINS) + 1e-4


def test_predict_returns_valid_distribution(untrained_predictor):
    for prompt in PROMPTS:
        assert_valid_prediction(untrained_predictor.predict(prompt), untrained_predictor.centers)


def test_save_and_reload_roundtrip(untrained_predictor, tmp_path):
    untrained_predictor.save(tmp_path)
    for name in ("encoder", "tokenizer", "head.pt", "bins.json", "config.json"):
        assert (tmp_path / name).exists()
    reloaded = LengthPredictor.load(tmp_path, device="cpu")
    np.testing.assert_allclose(reloaded.boundaries, untrained_predictor.boundaries)
    for before, after in zip(untrained_predictor.predict_batch(PROMPTS), reloaded.predict_batch(PROMPTS)):
        assert_valid_prediction(after, reloaded.centers)
        assert after["expected_output_tokens"] == pytest.approx(before["expected_output_tokens"], rel=1e-4)
        np.testing.assert_allclose(after["bin_probabilities"], before["bin_probabilities"], atol=1e-5)


def test_baseline_roundtrip(untrained_predictor, tmp_path):
    tok = untrained_predictor.tokenizer
    prompts = ["short", "a somewhat longer prompt here", "an even longer prompt with many more words in it"]
    base = PromptLengthBaseline(tok).fit(prompts, [20, 60, 120])
    base.save(tmp_path / "baseline.json")
    again = PromptLengthBaseline.load(tmp_path / "baseline.json", tok)
    np.testing.assert_allclose(again.predict(prompts), base.predict(prompts))
    assert base.coef > 0


ARTIFACTS = Path(Config().output_dir)


@pytest.mark.skipif(not (ARTIFACTS / "head.pt").exists(), reason="run `python -m ml.train` first")
def test_trained_artifacts_reload_and_predict():
    predictor = LengthPredictor.load(ARTIFACTS, device="cpu")
    assert (ARTIFACTS / "baseline.json").exists() and (ARTIFACTS / "eval_results.json").exists()
    assert_valid_prediction(predictor.predict(PROMPTS[0]), predictor.centers)
