# Output-length predictor (prototype)

DistilBERT → masked mean pooling → MLP head → 20 quantile-bin logits → softmax →
`expected_output_tokens = Σ p_i · center_i`, with `uncertainty` = entropy of the bin
distribution (in nats, max ln 20 ≈ 3.0).

> **The dataset in `data/synthetic_prompts.jsonl` is synthetic.** Its labels were invented to
> exercise the code path. None of the metrics it produces mean anything.

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python -m ml.data        # (re)write the synthetic dataset (train also does this if it's missing)
python -m ml.train       # train + baseline + val/test eval -> artifacts/distilbert_length_predictor/
python -m ml.evaluate    # reload saved artifacts and evaluate on the test split
python -m pytest         # sanity checks
```
Training flags: `--epochs --batch-size --lr --loss-lambda --pooling {mean,cls} --device --data --output-dir`.
Every other knob is in `ml/config.py`.

## API
```python
from ml.predictor import LengthPredictor
predictor = LengthPredictor.load("artifacts/distilbert_length_predictor")
predictor.predict("How do I reset my VPN password?")
# {"expected_output_tokens": float, "bin_probabilities": [20 floats], "uncertainty": float}
```

## Design notes
- Bin boundaries, bin centers (the median training target per bin) and the MSE scale come from
  the **training split only**.
- Loss is `λ·CE_soft + (1-λ)·MSE` with λ = 0.95. The MSE term is computed on
  `tokens / mse_scale`, where mse_scale defaults to the std of the training targets. Raw
  token-space MSE is around 1e4 and would swamp CE no matter what λ is.
- Soft labels follow `p(j) ∝ exp(-|j-i| / soft_label_scale)`, and scale 1.0 gives the spec'd `exp(-|j-i|)`.

## Real labels (Llama 3.1 8B, p90 target)
Real labels are produced by `datagen/` (see `datagen/README.md`). The handoff is already wired up:

```bash
python -m ml.train --real      # reads data/labels/llama31_8b_q4km/labels.jsonl
python -m ml.evaluate --real   # artifacts in artifacts/distilbert_length_predictor_llama31/
```
`Config.for_real_data()` sets these:

| Setting | Synthetic (default) | Real (`--real`) |
| --- | --- | --- |
| `data_path` | `data/synthetic_prompts.jsonl` | `data/labels/llama31_8b_q4km/labels.jsonl` |
| `target_field` | `target_output_tokens` | `target_p90_output_tokens` |
| `split_strategy` | `random` | `stratified` (by `category`, 70/15/15 within each) |
| `prompt_token_counter` (baseline) | `distilbert` | `llamacpp:<BACKEND_URL>` (the real Llama tokenizer, via llama-server `/tokenize`) |

Leakage guards: bins and the MSE scale come from the training split only. Exact duplicates are
removed during preprocessing. `assert_disjoint_splits` fails the run if the same prompt
(case- and whitespace-normalized) lands in two splits. `load_jsonl` refuses mock labels.

If llama-server isn't running at training time, use `--prompt-token-counter hf:meta-llama/Llama-3.1-8B-Instruct`
(gated, tokenizer files only) or fall back with `--prompt-token-counter distilbert`.

The synthetic path stays until real labels exist. To find the remaining spots, run
`grep -rn "REAL-DATA\|SYNTHETIC" ml/`.
