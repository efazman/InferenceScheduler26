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

## Moving to real labels
Run `grep -rn "REAL-DATA\|SYNTHETIC" ml/` to find every place that needs to change.
