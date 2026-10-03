"""Hyperparameters and paths for the output-length predictor.

Everything tunable lives here so no other module hardcodes assumptions.
The config is serialized to ``config.json`` next to the model artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- data -------------------------------------------------------------
    # Default = the SYNTHETIC pipeline-validation file. Real Llama labels are selected with
    # Config.for_real_data() / `python -m ml.train --real`, which reads REAL_LABELS_PATH.
    # REAL-DATA: once real labels exist, make for_real_data() the default and retire the synthetic path.
    data_path: str = str(PROJECT_ROOT / "data" / "synthetic_prompts.jsonl")
    target_field: str = "target_output_tokens"  # label field in the JSONL; real data: TARGET_FIELD
    split_strategy: str = "random"  # "random" | "stratified" (by "category")
    allow_mock_labels: bool = False  # MockBackend labels are refused unless this is set (tests only)
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # --- binning / labels ------------------------------------------------
    n_bins: int = 20
    # Soft label for bin j given true bin i is proportional to exp(-|j - i| / soft_label_scale).
    # 1.0 reproduces the EGTP-inspired exp(-|j - i|).
    soft_label_scale: float = 1.0

    # --- model -------------------------------------------------------------
    backbone: str = "distilbert-base-uncased"
    pooling: str = "mean"  # "mean" (masked mean pooling) or "cls"
    head_hidden_dim: int = 256  # 0 -> single linear layer
    dropout: float = 0.1
    max_length: int = 128  # REAL-DATA: revisit once real prompt length distribution is known

    # --- loss --------------------------------------------------------------
    # L = loss_lambda * CE_soft + (1 - loss_lambda) * MSE
    loss_lambda: float = 0.95
    # The MSE term is computed on (tokens / mse_scale) so it is commensurate with CE.
    # None -> use the std of the TRAINING-split targets (computed at train time and saved).
    mse_scale: Optional[float] = None

    # --- training ----------------------------------------------------------
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3  # REAL-DATA: increase (and add early stopping on val MAE) for real data
    batch_size: int = 8
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"

    # --- evaluation --------------------------------------------------------
    severe_underprediction_threshold: float = 0.5  # severe if pred < threshold * actual
    latency_warmup: int = 3

    # --- baseline ----------------------------------------------------------
    # How the prompt-length baseline counts prompt tokens:
    #   "distilbert"      the predictor's tokenizer (development fallback, no Llama files needed)
    #   "llamacpp:<url>"  llama-server's /tokenize endpoint (the real Llama 3.1 tokenizer, from the GGUF)
    #   "hf:<name|path>"  a Hugging Face tokenizer, e.g. hf:meta-llama/Llama-3.1-8B-Instruct (gated)
    prompt_token_counter: str = "distilbert"

    # --- output ------------------------------------------------------------
    output_dir: str = str(PROJECT_ROOT / "artifacts" / "distilbert_length_predictor")

    @classmethod
    def for_real_data(cls, **overrides) -> "Config":
        """Settings for training on labels from datagen.generate_labels (Llama 3.1 8B, p90 target)."""
        from datagen import config as gen

        real = dict(
            data_path=str(gen.REAL_LABELS_PATH),
            target_field=gen.TARGET_FIELD,
            split_strategy="stratified",
            prompt_token_counter=f"llamacpp:{gen.BACKEND_URL}",
            output_dir=str(PROJECT_ROOT / "artifacts" / "distilbert_length_predictor_llama31"),
        )
        return cls(**{**real, **overrides})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def resolve_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
