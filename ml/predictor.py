"""Public prediction API. The rest of the project should only need this module.

    from ml.predictor import LengthPredictor
    predictor = LengthPredictor.load("artifacts/distilbert_length_predictor")
    predictor.predict("How do I reset my VPN password?")
    # -> {"expected_output_tokens": float, "bin_probabilities": [20 floats], "uncertainty": float}

Artifact layout (written by ``save``):
    encoder/      DistilBERT weights + config (HF save_pretrained)
    tokenizer/    tokenizer files
    head.pt       prediction-head state dict
    bins.json     bin boundaries, bin centers, MSE scale
    config.json   full Config used for training
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ml.config import Config, resolve_device
from ml.model import LengthPredictorModel, entropy


class LengthPredictor:
    def __init__(self, model: LengthPredictorModel, tokenizer, config: Config,
                 boundaries: np.ndarray, centers: np.ndarray, mse_scale: float, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.boundaries = np.asarray(boundaries, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.mse_scale = float(mse_scale)
        self.device = device or resolve_device(config.device)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_batch(self, prompts: list[str]) -> list[dict]:
        self.model.eval()
        enc = self.tokenizer(prompts, padding=True, truncation=True,
                             max_length=self.config.max_length, return_tensors="pt").to(self.device)
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        probs = F.softmax(logits.float(), dim=-1)
        expected = (probs @ self.model.bin_centers).cpu().tolist()
        unc = entropy(probs).cpu().tolist()
        probs = probs.cpu().tolist()
        return [{"expected_output_tokens": float(e), "bin_probabilities": p, "uncertainty": float(u)}
                for e, p, u in zip(expected, probs, unc)]

    def predict(self, prompt: str) -> dict:
        return self.predict_batch([prompt])[0]

    # ------------------------------------------------------------------ io

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.encoder.save_pretrained(out / "encoder")
        self.tokenizer.save_pretrained(out / "tokenizer")
        torch.save(self.model.head.state_dict(), out / "head.pt")
        (out / "bins.json").write_text(json.dumps({
            "n_bins": len(self.centers),
            "boundaries": self.boundaries.tolist(),
            "centers": self.centers.tolist(),
            "mse_scale": self.mse_scale,
            "note": "Derived from the TRAINING split only.",
        }, indent=2))
        self.config.save(out / "config.json")

    @classmethod
    def load(cls, out_dir: str | Path, device: str | None = None) -> "LengthPredictor":
        out = Path(out_dir)
        config = Config.load(out / "config.json")
        bins = json.loads((out / "bins.json").read_text())
        model = LengthPredictorModel(
            backbone=str(out / "encoder"), n_bins=bins["n_bins"], bin_centers=bins["centers"],
            pooling=config.pooling, head_hidden_dim=config.head_hidden_dim, dropout=config.dropout,
        )
        model.head.load_state_dict(torch.load(out / "head.pt", map_location="cpu"))
        tokenizer = AutoTokenizer.from_pretrained(out / "tokenizer")
        return cls(model, tokenizer, config, bins["boundaries"], bins["centers"], bins["mse_scale"],
                   device=resolve_device(device or config.device))
