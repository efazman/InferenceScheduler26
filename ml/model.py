"""DistilBERT encoder + pooling + small MLP head -> logits over output-length bins.

Pooling choice: masked mean pooling over the last hidden state (default). DistilBERT has no
pretrained pooler/NSP head, so its [CLS] vector is not specially trained to summarize the input;
the mean of all non-padding token states is a simple, standard, and more robust sentence
representation. ``pooling="cls"`` is available for comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class LengthPredictorModel(nn.Module):
    def __init__(self, backbone: str, n_bins: int, bin_centers, pooling: str = "mean",
                 head_hidden_dim: int = 256, dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.encoder = (AutoModel.from_pretrained(backbone) if pretrained
                        else AutoModel.from_config(AutoConfig.from_pretrained(backbone)))
        hidden = self.encoder.config.hidden_size
        if head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, n_bins),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_bins))
        self.register_buffer("bin_centers", torch.as_tensor(bin_centers, dtype=torch.float32))

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape [batch, n_bins]."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.pool(hidden, attention_mask))

    def expected_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """sum_i p_i * center_i, shape [batch]."""
        return F.softmax(logits, dim=-1) @ self.bin_centers


def joint_loss(logits: torch.Tensor, soft_targets: torch.Tensor, expected: torch.Tensor,
               target_tokens: torch.Tensor, loss_lambda: float, mse_scale: float):
    """L = lambda * CE_soft + (1 - lambda) * MSE.

    CE_soft is cross-entropy against the soft bin distribution. MSE compares the expected token
    count to the true count, both divided by ``mse_scale`` so the two terms have similar
    magnitudes (raw token-space MSE would be ~1e4 and swamp CE regardless of lambda).
    Returns (total, ce, mse) for logging.
    """
    ce = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    mse = F.mse_loss(expected / mse_scale, target_tokens / mse_scale)
    return loss_lambda * ce + (1.0 - loss_lambda) * mse, ce, mse


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats along the last dim (max = ln(n_bins), about 3.0 for 20 bins)."""
    return -(probs * probs.clamp(min=1e-12).log()).sum(-1)
