"""Window-level Transformer policy for batched offloading inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class WindowedPolicyConfig:
    """Configuration for the windowed offloading policy model."""

    feature_dim: int = 24
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    num_actions: int = 4
    dropout: float = 0.0


class WindowedTransformerOffloadingPolicy(nn.Module):
    """Predict offloading actions for a batch of task scheduling windows.

    Input shape: `[batch_size, seq_len, feature_dim]`.
    Output shape: `[batch_size, seq_len, num_actions]`.
    """

    def __init__(self, config: WindowedPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or WindowedPolicyConfig()
        self.input_projection = nn.Linear(self.config.feature_dim, self.config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config.num_layers,
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, self.config.num_actions),
        )

    def forward(self, task_window: torch.Tensor) -> torch.Tensor:
        """Return per-task action scores for each scheduling window."""
        hidden = self.input_projection(task_window)
        encoded = self.encoder(hidden)
        return self.action_head(encoded)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
