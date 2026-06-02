"""Lightweight MLP policy model used for ONNX export."""

from __future__ import annotations

import torch
from torch import nn


class PolicyMLP(nn.Module):
    """Small feed-forward policy network for four offloading actions.

    The model consumes the normalized 10-feature offloading state from
    `OffloadingEnv` and returns four action scores.
    """

    def __init__(self, input_dim: int = 10, output_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return action logits or Q-values for a batch of observations."""
        return self.network(observation)

    def describe(self) -> dict[str, int]:
        """Return model shape metadata."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }
