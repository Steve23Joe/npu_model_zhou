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


class NpuFriendlyEncoderBlock(nn.Module):
    """Transformer-style encoder block implemented with primitive tensor ops.

    Optional attention masks must be broadcastable to `[batch, heads, seq, seq]`.
    The public policy model does not currently use masks.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attention_scale = self.head_dim**-0.5

        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def _apply_attention_mask(
        self,
        attention_scores: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return attention_scores

        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.to(device=attention_scores.device)
            return attention_scores.masked_fill(
                attention_mask,
                torch.finfo(attention_scores.dtype).min,
            )
        return attention_scores + attention_mask.to(
            device=attention_scores.device,
            dtype=attention_scores.dtype,
        )

    def _self_attention(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self._split_heads(self.q_projection(hidden))
        key = self._split_heads(self.k_projection(hidden))
        value = self._split_heads(self.v_projection(hidden))

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.attention_scale
        attention_scores = self._apply_attention_mask(attention_scores, attention_mask)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        context = torch.matmul(attention_weights, value)
        return self.output_projection(self._merge_heads(context))

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self._self_attention(self.norm_attention(hidden), attention_mask)
        hidden = hidden + self.residual_dropout(attended)
        fed_forward = self.feedforward(self.norm_feedforward(hidden))
        return hidden + self.residual_dropout(fed_forward)


class NpuFriendlyEncoder(nn.Module):
    """Stack of manual encoder blocks for batch-first `[batch, seq, dim]` tensors."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            NpuFriendlyEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return hidden


class WindowedTransformerOffloadingPolicy(nn.Module):
    """Predict offloading actions for a batch of task scheduling windows.

    Input shape: `[batch_size, seq_len, feature_dim]`.
    Output shape: `[batch_size, seq_len, num_actions]`.
    """

    def __init__(self, config: WindowedPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or WindowedPolicyConfig()
        self.input_projection = nn.Linear(self.config.feature_dim, self.config.d_model)
        self.encoder = NpuFriendlyEncoder(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
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
