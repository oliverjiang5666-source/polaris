"""
Transformer Encoder for Electricity Price Forecasting.

Two-path architecture following arXiv:2403.16108:
  Path A: price history sequence → Transformer encoder
  Path B: exogenous variables → embedding MLP
  Concatenate → output MLP → 96 price predictions

Adapted for Chinese 15-minute electricity markets.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from forecast.transformer_config import TransformerConfig


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1)]


class TransformerEPF(nn.Module):
    """
    Transformer Electricity Price Forecaster.

    Input:
        price_seq: (batch, window_size, n_price_features)  e.g. (B, 672, 3)
        exo_features: (batch, n_exo_features)              e.g. (B, 17)

    Output:
        (batch, output_steps)  e.g. (B, 96) — normalized price predictions
    """

    def __init__(self, config: TransformerConfig | None = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config

        # ---- Path A: Price history through Transformer ----
        self.price_proj = nn.Linear(config.n_price_features, config.embed_dim)
        self.pos_encoder = SinusoidalPE(config.window_size, config.embed_dim)
        self.input_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )
        self.price_norm = nn.LayerNorm(config.embed_dim)

        # ---- Path B: Exogenous variables through MLP ----
        self.exo_mlp = nn.Sequential(
            nn.Linear(config.n_exo_features, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
        )

        # ---- Output head ----
        self.output_head = nn.Sequential(
            nn.Linear(2 * config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.output_steps),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        price_seq: torch.Tensor,
        exo_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            price_seq:    (B, W, C) where W=window_size, C=n_price_features
            exo_features: (B, E) where E=n_exo_features

        Returns:
            (B, 96) normalized price predictions
        """
        # Path A: price sequence → Transformer
        h = self.price_proj(price_seq)          # (B, W, embed_dim)
        h = self.pos_encoder(h)                 # (B, W, embed_dim)
        h = self.input_dropout(h)
        h = self.transformer(h)                 # (B, W, embed_dim)
        h = self.price_norm(h)
        h_price = h[:, -1, :]                   # last token: (B, embed_dim)

        # Path B: exogenous → MLP
        h_exo = self.exo_mlp(exo_features)      # (B, embed_dim)

        # Fusion + output
        combined = torch.cat([h_price, h_exo], dim=-1)  # (B, 2*embed_dim)
        return self.output_head(combined)                # (B, 96)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
