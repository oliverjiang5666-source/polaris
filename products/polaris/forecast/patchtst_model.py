"""
PatchTST-EPF V3b/V5: Electricity Price Forecasting with PatchTST backbone.

V5 changes (from V3b):
  - Added linear skip connection: lag_96 → output directly (El Mahtout & Ziel 2025 insight)
  - Gated residual: output = NN_output + gate * linear(lag_96) + hour_bias + pos_bias
  - Controlled by config.use_skip_connection (default True for V5, False for V3b backward compat)

V3b architecture (unchanged):
  Path A: price_seq (B, 672, 8) → PatchEmbed → TransformerEncoder → MeanPool → (embed_dim)
  Path B: exo_features (B, 31+) → MLP → (embed_dim)
  Path C: price_lag_96 (B, 96) → Linear → (embed_dim)
  Concat A+B+C + target_hour_bias → MLP → 96 price predictions
"""
from __future__ import annotations

import torch
import torch.nn as nn

from forecast.transformer_config import TransformerConfig


class PatchEmbedding(nn.Module):
    """Convert time series into patches and project to embedding space."""

    def __init__(self, patch_size: int, n_channels: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * n_channels, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        P = self.patch_size
        n_patches = L // P
        x = x[:, :n_patches * P, :].reshape(B, n_patches, P * C)
        return self.dropout(self.proj(x))


class PatchTST_EPF(nn.Module):
    """
    PatchTST for Electricity Price Forecasting — V3b (simplified).

    Key improvements over V2 TransformerEPF:
      1. PatchTST backbone: 672→42 patches, 256x less attention compute
      2. Mean pooling over patches (no last-token bottleneck)
      3. All 31 features in Path B (vs 17 in V2)
      4. Price lag-96 as explicit Path C input
      5. Target-hour positional bias in output head
    """

    def __init__(self, config: TransformerConfig | None = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config

        n_seq = config.n_price_features  # 8
        patch_size = getattr(config, 'patch_size', 16)
        embed_dim = config.embed_dim
        n_patches = config.window_size // patch_size  # 672/16=42

        # ---- Path A: PatchTST backbone ----
        self.patch_embed = PatchEmbedding(patch_size, n_seq, embed_dim, config.dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.pool_norm = nn.LayerNorm(embed_dim)

        # ---- Path B: Exogenous features (all 31 dims) ----
        self.exo_mlp = nn.Sequential(
            nn.Linear(config.n_exo_features, embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # ---- Path C: Price lag-96 ----
        self.lag96_proj = nn.Sequential(
            nn.Linear(96, embed_dim),
            nn.GELU(),
        )

        # ---- Target-hour positional bias (96 output positions) ----
        # Learnable bias per output position, conditioned on hour-of-day
        self.output_pos_bias = nn.Parameter(torch.zeros(96))
        self.hour_bias = nn.Embedding(24, 4)  # 24 hours → 4 quarter-hour biases

        # ---- Output head: direct 96-value output ----
        fusion_dim = 3 * embed_dim
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, 96),
        )

        # ---- V5: Linear skip connection (El Mahtout & Ziel 2025) ----
        # Direct linear pathway: lag_96 → output, bypassing the NN bottleneck
        # Captures dominant autoregressive structure; NN only learns residual
        self.use_skip = getattr(config, 'use_skip_connection', False)
        if self.use_skip:
            self.skip_linear = nn.Linear(96, 96)
            self.skip_gate = nn.Parameter(torch.tensor(0.3))  # learnable blend ratio

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.zeros_(m.weight)  # start with zero bias

    def forward(
        self,
        price_seq: torch.Tensor,      # (B, 672, 8) — already normalized by dataset
        exo_features: torch.Tensor,    # (B, 31)
        price_lag_96: torch.Tensor,    # (B, 96)
        target_hours: torch.Tensor,    # (B, 24) — hour indices for each target hour
    ) -> torch.Tensor:
        """Returns: (B, 96) — predicted prices in normalized space."""
        B = price_seq.shape[0]

        # Path A: PatchTST
        x = self.patch_embed(price_seq)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.pool_norm(x)
        h_price = x.mean(dim=1)  # (B, embed_dim) — mean pooling

        # Path B: exogenous
        h_exo = self.exo_mlp(exo_features)

        # Path C: lag-96
        h_lag = self.lag96_proj(price_lag_96)

        # Fusion + direct 96-value output
        combined = torch.cat([h_price, h_exo, h_lag], dim=-1)
        output = self.output_head(combined)  # (B, 96)

        # Add target-hour positional bias
        # target_hours: (B, 24) → expand to (B, 96) via hour_bias
        hour_biases = self.hour_bias(target_hours)  # (B, 24, 4)
        hour_biases = hour_biases.reshape(B, 96)    # (B, 96)
        output = output + hour_biases + self.output_pos_bias

        # V5: Add linear skip connection from lag_96 directly to output
        if self.use_skip:
            skip_out = self.skip_linear(price_lag_96)
            gate = torch.sigmoid(self.skip_gate)
            output = (1 - gate) * output + gate * skip_out

        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
