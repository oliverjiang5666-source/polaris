"""
Transformer EPF configuration.

Hyperparameters and per-province configs for the Transformer
electricity price forecasting model.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    """Model architecture hyperparameters."""

    # --- Sequence ---
    window_size: int = 672          # 7 days of 15-min data
    output_steps: int = 96          # predict 1 day (96 x 15min)
    n_price_features: int = 8       # price + fundamental sequence features
    n_exo_features: int = 17        # exogenous features at prediction point (V2: 17)

    # --- Transformer ---
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.15

    # --- PatchTST (V3) ---
    patch_size: int = 16            # 16 steps = 4 hours per patch → 42 patches

    # --- Training ---
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 15              # early stopping patience
    grad_clip: float = 0.3
    scheduler_T0: int = 10          # cosine annealing warm restart period

    # --- Normalization ---
    norm_window: int = 96           # last 24h for adaptive normalization

    # --- Data augmentation ---
    noise_std_ratio: float = 0.02   # Gaussian noise std = ratio * price_std
    jitter_steps: int = 4           # random window shift during training

    # --- Rolling window (for provinces with price drift) ---
    rolling_train_days: int | None = None   # None = use all; 365 for Gansu

    # --- V5: Skip connection + Huber loss ---
    use_skip_connection: bool = False   # V5: linear skip from lag_96 to output
    loss_type: str = "mse"              # "mse", "huber", "mae"
    huber_delta: float = 50.0           # Huber loss delta (in normalized space ~1-2 std)

    # --- Device ---
    device: str = "cuda"            # "cuda" for cloud GPU, "mps" for Mac, "cpu" fallback


# Province-specific overrides
PROVINCE_CONFIGS: dict[str, dict] = {
    "shandong": {},                  # defaults work well
    "shanxi": {},
    "guangdong": {},
    "gansu": {
        "rolling_train_days": 365,   # address price drift
        "lr": 5e-5,                  # more conservative for volatile data
    },
}

# V3 PatchTST config: all 31 features in Path B
PROVINCE_CONFIGS_V3: dict[str, dict] = {
    "shandong": {},
    "shanxi": {},
    "guangdong": {},
    "gansu": {
        "rolling_train_days": 365,
        "lr": 5e-5,
    },
}


def get_config_v3(province: str | None = None) -> TransformerConfig:
    """Get V3 PatchTST config: 31 exo features, patch_size=16."""
    config = TransformerConfig(
        n_exo_features=31,      # all FEATURE_COLS
        patch_size=16,
        patience=20,
        batch_size=128,
    )
    if province and province in PROVINCE_CONFIGS_V3:
        for key, value in PROVINCE_CONFIGS_V3[province].items():
            setattr(config, key, value)
    return config


def get_config_v4(province: str | None = None) -> TransformerConfig:
    """Get V4 PatchTST config: 39 exo features (31 base + 8 causal weather/supply-demand)."""
    config = TransformerConfig(
        n_exo_features=39,      # FEATURE_COLS_V4 (31 + 8 causal)
        patch_size=16,
        patience=20,
        batch_size=128,
    )
    overrides = {
        "shandong": {},
        "shanxi": {},
        "guangdong": {},
        "gansu": {"rolling_train_days": 365, "lr": 5e-5},
    }
    if province and province in overrides:
        for key, value in overrides[province].items():
            setattr(config, key, value)
    return config


def get_config_v5(province: str | None = None) -> TransformerConfig:
    """Get V5 config: V4 features + skip connection + Huber loss."""
    config = TransformerConfig(
        n_exo_features=39,
        patch_size=16,
        patience=20,
        batch_size=128,
        use_skip_connection=True,
        loss_type="huber",
        huber_delta=1.5,         # in normalized space (~1.5 std)
    )
    overrides = {
        "shandong": {},
        "shanxi": {},
        "guangdong": {},
        "gansu": {"rolling_train_days": 365, "lr": 5e-5},
    }
    if province and province in overrides:
        for key, value in overrides[province].items():
            setattr(config, key, value)
    return config


def get_config(province: str | None = None) -> TransformerConfig:
    """Get TransformerConfig with province-specific overrides."""
    config = TransformerConfig()
    if province and province in PROVINCE_CONFIGS:
        for key, value in PROVINCE_CONFIGS[province].items():
            setattr(config, key, value)
    return config
