"""
PyTorch Dataset for Transformer electricity price forecasting.

Produces (price_window, exo_features, target_prices, norm_stats) tuples
using a sliding window over sequential time-series data.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from forecast.transformer_config import TransformerConfig


class EPFDataset(Dataset):
    """
    Sliding-window dataset for Transformer EPF.

    Each sample:
        price_window: (window_size, 3)  — [rt_price, da_price, spread], normalized
        exo_features: (n_exo,)          — exogenous features at prediction point
        target:       (96,)             — next 96 steps of rt_price, normalized
        norm_stats:   (2,)              — [mean, std] for denormalization
    """

    def __init__(
        self,
        rt_prices: np.ndarray,
        da_prices: np.ndarray,
        exo_features: np.ndarray,
        config: TransformerConfig | None = None,
        augment: bool = False,
    ):
        """
        Args:
            rt_prices:    (N,) real-time prices
            da_prices:    (N,) day-ahead prices
            exo_features: (N, n_exo) exogenous features matrix
            config:       model configuration
            augment:      enable data augmentation (training only)
        """
        self.config = config or TransformerConfig()
        self.augment = augment

        self.rt_prices = rt_prices.astype(np.float32)
        self.da_prices = da_prices.astype(np.float32)
        self.spreads = (da_prices - rt_prices).astype(np.float32)
        self.exo = exo_features.astype(np.float32)

        W = self.config.window_size  # 672
        T = self.config.output_steps  # 96

        # Valid start indices: need W steps before and T steps after
        self.n_total = len(rt_prices)
        self.valid_start = W  # first valid prediction point
        self.valid_end = self.n_total - T  # last valid prediction point

        if self.valid_end <= self.valid_start:
            raise ValueError(
                f"Not enough data: need at least {W + T} = {W + T} steps, "
                f"got {self.n_total}"
            )

    def __len__(self) -> int:
        return self.valid_end - self.valid_start

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        W = self.config.window_size
        T = self.config.output_steps
        norm_w = self.config.norm_window  # 96

        # Prediction point index
        idx = self.valid_start + i

        # Optional jitter for augmentation
        jitter = 0
        if self.augment and self.config.jitter_steps > 0:
            jitter = np.random.randint(
                -self.config.jitter_steps, self.config.jitter_steps + 1
            )
            idx = np.clip(idx + jitter, self.valid_start, self.valid_end - 1)

        # ---- Build price window: [idx-W : idx] ----
        win_start = idx - W
        rt_win = self.rt_prices[win_start:idx].copy()
        da_win = self.da_prices[win_start:idx].copy()
        sp_win = self.spreads[win_start:idx].copy()

        # ---- Compute normalization stats from last 24h ----
        norm_slice = rt_win[-norm_w:]
        mean = norm_slice.mean()
        std = norm_slice.std()
        if std < 1e-6:
            std = 1.0  # avoid division by zero

        # ---- Normalize price window ----
        rt_norm = (rt_win - mean) / std
        da_norm = (da_win - mean) / std
        sp_norm = (sp_win - mean) / std

        # Optional noise augmentation on price inputs
        if self.augment and self.config.noise_std_ratio > 0:
            noise_std = self.config.noise_std_ratio * std
            rt_norm += np.random.normal(0, noise_std / std, size=rt_norm.shape)

        # Stack into (W, 3)
        price_window = np.stack([rt_norm, da_norm, sp_norm], axis=-1)

        # ---- Exogenous features at prediction point ----
        exo = self.exo[idx].copy()

        # ---- Target: next 96 steps, normalized ----
        target = self.rt_prices[idx : idx + T].copy()
        target_norm = (target - mean) / std

        # ---- Norm stats for denormalization ----
        norm_stats = np.array([mean, std], dtype=np.float32)

        return (
            torch.from_numpy(price_window),
            torch.from_numpy(exo),
            torch.from_numpy(target_norm),
            torch.from_numpy(norm_stats),
        )


def build_datasets(
    df_train,
    df_val,
    config: TransformerConfig | None = None,
    exo_cols: list[str] | None = None,
) -> tuple[EPFDataset, EPFDataset]:
    """
    Build train and validation datasets from DataFrames.

    Args:
        df_train: training DataFrame with rt_price, da_price, and exo columns
        df_val:   validation DataFrame
        config:   TransformerConfig
        exo_cols: list of exogenous feature column names

    Returns:
        (train_dataset, val_dataset)
    """
    if config is None:
        config = TransformerConfig()

    if exo_cols is None:
        # Default exogenous features (existing 17 from features.py)
        exo_cols = [
            "hour_sin", "hour_cos",
            "weekday_sin", "weekday_cos",
            "month_sin", "month_cos",
            "load_norm", "load_change",
            "renewable_penetration",
            "wind_ratio", "solar_ratio",
            "net_load_norm",
            "tie_line_norm",
            "temperature_norm",
            "rt_price_ma_ratio",
            "rt_price_std_96",
            "da_price_ma_ratio",
        ]

    def _extract(df):
        rt = df["rt_price"].fillna(0).values
        da = df["da_price"].fillna(0).values if "da_price" in df.columns else np.zeros_like(rt)
        exo = df[exo_cols].fillna(0).values
        return rt, da, exo

    rt_train, da_train, exo_train = _extract(df_train)
    rt_val, da_val, exo_val = _extract(df_val)

    train_ds = EPFDataset(rt_train, da_train, exo_train, config, augment=True)
    val_ds = EPFDataset(rt_val, da_val, exo_val, config, augment=False)

    return train_ds, val_ds
