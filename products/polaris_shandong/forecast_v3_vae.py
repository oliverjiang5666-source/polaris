"""
Polaris 山东 · L2 预测层 v3 - Conditional VAE + 可微场景生成
==============================================================

与现有 L2（`core/regime_classifier.py` + bootstrap）**并行**，不替换。
接口统一: `generate_scenarios(target_day, df, n_scenarios)` → (96, R) 矩阵

设计:
  Encoder   q(z|x, c):  96 点价格 x + context c → latent z ∈ R^latent_dim
  Decoder   p(x|z, c):  z + context c → 96 点价格重建
  Prior     p(z|c):     标准 N(0, I) (独立于 c, 简化)

  Context c: 32 维包括 month sin/cos, weekday sin/cos, season one-hot,
                       is_weekend, is_holiday, 温度预报 mean/std,
                       风速预报 mean/std, 光伏 mean/std,
                       历史 7/14/30 日价格滑动 mean/std, 昨日价均值...

训练:
  L = L_recon (MSE) + β·KL(q||p)   (β-VAE)
  β 从 0 warm-up 到 1.0

生成:
  target_day → context c_target
  z_i ~ N(0, I), i = 1..R
  scenario_i = Decoder(z_i, c_target)
  → (96, R) matrix, 传给 Tensor DP 作为 price scenarios

DFL (Phase 2, 未实现):
  优化 Decoder 直接最小化 Tensor DP(生成场景) 下的 expected regret
  需要可微 DP 或 perturb-and-MAP

Usage:
    from products.polaris_shandong.forecast_v3_vae import VAEScenarioGenerator
    gen = VAEScenarioGenerator()
    gen.fit(df_train, price_col="rt_price")
    scenarios_96_R = gen.generate(target_day_idx=1800, df=df, n_scenarios=500)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from loguru import logger


STEPS_PER_DAY = 96


# ============================================================
# Config
# ============================================================

@dataclass
class VAEConfig:
    latent_dim: int = 16
    context_dim: int = 32
    hidden_dim: int = 128

    # 训练
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 80
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_epochs: int = 20

    # 场景生成
    n_scenarios_default: int = 500

    # Normalization
    price_clip_min: float = -200.0
    price_clip_max: float = 2000.0

    # Device
    device: str = "cpu"


# ============================================================
# Context feature builder
# ============================================================

def build_context(df: pd.DataFrame, target_day_idx: int, context_dim: int = 32) -> np.ndarray:
    """
    为给定 target_day 构造 context 向量 (32 维).

    特征:
      hour sin/cos (取 day start, 通常 0)     - 2
      weekday sin/cos                          - 2
      month sin/cos                            - 2
      season one-hot                           - 4
      is_weekend, is_holiday                   - 2
      温度预报 mean/std (target day 96 点)     - 2
      风速预报 mean/std                        - 2
      光照 mean/std                            - 2
      昨日价均值/std                           - 2
      近 7 日价均值/std                        - 2
      近 14 日价均值/std                       - 2
      近 30 日价均值/std                       - 2
      padding to 32                            - 4
    """
    ctx = np.zeros(context_dim, dtype=np.float32)
    idx = target_day_idx * STEPS_PER_DAY
    if idx >= len(df):
        return ctx

    ts = df.index[idx]

    # 日历
    ctx[0] = 0.0                            # hour sin (day start)
    ctx[1] = 1.0                            # hour cos
    wd = ts.weekday()
    ctx[2] = float(np.sin(2 * np.pi * wd / 7))
    ctx[3] = float(np.cos(2 * np.pi * wd / 7))
    mo = ts.month
    ctx[4] = float(np.sin(2 * np.pi * mo / 12))
    ctx[5] = float(np.cos(2 * np.pi * mo / 12))

    # Season one-hot
    season = (mo - 1) // 3 % 4
    ctx[6 + season] = 1.0

    ctx[10] = float(wd >= 5)
    # 节假日简单占位 (真实可查表)
    ctx[11] = 0.0

    # 天气 (target day)
    day_slice = df.iloc[idx:idx + STEPS_PER_DAY]
    for col, base in [("temperature_2m", 12), ("wind_speed_10m", 14), ("shortwave_radiation", 16)]:
        if col in day_slice.columns:
            v = day_slice[col].fillna(0).values
            ctx[base] = float(v.mean())
            ctx[base + 1] = float(v.std())

    # 历史价 lags
    for lag_days, base in [(1, 18), (7, 20), (14, 22), (30, 24)]:
        lag_start = max(0, idx - lag_days * STEPS_PER_DAY)
        lag_end = idx
        lag_prices = df["rt_price"].iloc[lag_start:lag_end].fillna(0).values
        if len(lag_prices) > 0:
            ctx[base] = float(lag_prices.mean())
            ctx[base + 1] = float(lag_prices.std())

    return ctx


# ============================================================
# Model
# ============================================================

class PriceVAE(nn.Module):
    """
    Conditional VAE:
      Encoder q(z|x, c): concat(x, c) -> hidden -> (μ, log σ²)
      Decoder p(x|z, c): concat(z, c) -> hidden -> 96 点价格
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        cfg = config
        self.cfg = cfg

        # Encoder: (96 + context_dim) -> hidden -> (μ, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(STEPS_PER_DAY + cfg.context_dim, cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        # Decoder: (latent + context_dim) -> hidden -> 96
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.context_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim * 2, STEPS_PER_DAY),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar


# ============================================================
# Scenario Generator
# ============================================================

class VAEScenarioGenerator:
    """
    与 regime-conditioned bootstrap **并行**的 L2 预测层.

    Fit on historical (x, c) pairs where x is 96-point price, c is day context.
    Generate scenarios by sampling z ~ N(0, I) and decoding.

    统一接口匹配现有 regime_classifier:
        gen.fit(df_train, price_col="rt_price")
        scenarios, probs = gen.sample_scenarios(target_day_idx, df, n_scenarios)
    """

    def __init__(self, config: VAEConfig | None = None):
        self.cfg = config or VAEConfig()
        self.model: PriceVAE | None = None

        # Normalization (z-score)
        self.price_mean: float = 0.0
        self.price_std: float = 1.0
        self.context_mean: np.ndarray | None = None
        self.context_std: np.ndarray | None = None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        price_col: str = "rt_price",
        train_days: int | None = None,
    ) -> dict:
        """
        训练 VAE.
        train_days: 最近 N 天作训练 (None = 全部)
        """
        cfg = self.cfg
        device = torch.device(cfg.device)

        # 构建 (x, c) 对
        n_total = len(df) // STEPS_PER_DAY
        prices = df[price_col].ffill().fillna(0).values.astype(np.float32)
        prices = np.clip(prices, cfg.price_clip_min, cfg.price_clip_max)
        prices_days = prices[: n_total * STEPS_PER_DAY].reshape(n_total, STEPS_PER_DAY)

        valid_days_start = 30                          # 需要 30 日 lag
        valid_days_end = n_total - 1                   # 保留 1 天作未来 lag
        if train_days is not None:
            valid_days_start = max(valid_days_start, valid_days_end - train_days)

        xs, cs = [], []
        for d in range(valid_days_start, valid_days_end):
            xs.append(prices_days[d])
            cs.append(build_context(df, d, cfg.context_dim))
        xs = np.asarray(xs, dtype=np.float32)             # (N, 96)
        cs = np.asarray(cs, dtype=np.float32)             # (N, context_dim)

        # Normalization
        self.price_mean = float(xs.mean())
        self.price_std = float(xs.std()) + 1e-6
        x_norm = (xs - self.price_mean) / self.price_std

        self.context_mean = cs.mean(axis=0)
        self.context_std = cs.std(axis=0) + 1e-6
        c_norm = (cs - self.context_mean) / self.context_std

        logger.info(
            f"VAE fit: {len(xs)} days, price mean/std {self.price_mean:.1f}/{self.price_std:.1f}, "
            f"context dim {cs.shape[1]}"
        )

        # DataLoader
        ds = TensorDataset(
            torch.from_numpy(x_norm), torch.from_numpy(c_norm)
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

        # Model
        self.model = PriceVAE(cfg).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        history = {"epoch": [], "recon": [], "kl": [], "total": []}

        for ep in range(cfg.epochs):
            # β warmup
            if ep < cfg.beta_warmup_epochs:
                beta = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * ep / cfg.beta_warmup_epochs
            else:
                beta = cfg.beta_end

            self.model.train()
            recon_losses, kl_losses = [], []
            for xb, cb in loader:
                xb = xb.to(device)
                cb = cb.to(device)

                x_recon, mu, logvar = self.model(xb, cb)
                recon = F.mse_loss(x_recon, xb, reduction="sum") / xb.size(0)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
                loss = recon + beta * kl

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                recon_losses.append(recon.item())
                kl_losses.append(kl.item())

            avg_recon = float(np.mean(recon_losses))
            avg_kl = float(np.mean(kl_losses))
            history["epoch"].append(ep)
            history["recon"].append(avg_recon)
            history["kl"].append(avg_kl)
            history["total"].append(avg_recon + beta * avg_kl)

            if (ep + 1) % 10 == 0 or ep == 0:
                logger.info(
                    f"  Epoch {ep+1}/{cfg.epochs}: recon={avg_recon:.4f}, "
                    f"kl={avg_kl:.4f}, beta={beta:.2f}"
                )

        self.model.eval()
        return history

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    def sample_scenarios(
        self,
        target_day_idx: int,
        df: pd.DataFrame,
        n_scenarios: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        生成 target day 的 K 条价格场景.

        Returns:
            scenarios: (96, K) matrix of prices
            probs:     (96, K) uniform 1/K (VAE 采样是等权)
        """
        assert self.model is not None, "fit() first"
        cfg = self.cfg
        device = torch.device(cfg.device)
        K = n_scenarios or cfg.n_scenarios_default

        # Context
        c = build_context(df, target_day_idx, cfg.context_dim)
        c_norm = (c - self.context_mean) / self.context_std
        c_tensor = torch.from_numpy(c_norm.astype(np.float32)).unsqueeze(0).to(device)
        c_batch = c_tensor.repeat(K, 1)                  # (K, context_dim)

        # Sample z
        with torch.no_grad():
            z = torch.randn(K, cfg.latent_dim, device=device)
            x_recon = self.model.decode(z, c_batch)      # (K, 96)
            x_recon_np = x_recon.cpu().numpy()

        # Denormalize
        scenarios = x_recon_np * self.price_std + self.price_mean
        scenarios = np.clip(scenarios, cfg.price_clip_min, cfg.price_clip_max)

        # Shape to (96, K) for Tensor DP
        scenarios_96_K = scenarios.T.astype(np.float64)
        probs_96_K = np.ones_like(scenarios_96_K) / K

        return scenarios_96_K, probs_96_K


# ============================================================
# Head-to-head eval helper (for A/B vs regime-conditioned)
# ============================================================

def ab_test_vae_vs_regime(
    df: pd.DataFrame,
    regime_scenario_generator_fn,         # callable(day_idx) -> (price_96_R, probs_96_R)
    vae_generator: VAEScenarioGenerator,
    test_days: list[int],
    dp_solve_fn,                          # callable(scenarios, probs, actual_prices) -> revenue
    actual_prices_fn,                      # callable(day_idx) -> 96 点真实价
) -> pd.DataFrame:
    """
    对 test_days 并行跑两个 scenario generator, 每天算 revenue.
    """
    rows = []
    for d in test_days:
        # Regime-conditioned
        scen_r, p_r = regime_scenario_generator_fn(d)
        actual_prices = actual_prices_fn(d)
        rev_r = dp_solve_fn(scen_r, p_r, actual_prices)

        # VAE
        scen_v, p_v = vae_generator.sample_scenarios(d, df, n_scenarios=scen_r.shape[1])
        rev_v = dp_solve_fn(scen_v, p_v, actual_prices)

        rows.append({
            "day_idx": d,
            "rev_regime": rev_r,
            "rev_vae": rev_v,
            "vae_vs_regime": rev_v - rev_r,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    """Smoke test: fit VAE on shandong data + generate scenarios"""
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]

    df = pd.read_parquet(ROOT / "data" / "china" / "processed" / "shandong_oracle.parquet")
    logger.info(f"Loaded shandong: {len(df)} rows, {len(df)//96} days")

    # Fit on historical data (before 2025)
    df_train = df.loc[:"2024-12-31"]
    gen = VAEScenarioGenerator(VAEConfig(epochs=20))   # fast smoke: 20 epochs
    gen.fit(df_train, price_col="rt_price")

    # Generate scenarios for 2025-06-15 (midsummer random day)
    target_date = pd.Timestamp("2025-06-15")
    day_starts = df.index[::96].normalize()
    target_day = int((day_starts >= target_date).argmax())

    scenarios, _ = gen.sample_scenarios(target_day, df, n_scenarios=500)
    actual = df["rt_price"].iloc[target_day * 96:(target_day + 1) * 96].values

    logger.info(
        f"\nTarget day: {df.index[target_day*96].date()}"
    )
    logger.info(
        f"  Actual  mean {actual.mean():.1f}, std {actual.std():.1f}, "
        f"range [{actual.min():.1f}, {actual.max():.1f}]"
    )
    logger.info(
        f"  Scenarios mean {scenarios.mean():.1f}, std {scenarios.std():.1f}, "
        f"range [{scenarios.min():.1f}, {scenarios.max():.1f}]"
    )
    # Coverage: actual 每时段是否落在 scenarios p5-p95 区间
    p5 = np.percentile(scenarios, 5, axis=1)            # (96,)
    p95 = np.percentile(scenarios, 95, axis=1)
    in_band = ((actual >= p5) & (actual <= p95)).mean()
    logger.info(f"  Coverage (actual ∈ scenarios p5-p95 per step): {in_band*100:.1f}%  (ideal 90%)")
