"""
PatchTST-EPF Training + MPC Evaluation (V3).

Fixes over V2:
  1. CAUSAL rolling normalization (no future leakage)
  2. All 31 features in Path B
  3. Price lag-96 as explicit input
  4. Target-hour positional encoding
  5. PatchTST backbone with RevIN
  6. Two-stage output: 24h → 96 fifteen-minute
  7. Mean pooling instead of last-token

Usage:
    PYTHONPATH=. python3 scripts/12_patchtst_train.py --province shandong
    PYTHONPATH=. python3 scripts/12_patchtst_train.py --all
"""
from __future__ import annotations

import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from data.china.features import FEATURE_COLS
from forecast.transformer_config import get_config_v3, TransformerConfig
from forecast.patchtst_model import PatchTST_EPF
from forecast.lgbm_forecaster import LGBMForecaster
from forecast.mpc_controller import (
    MPCController, simulate_mpc, simulate_threshold,
    simulate_oracle_continuous, simulate_oracle_discrete,
)

PROCESSED_DIR = Path("data/china/processed")
MODEL_DIR = Path("models")

# V3: use ALL 31 features for Path B
EXO_COLS = FEATURE_COLS  # 31 features

# 8-channel sequence features (same as V2)
SEQ_COLS = [
    "rt_price", "da_price", "da_rt_spread",
    "load_norm", "renewable_penetration", "net_load_norm",
    "wind_ratio", "solar_ratio",
]


# ============================================================
# V3 Dataset: causal normalization, price_lag_96, target_hours
# ============================================================

class EPFDatasetV3(Dataset):
    """
    PatchTST dataset with:
      - CAUSAL rolling normalization (fixes V2's centered convolution bug)
      - Price lag-96: yesterday's prices at each target quarter-hour
      - Target hour indices for the output positions
    """

    def __init__(self, df, exo_cols, config, augment=False, stride=1):
        self.config = config
        self.augment = augment

        W = config.window_size   # 672
        T = config.output_steps  # 96
        N = len(df)

        # Sequence features (N, 8)
        seq_data = []
        for col in SEQ_COLS:
            if col in df.columns:
                seq_data.append(df[col].fillna(0).values.astype(np.float32))
            else:
                seq_data.append(np.zeros(N, dtype=np.float32))
        self.seq = np.stack(seq_data, axis=-1)

        # Price for target and normalization
        self.rt = df["rt_price"].fillna(0).values.astype(np.float32)

        # Exogenous features (N, 31)
        self.exo = df[exo_cols].fillna(0).values.astype(np.float32)

        # Hour-of-day for each row (precomputed for fast target_hours)
        if hasattr(df.index, 'hour'):
            self.hours = df.index.hour.values.astype(np.int64)
        else:
            self.hours = np.zeros(N, dtype=np.int64)

        # Precompute target_hours for all positions: (N, 24)
        hour_offsets = np.arange(24, dtype=np.int64)
        self.target_hours_all = (self.hours[:, None] + hour_offsets[None, :]) % 24  # (N, 24)

        # Valid indices: need W history + T future + 96 for lag
        min_start = max(W, 96)
        self.indices = np.arange(min_start, N - T, stride)
        logger.info(f"    Dataset: {len(self.indices)} samples (stride={stride}, N={N})")

        # Precompute CAUSAL rolling stats (FIX: no future leakage)
        # Vectorized: use cumsum for O(N) computation, no Python loop
        norm_w = config.norm_window  # 96
        rt_padded = np.concatenate([np.zeros(norm_w - 1, dtype=np.float32), self.rt])
        cumsum = np.cumsum(rt_padded)
        cumsum_sq = np.cumsum(rt_padded ** 2)
        n_vals = np.minimum(np.arange(1, N + 1, dtype=np.float32), norm_w)
        ends = np.arange(norm_w - 1, N + norm_w - 1)
        starts = ends - n_vals.astype(int)
        self.rolling_mean = ((cumsum[ends] - cumsum[starts]) / n_vals).astype(np.float32)
        mean_sq = ((cumsum_sq[ends] - cumsum_sq[starts]) / n_vals).astype(np.float32)
        variance = np.maximum(mean_sq - self.rolling_mean ** 2, 0)
        self.rolling_std = np.sqrt(variance + 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        W = self.config.window_size
        T = self.config.output_steps

        idx = self.indices[i]

        if self.augment and self.config.jitter_steps > 0:
            jitter = np.random.randint(-self.config.jitter_steps, self.config.jitter_steps + 1)
            idx = max(max(W, 96), min(idx + jitter, len(self.rt) - T - 1))

        mean = self.rolling_mean[idx]
        std = max(self.rolling_std[idx], 1.0)

        # === 8-channel sequence window (W, 8) ===
        s = idx - W
        seq_win = self.seq[s:idx].copy()

        # Normalize price channels (0,1,2) with causal adaptive stats
        seq_win[:, 0] = (seq_win[:, 0] - mean) / std
        seq_win[:, 1] = (seq_win[:, 1] - mean) / std
        seq_win[:, 2] = seq_win[:, 2] / std

        if self.augment and self.config.noise_std_ratio > 0:
            noise = np.random.normal(0, self.config.noise_std_ratio, size=W).astype(np.float32)
            seq_win[:, 0] += noise

        # === Exogenous features (31,) ===
        exo = self.exo[idx].copy()

        # === Price lag-96: yesterday's prices at each of the 96 target positions ===
        # Target positions: idx, idx+1, ..., idx+95
        # Lag-96 positions: idx-96, idx-95, ..., idx-1
        lag_start = idx - 96
        if lag_start >= 0:
            price_lag_96 = self.rt[lag_start:lag_start + 96].copy()
            price_lag_96 = (price_lag_96 - mean) / std  # normalize with same stats
        else:
            price_lag_96 = np.zeros(96, dtype=np.float32)

        # === Target hours (24,): precomputed, just lookup ===
        target_hours = self.target_hours_all[idx]

        # === Target (96,): normalized prices ===
        target = (self.rt[idx:idx + T] - mean) / std

        # Norm stats for denormalization
        norm_stats = np.array([mean, std], dtype=np.float32)

        return (
            torch.from_numpy(seq_win),
            torch.from_numpy(exo),
            torch.from_numpy(price_lag_96),
            torch.from_numpy(target_hours),
            torch.from_numpy(target),
            torch.from_numpy(norm_stats),
        )


# ============================================================
# Data loading (same as V2)
# ============================================================

def load_and_split(province, test_days=365, val_days=180):
    path = PROCESSED_DIR / f"{province}_oracle.parquet"
    df = pd.read_parquet(path)
    logger.info(f"Loaded {province}: {len(df):,} rows ({len(df) // 96} days)")

    test_start = len(df) - 96 * test_days
    if test_start < 96 * (val_days + 180):
        test_start = len(df) // 2
        val_days = min(val_days, (test_start // 96) // 3)

    val_start = test_start - 96 * val_days

    df_train = df.iloc[:val_start]
    df_val = df.iloc[val_start:test_start]
    df_test = df.iloc[test_start:]

    logger.info(f"  Train: {len(df_train) // 96}d | Val: {len(df_val) // 96}d | Test: {len(df_test) // 96}d")
    return df_train, df_val, df_test


# ============================================================
# Training
# ============================================================

def train_patchtst(province, df_train, df_val, config):
    logger.info(f"\n--- Training PatchTST-EPF V3 for {province} ---")
    logger.info(f"  Config: embed={config.embed_dim}, heads={config.n_heads}, "
                f"layers={config.n_layers}, ff={config.ff_dim}, patch={config.patch_size}")
    logger.info(f"  Exo features: {config.n_exo_features}, Window: {config.window_size}")

    if config.rolling_train_days is not None:
        max_rows = 96 * config.rolling_train_days
        if len(df_train) > max_rows:
            df_train = df_train.iloc[-max_rows:]
            logger.info(f"  Rolling window: last {config.rolling_train_days} days")

    train_ds = EPFDatasetV3(df_train, EXO_COLS, config, augment=True, stride=1)
    val_ds = EPFDatasetV3(df_val, EXO_COLS, config, augment=False, stride=96)

    logger.info(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = PatchTST_EPF(config).to(device)
    logger.info(f"  Parameters: {model.count_parameters():,}, Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler_T0)
    criterion = nn.MSELoss()

    n_workers = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=(device.type == "cuda"),
                              persistent_workers=(n_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=(device.type == "cuda"),
                            persistent_workers=(n_workers > 0))

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    t0 = time.time()
    for epoch in range(config.max_epochs):
        # Train
        model.train()
        train_losses = []
        for seq, exo, lag96, tgt_hours, target, ns in train_loader:
            seq = seq.to(device)
            exo = exo.to(device)
            lag96 = lag96.to(device)
            tgt_hours = tgt_hours.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(seq, exo, lag96, tgt_hours)
            loss = criterion(pred, target)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_maes = []
        with torch.no_grad():
            for seq, exo, lag96, tgt_hours, target, ns in val_loader:
                seq = seq.to(device)
                exo = exo.to(device)
                lag96 = lag96.to(device)
                tgt_hours = tgt_hours.to(device)
                target = target.to(device)
                ns = ns.to(device)

                pred = model(seq, exo, lag96, tgt_hours)
                mean, std = ns[:, 0:1], ns[:, 1:2]
                pred_real = pred * std + mean
                target_real = target * std + mean
                mae = torch.abs(pred_real - target_real).mean().item()
                val_maes.append(mae)

        avg_train = np.mean(train_losses)
        val_mae = np.mean(val_maes)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            logger.info(f"  Epoch {epoch + 1:3d}: loss={avg_train:.6f}, val_MAE={val_mae:.1f} 元/MWh{marker}")

        if patience_counter >= config.patience:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break

    elapsed = time.time() - t0
    logger.info(f"  Training: {elapsed:.0f}s, Best val MAE: {best_val_mae:.1f} 元/MWh")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    MODEL_DIR.mkdir(exist_ok=True)
    save_path = MODEL_DIR / f"patchtst_{province}.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, save_path)
    logger.info(f"  Saved: {save_path}")

    return model, config


# ============================================================
# PatchTST Forecaster (for MPC integration)
# ============================================================

class PatchTSTForecaster:
    """Drop-in replacement for LGBMForecaster, using PatchTST-EPF."""

    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.model.eval()

        # Set externally
        self._full_prices = None
        self._full_da_prices = None
        self._full_exo = None
        self._full_seq = None
        self._full_hours = None

    def predict(self, features_t, idx, horizon=96):
        W = self.config.window_size
        norm_w = self.config.norm_window

        if self._full_prices is None:
            raise RuntimeError("Set _full_prices before calling predict()")

        # --- Sequence window (W, 8) ---
        win_start = max(0, idx - W + 1)
        win_end = idx + 1

        if self._full_seq is not None:
            seq_win = self._full_seq[win_start:win_end].astype(np.float32)
        else:
            rt_win = self._full_prices[win_start:win_end].astype(np.float32)
            da_win = self._full_da_prices[win_start:win_end].astype(np.float32) if self._full_da_prices is not None else np.zeros_like(rt_win)
            sp_win = da_win - rt_win
            seq_win = np.stack([rt_win, da_win, sp_win], axis=-1)
            # Pad to 8 channels
            if seq_win.shape[1] < 8:
                seq_win = np.pad(seq_win, ((0, 0), (0, 8 - seq_win.shape[1])), mode="constant")

        # Pad if not enough history
        if len(seq_win) < W:
            pad_len = W - len(seq_win)
            seq_win = np.pad(seq_win, ((pad_len, 0), (0, 0)), mode="edge")

        # --- Causal normalization (match training) ---
        rt_col = self._full_prices[max(0, idx + 1 - norm_w):idx + 1].astype(np.float32)
        mean = rt_col.mean()
        std = rt_col.std()
        if std < 1e-6:
            std = 1.0

        seq_norm = seq_win.copy()
        seq_norm[:, 0] = (seq_norm[:, 0] - mean) / std
        seq_norm[:, 1] = (seq_norm[:, 1] - mean) / std
        seq_norm[:, 2] = seq_norm[:, 2] / std

        # --- Exogenous features (31,) ---
        if self._full_exo is not None and idx < len(self._full_exo):
            exo = self._full_exo[idx].astype(np.float32)
        else:
            exo = np.zeros(self.config.n_exo_features, dtype=np.float32)
            n = min(len(features_t), self.config.n_exo_features)
            exo[:n] = features_t[:n]

        # --- Price lag-96 (yesterday's prices, normalized) ---
        lag_start = max(0, idx - 95)
        lag_end = idx + 1
        lag_raw = self._full_prices[lag_start:lag_end].astype(np.float32)
        if len(lag_raw) < 96:
            lag_raw = np.pad(lag_raw, (96 - len(lag_raw), 0), mode="edge")
        price_lag_96 = (lag_raw - mean) / std

        # --- Target hours (24,) ---
        if self._full_hours is not None and idx < len(self._full_hours):
            base_hour = int(self._full_hours[idx])
        else:
            base_hour = (idx // 4) % 24  # rough estimate
        target_hours = np.array([(base_hour + h) % 24 for h in range(24)], dtype=np.int64)

        # --- Inference ---
        with torch.no_grad():
            seq_t = torch.from_numpy(seq_norm).unsqueeze(0).to(self.device)
            exo_t = torch.from_numpy(exo).unsqueeze(0).to(self.device)
            lag_t = torch.from_numpy(price_lag_96).unsqueeze(0).to(self.device)
            hrs_t = torch.from_numpy(target_hours).unsqueeze(0).to(self.device)

            pred_norm = self.model(seq_t, exo_t, lag_t, hrs_t).cpu().numpy()[0]

        # --- Denormalize ---
        pred = pred_norm * std + mean
        pred = np.clip(pred, -500, 50000)

        if horizon <= len(pred):
            return pred[:horizon]
        else:
            return np.pad(pred, (0, horizon - len(pred)), mode="edge")


# ============================================================
# MPC Evaluation
# ============================================================

def evaluate(province, df_train, df_val, df_test, model, config):
    battery = BatteryConfig()
    df_trainval = pd.concat([df_train, df_val])

    features_test = df_test[FEATURE_COLS].fillna(0).values.astype(np.float32)
    prices_test = df_test["rt_price"].fillna(0).values.astype(np.float32)

    prices_tv = df_trainval["rt_price"].fillna(0).values.astype(np.float32)
    full_prices = np.concatenate([prices_tv, prices_test])
    test_offset = len(prices_tv)

    da_tv = df_trainval["da_price"].fillna(0).values.astype(np.float32) if "da_price" in df_trainval.columns else np.zeros(len(prices_tv), dtype=np.float32)
    da_test = df_test["da_price"].fillna(0).values.astype(np.float32) if "da_price" in df_test.columns else np.zeros(len(prices_test), dtype=np.float32)
    full_da = np.concatenate([da_tv, da_test])

    exo_tv = df_trainval[EXO_COLS].fillna(0).values.astype(np.float32)
    exo_test = df_test[EXO_COLS].fillna(0).values.astype(np.float32)
    full_exo = np.concatenate([exo_tv, exo_test])

    # Build full sequence features (8 channels)
    def _build_seq(df_part):
        cols = []
        for col in SEQ_COLS:
            if col in df_part.columns:
                cols.append(df_part[col].fillna(0).values.astype(np.float32))
            else:
                cols.append(np.zeros(len(df_part), dtype=np.float32))
        return np.stack(cols, axis=-1)
    full_seq = np.concatenate([_build_seq(df_trainval), _build_seq(df_test)])

    # Build hour indices
    hours_tv = df_trainval.index.hour.values if hasattr(df_trainval.index, 'hour') else np.zeros(len(df_trainval), dtype=np.int64)
    hours_test = df_test.index.hour.values if hasattr(df_test.index, 'hour') else np.zeros(len(df_test), dtype=np.int64)
    full_hours = np.concatenate([hours_tv, hours_test])

    n_train_days = len(prices_tv) // 96
    daily_profile = prices_tv[:n_train_days * 96].reshape(n_train_days, 96).mean(axis=0)

    # Device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 1. Threshold
    logger.info("\n--- Threshold ---")
    ma96_tv = df_trainval.get("rt_price_ma_96", pd.Series(dtype=float))
    ma96_tv = ma96_tv.fillna(pd.Series(prices_tv).rolling(96, min_periods=1).mean()).values
    ma96_test = df_test.get("rt_price_ma_96", pd.Series(dtype=float))
    ma96_test = ma96_test.fillna(pd.Series(prices_test).rolling(96, min_periods=1).mean()).values

    best_rev, best_cr, best_dr = -np.inf, 0.65, 1.35
    for cr in np.arange(0.50, 0.86, 0.05):
        for dr in np.arange(1.15, 1.56, 0.05):
            r = simulate_threshold(prices_tv, ma96_tv, battery, cr, dr)["revenue"]
            if r > best_rev:
                best_rev, best_cr, best_dr = r, cr, dr
    th = simulate_threshold(prices_test, ma96_test, battery, best_cr, best_dr)
    logger.info(f"  Revenue: {th['revenue']:,.0f}")

    # 2. Oracle
    logger.info("\n--- Oracle ---")
    oracle_cont = simulate_oracle_continuous(prices_test, battery)
    oracle_disc = simulate_oracle_discrete(prices_test, battery)

    # 3. LightGBM MPC
    logger.info("\n--- LightGBM MPC ---")
    t0 = time.time()
    features_tv = df_trainval[FEATURE_COLS].fillna(0).values.astype(np.float32)
    lgbm = LGBMForecaster(n_estimators=300, max_depth=6, learning_rate=0.05)
    lgbm.fit(features_tv, prices_tv)
    lgbm._full_prices = full_prices

    class LGBMWrap:
        def predict(self, f, idx, horizon=96):
            return lgbm.predict(f, test_offset + idx, horizon)

    lgbm_ctrl = MPCController(LGBMWrap(), battery, continuous=True, daily_profile=daily_profile)
    lgbm_r = simulate_mpc(lgbm_ctrl, features_test, prices_test, battery, replan_every=4)
    logger.info(f"  Revenue: {lgbm_r['revenue']:,.0f}  ({time.time() - t0:.0f}s)")

    # 4. PatchTST MPC
    logger.info("\n--- PatchTST V3 MPC ---")
    t0 = time.time()
    forecaster = PatchTSTForecaster(config, model, device)
    forecaster._full_prices = full_prices
    forecaster._full_da_prices = full_da
    forecaster._full_exo = full_exo
    forecaster._full_seq = full_seq
    forecaster._full_hours = full_hours

    class PTSTWrap:
        def predict(self, f, idx, horizon=96):
            return forecaster.predict(f, test_offset + idx, horizon)

    ptst_ctrl = MPCController(PTSTWrap(), battery, continuous=True, daily_profile=daily_profile)
    ptst_r = simulate_mpc(ptst_ctrl, features_test, prices_test, battery, replan_every=4, log_every=90)
    logger.info(f"  Revenue: {ptst_r['revenue']:,.0f}  ({time.time() - t0:.0f}s)")

    # Summary
    test_d = len(prices_test) // 96
    ann = lambda r: r / test_d * 365
    ptst_vs_lgbm = (ptst_r["revenue"] - lgbm_r["revenue"]) / abs(lgbm_r["revenue"]) * 100

    logger.info(f"\n{'=' * 80}")
    logger.info(f"  {province.upper()} — {test_d} test days (PatchTST V3)")
    logger.info(f"{'=' * 80}")
    logger.info(f"  {'Method':<25} {'Revenue':>14} {'Annual':>14} {'vs TH':>10}")
    logger.info(f"  {'-' * 25} {'-' * 14} {'-' * 14} {'-' * 10}")

    for name, rev in [
        ("Threshold", th["revenue"]),
        ("LightGBM MPC", lgbm_r["revenue"]),
        ("PatchTST V3 MPC", ptst_r["revenue"]),
        ("Oracle (discrete)", oracle_disc["revenue"]),
        ("Oracle (continuous)", oracle_cont["revenue"]),
    ]:
        vs = (rev - th["revenue"]) / abs(th["revenue"]) * 100 if th["revenue"] != 0 else 0
        logger.info(f"  {name:<25} {rev:>14,.0f} {ann(rev):>14,.0f} {vs:>+9.1f}%")

    logger.info(f"\n  >>> PatchTST V3 vs LightGBM: {ptst_vs_lgbm:+.1f}% <<<")

    return {
        "province": province, "test_days": test_d,
        "threshold": th["revenue"], "lgbm_mpc": lgbm_r["revenue"],
        "patchtst_mpc": ptst_r["revenue"],
        "oracle_disc": oracle_disc["revenue"], "oracle_cont": oracle_cont["revenue"],
        "ptst_vs_lgbm_pct": ptst_vs_lgbm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="shandong")
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--val-days", type=int, default=180)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    provinces = ["shandong", "shanxi", "guangdong", "gansu"] if args.all else [args.province]
    results = []

    for prov in provinces:
        logger.info(f"\n{'#' * 80}\n# {prov.upper()} (PatchTST V3)\n{'#' * 80}")
        config = get_config_v3(prov)
        df_train, df_val, df_test = load_and_split(prov, args.test_days, args.val_days)
        model, config = train_patchtst(prov, df_train, df_val, config)
        r = evaluate(prov, df_train, df_val, df_test, model, config)
        results.append(r)

    if len(results) > 1:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"  FINAL — PatchTST V3 vs LightGBM")
        logger.info(f"{'=' * 80}")
        for r in results:
            logger.info(f"  {r['province']:<12} LGBM={r['lgbm_mpc']:>12,.0f}  PTST={r['patchtst_mpc']:>12,.0f}  diff={r['ptst_vs_lgbm_pct']:>+6.1f}%")
        pd.DataFrame(results).to_csv(PROCESSED_DIR / "patchtst_v3_results.csv", index=False)


if __name__ == "__main__":
    main()
