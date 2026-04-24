"""
V5 Validation Pipeline: Rolling Forward + De-Oracle Test.

Validates that V5 ensemble gains are real, not artifacts of:
  1. Alpha overfitting to a fixed test set (Rolling Forward)
  2. Oracle features unavailable in production (De-Oracle)

Only runs Shandong for speed. ~4.5 hours on GPU.

Usage:
    PYTHONPATH=. python3 scripts/16_validate_v5.py
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from data.china.features import FEATURE_COLS, FEATURE_COLS_V4
from forecast.transformer_config import get_config_v5, TransformerConfig
from forecast.patchtst_model import PatchTST_EPF
from forecast.lgbm_forecaster import LGBMForecaster
from forecast.mpc_controller import (
    MPCController, simulate_mpc, simulate_threshold,
)

PROCESSED_DIR = Path("data/china/processed")
MODEL_DIR = Path("models")
PROVINCE = "shandong"

EXO_COLS = FEATURE_COLS_V4
SEQ_COLS = [
    "rt_price", "da_price", "da_rt_spread",
    "load_norm", "renewable_penetration", "net_load_norm",
    "wind_ratio", "solar_ratio",
]

# Oracle features that use actual values (not available in real-time)
ORACLE_COLS = [
    "load_norm", "load_change",
    "renewable_penetration", "wind_ratio", "solar_ratio",
    "net_load_norm", "tie_line_norm", "temperature_norm",
    # V4 causal features (derived from actual weather/generation)
    "wind_speed_norm", "solar_radiation_norm", "temp_load_interaction",
    "wind_ramp", "solar_ramp", "net_load_ramp",
    "supply_demand_tightness", "renewable_forecast_proxy",
]


# ============================================================
# Dataset (copied from V5 script for self-containment)
# ============================================================

class EPFDatasetV3(Dataset):
    def __init__(self, df, exo_cols, config, augment=False, stride=1):
        self.config = config
        self.augment = augment
        W = config.window_size
        T = config.output_steps
        N = len(df)

        seq_data = []
        for col in SEQ_COLS:
            if col in df.columns:
                seq_data.append(df[col].fillna(0).values.astype(np.float32))
            else:
                seq_data.append(np.zeros(N, dtype=np.float32))
        self.seq = np.stack(seq_data, axis=-1)

        self.rt = df["rt_price"].fillna(0).values.astype(np.float32)
        self.exo = df[exo_cols].fillna(0).values.astype(np.float32)

        if hasattr(df.index, 'hour'):
            self.hours = df.index.hour.values.astype(np.int64)
        else:
            self.hours = np.zeros(N, dtype=np.int64)

        hour_offsets = np.arange(24, dtype=np.int64)
        self.target_hours_all = (self.hours[:, None] + hour_offsets[None, :]) % 24

        min_start = max(W, 96)
        self.indices = np.arange(min_start, N - T, stride)

        norm_w = config.norm_window
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

        s = idx - W
        seq_win = self.seq[s:idx].copy()
        seq_win[:, 0] = (seq_win[:, 0] - mean) / std
        seq_win[:, 1] = (seq_win[:, 1] - mean) / std
        seq_win[:, 2] = seq_win[:, 2] / std

        if self.augment and self.config.noise_std_ratio > 0:
            noise = np.random.normal(0, self.config.noise_std_ratio, size=W).astype(np.float32)
            seq_win[:, 0] += noise

        exo = self.exo[idx].copy()

        lag_start = idx - 96
        if lag_start >= 0:
            price_lag_96 = self.rt[lag_start:lag_start + 96].copy()
            price_lag_96 = (price_lag_96 - mean) / std
        else:
            price_lag_96 = np.zeros(96, dtype=np.float32)

        target_hours = self.target_hours_all[idx]
        target = (self.rt[idx:idx + T] - mean) / std
        norm_stats = np.array([mean, std], dtype=np.float32)

        return (
            torch.from_numpy(seq_win), torch.from_numpy(exo),
            torch.from_numpy(price_lag_96), torch.from_numpy(target_hours),
            torch.from_numpy(target), torch.from_numpy(norm_stats),
        )


# ============================================================
# PatchTST Forecaster (from V5)
# ============================================================

class PatchTSTForecaster:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.model.eval()
        self._full_prices = None
        self._full_da_prices = None
        self._full_exo = None
        self._full_seq = None
        self._full_hours = None

    def predict(self, features_t, idx, horizon=96):
        W = self.config.window_size
        norm_w = self.config.norm_window

        win_start = max(0, idx - W + 1)
        win_end = idx + 1

        if self._full_seq is not None:
            seq_win = self._full_seq[win_start:win_end].astype(np.float32)
        else:
            seq_win = np.zeros((win_end - win_start, 8), dtype=np.float32)

        if len(seq_win) < W:
            pad_len = W - len(seq_win)
            seq_win = np.pad(seq_win, ((pad_len, 0), (0, 0)), mode="edge")

        rt_col = self._full_prices[max(0, idx + 1 - norm_w):idx + 1].astype(np.float32)
        mean = rt_col.mean()
        std = max(rt_col.std(), 1e-6)

        seq_norm = seq_win.copy()
        seq_norm[:, 0] = (seq_norm[:, 0] - mean) / std
        seq_norm[:, 1] = (seq_norm[:, 1] - mean) / std
        seq_norm[:, 2] = seq_norm[:, 2] / std

        if self._full_exo is not None and idx < len(self._full_exo):
            exo = self._full_exo[idx].astype(np.float32)
        else:
            exo = np.zeros(self.config.n_exo_features, dtype=np.float32)

        lag_start = max(0, idx - 95)
        lag_end = idx + 1
        lag_raw = self._full_prices[lag_start:lag_end].astype(np.float32)
        if len(lag_raw) < 96:
            lag_raw = np.pad(lag_raw, (96 - len(lag_raw), 0), mode="edge")
        price_lag_96 = (lag_raw - mean) / std

        if self._full_hours is not None and idx < len(self._full_hours):
            base_hour = int(self._full_hours[idx])
        else:
            base_hour = (idx // 4) % 24
        target_hours = np.array([(base_hour + h) % 24 for h in range(24)], dtype=np.int64)

        with torch.no_grad():
            seq_t = torch.from_numpy(seq_norm).unsqueeze(0).to(self.device)
            exo_t = torch.from_numpy(exo).unsqueeze(0).to(self.device)
            lag_t = torch.from_numpy(price_lag_96).unsqueeze(0).to(self.device)
            hrs_t = torch.from_numpy(target_hours).unsqueeze(0).to(self.device)
            pred_norm = self.model(seq_t, exo_t, lag_t, hrs_t).cpu().numpy()[0]

        pred = pred_norm * std + mean
        pred = np.clip(pred, -500, 50000)
        return pred[:horizon] if horizon <= len(pred) else np.pad(pred, (0, horizon - len(pred)), mode="edge")


# ============================================================
# Training function
# ============================================================

def train_patchtst(df_train, df_val, config):
    if config.rolling_train_days is not None:
        max_rows = 96 * config.rolling_train_days
        if len(df_train) > max_rows:
            df_train = df_train.iloc[-max_rows:]

    train_ds = EPFDatasetV3(df_train, EXO_COLS, config, augment=True, stride=1)
    val_ds = EPFDatasetV3(df_val, EXO_COLS, config, augment=False, stride=96)

    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = PatchTST_EPF(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler_T0)
    criterion = nn.HuberLoss(delta=config.huber_delta)

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
        model.train()
        for seq, exo, lag96, tgt_hours, target, ns in train_loader:
            seq, exo, lag96, tgt_hours, target = (
                seq.to(device), exo.to(device), lag96.to(device),
                tgt_hours.to(device), target.to(device),
            )
            optimizer.zero_grad()
            pred = model(seq, exo, lag96, tgt_hours)
            loss = criterion(pred, target)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_maes = []
        with torch.no_grad():
            for seq, exo, lag96, tgt_hours, target, ns in val_loader:
                seq, exo, lag96, tgt_hours, target, ns = (
                    seq.to(device), exo.to(device), lag96.to(device),
                    tgt_hours.to(device), target.to(device), ns.to(device),
                )
                pred = model(seq, exo, lag96, tgt_hours)
                mean, std = ns[:, 0:1], ns[:, 1:2]
                mae = torch.abs(pred * std + mean - (target * std + mean)).mean().item()
                val_maes.append(mae)

        val_mae = np.mean(val_maes)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    elapsed = time.time() - t0
    logger.info(f"  PatchTST trained: {elapsed:.0f}s, {epoch+1} epochs, best MAE={best_val_mae:.1f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, device


# ============================================================
# MPC evaluation for a single period
# ============================================================

def eval_period(df_trainval, df_test, model, config, device, alpha_search_prices=None):
    """Evaluate LGBM, PatchTST, and Ensemble on df_test. Return revenues."""
    battery = BatteryConfig()

    features_test = df_test[FEATURE_COLS].fillna(0).values.astype(np.float32)
    prices_test = df_test["rt_price"].fillna(0).values.astype(np.float32)
    prices_tv = df_trainval["rt_price"].fillna(0).values.astype(np.float32)
    full_prices = np.concatenate([prices_tv, prices_test])
    test_offset = len(prices_tv)

    # Full arrays for PatchTST
    exo_tv = df_trainval[EXO_COLS].fillna(0).values.astype(np.float32)
    exo_test = df_test[EXO_COLS].fillna(0).values.astype(np.float32)
    full_exo = np.concatenate([exo_tv, exo_test])

    def _build_seq(df_part):
        cols = []
        for col in SEQ_COLS:
            if col in df_part.columns:
                cols.append(df_part[col].fillna(0).values.astype(np.float32))
            else:
                cols.append(np.zeros(len(df_part), dtype=np.float32))
        return np.stack(cols, axis=-1)
    full_seq = np.concatenate([_build_seq(df_trainval), _build_seq(df_test)])

    hours_tv = df_trainval.index.hour.values if hasattr(df_trainval.index, 'hour') else np.zeros(len(df_trainval), dtype=np.int64)
    hours_test = df_test.index.hour.values if hasattr(df_test.index, 'hour') else np.zeros(len(df_test), dtype=np.int64)
    full_hours = np.concatenate([hours_tv, hours_test])

    n_train_days = len(prices_tv) // 96
    daily_profile = prices_tv[:n_train_days * 96].reshape(n_train_days, 96).mean(axis=0)

    # LightGBM
    features_tv = df_trainval[FEATURE_COLS].fillna(0).values.astype(np.float32)
    lgbm = LGBMForecaster(n_estimators=300, max_depth=6, learning_rate=0.05)
    lgbm.fit(features_tv, prices_tv)
    lgbm._full_prices = full_prices

    class LGBMWrap:
        def predict(self, f, idx, horizon=96):
            return lgbm.predict(f, test_offset + idx, horizon)

    lgbm_ctrl = MPCController(LGBMWrap(), battery, continuous=True, daily_profile=daily_profile)
    lgbm_r = simulate_mpc(lgbm_ctrl, features_test, prices_test, battery, replan_every=4)

    # PatchTST
    forecaster = PatchTSTForecaster(config, model, device)
    forecaster._full_prices = full_prices
    forecaster._full_exo = full_exo
    forecaster._full_seq = full_seq
    forecaster._full_hours = full_hours

    class PTSTWrap:
        def predict(self, f, idx, horizon=96):
            return forecaster.predict(f, test_offset + idx, horizon)

    ptst_ctrl = MPCController(PTSTWrap(), battery, continuous=True, daily_profile=daily_profile)
    ptst_r = simulate_mpc(ptst_ctrl, features_test, prices_test, battery, replan_every=4)

    # Ensemble: search alpha on alpha_search_prices if provided, else on test
    if alpha_search_prices is not None:
        # Search alpha on a SEPARATE period (previous quarter)
        search_features = alpha_search_prices["features"]
        search_prices = alpha_search_prices["prices"]
        search_offset = alpha_search_prices["offset"]

        class LGBMSearch:
            def predict(self, f, idx, horizon=96):
                return lgbm.predict(f, search_offset + idx, horizon)

        search_forecaster = PatchTSTForecaster(config, model, device)
        search_forecaster._full_prices = full_prices
        search_forecaster._full_exo = full_exo
        search_forecaster._full_seq = full_seq
        search_forecaster._full_hours = full_hours

        class PTSTSearch:
            def predict(self, f, idx, horizon=96):
                return search_forecaster.predict(f, search_offset + idx, horizon)

        best_a, best_rev = 0.6, -np.inf
        for a_pct in [50, 55, 60, 65, 70]:
            a = a_pct / 100.0

            class EnsSearch:
                def __init__(self, alpha):
                    self._a = alpha
                def predict(self, f, idx, horizon=96):
                    p1 = lgbm.predict(f, search_offset + idx, horizon)
                    p2 = search_forecaster.predict(f, search_offset + idx, horizon)
                    return self._a * p1 + (1 - self._a) * p2

            ens_ctrl = MPCController(EnsSearch(a), battery, continuous=True, daily_profile=daily_profile)
            r = simulate_mpc(ens_ctrl, search_features, search_prices, battery, replan_every=4)
            if r["revenue"] > best_rev:
                best_rev = r["revenue"]
                best_a = a
    else:
        best_a = 0.65  # default

    # Evaluate ensemble with chosen alpha on TEST period
    class EnsWrap:
        def __init__(self, alpha):
            self._a = alpha
        def predict(self, f, idx, horizon=96):
            p1 = lgbm.predict(f, test_offset + idx, horizon)
            p2 = forecaster.predict(f, test_offset + idx, horizon)
            return self._a * p1 + (1 - self._a) * p2

    ens_ctrl = MPCController(EnsWrap(best_a), battery, continuous=True, daily_profile=daily_profile)
    ens_r = simulate_mpc(ens_ctrl, features_test, prices_test, battery, replan_every=4)

    return {
        "lgbm": lgbm_r["revenue"],
        "ptst": ptst_r["revenue"],
        "ensemble": ens_r["revenue"],
        "alpha": best_a,
        "test_days": len(prices_test) // 96,
    }


# ============================================================
# De-Oracle: shift oracle features by 96 steps
# ============================================================

def de_oracle(df):
    """Replace oracle features with t-96 (yesterday) values."""
    df = df.copy()
    for col in ORACLE_COLS:
        if col in df.columns:
            df[col] = df[col].shift(96).fillna(0)
    return df


# ============================================================
# Main validation
# ============================================================

def main():
    logger.info("=" * 80)
    logger.info("  V5 VALIDATION PIPELINE — Shandong")
    logger.info("=" * 80)

    # Load full data
    path = PROCESSED_DIR / f"{PROVINCE}_oracle.parquet"
    df_full = pd.read_parquet(path)
    logger.info(f"Loaded {PROVINCE}: {len(df_full):,} rows ({len(df_full) // 96} days)")

    config = get_config_v5(PROVINCE)
    test_days = 365
    val_days = 180
    quarter_days = 91

    test_start = len(df_full) - 96 * test_days
    val_start = test_start - 96 * val_days

    df_test_full = df_full.iloc[test_start:]

    # ================================================================
    # PART 1: Rolling Forward Validation
    # ================================================================
    logger.info(f"\n{'#' * 80}")
    logger.info(f"# PART 1: ROLLING FORWARD VALIDATION")
    logger.info(f"{'#' * 80}")

    quarters = []
    q_starts = []
    for q in range(4):
        q_start = test_start + q * 96 * quarter_days
        q_end = test_start + (q + 1) * 96 * quarter_days if q < 3 else len(df_full)
        quarters.append((q_start, q_end))
        q_starts.append(q_start)
        q_days = (q_end - q_start) // 96
        logger.info(f"  Q{q+1}: rows [{q_start}:{q_end}] = {q_days} days")

    rolling_results = []

    for q_idx, (q_start, q_end) in enumerate(quarters):
        q_label = f"Q{q_idx+1}"
        q_days = (q_end - q_start) // 96
        logger.info(f"\n--- {q_label}: {q_days} days ---")

        # Train on everything before this quarter
        df_train = df_full.iloc[:q_start - 96 * val_days]
        df_val = df_full.iloc[q_start - 96 * val_days:q_start]
        df_test_q = df_full.iloc[q_start:q_end]

        logger.info(f"  Train: {len(df_train)//96}d, Val: {len(df_val)//96}d, Test: {q_days}d")

        # Train PatchTST
        t0 = time.time()
        model, device = train_patchtst(df_train, df_val, config)
        logger.info(f"  Training done ({time.time()-t0:.0f}s)")

        # Alpha search: use previous quarter (or val set for Q1)
        if q_idx == 0:
            # For Q1, search alpha on val set
            alpha_search = {
                "features": df_val[FEATURE_COLS].fillna(0).values.astype(np.float32),
                "prices": df_val["rt_price"].fillna(0).values.astype(np.float32),
                "offset": len(df_train),
            }
        else:
            # For Q2+, search alpha on previous quarter
            prev_start, prev_end = quarters[q_idx - 1]
            df_prev = df_full.iloc[prev_start:prev_end]
            alpha_search = {
                "features": df_prev[FEATURE_COLS].fillna(0).values.astype(np.float32),
                "prices": df_prev["rt_price"].fillna(0).values.astype(np.float32),
                "offset": prev_start,
            }

        # Evaluate
        df_trainval = pd.concat([df_train, df_val])
        t0 = time.time()
        r = eval_period(df_trainval, df_test_q, model, config, device, alpha_search)
        elapsed = time.time() - t0

        ens_vs = (r["ensemble"] - r["lgbm"]) / abs(r["lgbm"]) * 100
        logger.info(f"  {q_label}: LGBM={r['lgbm']:,.0f}  ENS={r['ensemble']:,.0f}  "
                    f"diff={ens_vs:+.1f}%  alpha={r['alpha']:.2f}  ({elapsed:.0f}s)")
        rolling_results.append({**r, "quarter": q_label, "ens_vs_lgbm": ens_vs})

    # Rolling summary
    total_lgbm = sum(r["lgbm"] for r in rolling_results)
    total_ens = sum(r["ensemble"] for r in rolling_results)
    total_vs = (total_ens - total_lgbm) / abs(total_lgbm) * 100

    logger.info(f"\n{'=' * 80}")
    logger.info(f"  ROLLING FORWARD SUMMARY (Shandong)")
    logger.info(f"{'=' * 80}")
    for r in rolling_results:
        logger.info(f"  {r['quarter']} ({r['test_days']}d): "
                    f"LGBM={r['lgbm']:>12,.0f}  ENS={r['ensemble']:>12,.0f}  "
                    f"diff={r['ens_vs_lgbm']:>+6.1f}%  alpha={r['alpha']:.2f}")
    logger.info(f"  {'Total':>5}: "
                f"LGBM={total_lgbm:>12,.0f}  ENS={total_ens:>12,.0f}  "
                f"diff={total_vs:>+6.1f}%")

    wins = sum(1 for r in rolling_results if r["ens_vs_lgbm"] > 0)
    logger.info(f"\n  Ensemble wins {wins}/4 quarters")

    # ================================================================
    # PART 2: De-Oracle Test
    # ================================================================
    logger.info(f"\n{'#' * 80}")
    logger.info(f"# PART 2: DE-ORACLE TEST")
    logger.info(f"{'#' * 80}")

    # 2a. Oracle baseline (same as V5 original, fixed split)
    df_train_orig = df_full.iloc[:val_start]
    df_val_orig = df_full.iloc[val_start:test_start]
    df_test_orig = df_full.iloc[test_start:]

    logger.info("\n--- Oracle (original data) ---")
    model_oracle, device = train_patchtst(df_train_orig, df_val_orig, config)
    df_trainval_orig = pd.concat([df_train_orig, df_val_orig])
    r_oracle = eval_period(df_trainval_orig, df_test_orig, model_oracle, config, device)

    oracle_vs = (r_oracle["ensemble"] - r_oracle["lgbm"]) / abs(r_oracle["lgbm"]) * 100
    logger.info(f"  Oracle:  LGBM={r_oracle['lgbm']:,.0f}  ENS={r_oracle['ensemble']:,.0f}  diff={oracle_vs:+.1f}%")

    # 2b. De-oracle: shift oracle features by 96 steps
    logger.info("\n--- De-Oracle (shifted features) ---")
    df_full_deoracle = de_oracle(df_full)

    df_train_do = df_full_deoracle.iloc[:val_start]
    df_val_do = df_full_deoracle.iloc[val_start:test_start]
    df_test_do = df_full_deoracle.iloc[test_start:]

    model_do, device = train_patchtst(df_train_do, df_val_do, config)
    df_trainval_do = pd.concat([df_train_do, df_val_do])
    r_do = eval_period(df_trainval_do, df_test_do, model_do, config, device)

    do_vs = (r_do["ensemble"] - r_do["lgbm"]) / abs(r_do["lgbm"]) * 100
    logger.info(f"  De-Oracle: LGBM={r_do['lgbm']:,.0f}  ENS={r_do['ensemble']:,.0f}  diff={do_vs:+.1f}%")

    # De-oracle summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  DE-ORACLE SUMMARY (Shandong)")
    logger.info(f"{'=' * 80}")
    logger.info(f"  {'Mode':<15} {'LGBM':>14} {'Ensemble':>14} {'ENS vs LGBM':>12}")
    logger.info(f"  {'-'*15} {'-'*14} {'-'*14} {'-'*12}")
    logger.info(f"  {'Oracle':<15} {r_oracle['lgbm']:>14,.0f} {r_oracle['ensemble']:>14,.0f} {oracle_vs:>+11.1f}%")
    logger.info(f"  {'De-Oracle':<15} {r_do['lgbm']:>14,.0f} {r_do['ensemble']:>14,.0f} {do_vs:>+11.1f}%")

    lgbm_degrade = (r_do["lgbm"] - r_oracle["lgbm"]) / abs(r_oracle["lgbm"]) * 100
    ens_degrade = (r_do["ensemble"] - r_oracle["ensemble"]) / abs(r_oracle["ensemble"]) * 100
    logger.info(f"\n  LGBM degradation:     {lgbm_degrade:+.1f}%")
    logger.info(f"  Ensemble degradation: {ens_degrade:+.1f}%")
    logger.info(f"  Ensemble advantage retained: {'YES' if do_vs > 0 else 'NO'}")


if __name__ == "__main__":
    main()
