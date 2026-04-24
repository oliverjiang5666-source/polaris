"""
30-Day Paper Trading Simulation.

Simulates daily trading decisions using the V5 ensemble (LightGBM + PatchTST).
Uses the last 30 days of existing data, revealing one day at a time.

For each day:
  1. Train LightGBM on all data up to that day (seconds, CPU)
  2. Use pre-trained PatchTST V5 weights for inference
  3. Ensemble = 0.70 * LGBM + 0.30 * PatchTST
  4. MPC generates charging/discharging schedule
  5. Compare vs pure LGBM and Oracle (hindsight)

Usage:
    PYTHONPATH=. python3 scripts/17_paper_trade_sim.py
    PYTHONPATH=. python3 scripts/17_paper_trade_sim.py --days 14
"""
from __future__ import annotations

import time
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from data.china.features import FEATURE_COLS, FEATURE_COLS_V4
from forecast.transformer_config import get_config_v5
from forecast.patchtst_model import PatchTST_EPF
from forecast.lgbm_forecaster import LGBMForecaster
from forecast.mpc_controller import (
    MPCController, simulate_mpc, simulate_oracle_continuous,
)
from oracle.lp_oracle import solve_day

PROCESSED_DIR = Path("data/china/processed")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/paper_trading")
PROVINCE = "shandong"
ALPHA = 0.70  # LightGBM weight in ensemble (from Rolling Forward validation)

EXO_COLS = FEATURE_COLS_V4
SEQ_COLS = [
    "rt_price", "da_price", "da_rt_spread",
    "load_norm", "renewable_penetration", "net_load_norm",
    "wind_ratio", "solar_ratio",
]


class PatchTSTForecaster:
    """Lightweight forecaster using pre-trained PatchTST weights."""

    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.model.eval()
        self._full_prices = None
        self._full_exo = None
        self._full_seq = None
        self._full_hours = None

    def predict(self, features_t, idx, horizon=96):
        W = self.config.window_size
        norm_w = self.config.norm_window

        win_start = max(0, idx - W + 1)
        win_end = idx + 1
        seq_win = self._full_seq[win_start:win_end].astype(np.float32)

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

        exo = self._full_exo[idx].astype(np.float32) if idx < len(self._full_exo) else np.zeros(self.config.n_exo_features, dtype=np.float32)

        lag_start = max(0, idx - 95)
        lag_raw = self._full_prices[lag_start:idx + 1].astype(np.float32)
        if len(lag_raw) < 96:
            lag_raw = np.pad(lag_raw, (96 - len(lag_raw), 0), mode="edge")
        price_lag_96 = (lag_raw - mean) / std

        base_hour = int(self._full_hours[idx]) if idx < len(self._full_hours) else (idx // 4) % 24
        target_hours = np.array([(base_hour + h) % 24 for h in range(24)], dtype=np.int64)

        with torch.no_grad():
            seq_t = torch.from_numpy(seq_norm).unsqueeze(0).to(self.device)
            exo_t = torch.from_numpy(exo).unsqueeze(0).to(self.device)
            lag_t = torch.from_numpy(price_lag_96).unsqueeze(0).to(self.device)
            hrs_t = torch.from_numpy(target_hours).unsqueeze(0).to(self.device)
            pred_norm = self.model(seq_t, exo_t, lag_t, hrs_t).cpu().numpy()[0]

        pred = pred_norm * std + mean
        return np.clip(pred[:horizon], -500, 50000)


def simulate_one_day(prices_day, schedule_power, battery):
    """Simulate battery for one day (96 steps) given a power schedule."""
    from forecast.mpc_controller import _step_battery
    soc = 0.5
    revenue = 0.0
    for t in range(min(96, len(prices_day), len(schedule_power))):
        power = float(schedule_power[t])
        soc, net_rev, _ = _step_battery(power, prices_day[t], soc, battery, degradation_per_mwh=2.0)
        revenue += net_rev
    return revenue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    parser.add_argument("--province", default="shandong")
    args = parser.parse_args()

    province = args.province
    sim_days = args.days

    logger.info("=" * 70)
    logger.info(f"  PAPER TRADING SIMULATION — {province.upper()}")
    logger.info(f"  Simulating last {sim_days} days, alpha={ALPHA}")
    logger.info("=" * 70)

    # Load data
    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")
    logger.info(f"Loaded: {len(df):,} rows, {df.index.min()} to {df.index.max()}")

    # Load pre-trained PatchTST
    config = get_config_v5(province)
    config.device = "cpu"  # local inference

    model_path = MODEL_DIR / f"patchtst_{province}.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = PatchTST_EPF(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    device = torch.device("cpu")
    logger.info(f"Loaded PatchTST: {model.count_parameters():,} params from {model_path}")

    battery = BatteryConfig()

    # Build full arrays for forecaster
    full_prices = df["rt_price"].fillna(0).values.astype(np.float32)
    full_exo = df[EXO_COLS].fillna(0).values.astype(np.float32)

    seq_data = []
    for col in SEQ_COLS:
        if col in df.columns:
            seq_data.append(df[col].fillna(0).values.astype(np.float32))
        else:
            seq_data.append(np.zeros(len(df), dtype=np.float32))
    full_seq = np.stack(seq_data, axis=-1)
    full_hours = df.index.hour.values.astype(np.int64)

    # Simulation period: last N days
    total_steps = len(df)
    sim_start = total_steps - sim_days * 96
    n_train_days_base = sim_start // 96

    daily_profile = full_prices[:sim_start].reshape(-1, 96)[-min(n_train_days_base, 365):].mean(axis=0)

    logger.info(f"Sim period: day {sim_start//96+1} to {total_steps//96}")
    logger.info(f"Train base: {n_train_days_base} days")
    logger.info("")

    # Daily simulation
    results = []

    for d in range(sim_days):
        day_start = sim_start + d * 96
        day_end = day_start + 96
        if day_end > total_steps:
            break

        date_str = str(df.index[day_start].date())
        prices_today = full_prices[day_start:day_end]

        # -- Train LightGBM on all data up to today --
        train_end = day_start
        features_train = df.iloc[:train_end][FEATURE_COLS].fillna(0).values.astype(np.float32)
        prices_train = full_prices[:train_end]

        lgbm = LGBMForecaster(n_estimators=300, max_depth=6, learning_rate=0.05, verbose=-1)
        lgbm.fit(features_train, prices_train)
        lgbm._full_prices = full_prices

        # -- Setup PatchTST forecaster --
        ptst = PatchTSTForecaster(config, model, device)
        ptst._full_prices = full_prices
        ptst._full_exo = full_exo
        ptst._full_seq = full_seq
        ptst._full_hours = full_hours

        # -- Get features for today's first timestep --
        features_t = df.iloc[day_start][FEATURE_COLS].fillna(0).values.astype(np.float32)

        # -- Forecasts --
        pred_lgbm = lgbm.predict(features_t, day_start, horizon=96)
        pred_ptst = ptst.predict(features_t, day_start, horizon=96)
        pred_ens = ALPHA * pred_lgbm + (1 - ALPHA) * pred_ptst

        # -- MPC schedules --
        def make_schedule(forecast):
            # Extend with daily profile
            step_in_day = day_start % 96
            extended = np.roll(daily_profile, -((step_in_day + 96) % 96))
            scale = np.mean(forecast) / (np.mean(daily_profile) + 1e-8)
            extended_prices = np.concatenate([forecast, extended * scale])
            result = solve_day(extended_prices, battery, init_soc=0.5)
            return result["net_power"][:96]

        sched_lgbm = make_schedule(pred_lgbm)
        sched_ens = make_schedule(pred_ens)

        # -- Simulate actual revenue --
        rev_lgbm = simulate_one_day(prices_today, sched_lgbm, battery)
        rev_ens = simulate_one_day(prices_today, sched_ens, battery)

        # -- Oracle (hindsight) --
        oracle_result = solve_day(prices_today, battery, init_soc=0.5)
        rev_oracle = oracle_result["revenue"]

        # -- Log --
        ens_vs_lgbm = (rev_ens - rev_lgbm) / abs(rev_lgbm) * 100 if rev_lgbm != 0 else 0
        ens_capture = rev_ens / rev_oracle * 100 if rev_oracle > 0 else 0

        results.append({
            "date": date_str,
            "rev_ensemble": rev_ens,
            "rev_lgbm": rev_lgbm,
            "rev_oracle": rev_oracle,
            "ens_vs_lgbm_pct": ens_vs_lgbm,
            "oracle_capture_pct": ens_capture,
        })

        marker = "+" if rev_ens > rev_lgbm else "-"
        logger.info(
            f"  {date_str}  ENS={rev_ens:>10,.0f}  LGBM={rev_lgbm:>10,.0f}  "
            f"Oracle={rev_oracle:>10,.0f}  vs_LGBM={ens_vs_lgbm:>+5.1f}%  "
            f"capture={ens_capture:>4.0f}%  [{marker}]"
        )

    # ============================================================
    # Summary
    # ============================================================
    df_results = pd.DataFrame(results)
    total_ens = df_results["rev_ensemble"].sum()
    total_lgbm = df_results["rev_lgbm"].sum()
    total_oracle = df_results["rev_oracle"].sum()
    total_vs = (total_ens - total_lgbm) / abs(total_lgbm) * 100
    total_capture = total_ens / total_oracle * 100

    win_days = (df_results["rev_ensemble"] > df_results["rev_lgbm"]).sum()
    lose_days = (df_results["rev_ensemble"] < df_results["rev_lgbm"]).sum()
    tie_days = len(df_results) - win_days - lose_days

    # Max consecutive losing days
    is_losing = (df_results["rev_ensemble"] < df_results["rev_lgbm"]).values
    max_losing_streak = 0
    current_streak = 0
    for x in is_losing:
        if x:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  PAPER TRADING SUMMARY — {province.upper()} ({len(df_results)} days)")
    logger.info(f"{'=' * 70}")
    logger.info(f"  {'':>20} {'Ensemble':>14} {'LightGBM':>14} {'Oracle':>14}")
    logger.info(f"  {'Total Revenue':>20} {total_ens:>14,.0f} {total_lgbm:>14,.0f} {total_oracle:>14,.0f}")
    logger.info(f"  {'Daily Average':>20} {total_ens/len(df_results):>14,.0f} {total_lgbm/len(df_results):>14,.0f} {total_oracle/len(df_results):>14,.0f}")
    logger.info(f"")
    logger.info(f"  Ensemble vs LightGBM:  {total_vs:+.2f}%")
    logger.info(f"  Oracle Capture Rate:   {total_capture:.1f}%")
    logger.info(f"  Win/Lose/Tie:          {win_days}/{lose_days}/{tie_days}")
    logger.info(f"  Max Losing Streak:     {max_losing_streak} days")
    logger.info(f"")

    # Pass/Fail criteria
    pass_count = 0
    checks = [
        (f"Win rate > 50%:          {win_days}/{len(df_results)} = {win_days/len(df_results)*100:.0f}%",
         win_days > len(df_results) / 2),
        (f"Total ENS > LGBM:        {total_vs:+.2f}%",
         total_ens > total_lgbm),
        (f"Max losing streak <= 5:  {max_losing_streak} days",
         max_losing_streak <= 5),
        (f"Oracle capture 40-60%:   {total_capture:.1f}%",
         40 <= total_capture <= 65),
    ]

    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {desc}")
        if passed:
            pass_count += 1

    logger.info(f"\n  Result: {pass_count}/4 checks passed")
    verdict = "READY for live paper trading" if pass_count >= 3 else "NEEDS improvement"
    logger.info(f"  Verdict: {verdict}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"sim_{province}_{sim_days}d.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\n  Results saved: {csv_path}")


if __name__ == "__main__":
    main()
