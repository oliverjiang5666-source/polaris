"""
MPC Evaluation Pipeline — Price Forecast + LP Planning

Compares:
  - Threshold (discrete, rule-based)
  - Naive MPC (yesterday same hour, continuous LP)
  - LightGBM MPC (learned forecast, continuous LP)
  - Perfect MPC (oracle prices, continuous LP)
  - Oracle discrete (LP + quantization, daily reset)
  - Oracle continuous (LP, daily reset — theoretical upper bound)

Usage:
    PYTHONPATH=. python3 scripts/09_mpc_eval.py [--province shandong] [--test-days 365]
    PYTHONPATH=. python3 scripts/09_mpc_eval.py --all
"""
from __future__ import annotations

import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from data.china.features import FEATURE_COLS
from forecast.lgbm_forecaster import LGBMForecaster
from forecast.mpc_controller import (
    MPCController, simulate_mpc, simulate_threshold,
    simulate_oracle_continuous, simulate_oracle_discrete,
)

PROCESSED_DIR = Path("data/china/processed")


def load_and_split(province: str, test_days: int = 365):
    path = PROCESSED_DIR / f"{province}_oracle.parquet"
    df = pd.read_parquet(path)
    logger.info(f"Loaded {province}: {len(df):,} rows")

    test_start = len(df) - 96 * test_days
    if test_start < 96 * 180:
        test_start = len(df) // 2

    df_train = df.iloc[:test_start]
    df_test = df.iloc[test_start:].reset_index(drop=True)
    logger.info(f"  Train: {len(df_train)//96} days | Test: {len(df_test)//96} days")
    return df_train, df_test


def grid_search_threshold(prices, ma96, battery):
    best_rev, best_cr, best_dr = -np.inf, 0.65, 1.35
    for cr in np.arange(0.50, 0.86, 0.05):
        for dr in np.arange(1.15, 1.56, 0.05):
            r = simulate_threshold(prices, ma96, battery, cr, dr)["revenue"]
            if r > best_rev:
                best_rev, best_cr, best_dr = r, cr, dr
    return best_cr, best_dr


def run_province(province: str, test_days: int = 365):
    battery = BatteryConfig()
    df_train, df_test = load_and_split(province, test_days)

    features_train = df_train[FEATURE_COLS].fillna(0).values.astype(np.float32)
    features_test = df_test[FEATURE_COLS].fillna(0).values.astype(np.float32)
    prices_train = df_train["rt_price"].fillna(0).values.astype(np.float32)
    prices_test = df_test["rt_price"].fillna(0).values.astype(np.float32)
    full_prices = np.concatenate([prices_train, prices_test])
    test_offset = len(prices_train)

    # Compute average daily price profile from training data (96 values).
    # This gives the MPC a realistic "tomorrow" for terminal value.
    n_train_days = len(prices_train) // 96
    daily_prices = prices_train[:n_train_days * 96].reshape(n_train_days, 96)
    daily_profile = daily_prices.mean(axis=0)
    logger.info(f"  Daily profile: min={daily_profile.min():.0f}, max={daily_profile.max():.0f}, "
                f"mean={daily_profile.mean():.0f}, spread={daily_profile.max()-daily_profile.min():.0f}")

    # ---- 1. Threshold ----
    logger.info("\n--- Threshold ---")
    ma96_test = df_test["rt_price_ma_96"].fillna(
        df_test["rt_price"].rolling(96, min_periods=1).mean()
    ).values
    best_cr, best_dr = grid_search_threshold(
        df_train["rt_price"].values,
        df_train["rt_price_ma_96"].fillna(df_train["rt_price"].rolling(96, min_periods=1).mean()).values,
        battery,
    )
    th = simulate_threshold(prices_test, ma96_test, battery, best_cr, best_dr)
    logger.info(f"  Revenue: {th['revenue']:,.0f} (CR={best_cr:.2f}, DR={best_dr:.2f})")

    # ---- 2. Oracle (continuous + discrete) ----
    logger.info("\n--- Oracle ---")
    oracle_cont = simulate_oracle_continuous(prices_test, battery)
    oracle_disc = simulate_oracle_discrete(prices_test, battery)
    logger.info(f"  Continuous: {oracle_cont['revenue']:,.0f}")
    logger.info(f"  Discrete:   {oracle_disc['revenue']:,.0f}  ({oracle_disc['revenue']/oracle_cont['revenue']*100:.1f}% of continuous)")

    # ---- 3. Perfect MPC (continuous, replans every step) ----
    logger.info("\n--- Perfect MPC (continuous) ---")

    class PerfectForecaster:
        def predict(self, features_t, idx, horizon=96):
            start = idx + 1
            end = min(start + horizon, len(prices_test))
            f = prices_test[start:end]
            if len(f) < horizon:
                f = np.pad(f, (0, horizon - len(f)), constant_values=f[-1] if len(f) > 0 else 300)
            return f

    t0 = time.time()
    perf_ctrl = MPCController(PerfectForecaster(), battery, continuous=True, daily_profile=daily_profile)
    perf = simulate_mpc(perf_ctrl, features_test, prices_test, battery, replan_every=4)
    logger.info(f"  Revenue: {perf['revenue']:,.0f}  ({time.time()-t0:.0f}s)")
    logger.info(f"  vs Oracle continuous: {perf['revenue']/oracle_cont['revenue']*100:.1f}%")

    # ---- 4. Naive MPC (yesterday same hour, continuous) ----
    logger.info("\n--- Naive MPC (yesterday same hour) ---")

    class NaiveForecaster:
        def predict(self, features_t, idx, horizon=96):
            f = np.zeros(horizon)
            for h in range(horizon):
                past = test_offset + idx + h + 1 - 96
                f[h] = full_prices[past] if 0 <= past < len(full_prices) else full_prices[test_offset + idx]
            return f

    t0 = time.time()
    naive_ctrl = MPCController(NaiveForecaster(), battery, continuous=True, daily_profile=daily_profile)
    naive = simulate_mpc(naive_ctrl, features_test, prices_test, battery, replan_every=4)
    logger.info(f"  Revenue: {naive['revenue']:,.0f}  ({time.time()-t0:.0f}s)")

    # ---- 5. LightGBM MPC ----
    logger.info("\n--- LightGBM MPC ---")
    logger.info("Training forecaster...")
    t0 = time.time()
    lgbm = LGBMForecaster(n_estimators=300, max_depth=6, learning_rate=0.05)
    lgbm.fit(features_train, prices_train)
    train_time = time.time() - t0
    logger.info(f"  Training: {train_time:.1f}s")

    lgbm._full_prices = full_prices

    class LGBMWrapper:
        def predict(self, features_t, idx, horizon=96):
            return lgbm.predict(features_t, test_offset + idx, horizon)

    t0 = time.time()
    lgbm_ctrl = MPCController(LGBMWrapper(), battery, continuous=True, daily_profile=daily_profile)
    lgbm_result = simulate_mpc(lgbm_ctrl, features_test, prices_test, battery,
                               replan_every=4, log_every=90)
    eval_time = time.time() - t0
    logger.info(f"  Revenue: {lgbm_result['revenue']:,.0f}  ({eval_time:.0f}s)")

    lgbm_soc = {"revenue": lgbm_result["revenue"]}  # same as continuous for now
    perf_soc = {"revenue": perf["revenue"]}

    # ---- SUMMARY ----
    test_d = len(prices_test) // 96
    ann = lambda r: r / test_d * 365

    logger.info(f"\n{'='*80}")
    logger.info(f"  {province.upper()} — {test_d} test days")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Method':<30} {'Revenue':>14} {'Annual':>14} {'vs TH':>10} {'% Oracle*':>10}")
    logger.info(f"  {'-'*30} {'-'*14} {'-'*14} {'-'*10} {'-'*10}")

    # Use discrete oracle as the "achievable upper bound"
    rows = [
        ("Threshold (discrete)", th["revenue"]),
        ("Naive MPC (YSH)", naive["revenue"]),
        ("LightGBM MPC", lgbm_result["revenue"]),
        ("Perfect MPC", perf["revenue"]),
        ("Oracle discrete*", oracle_disc["revenue"]),
        ("Oracle continuous", oracle_cont["revenue"]),
    ]

    for name, rev in rows:
        vs_th = (rev - th["revenue"]) / abs(th["revenue"]) * 100 if th["revenue"] != 0 else 0
        pct_orc = rev / oracle_disc["revenue"] * 100 if oracle_disc["revenue"] != 0 else 0
        logger.info(f"  {name:<30} {rev:>14,.0f} {ann(rev):>14,.0f} {vs_th:>+9.1f}% {pct_orc:>9.1f}%")

    logger.info(f"\n  * Oracle discrete = achievable upper bound with 5-action space")
    logger.info(f"    Oracle continuous = theoretical upper bound (LP solution)")

    return {
        "province": province,
        "test_days": test_d,
        "threshold": th["revenue"],
        "naive_mpc": naive["revenue"],
        "lgbm_mpc_cont": lgbm_result["revenue"],
        "lgbm_mpc_soc": lgbm_soc["revenue"],
        "perfect_mpc": perf["revenue"],
        "perfect_mpc_soc": perf_soc["revenue"],
        "oracle_disc": oracle_disc["revenue"],
        "oracle_cont": oracle_cont["revenue"],
        "lgbm_soc_vs_th_pct": (lgbm_soc["revenue"] - th["revenue"]) / abs(th["revenue"]) * 100,
        "lgbm_soc_pct_oracle": lgbm_soc["revenue"] / oracle_disc["revenue"] * 100,
        "best_cr": best_cr,
        "best_dr": best_dr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="shandong")
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    provinces = ["shandong", "shanxi", "guangdong", "gansu"] if args.all else [args.province]
    results = []

    for prov in provinces:
        logger.info(f"\n{'#'*80}\n# {prov.upper()}\n{'#'*80}")
        r = run_province(prov, args.test_days)
        results.append(r)

    if len(results) > 1:
        df = pd.DataFrame(results)
        out = PROCESSED_DIR / "mpc_results.csv"
        df.to_csv(out, index=False)
        logger.info(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
