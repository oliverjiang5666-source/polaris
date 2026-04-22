"""
DA Price vs LightGBM vs Perfect Forecast — Value of Prediction Experiment

Answers: how much value does AI price prediction actually add over
the free day-ahead price that every market participant already has?

Three forecast sources, all fed into MPC (LP + continuous power + battery sim):
  1. DA forecast:      da_price[t+1 : t+97] (published day-ahead, free)
  2. LightGBM forecast: trained multi-horizon predictor
  3. Perfect forecast:  rt_price[t+1 : t+97] (theoretical upper bound)

Plus Threshold baseline for reference.

Usage:
    PYTHONPATH=. python3 scripts/10_da_vs_lgbm.py
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from data.china.features import FEATURE_COLS
from forecast.lgbm_forecaster import LGBMForecaster
from forecast.mpc_controller import (
    MPCController, simulate_mpc, simulate_threshold,
)

PROCESSED_DIR = Path("data/china/processed")
PROVINCES = ["shandong", "shanxi", "guangdong", "gansu"]
PROVINCE_CN = {"shandong": "山东", "shanxi": "山西", "guangdong": "广东", "gansu": "甘肃"}


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


def run_province(province: str, test_days: int = 365) -> dict:
    battery = BatteryConfig()
    df_train, df_test = load_and_split(province, test_days)

    features_train = df_train[FEATURE_COLS].fillna(0).values.astype(np.float32)
    features_test = df_test[FEATURE_COLS].fillna(0).values.astype(np.float32)
    prices_train = df_train["rt_price"].fillna(0).values.astype(np.float32)
    prices_test = df_test["rt_price"].fillna(0).values.astype(np.float32)
    da_prices_test = df_test["da_price"].fillna(0).values.astype(np.float32)
    full_prices = np.concatenate([prices_train, prices_test])
    full_da = np.concatenate([
        df_train["da_price"].fillna(0).values.astype(np.float32),
        da_prices_test,
    ])
    test_offset = len(prices_train)

    # Daily profile from training data
    n_train_days = len(prices_train) // 96
    daily_prices = prices_train[:n_train_days * 96].reshape(n_train_days, 96)
    daily_profile = daily_prices.mean(axis=0)

    # DA price quality check
    da_zero_pct = (da_prices_test == 0).sum() / len(da_prices_test) * 100
    logger.info(f"  DA price zeros in test: {da_zero_pct:.1f}%")

    # ---- 1. Threshold baseline ----
    logger.info("--- Threshold ---")
    ma96_test = df_test["rt_price_ma_96"].fillna(
        df_test["rt_price"].rolling(96, min_periods=1).mean()
    ).values
    best_cr, best_dr = grid_search_threshold(
        df_train["rt_price"].values,
        df_train["rt_price_ma_96"].fillna(
            df_train["rt_price"].rolling(96, min_periods=1).mean()
        ).values,
        battery,
    )
    th = simulate_threshold(prices_test, ma96_test, battery, best_cr, best_dr)
    logger.info(f"  Revenue: {th['revenue']:,.0f}")

    # ---- 2. DA Price MPC ----
    logger.info("--- DA Price MPC ---")

    class DAForecaster:
        """Use day-ahead prices as forecast. Fill zeros with rt_price."""
        def predict(self, features_t, idx, horizon=96):
            forecast = np.zeros(horizon)
            for h in range(horizon):
                global_idx = test_offset + idx + h + 1
                if global_idx < len(full_da):
                    val = full_da[global_idx]
                    if val == 0 and global_idx < len(full_prices):
                        # DA=0 likely means missing; use last known rt_price
                        val = full_prices[min(test_offset + idx, len(full_prices) - 1)]
                    forecast[h] = val
                else:
                    forecast[h] = full_da[min(global_idx, len(full_da) - 1)]
            return forecast

    t0 = time.time()
    da_ctrl = MPCController(DAForecaster(), battery, continuous=True, daily_profile=daily_profile)
    da_result = simulate_mpc(da_ctrl, features_test, prices_test, battery,
                             replan_every=4, log_every=90)
    da_time = time.time() - t0
    logger.info(f"  Revenue: {da_result['revenue']:,.0f}  ({da_time:.0f}s)")

    # ---- 3. LightGBM MPC ----
    logger.info("--- LightGBM MPC ---")
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
    lgbm_time = time.time() - t0
    logger.info(f"  Revenue: {lgbm_result['revenue']:,.0f}  ({lgbm_time:.0f}s)")

    # ---- 4. Perfect MPC ----
    logger.info("--- Perfect MPC ---")

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
    perf_result = simulate_mpc(perf_ctrl, features_test, prices_test, battery, replan_every=4)
    perf_time = time.time() - t0
    logger.info(f"  Revenue: {perf_result['revenue']:,.0f}  ({perf_time:.0f}s)")

    # ---- SUMMARY ----
    test_d = len(prices_test) // 96
    ann = lambda r: r / test_d * 365

    th_rev = th["revenue"]
    da_rev = da_result["revenue"]
    lgbm_rev = lgbm_result["revenue"]
    perf_rev = perf_result["revenue"]

    logger.info(f"\n{'='*90}")
    logger.info(f"  {province.upper()} ({PROVINCE_CN[province]}) — {test_d} test days")
    logger.info(f"{'='*90}")
    logger.info(f"  {'Method':<25} {'Test Revenue':>14} {'Annual (万)':>12} {'vs Threshold':>12} {'vs DA MPC':>12}")
    logger.info(f"  {'-'*25} {'-'*14} {'-'*12} {'-'*12} {'-'*12}")

    rows = [
        ("Threshold", th_rev),
        ("DA Price MPC", da_rev),
        ("LightGBM MPC", lgbm_rev),
        ("Perfect MPC", perf_rev),
    ]

    for name, rev in rows:
        vs_th = (rev - th_rev) / abs(th_rev) * 100 if th_rev != 0 else 0
        vs_da = (rev - da_rev) / abs(da_rev) * 100 if da_rev != 0 else 0
        annual_wan = ann(rev) / 1e4
        logger.info(f"  {name:<25} {rev:>14,.0f} {annual_wan:>12,.0f} {vs_th:>+11.1f}% {vs_da:>+11.1f}%")

    # Value decomposition
    logger.info(f"\n  --- Value Decomposition ---")
    total_gap = perf_rev - th_rev
    da_value = da_rev - th_rev
    lgbm_incremental = lgbm_rev - da_rev
    remaining_gap = perf_rev - lgbm_rev

    logger.info(f"  Total gap (Perfect - Threshold):   {total_gap:>12,.0f}  (100%)")
    if total_gap != 0:
        logger.info(f"  DA Price contribution:             {da_value:>12,.0f}  ({da_value/total_gap*100:>5.1f}%)")
        logger.info(f"  LightGBM incremental over DA:      {lgbm_incremental:>12,.0f}  ({lgbm_incremental/total_gap*100:>5.1f}%)")
        logger.info(f"  Remaining gap (LightGBM→Perfect):  {remaining_gap:>12,.0f}  ({remaining_gap/total_gap*100:>5.1f}%)")

    return {
        "province": province,
        "test_days": test_d,
        "threshold": th_rev,
        "da_mpc": da_rev,
        "lgbm_mpc": lgbm_rev,
        "perfect_mpc": perf_rev,
        "da_vs_th_pct": (da_rev - th_rev) / abs(th_rev) * 100,
        "lgbm_vs_th_pct": (lgbm_rev - th_rev) / abs(th_rev) * 100,
        "lgbm_vs_da_pct": (lgbm_rev - da_rev) / abs(da_rev) * 100 if da_rev != 0 else 0,
        "da_zero_pct": da_zero_pct,
    }


def main():
    results = []
    for prov in PROVINCES:
        logger.info(f"\n{'#'*90}\n# {prov.upper()} ({PROVINCE_CN[prov]})\n{'#'*90}")
        r = run_province(prov)
        results.append(r)

    # ---- CROSS-PROVINCE SUMMARY ----
    logger.info(f"\n\n{'='*100}")
    logger.info(f"  CROSS-PROVINCE SUMMARY — DA vs LightGBM vs Perfect")
    logger.info(f"{'='*100}")

    ann = lambda r, d: r / d * 365

    logger.info(f"\n  {'Province':<10} {'Threshold':>12} {'DA MPC':>12} {'LGBM MPC':>12} {'Perfect':>12} "
                f"{'DA vs TH':>10} {'LGBM vs TH':>10} {'LGBM vs DA':>10}")
    logger.info(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        prov = r["province"]
        d = r["test_days"]
        th_a = ann(r["threshold"], d) / 1e4
        da_a = ann(r["da_mpc"], d) / 1e4
        lg_a = ann(r["lgbm_mpc"], d) / 1e4
        pf_a = ann(r["perfect_mpc"], d) / 1e4
        logger.info(f"  {PROVINCE_CN[prov]:<10} {th_a:>10,.0f}万 {da_a:>10,.0f}万 {lg_a:>10,.0f}万 {pf_a:>10,.0f}万 "
                    f"{r['da_vs_th_pct']:>+9.1f}% {r['lgbm_vs_th_pct']:>+9.1f}% {r['lgbm_vs_da_pct']:>+9.1f}%")

    logger.info(f"\n  --- KEY QUESTION: How much does AI prediction add over DA price? ---")
    for r in results:
        prov = PROVINCE_CN[r["province"]]
        d = r["test_days"]
        da_annual = ann(r["da_mpc"], d) / 1e4
        lgbm_annual = ann(r["lgbm_mpc"], d) / 1e4
        diff = lgbm_annual - da_annual
        logger.info(f"  {prov}: LightGBM - DA = {diff:+,.0f}万/年 ({r['lgbm_vs_da_pct']:+.1f}%)")

    # Save results
    df = pd.DataFrame(results)
    out = PROCESSED_DIR / "da_vs_lgbm_results.csv"
    df.to_csv(out, index=False)
    logger.info(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
