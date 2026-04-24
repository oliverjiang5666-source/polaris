"""
Regime Strategy Backtest: 365-day comparison.

Compares zone/regime strategy vs V5 ensemble vs LightGBM vs Oracle.

Usage:
    PYTHONPATH=. python3 scripts/20_regime_backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from forecast.mpc_controller import _step_battery
from oracle.lp_oracle import solve_day

PROCESSED_DIR = Path("data/china/processed")
MODEL_DIR = Path("models")
PROVINCE = "shandong"


def main():
    battery = BatteryConfig()

    logger.info("=" * 70)
    logger.info("  REGIME STRATEGY BACKTEST — SHANDONG")
    logger.info("=" * 70)

    # Load regime table
    table = np.load(MODEL_DIR / f"regime_table_{PROVINCE}.npz")
    best_power_idx = table["best_power_idx"]   # (6, 96, 20)
    power_levels = table["power_levels"]        # (101,)
    soc_levels = table["soc_levels"]            # (20,)
    regime_profiles = table["regime_profiles"]  # (6, 96)
    price_matrix = table["price_matrix"]        # (n_days, 96)
    all_labels = table["labels"]                # (n_days,)
    n_days = len(price_matrix)

    # Load predictions
    preds = np.load(MODEL_DIR / f"regime_predictions_{PROVINCE}.npz")
    y_test = preds["y_test"]
    y_pred = preds["y_pred"]
    test_start_day = int(preds["test_start_day"])
    test_days = len(y_test)

    n_soc = len(soc_levels)
    soc_min = soc_levels[0]
    soc_max = soc_levels[-1]

    logger.info(f"Test period: day {test_start_day+1} to {n_days} ({test_days} days)")
    logger.info(f"Power levels: {len(power_levels)} ({power_levels[0]:.0f} to {power_levels[-1]:.0f} MW)")
    logger.info(f"SOC levels: {n_soc} ({soc_min:.0%} to {soc_max:.0%})")
    logger.info(f"Classification accuracy: {(y_pred == y_test).sum()}/{test_days} = {(y_pred == y_test).mean():.1%}")

    # ================================================================
    # Simulate each day
    # ================================================================
    results = []

    for i in range(test_days):
        d = test_start_day + i
        day_prices = price_matrix[d]
        predicted_regime = y_pred[i]
        actual_regime = y_test[i]
        date_str = f"Day {d+1}"

        # --- Strategy 1: Regime strategy (predicted) ---
        soc = 0.5
        rev_regime = 0.0
        for t in range(96):
            # Find nearest SOC index
            s_idx = int(np.clip(np.round((soc - soc_min) / (soc_max - soc_min) * (n_soc - 1)), 0, n_soc - 1))
            p_idx = best_power_idx[predicted_regime][t][s_idx]
            power_mw = power_levels[p_idx]
            soc, net_rev, _ = _step_battery(power_mw, day_prices[t], soc, battery)
            rev_regime += net_rev

        # --- Strategy 2: Regime strategy (perfect classification) ---
        soc = 0.5
        rev_perfect = 0.0
        for t in range(96):
            s_idx = int(np.clip(np.round((soc - soc_min) / (soc_max - soc_min) * (n_soc - 1)), 0, n_soc - 1))
            p_idx = best_power_idx[actual_regime][t][s_idx]
            power_mw = power_levels[p_idx]
            soc, net_rev, _ = _step_battery(power_mw, day_prices[t], soc, battery)
            rev_perfect += net_rev

        # --- Strategy 3: Oracle ---
        oracle_result = solve_day(day_prices, battery, init_soc=0.5)
        rev_oracle = oracle_result["revenue"]

        results.append({
            "day": d + 1,
            "pred_regime": predicted_regime,
            "actual_regime": actual_regime,
            "correct": predicted_regime == actual_regime,
            "rev_regime": rev_regime,
            "rev_perfect": rev_perfect,
            "rev_oracle": rev_oracle,
        })

        if (i + 1) % 30 == 0:
            logger.info(f"  Processed {i+1}/{test_days} days...")

    df_r = pd.DataFrame(results)

    # ================================================================
    # Load V5 paper trading results for comparison
    # ================================================================
    v5_path = Path("data/paper_trading/sim_shandong_30d.csv")
    has_v5 = v5_path.exists()

    # ================================================================
    # Summary
    # ================================================================
    total_regime = df_r["rev_regime"].sum()
    total_perfect = df_r["rev_perfect"].sum()
    total_oracle = df_r["rev_oracle"].sum()

    capture_regime = total_regime / total_oracle * 100
    capture_perfect = total_perfect / total_oracle * 100

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  BACKTEST RESULTS — {PROVINCE.upper()} ({test_days} days)")
    logger.info(f"{'=' * 70}")

    logger.info(f"\n  {'Method':<35} {'Total Revenue':>14} {'Daily Avg':>12} {'Capture':>8}")
    logger.info(f"  {'-'*35} {'-'*14} {'-'*12} {'-'*8}")
    logger.info(f"  {'Oracle (per-day optimal)':<35} {total_oracle:>14,.0f} {total_oracle/test_days:>12,.0f} {'100.0%':>8}")
    logger.info(f"  {'Regime (perfect classification)':<35} {total_perfect:>14,.0f} {total_perfect/test_days:>12,.0f} {capture_perfect:>7.1f}%")
    logger.info(f"  {'Regime (predicted, day-ahead)':<35} {total_regime:>14,.0f} {total_regime/test_days:>12,.0f} {capture_regime:>7.1f}%")

    # Win rate: days where regime > 0
    positive_days = (df_r["rev_regime"] > 0).sum()
    logger.info(f"\n  Profitable days: {positive_days}/{test_days} ({positive_days/test_days:.1%})")

    # Classification impact
    correct_mask = df_r["correct"]
    wrong_mask = ~df_r["correct"]
    if correct_mask.sum() > 0:
        avg_correct = df_r.loc[correct_mask, "rev_regime"].mean()
        avg_wrong = df_r.loc[wrong_mask, "rev_regime"].mean() if wrong_mask.sum() > 0 else 0
        logger.info(f"\n  When classification is CORRECT ({correct_mask.sum()} days): avg revenue = {avg_correct:,.0f}/day")
        logger.info(f"  When classification is WRONG   ({wrong_mask.sum()} days): avg revenue = {avg_wrong:,.0f}/day")

    # Per-regime breakdown
    logger.info(f"\n  Per-regime performance:")
    logger.info(f"  {'Regime':>8} {'Days':>6} {'Accuracy':>10} {'Avg Rev':>12} {'Avg Oracle':>12} {'Capture':>8}")
    for c in range(best_power_idx.shape[0]):
        mask = df_r["actual_regime"] == c
        if mask.sum() == 0:
            continue
        n = mask.sum()
        acc = (df_r.loc[mask, "correct"]).mean()
        avg_rev = df_r.loc[mask, "rev_regime"].mean()
        avg_oracle = df_r.loc[mask, "rev_oracle"].mean()
        cap = avg_rev / avg_oracle * 100 if avg_oracle > 0 else 0
        logger.info(f"  {c:>8} {n:>6} {acc:>9.1%} {avg_rev:>12,.0f} {avg_oracle:>12,.0f} {cap:>7.1f}%")

    # Comparison with V5 ensemble (from our earlier experiments)
    # V5 ensemble mountain: 44,682,220 (365 days), daily avg ~122,417
    # V5 ensemble modular test: 4,382,636 (30 days), daily avg 146,088
    v5_annual_rev = 44_682_220  # from V5 results
    lgbm_annual_rev = 44_159_515  # from V5 results

    logger.info(f"\n  {'=' * 50}")
    logger.info(f"  COMPARISON WITH PREVIOUS METHODS")
    logger.info(f"  {'=' * 50}")
    logger.info(f"  {'Method':<35} {'Annual Revenue':>14} {'Capture':>8}")
    logger.info(f"  {'-'*35} {'-'*14} {'-'*8}")
    logger.info(f"  {'Oracle':<35} {total_oracle:>14,.0f} {'100.0%':>8}")
    logger.info(f"  {'Regime (predicted)':<35} {total_regime:>14,.0f} {capture_regime:>7.1f}%")
    logger.info(f"  {'V5 Ensemble (LGBM+PatchTST)':<35} {v5_annual_rev:>14,.0f} {v5_annual_rev/total_oracle*100:>7.1f}%")
    logger.info(f"  {'Pure LightGBM':<35} {lgbm_annual_rev:>14,.0f} {lgbm_annual_rev/total_oracle*100:>7.1f}%")

    regime_vs_v5 = (total_regime - v5_annual_rev) / abs(v5_annual_rev) * 100
    regime_vs_lgbm = (total_regime - lgbm_annual_rev) / abs(lgbm_annual_rev) * 100
    logger.info(f"\n  Regime vs V5 Ensemble: {regime_vs_v5:+.1f}%")
    logger.info(f"  Regime vs Pure LightGBM: {regime_vs_lgbm:+.1f}%")

    verdict = "REGIME WINS" if total_regime > v5_annual_rev else "V5 ENSEMBLE STILL BETTER"
    logger.info(f"\n  Verdict: {verdict}")

    # Save results
    output_path = PROCESSED_DIR / f"regime_backtest_{PROVINCE}.csv"
    df_r.to_csv(output_path, index=False)
    logger.info(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
