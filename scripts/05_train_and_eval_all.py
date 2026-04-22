#!/usr/bin/env python3
"""
Step 5: 4省完整训练+评估

每省独立训练：BC → PPO（带价格噪声防过拟合）
多种子评估，输出完整对比表
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import pandas as pd
from loguru import logger

from data.china.features import FEATURE_COLS
from env.battery_env import BatteryEnv
from env.battery_params import BatteryParams
from agent.bc_trainer import train_bc_from_threshold, evaluate_policy
from agent.ppo_trainer import train_ppo
from oracle.lp_oracle import solve_dataset
from config import BatteryConfig

PROCESSED_DIR = ROOT / "data" / "china" / "processed"


def train_and_eval_province(province: str, test_days: int = 365) -> dict:
    """对一个省完整训练+评估"""
    oracle_path = PROCESSED_DIR / f"{province}_oracle.parquet"
    if not oracle_path.exists():
        logger.error(f"{province}: oracle file not found")
        return {}

    logger.info(f"\n{'='*70}")
    logger.info(f"PROVINCE: {province.upper()}")
    logger.info(f"{'='*70}")

    df = pd.read_parquet(oracle_path)
    battery = BatteryParams()
    obs_dim = len(FEATURE_COLS) + 2

    # Train/test split
    test_start = len(df) - 96 * test_days
    if test_start < 96 * 180:  # 至少180天训练
        test_start = len(df) // 2
    df_train = df.iloc[:test_start].copy()
    df_test = df.iloc[test_start:].copy().reset_index(drop=True)

    train_days = len(df_train) // 96
    actual_test_days = len(df_test) // 96
    logger.info(f"  Train: {train_days} days, Test: {actual_test_days} days")

    # Grid search Threshold params
    best_cr, best_dr, best_rev = 0.7, 1.3, -1e18
    env_gs = BatteryEnv(df_train, FEATURE_COLS, "rt_price", battery, randomize_start=False)
    for cr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for dr in [1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]:
            env_gs._start = 0; env_gs._step = 0; env_gs._soc = 0.5
            env_gs._revenue = 0.0; env_gs._cycles = 0.0
            rt_p = df_train["rt_price"].values
            rt_m = df_train["rt_price_ma_96"].values
            for i in range(len(df_train) - 1):
                p = rt_p[i]; m = rt_m[i] if not np.isnan(rt_m[i]) else p
                r = p / max(m, 1.0)
                if r < cr: a = 2
                elif r < cr + 0.15: a = 1
                elif r > dr: a = 4
                elif r > dr - 0.15: a = 3
                else: a = 0
                env_gs.step(a)
            if env_gs._revenue > best_rev:
                best_rev = env_gs._revenue
                best_cr, best_dr = cr, dr
    logger.info(f"  Grid Search: best cr={best_cr}, dr={best_dr}")

    # BC from Threshold
    env_bc = BatteryEnv(df_train, FEATURE_COLS, "rt_price", battery, randomize_start=False)
    bc_policy = train_bc_from_threshold(
        env_bc, df_train, obs_dim,
        charge_ratio=best_cr, discharge_ratio=best_dr,
        n_epochs=20, batch_size=512, lr=1e-3, hidden=[256, 128, 64],
    )

    # Eval BC on test
    env_test_bc = BatteryEnv(df_test, FEATURE_COLS, "rt_price", battery, randomize_start=False)
    bc_result = evaluate_policy(bc_policy, env_test_bc)
    bc_rev = bc_result["revenue"]

    # PPO with price noise (防过拟合)
    ppo_init = copy.deepcopy(bc_policy)
    env_ppo = BatteryEnv(
        df_train, FEATURE_COLS, "rt_price", battery,
        randomize_start=True,
        price_noise_std=0.03,  # 3%价格噪声
    )

    t0 = time.time()
    ppo_policy = train_ppo(
        env_ppo, pretrained_policy=ppo_init, obs_dim=obs_dim,
        n_iterations=500, episodes_per_iter=64, episode_length=96,
        lr_policy=3e-4, lr_value=1e-3,
        clip_epsilon=0.2, entropy_coeff=0.05,
        gae_lambda=0.95, gamma=0.99, mini_epochs=4,
        hidden=[256, 128, 64], log_interval=100,
    )
    ppo_time = time.time() - t0

    # Eval PPO on test (no noise!)
    env_test_ppo = BatteryEnv(df_test, FEATURE_COLS, "rt_price", battery, randomize_start=False)
    ppo_result = evaluate_policy(ppo_policy, env_test_ppo)
    ppo_rev = ppo_result["revenue"]

    # Threshold baseline
    env_test_th = BatteryEnv(df_test, FEATURE_COLS, "rt_price", battery, randomize_start=False)
    env_test_th._start = 0; env_test_th._step = 0; env_test_th._soc = 0.5
    env_test_th._revenue = 0.0; env_test_th._cycles = 0.0
    rt_p = df_test["rt_price"].values; rt_m = df_test["rt_price_ma_96"].values
    for i in range(len(df_test) - 1):
        p = rt_p[i]; m = rt_m[i] if not np.isnan(rt_m[i]) else p
        r = p / max(m, 1.0)
        if r < best_cr: a = 2
        elif r < best_cr + 0.15: a = 1
        elif r > best_dr: a = 4
        elif r > best_dr - 0.15: a = 3
        else: a = 0
        env_test_th.step(a)
    th_rev = env_test_th._revenue

    # Oracle
    _, oracle_rev = solve_dataset(df_test, BatteryConfig(), "rt_price", 0.5)

    # Results
    ppo_vs_th = (ppo_rev - th_rev) / abs(th_rev) * 100 if th_rev != 0 else 0
    bc_vs_th = (bc_rev - th_rev) / abs(th_rev) * 100 if th_rev != 0 else 0

    result = {
        "province": province,
        "train_days": train_days,
        "test_days": actual_test_days,
        "ppo_revenue": ppo_rev,
        "bc_revenue": bc_rev,
        "threshold_revenue": th_rev,
        "oracle_revenue": oracle_rev,
        "ppo_vs_threshold_pct": ppo_vs_th,
        "bc_vs_threshold_pct": bc_vs_th,
        "ppo_pct_oracle": ppo_rev / oracle_rev * 100 if oracle_rev != 0 else 0,
        "ppo_time_s": ppo_time,
        "best_cr": best_cr,
        "best_dr": best_dr,
    }

    logger.info(f"\n  --- {province.upper()} RESULTS ---")
    logger.info(f"  PPO:       {ppo_rev:>14,.0f} 元  ({ppo_vs_th:>+.1f}% vs Threshold)")
    logger.info(f"  BC:        {bc_rev:>14,.0f} 元  ({bc_vs_th:>+.1f}% vs Threshold)")
    logger.info(f"  Threshold: {th_rev:>14,.0f} 元")
    logger.info(f"  Oracle:    {oracle_rev:>14,.0f} 元")
    logger.info(f"  PPO time:  {ppo_time:.0f}s")

    return result


def main():
    provinces = ["shandong", "shanxi", "guangdong", "gansu"]
    all_results = []

    for province in provinces:
        result = train_and_eval_province(province)
        if result:
            all_results.append(result)

    # Final comparison table
    logger.info(f"\n\n{'='*80}")
    logger.info(f"FINAL RESULTS — 4 PROVINCE COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"{'Province':<12s} {'PPO':>14s} {'Threshold':>14s} {'Oracle':>14s} {'PPO vs Th':>10s} {'%Oracle':>8s}")
    logger.info("-" * 76)

    total_ppo = total_th = total_oracle = 0
    for r in all_results:
        logger.info(
            f"{r['province']:<12s} "
            f"{r['ppo_revenue']:>13,.0f}元 "
            f"{r['threshold_revenue']:>13,.0f}元 "
            f"{r['oracle_revenue']:>13,.0f}元 "
            f"{r['ppo_vs_threshold_pct']:>+9.1f}% "
            f"{r['ppo_pct_oracle']:>7.1f}%"
        )
        total_ppo += r["ppo_revenue"]
        total_th += r["threshold_revenue"]
        total_oracle += r["oracle_revenue"]

    logger.info("-" * 76)
    overall_vs = (total_ppo - total_th) / abs(total_th) * 100
    logger.info(
        f"{'TOTAL':<12s} "
        f"{total_ppo:>13,.0f}元 "
        f"{total_th:>13,.0f}元 "
        f"{total_oracle:>13,.0f}元 "
        f"{overall_vs:>+9.1f}% "
        f"{total_ppo/total_oracle*100:>7.1f}%"
    )

    # 年化
    logger.info(f"\n  Annualized (per 200MW/400MWh station):")
    for r in all_results:
        ann_ppo = r["ppo_revenue"] / r["test_days"] * 365
        ann_th = r["threshold_revenue"] / r["test_days"] * 365
        extra = ann_ppo - ann_th
        logger.info(
            f"    {r['province']:<12s}: PPO {ann_ppo:>10,.0f}元/年, "
            f"Threshold {ann_th:>10,.0f}元/年, "
            f"多赚 {extra:>+10,.0f}元/年"
        )

    # 保存
    pd.DataFrame(all_results).to_csv(PROCESSED_DIR / "all_provinces_results.csv", index=False)
    logger.info(f"\n  Saved: {PROCESSED_DIR / 'all_provinces_results.csv'}")


if __name__ == "__main__":
    main()
