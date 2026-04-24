#!/usr/bin/env python3
"""
Step 4: Walk-Forward回测

对每个省份执行：
1. 生成walk-forward窗口
2. 每个窗口：LP Oracle → BC训练 → 测试
3. 对比：BC-Oracle vs Threshold vs Oracle上界
4. 输出结果表
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config import BatteryConfig, BCConfig, BacktestConfig, ThresholdSearchConfig
from data.china.features import FEATURE_COLS
from env.battery_env import BatteryEnv, ACTION_POWER_RATIOS
from env.battery_params import BatteryParams
from oracle.lp_oracle import solve_dataset
from agent.bc_trainer import train_bc_from_oracle, train_bc_from_threshold, evaluate_policy
from backtest.walk_forward import generate_windows

PROCESSED_DIR = ROOT / "data" / "china" / "processed"


def run_threshold_baseline(
    env: BatteryEnv,
    df_test: pd.DataFrame,
    charge_ratio: float = 0.70,
    discharge_ratio: float = 1.30,
) -> dict:
    """运行Threshold基线策略"""
    rt_price = df_test["rt_price"].values
    rt_ma_96 = df_test["rt_price_ma_96"].values

    env._start = 0
    env._step = 0
    env._soc = 0.5
    env._revenue = 0.0
    env._cycles = 0.0

    for i in range(len(df_test) - 1):
        price = rt_price[i]
        ma96 = rt_ma_96[i] if not np.isnan(rt_ma_96[i]) else price
        ratio = price / max(ma96, 1.0)

        if ratio < charge_ratio:
            action = 2
        elif ratio < charge_ratio + 0.15:
            action = 1
        elif ratio > discharge_ratio:
            action = 4
        elif ratio > discharge_ratio - 0.15:
            action = 3
        else:
            action = 0

        env.step(action)

    return {"revenue": env._revenue, "cycles": env._cycles}


def run_oracle_upper_bound(df_test: pd.DataFrame, battery: BatteryParams) -> float:
    """计算Oracle上界（用测试集数据，完美后见之明）"""
    _, oracle_revenue = solve_dataset(
        df_test, BatteryConfig(), price_col="rt_price", init_soc=0.5
    )
    return oracle_revenue


def backtest_province(province: str):
    """对一个省份执行完整的walk-forward回测"""
    bc_cfg = BCConfig()
    bt_cfg = BacktestConfig()

    # 加载数据
    oracle_path = PROCESSED_DIR / f"{province}_oracle.parquet"
    if not oracle_path.exists():
        logger.error(f"{oracle_path} not found. Run 01_solve_oracle.py first.")
        return

    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTEST: {province.upper()}")
    logger.info(f"{'='*70}")

    df = pd.read_parquet(oracle_path)
    logger.info(f"Loaded: {len(df):,} rows")

    # 电池参数
    battery = BatteryParams()

    # 生成walk-forward窗口
    windows = generate_windows(
        n_total=len(df),
        min_train_days=bt_cfg.min_train_days,
        test_days=bt_cfg.test_days,
        stride_days=bt_cfg.stride_days,
        expanding=bt_cfg.expanding_window,
    )

    if not windows:
        logger.warning("No valid windows generated!")
        return

    results = []
    obs_dim = len(FEATURE_COLS) + 2  # 31 features + SOC + revenue_norm = 33
    ts_cfg = ThresholdSearchConfig()

    for w in windows:
        df_train = df.iloc[w.train_start:w.train_end].copy()
        df_test = df.iloc[w.test_start:w.test_end].copy()

        train_days = len(df_train) // 96
        test_days_n = len(df_test) // 96

        logger.info(f"\n--- Window {w.window_id}: train={train_days}d, test={test_days_n}d ---")

        # Step 1: Grid Search最优Threshold参数
        best_cr, best_dr, best_rev = 0.7, 1.3, -1e18
        env_gs = BatteryEnv(df_train, FEATURE_COLS, "rt_price", battery, randomize_start=False)
        for cr in ts_cfg.charge_ratios:
            for dr in ts_cfg.discharge_ratios:
                res = run_threshold_baseline(env_gs, df_train, cr, dr)
                if res["revenue"] > best_rev:
                    best_rev = res["revenue"]
                    best_cr, best_dr = cr, dr
        logger.info(f"  Grid Search: best cr={best_cr}, dr={best_dr}, rev={best_rev:,.0f}")

        # Step 2: BC from Oracle（修复版：无多SOC增广，温和class weights）
        oracle_actions_train = df_train["oracle_action"].values
        env_train = BatteryEnv(df_train, FEATURE_COLS, "rt_price", battery, randomize_start=False)
        t0 = time.time()
        policy = train_bc_from_oracle(
            env_train, oracle_actions_train, obs_dim,
            init_socs=[0.5],  # 只用0.5匹配Oracle
            n_epochs=bc_cfg.epochs,
            batch_size=bc_cfg.batch_size,
            lr=bc_cfg.lr,
            hidden=bc_cfg.hidden_dims,
            use_layer_norm=bc_cfg.use_layer_norm,
        )
        train_time = time.time() - t0

        # 测试：BC-Oracle策略
        df_test_reset = df_test.reset_index(drop=True)
        env_test = BatteryEnv(df_test_reset, FEATURE_COLS, "rt_price", battery, randomize_start=False)
        bc_result = evaluate_policy(policy, env_test)

        # 测试：Threshold基线（用最优参数）
        env_test2 = BatteryEnv(df_test_reset, FEATURE_COLS, "rt_price", battery, randomize_start=False)
        thresh_result = run_threshold_baseline(env_test2, df_test_reset, best_cr, best_dr)

        # 测试：Oracle上界
        oracle_rev = run_oracle_upper_bound(df_test_reset, battery)

        # 记录
        bc_rev = bc_result["revenue"]
        th_rev = thresh_result["revenue"]
        bc_vs_th = (bc_rev - th_rev) / abs(th_rev) * 100 if th_rev != 0 else 0
        bc_vs_oracle = bc_rev / oracle_rev * 100 if oracle_rev != 0 else 0

        results.append({
            "window": w.window_id,
            "train_days": train_days,
            "test_days": test_days_n,
            "bc_revenue": bc_rev,
            "threshold_revenue": th_rev,
            "oracle_revenue": oracle_rev,
            "bc_vs_threshold_pct": bc_vs_th,
            "bc_pct_of_oracle": bc_vs_oracle,
            "train_time_s": train_time,
            "bc_win": bc_rev > th_rev,
            "best_cr": best_cr,
            "best_dr": best_dr,
        })

        logger.info(
            f"  BC: {bc_rev:>12,.0f}元  "
            f"Threshold: {th_rev:>12,.0f}元  "
            f"Oracle: {oracle_rev:>12,.0f}元  "
            f"BC vs Th: {bc_vs_th:>+.1f}%  "
            f"{'✅' if bc_rev > th_rev else '❌'}"
        )

    # 汇总
    results_df = pd.DataFrame(results)
    total_bc = results_df["bc_revenue"].sum()
    total_th = results_df["threshold_revenue"].sum()
    total_oracle = results_df["oracle_revenue"].sum()
    win_rate = results_df["bc_win"].mean() * 100
    overall_vs_th = (total_bc - total_th) / abs(total_th) * 100 if total_th != 0 else 0

    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: {province.upper()}")
    logger.info(f"{'='*70}")
    logger.info(f"  Windows:        {len(results)}")
    logger.info(f"  BC Total:       {total_bc:>14,.0f} 元")
    logger.info(f"  Threshold Total:{total_th:>14,.0f} 元")
    logger.info(f"  Oracle Total:   {total_oracle:>14,.0f} 元")
    logger.info(f"  BC vs Threshold:{overall_vs_th:>+.1f}%")
    logger.info(f"  BC % of Oracle: {total_bc/total_oracle*100:.1f}%")
    logger.info(f"  BC Win Rate:    {win_rate:.0f}%")

    # 年化
    total_test_days = results_df["test_days"].sum()
    bc_annual = total_bc / total_test_days * 365
    th_annual = total_th / total_test_days * 365
    oracle_annual = total_oracle / total_test_days * 365
    logger.info(f"\n  Annualized:")
    logger.info(f"    BC:        {bc_annual:>14,.0f} 元/年")
    logger.info(f"    Threshold: {th_annual:>14,.0f} 元/年")
    logger.info(f"    Oracle:    {oracle_annual:>14,.0f} 元/年")
    logger.info(f"    BC 多赚:   {bc_annual - th_annual:>+14,.0f} 元/年 vs Threshold")

    # 保存结果
    out_path = PROCESSED_DIR / f"{province}_backtest_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"\n  Results saved: {out_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", type=str, default="shandong",
                        choices=["shandong", "shanxi", "guangdong", "gansu", "all"])
    args = parser.parse_args()

    if args.province == "all":
        provinces = ["shandong", "shanxi", "guangdong", "gansu"]
    else:
        provinces = [args.province]

    all_results = {}
    for province in provinces:
        result = backtest_province(province)
        if result is not None:
            all_results[province] = result

    if len(all_results) > 1:
        logger.info(f"\n{'='*70}")
        logger.info("CROSS-PROVINCE COMPARISON")
        logger.info(f"{'='*70}")
        logger.info(f"{'Province':<12s} {'BC vs Threshold':>16s} {'BC % of Oracle':>16s} {'Win Rate':>10s}")
        logger.info("-" * 56)
        for prov, df in all_results.items():
            total_bc = df["bc_revenue"].sum()
            total_th = df["threshold_revenue"].sum()
            total_or = df["oracle_revenue"].sum()
            vs_th = (total_bc - total_th) / abs(total_th) * 100
            pct_or = total_bc / total_or * 100
            win = df["bc_win"].mean() * 100
            logger.info(f"{prov:<12s} {vs_th:>+15.1f}% {pct_or:>15.1f}% {win:>9.0f}%")


if __name__ == "__main__":
    main()
