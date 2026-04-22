#!/usr/bin/env python3
"""
Step 1: 对4省数据求解LP Oracle

为每一天计算数学最优充放电方案，作为BC训练的完美老师。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger
from oracle.lp_oracle import solve_dataset
from config import BatteryConfig

PROCESSED_DIR = ROOT / "data" / "china" / "processed"


def main():
    battery = BatteryConfig()

    logger.info("=" * 60)
    logger.info("Step 1: Solving LP Oracle for all provinces")
    logger.info(f"Battery: {battery.capacity_mw}MW / {battery.capacity_mwh}MWh")
    logger.info("=" * 60)

    provinces = ["shandong", "shanxi", "guangdong", "gansu"]
    results_summary = []

    for province in provinces:
        feat_path = PROCESSED_DIR / f"{province}_features.parquet"
        if not feat_path.exists():
            logger.warning(f"  Skipping {province}: {feat_path} not found")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Solving Oracle for {province}")
        logger.info(f"{'='*60}")

        df = pd.read_parquet(feat_path)
        logger.info(f"  Loaded: {len(df):,} rows")

        # 过滤掉rt_price为NaN的行——不能用来做Oracle
        valid_mask = df["rt_price"].notna()
        valid_pct = valid_mask.mean() * 100
        logger.info(f"  Valid price rows: {valid_mask.sum():,} ({valid_pct:.1f}%)")

        t0 = time.time()
        df_oracle, total_revenue = solve_dataset(df, battery, price_col="rt_price")
        elapsed = time.time() - t0

        # 统计
        n_days = len(df) // 96
        actions = df_oracle["oracle_action"].values
        action_counts = pd.Series(actions).value_counts().sort_index()

        annual_revenue = total_revenue / (n_days / 365) if n_days > 0 else 0

        logger.info(f"  Solve time: {elapsed:.1f}s ({elapsed/max(n_days,1)*1000:.1f}ms/day)")
        logger.info(f"  Total revenue: {total_revenue:,.0f} 元")
        logger.info(f"  Annualized:    {annual_revenue:,.0f} 元/年")
        logger.info(f"  Days solved:   {n_days}")
        logger.info(f"  Action distribution:")
        for action_id, count in action_counts.items():
            pct = count / len(actions) * 100
            action_name = {0: "wait", 1: "slow_charge", 2: "fast_charge",
                          3: "slow_discharge", 4: "fast_discharge"}[action_id]
            logger.info(f"    {action_id} ({action_name:<16s}): {count:>8,} ({pct:>5.1f}%)")

        # 保存
        out_path = PROCESSED_DIR / f"{province}_oracle.parquet"
        df_oracle.to_parquet(out_path)
        logger.info(f"  Saved: {out_path}")

        results_summary.append({
            "province": province,
            "days": n_days,
            "total_revenue": total_revenue,
            "annual_revenue": annual_revenue,
            "solve_time_s": elapsed,
        })

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info("ORACLE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Province':<12s} {'Days':>6s} {'Total Revenue':>15s} {'Annual Revenue':>15s} {'Time':>8s}")
    logger.info("-" * 60)
    for r in results_summary:
        logger.info(
            f"{r['province']:<12s} "
            f"{r['days']:>6d} "
            f"{r['total_revenue']:>14,.0f}元 "
            f"{r['annual_revenue']:>14,.0f}元 "
            f"{r['solve_time_s']:>7.1f}s"
        )


if __name__ == "__main__":
    main()
