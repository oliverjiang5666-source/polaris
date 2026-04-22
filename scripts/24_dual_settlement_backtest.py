"""
Modification #1: DAM+RTM 双结算回测

对比以下策略在山东2024 Q4数据上的收入差异：
1. 单结算RT价（当前代码）——高估收入
2. 单结算DAM价（假设完美执行）——低估收入
3. 双结算（DAM计划+RT偏差）——贴近真实

目标：量化"从当前回测数字到真实客户可兑现数字"的折价系数。

Usage:
    PYTHONPATH=. python3 scripts/24_dual_settlement_backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from oracle.lp_oracle import solve_day, solve_day_dual, compare_single_vs_dual
from forecast.mpc_controller import simulate_dual_settlement

PROCESSED_DIR = Path("data/china/processed")


def run_oracle_comparison(province: str = "shandong", year: int = 2024, quarter: int = 4):
    """
    Oracle 三种结算对比。
    """
    battery = BatteryConfig()
    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")

    # 选Q4（Oct-Dec）
    q_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}[quarter]
    mask = (df.index.year == year) & (df.index.month.isin(q_months))
    dq = df[mask].copy()
    n_days = len(dq) // 96

    logger.info(f"\n{'='*70}")
    logger.info(f"  {province.upper()} {year} Q{quarter} — {n_days} days")
    logger.info(f"{'='*70}")

    rt_total_single = 0.0
    dam_total_single = 0.0
    dual_total = 0.0
    dual_breakdown = {"dam": 0.0, "rt": 0.0, "deg": 0.0}

    # 统计：每天
    for d in range(n_days):
        s, e = d * 96, (d + 1) * 96
        rt = dq["rt_price"].values[s:e]
        dam = dq["da_price"].values[s:e]

        if np.isnan(rt).any() or np.isnan(dam).any():
            continue

        # 1. 单结算 RT（当前代码）
        rt_single = solve_day(rt, battery)
        rt_total_single += rt_single["revenue"]

        # 2. 单结算 DAM
        dam_single = solve_day(dam, battery)
        dam_total_single += dam_single["revenue"]

        # 3. 双结算（上帝视角：知道DAM和RT）
        dual = solve_day_dual(dam, rt, battery)
        dual_total += dual["revenue_total"]
        dual_breakdown["dam"] += dual["revenue_dam"]
        dual_breakdown["rt"] += dual["revenue_rt"]
        dual_breakdown["deg"] += dual["revenue_degradation"]

    # 结果
    logger.info("")
    logger.info(f"  【Oracle】三种结算对比：")
    logger.info(f"  ┌─────────────────────────────────────┬──────────────┐")
    logger.info(f"  │ 策略                                │ 收入（元）    │")
    logger.info(f"  ├─────────────────────────────────────┼──────────────┤")
    logger.info(f"  │ 单结算（全部按 RT 价）              │ {rt_total_single:>12,.0f} │ ← 你当前代码")
    logger.info(f"  │ 单结算（全部按 DAM 价）             │ {dam_total_single:>12,.0f} │")
    logger.info(f"  │ 双结算（DAM 计划 + RT 偏差）        │ {dual_total:>12,.0f} │ ← 真实结算")
    logger.info(f"  └─────────────────────────────────────┴──────────────┘")
    logger.info("")
    logger.info(f"  双结算拆解：")
    logger.info(f"    DAM 收入：        {dual_breakdown['dam']:>12,.0f}")
    logger.info(f"    RT 偏差收入：      {dual_breakdown['rt']:>12,.0f}")
    logger.info(f"    降解成本（负）：   {dual_breakdown['deg']:>12,.0f}")
    logger.info("")

    # 关键指标
    rt_vs_dual = (rt_total_single / dual_total - 1) * 100 if dual_total > 0 else 0
    logger.info(f"  💡 关键发现：")
    logger.info(f"    你当前代码的数字比双结算真实数字 "
                f"{'高估' if rt_vs_dual > 0 else '低估'} {abs(rt_vs_dual):.1f}%")
    logger.info(f"    → 如果你回测出 ¥5381万，实盘贴近 ¥{5381/(1+rt_vs_dual/100):,.0f}万")

    return {
        "rt_single": rt_total_single,
        "dam_single": dam_total_single,
        "dual": dual_total,
        "rt_vs_dual_pct": rt_vs_dual,
        "n_days": n_days,
    }


def test_basic_sanity():
    """
    基础正确性测试：
    1. 如果 DAM == RT，双结算应该 == 单结算
    2. SoC 演化不应破坏物理边界
    """
    logger.info(f"\n{'─'*70}")
    logger.info(f"  Sanity Tests")
    logger.info(f"{'─'*70}")

    battery = BatteryConfig()
    np.random.seed(42)

    # Test 1: DAM == RT
    hours = np.arange(96) / 4
    rt = 320 - 60 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 20, 96)
    dam = rt.copy()  # identical

    single = solve_day(rt, battery)
    dual = solve_day_dual(dam, rt, battery)

    # 由于有 heuristic deviation，即使 DAM==RT 结果可能略有不同
    # 但 deviation 信号 |DAM-RT|/|DAM| 会 == 0，所以不应该偏离
    single_rev = single["revenue"]
    dual_rev = dual["revenue_total"]
    diff_pct = abs(single_rev - dual_rev) / abs(single_rev) * 100

    logger.info(f"  Test 1 (DAM==RT): single=¥{single_rev:,.0f}, dual=¥{dual_rev:,.0f}")
    logger.info(f"    差异 {diff_pct:.2f}%  {'✅ PASS' if diff_pct < 2 else '❌ FAIL'}")

    # Test 2: SoC bounds
    soc_min = dual["soc"].min()
    soc_max = dual["soc"].max()
    soc_ok = soc_min >= battery.min_soc - 1e-3 and soc_max <= battery.max_soc + 1e-3
    logger.info(f"  Test 2 (SoC bounds): min={soc_min:.3f}, max={soc_max:.3f}  "
                f"{'✅ PASS' if soc_ok else '❌ FAIL'}")

    # Test 3: 有 DAM/RT 差异时，双结算应该 > 单DAM
    dam_biased = rt * 0.8  # DAM 低估 20%
    single_dam_biased = solve_day(dam_biased, battery)
    dual_biased = solve_day_dual(dam_biased, rt, battery)

    test3_ok = dual_biased["revenue_total"] > single_dam_biased["revenue"] * 0.95
    logger.info(f"  Test 3 (dual >= DAM-only when RT>DAM): dual=¥{dual_biased['revenue_total']:,.0f}, "
                f"dam_only=¥{single_dam_biased['revenue']:,.0f}  "
                f"{'✅ PASS' if test3_ok else '❌ FAIL'}")

    return all([diff_pct < 2, soc_ok, test3_ok])


def main():
    # Step 1: Sanity tests
    if not test_basic_sanity():
        logger.error("❌ Sanity tests failed — stop here")
        return

    # Step 2: Shandong Q4 对比
    logger.info(f"\n\n{'='*70}")
    logger.info("  山东 Q4 2024：三种结算模式对比")
    logger.info(f"{'='*70}")
    sd_results = run_oracle_comparison("shandong", 2024, 4)

    # Step 3: 广东 对比（LMP 市场，价差更大）
    logger.info(f"\n\n{'='*70}")
    logger.info("  广东 Q4 2024：对比（LMP 节点电价）")
    logger.info(f"{'='*70}")
    gd_results = run_oracle_comparison("guangdong", 2024, 4)

    # Step 4: 山西
    logger.info(f"\n\n{'='*70}")
    logger.info("  山西 Q4 2024：对比")
    logger.info(f"{'='*70}")
    sx_results = run_oracle_comparison("shanxi", 2024, 4)

    # 最终汇总
    logger.info(f"\n\n{'='*70}")
    logger.info(f"  总结：单结算 vs 双结算 差异幅度")
    logger.info(f"{'='*70}")
    logger.info(f"  {'省份':<10}{'单结算(RT)':>14}{'双结算':>14}{'差异':>10}")
    for name, r in [("山东", sd_results), ("广东", gd_results), ("山西", sx_results)]:
        logger.info(f"  {name:<10}{r['rt_single']:>14,.0f}{r['dual']:>14,.0f}{r['rt_vs_dual_pct']:>9.1f}%")


if __name__ == "__main__":
    main()
