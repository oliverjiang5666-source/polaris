"""
K-Mystery 调查：为什么 K=50/100 > K=200？

Mac #1 跑完 3 个配置的初步数据：
    baseline (K=200): ¥2834万/年
    K_050    (K=50):  ¥3903万/年  ← 比 baseline +38%！
    K_100    (K=100): ¥4054万/年  ← 比 baseline +43%！

理论上 K 越大 Monte Carlo 逼近越准，不应该反向。
需要诊断是：
  (A) 真实现象：Bootstrap 场景在 K 大时"过度平滑"丢失尾部
  (B) Bug：场景生成器/MILP 建模/随机种子有问题
  (C) 收敛噪声：单次测试有方差，多 seed 平均后差异消失

本脚本 controlled 实验：
  - 取山东 20 天的 test subset
  - 对每天：生成 K ∈ {20, 50, 100, 200, 500, 1000} 的场景
  - 记录每组的：
      场景统计：mean, std, P5, P95（跨场景 + 跨时段）
      MILP 解：DAM 承诺均值/std、平均 actual_power、objective
      实盘收入（用真实 RT 算）
  - 跑 3 个 seed 看噪声
  - 输出：CSV + 快速诊断报告

用法：
    PYTHONPATH=. python3 scripts/30_k_mystery_investigation.py

不要在 Mac #1 实验跑完前运行——会竞争 CPU。Mac #2 也在跑，建议在 Mac #1 结束后运行。
或者用很小的 sample（默认 5 天 × 3 K × 2 seed = 30 MILP）本地快速跑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from loguru import logger
from dataclasses import asdict

from optimization.milp.data_loader import load_province, split_walkforward
from optimization.milp.scenario_generator import (
    RegimeClassifier, generate_scenarios_bootstrap,
)
from optimization.milp.milp_formulation import (
    build_two_stage_lp, BatteryParams, MILPConfig, extract_solution, simulate_on_actual,
)
from optimization.milp.stochastic_solver import SolverAdapter


OUTPUT = Path("runs/k_mystery")
OUTPUT.mkdir(parents=True, exist_ok=True)


def investigate_k_effect(
    province: str = "shandong",
    test_day_start: int = 1700,
    n_test_days: int = 20,
    K_values: list[int] = (20, 50, 100, 200, 500),
    n_seeds: int = 2,
    solver_backend: str = "appsi_highs",
):
    """
    对指定测试天数，跑不同 K 下的 MILP，观察结果。
    """
    data = load_province(province)
    battery = BatteryParams()
    config = MILPConfig(deviation_bound=0.10)

    # 训练 classifier 一次（用 test_day_start 之前所有数据）
    classifier = RegimeClassifier(n_regimes=12)
    logger.info(f"Training classifier on days [0, {test_day_start})...")
    t0 = time.time()
    classifier.fit(data, train_day_end=test_day_start)
    logger.info(f"  done in {time.time() - t0:.1f}s")

    records = []

    for seed in range(n_seeds):
        rng_base = 42 + seed * 1000
        logger.info(f"\n=== Seed {seed} ===")

        for day_idx in range(test_day_start, test_day_start + n_test_days):
            actual_rt = data.rt_prices[day_idx]
            actual_dam = data.dam_prices[day_idx]

            for K in K_values:
                rng = np.random.default_rng(rng_base + day_idx)  # Per-day seed
                scen = generate_scenarios_bootstrap(
                    classifier, data, target_day=day_idx, K=K, rng=rng,
                )

                # 场景统计
                sc_mean = float(scen.rt_scenarios.mean())
                sc_std = float(scen.rt_scenarios.std())
                sc_p5 = float(np.percentile(scen.rt_scenarios, 5))
                sc_p95 = float(np.percentile(scen.rt_scenarios, 95))
                sc_range = sc_p95 - sc_p5

                # 每时段的跨场景 spread
                per_step_std = scen.rt_scenarios.std(axis=0).mean()

                # 跑 MILP
                t1 = time.time()
                model = build_two_stage_lp(
                    scen.dam_forecast, scen.rt_scenarios, scen.weights,
                    battery, config,
                )
                adapter = SolverAdapter(backend=solver_backend, threads=2, verbose=False)
                result = adapter.solve(model, time_limit=180)
                solve_time = time.time() - t1

                if result.status != "optimal":
                    logger.warning(f"  day={day_idx} K={K} seed={seed}: {result.status}")
                    continue

                sol = extract_solution(model, K, 96)
                p_dam = sol["p_dam"]

                # 在真实 RT/DAM 上仿真
                sim = simulate_on_actual(p_dam, actual_rt, actual_dam, battery, config)

                records.append({
                    "seed": seed,
                    "day_idx": day_idx,
                    "K": K,
                    "actual_rt_mean": float(actual_rt.mean()),
                    "actual_rt_range": float(actual_rt.max() - actual_rt.min()),
                    "scenario_mean": sc_mean,
                    "scenario_std": sc_std,
                    "scenario_range_p95p5": sc_range,
                    "per_step_cross_scenario_std": float(per_step_std),
                    "milp_objective": result.objective,
                    "p_dam_mean": float(p_dam.mean()),
                    "p_dam_std": float(p_dam.std()),
                    "p_dam_max": float(p_dam.max()),
                    "p_dam_min": float(p_dam.min()),
                    "p_dam_aggressive_mw": float(np.abs(p_dam).max()),
                    "solve_time": solve_time,
                    "actual_revenue": sim["revenue_total"],
                    "actual_revenue_dam": sim["revenue_dam"],
                    "actual_revenue_dev": sim["revenue_dev"],
                })

                logger.info(
                    f"  seed={seed} day={day_idx} K={K:>4d}: "
                    f"obj=¥{result.objective:>8,.0f}  actual=¥{sim['revenue_total']:>8,.0f}  "
                    f"p_dam_max={float(np.abs(p_dam).max()):>5.0f}MW  solve={solve_time:.1f}s"
                )

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT / f"{province}_k_mystery.csv", index=False)
    return df


def analyze(df: pd.DataFrame):
    """分析 K 的影响"""
    logger.info(f"\n{'=' * 70}")
    logger.info("K-Mystery Analysis")
    logger.info(f"{'=' * 70}")

    # 按 K 聚合
    by_k = df.groupby("K").agg(
        actual_revenue=("actual_revenue", "mean"),
        actual_revenue_std=("actual_revenue", "std"),
        milp_objective=("milp_objective", "mean"),
        p_dam_aggressive=("p_dam_aggressive_mw", "mean"),
        p_dam_std=("p_dam_std", "mean"),
        scenario_std=("scenario_std", "mean"),
        per_step_cross_std=("per_step_cross_scenario_std", "mean"),
        solve_time=("solve_time", "mean"),
    ).round(2)

    logger.info(f"\n按 K 聚合（所有 seed × day 平均）：")
    print(by_k.to_string())

    # 对比 K=最小 vs K=最大 的配对差
    K_vals = sorted(df["K"].unique())
    k_min, k_max = K_vals[0], K_vals[-1]

    paired = df.pivot_table(
        index=["seed", "day_idx"],
        columns="K",
        values="actual_revenue",
        aggfunc="first",
    )
    if k_min in paired and k_max in paired:
        diff = paired[k_max] - paired[k_min]
        logger.info(f"\n配对差 (K={k_max}) - (K={k_min}):")
        logger.info(f"  均值: ¥{diff.mean():>10,.0f}")
        logger.info(f"  中位数: ¥{diff.median():>10,.0f}")
        logger.info(f"  >0 的比例: {(diff > 0).mean() * 100:.1f}%")
        if (diff < 0).mean() > 0.5:
            logger.info(f"  → ⚠️  K 越大收入越低 在 >50% 的天内成立，不是噪声")

    # DAM 承诺攻击性随 K 变化
    logger.info(f"\n DAM 承诺随 K 变化：")
    for K in K_vals:
        sub = df[df["K"] == K]
        logger.info(
            f"  K={K:>4d}: |p_dam|_max={sub['p_dam_aggressive_mw'].mean():>5.1f} MW, "
            f"p_dam_std={sub['p_dam_std'].mean():>5.1f}, "
            f"场景 std={sub['scenario_std'].mean():>5.1f}"
        )

    # Hypothesis check
    logger.info(f"\n{'=' * 70}")
    logger.info("诊断假设：")
    logger.info(f"{'=' * 70}")

    # 假设 A：K 大 → 场景分布更接近"平均"，std 应减小
    low_std = by_k.loc[k_min, "scenario_std"]
    high_std = by_k.loc[k_max, "scenario_std"]
    logger.info(f"  A) 场景 std 从 K={k_min} 到 K={k_max}: {low_std:.1f} → {high_std:.1f}")
    if high_std < low_std * 0.95:
        logger.info(f"     ✅ 场景 std 显著减小 → Monte Carlo 平滑效应")
    else:
        logger.info(f"     ❌ 场景 std 没有显著变化 → 不是纯平滑")

    # 假设 B：DAM 承诺是否因为 K 大变得更保守
    low_agg = by_k.loc[k_min, "p_dam_aggressive"]
    high_agg = by_k.loc[k_max, "p_dam_aggressive"]
    logger.info(f"  B) DAM |p_dam|_max 从 K={k_min} 到 K={k_max}: {low_agg:.1f} MW → {high_agg:.1f} MW")
    if high_agg < low_agg * 0.95:
        logger.info(f"     ✅ K 大时 DAM 承诺变保守（更靠近 0）→ 丢失攻击性")
    else:
        logger.info(f"     ❌ DAM 承诺攻击性没有显著变化")

    # 假设 C：MILP obj 和 actual 之差（planning-execution gap）
    gap_low = (by_k.loc[k_min, "milp_objective"] - by_k.loc[k_min, "actual_revenue"])
    gap_high = (by_k.loc[k_max, "milp_objective"] - by_k.loc[k_max, "actual_revenue"])
    logger.info(f"  C) MILP obj - actual gap，K={k_min}: ¥{gap_low:,.0f}, K={k_max}: ¥{gap_high:,.0f}")


def main():
    df = investigate_k_effect(
        province="shandong",
        test_day_start=1700,
        n_test_days=10,          # 10 天
        K_values=[20, 50, 100, 200, 500],
        n_seeds=2,                # 2 seed
    )
    analyze(df)
    logger.info(f"\n结果保存到 {OUTPUT}/shandong_k_mystery.csv")


if __name__ == "__main__":
    main()
