"""
并行 walk-forward 回测（multiprocessing）

每天的 MILP 求解独立 → 天然并行
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from multiprocessing import Pool, get_context
from functools import partial
from loguru import logger
import time
import pickle

from optimization.milp.data_loader import ProvinceData, split_walkforward, load_province
from optimization.milp.scenario_generator import (
    RegimeClassifier, generate_scenarios_bootstrap, ScenarioSet,
)
from optimization.milp.milp_formulation import (
    BatteryParams, MILPConfig, build_two_stage_lp, extract_solution, simulate_on_actual,
)
from optimization.milp.stochastic_solver import SolverAdapter, SolverResult


# ============================================================
# 单日任务（可 pickle 的纯函数，供子进程调用）
# ============================================================


def _solve_one_day(args: dict) -> dict:
    """
    单日 MILP 求解（作为子进程的工作单元）。
    输入全部为可 pickle 的数据。
    """
    day_idx = args["day_idx"]
    dam_forecast = args["dam_forecast"]
    rt_scenarios = args["rt_scenarios"]
    weights = args["weights"]
    actual_dam = args["actual_dam"]
    actual_rt = args["actual_rt"]
    battery = args["battery"]
    config = args["config"]
    solver_backend = args.get("solver_backend", "appsi_highs")
    solver_threads = args.get("solver_threads", 2)
    time_limit = args.get("time_limit", 120)

    t0 = time.time()

    # 1. 构建 MILP
    model = build_two_stage_lp(dam_forecast, rt_scenarios, weights, battery, config)

    # 2. 求解
    adapter = SolverAdapter(backend=solver_backend, threads=solver_threads, verbose=False)
    result = adapter.solve(model, time_limit=time_limit)

    if result.status != "optimal":
        return {
            "day_idx": day_idx,
            "status": result.status,
            "milp_objective": None,
            "actual_revenue": 0.0,
            "solve_time": result.solve_time,
            "total_time": time.time() - t0,
        }

    # 3. 提取解
    K, T = rt_scenarios.shape
    sol = extract_solution(model, K, T)
    p_dam = sol["p_dam"]

    # 4. 在真实价格上仿真
    sim = simulate_on_actual(p_dam, actual_rt, actual_dam, battery, config)

    return {
        "day_idx": day_idx,
        "status": result.status,
        "milp_objective": result.objective,
        "actual_revenue": sim["revenue_total"],
        "actual_revenue_dam": sim["revenue_dam"],
        "actual_revenue_dev": sim["revenue_dev"],
        "actual_degradation": sim["degradation"],
        "solve_time": result.solve_time,
        "total_time": time.time() - t0,
        "p_dam_mean": float(p_dam.mean()),
        "p_dam_std": float(p_dam.std()),
    }


# ============================================================
# 批量准备：生成所有天的场景 + 数据
# ============================================================


def prepare_tasks(
    province_data: ProvinceData,
    classifier: RegimeClassifier,
    test_days: list[int],
    K: int,
    battery: BatteryParams,
    config: MILPConfig,
    solver_backend: str,
    solver_threads: int,
    time_limit: float,
    rng_seed: int = 42,
) -> list[dict]:
    """
    为每个测试天准备任务字典（预生成场景）。

    这样子进程只做 MILP 求解，不涉及分类器（分类器不能 pickle 到子进程）。
    """
    tasks = []
    rng = np.random.default_rng(rng_seed)

    for d in test_days:
        # 生成场景
        scen = generate_scenarios_bootstrap(
            classifier, province_data, target_day=d, K=K, rng=rng,
        )

        tasks.append({
            "day_idx": d,
            "dam_forecast": scen.dam_forecast,
            "rt_scenarios": scen.rt_scenarios,
            "weights": scen.weights,
            "actual_dam": province_data.dam_prices[d],
            "actual_rt": province_data.rt_prices[d],
            "battery": battery,
            "config": config,
            "solver_backend": solver_backend,
            "solver_threads": solver_threads,
            "time_limit": time_limit,
        })

    return tasks


# ============================================================
# 主回测函数
# ============================================================


@dataclass
class BacktestResult:
    province: str
    config_name: str
    total_revenue: float
    per_day_revenue: np.ndarray      # [n_days]
    per_day_status: list[str]
    per_day_solve_time: np.ndarray
    n_failed: int
    total_solve_time: float
    total_wall_time: float


def walk_forward_milp(
    province: str,
    K: int = 200,
    battery: BatteryParams | None = None,
    config: MILPConfig | None = None,
    solver_backend: str = "appsi_highs",
    solver_threads: int = 2,
    n_workers: int = 4,
    time_limit: float = 120.0,
    test_days_override: list[int] | None = None,
    config_name: str = "baseline",
) -> BacktestResult:
    """
    在单个省上跑 walk-forward MILP 回测。

    - 按季度 retrain 分类器（和 scripts/22 保持一致）
    - 每天生成 K 场景，解 MILP
    - 多进程并行
    """
    if battery is None:
        battery = BatteryParams()
    if config is None:
        config = MILPConfig()

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  {province.upper()} — {config_name} (K={K}, "
                f"dev={config.deviation_bound:.0%}, deg={battery.deg_cost:.0f})")
    logger.info(f"{'=' * 70}")

    data = load_province(province)
    quarters = split_walkforward(data)

    if test_days_override is not None:
        # 使用户指定 days 做 smoke test
        test_days_by_quarter = [test_days_override]
        quarters = [(test_days_override[0], test_days_override[-1] + 1)]
    else:
        test_days_by_quarter = [list(range(qs, qe)) for qs, qe in quarters]

    all_results = []
    total_t0 = time.time()

    for qi, (quarter_range, (qs, qe)) in enumerate(zip(test_days_by_quarter, quarters)):
        logger.info(f"\n  Q{qi + 1} [{qs}, {qe}) — {len(quarter_range)} days")

        # 训练分类器：只用当前 quarter 之前的数据
        t_train = time.time()
        classifier = RegimeClassifier(n_regimes=12)
        classifier.fit(data, train_day_end=qs)
        logger.info(f"    Classifier trained in {time.time()-t_train:.1f}s")

        # 准备任务
        t_prep = time.time()
        tasks = prepare_tasks(
            data, classifier, quarter_range, K=K,
            battery=battery, config=config,
            solver_backend=solver_backend, solver_threads=solver_threads,
            time_limit=time_limit,
        )
        logger.info(f"    Tasks prepared in {time.time()-t_prep:.1f}s "
                    f"(scenarios: {len(tasks)} × K={K})")

        # 并行求解
        t_solve = time.time()
        ctx = get_context("spawn")  # macOS 默认，避免 fork 问题
        with ctx.Pool(processes=n_workers) as pool:
            quarter_results = pool.map(_solve_one_day, tasks)

        solve_elapsed = time.time() - t_solve
        success = sum(1 for r in quarter_results if r["status"] == "optimal")
        total_rev = sum(r.get("actual_revenue", 0) for r in quarter_results)

        logger.info(f"    Q{qi + 1} done in {solve_elapsed:.1f}s  "
                    f"(success: {success}/{len(tasks)}, "
                    f"avg: {solve_elapsed / max(len(tasks), 1):.1f}s/day)")
        logger.info(f"    Q{qi + 1} revenue: ¥{total_rev:,.0f}")

        all_results.extend(quarter_results)

    # 汇总
    total_wall = time.time() - total_t0
    per_day_rev = np.array([r.get("actual_revenue", 0) for r in all_results])
    per_day_status = [r["status"] for r in all_results]
    per_day_solve_time = np.array([r["solve_time"] for r in all_results])
    n_failed = sum(1 for s in per_day_status if s != "optimal")
    total_solve_time = per_day_solve_time.sum()
    total_revenue = per_day_rev.sum()

    logger.info(f"\n  TOTAL: ¥{total_revenue:,.0f}  "
                f"(failed: {n_failed}/{len(all_results)}, "
                f"solve: {total_solve_time/60:.1f}min, wall: {total_wall/60:.1f}min)")

    return BacktestResult(
        province=province,
        config_name=config_name,
        total_revenue=float(total_revenue),
        per_day_revenue=per_day_rev,
        per_day_status=per_day_status,
        per_day_solve_time=per_day_solve_time,
        n_failed=n_failed,
        total_solve_time=float(total_solve_time),
        total_wall_time=float(total_wall),
    )


def save_result(result: BacktestResult, output_dir: Path):
    """保存结果到 pickle + CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{result.province}_{result.config_name}"

    with open(output_dir / f"{fname}.pkl", "wb") as f:
        pickle.dump(result, f)

    pd.DataFrame({
        "day_idx": range(len(result.per_day_revenue)),
        "revenue": result.per_day_revenue,
        "status": result.per_day_status,
        "solve_time": result.per_day_solve_time,
    }).to_csv(output_dir / f"{fname}_per_day.csv", index=False)


if __name__ == "__main__":
    """快速 smoke test：山东 5 天"""
    battery = BatteryParams()
    config = MILPConfig(deviation_bound=0.10, cycle_limit=None)

    # 用山东的第 1700-1704 天做测试
    result = walk_forward_milp(
        province="shandong",
        K=50,  # smoke test 用小 K
        battery=battery,
        config=config,
        n_workers=4,
        time_limit=60,
        test_days_override=list(range(1700, 1705)),
        config_name="smoke_test",
    )

    logger.info(f"\nSmoke test 成功")
    logger.info(f"总收入: ¥{result.total_revenue:,.0f}")
    logger.info(f"每日: {result.per_day_revenue}")
