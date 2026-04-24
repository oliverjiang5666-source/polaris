"""
实验矩阵编排器

从一个 YAML/字典配置驱动多个实验，结果保存到 runs/ 目录。
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, asdict
from loguru import logger
import json
import pickle
from datetime import datetime

from optimization.milp.milp_formulation import BatteryParams, MILPConfig
from optimization.milp.parallel_backtest import walk_forward_milp, save_result, BacktestResult


@dataclass
class ExperimentSpec:
    """单个实验规格"""
    name: str
    province: str
    K: int = 200
    deviation_bound: float = 0.10
    deg_cost: float = 2.0
    final_soc_min: float | None = 0.3
    cycle_limit: float | None = None
    scenario_method: str = "bootstrap"  # "bootstrap" / "quantile"
    use_cvar: bool = False
    cvar_weight: float = 0.3


def run_experiment(
    spec: ExperimentSpec,
    solver_backend: str = "appsi_highs",
    n_workers: int = 4,
    solver_threads: int = 2,
    time_limit: float = 120.0,
    output_dir: Path = Path("runs/milp_experiments"),
    test_days_override: list[int] | None = None,
) -> BacktestResult:
    """执行单个实验"""
    battery = BatteryParams(deg_cost=spec.deg_cost)
    config = MILPConfig(
        deviation_bound=spec.deviation_bound,
        final_soc_min=spec.final_soc_min,
        cycle_limit=spec.cycle_limit,
        use_cvar=spec.use_cvar,
        cvar_weight=spec.cvar_weight,
    )

    result = walk_forward_milp(
        province=spec.province,
        K=spec.K,
        battery=battery,
        config=config,
        solver_backend=solver_backend,
        solver_threads=solver_threads,
        n_workers=n_workers,
        time_limit=time_limit,
        config_name=spec.name,
        test_days_override=test_days_override,
    )

    save_result(result, output_dir)

    # 保存 spec
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{spec.province}_{spec.name}_spec.json", "w") as f:
        json.dump(asdict(spec), f, indent=2)

    return result


# ============================================================
# 预定义实验矩阵
# ============================================================


def build_shandong_ablation_matrix() -> list[ExperimentSpec]:
    """山东完整消融矩阵"""
    specs = []

    # Baseline
    specs.append(ExperimentSpec(
        name="baseline", province="shandong",
        K=200, deviation_bound=0.10, deg_cost=2.0,
    ))

    # Deviation bound 消融
    for dev in [0.0, 0.20, 0.50]:
        specs.append(ExperimentSpec(
            name=f"dev_{int(dev*100):02d}", province="shandong",
            K=200, deviation_bound=dev, deg_cost=2.0,
        ))

    # Degradation cost 消融
    for deg in [8.0, 20.0, 50.0]:
        specs.append(ExperimentSpec(
            name=f"deg_{int(deg):02d}", province="shandong",
            K=200, deviation_bound=0.10, deg_cost=deg,
        ))

    # K 消融
    for k in [50, 100, 500]:
        specs.append(ExperimentSpec(
            name=f"K_{k:03d}", province="shandong",
            K=k, deviation_bound=0.10, deg_cost=2.0,
        ))

    # Scenario 方法
    specs.append(ExperimentSpec(
        name="scen_quantile", province="shandong",
        K=200, scenario_method="quantile",
    ))

    # Final SoC
    specs.append(ExperimentSpec(
        name="no_final_soc", province="shandong",
        K=200, deviation_bound=0.10, deg_cost=2.0, final_soc_min=None,
    ))

    # 循环约束
    specs.append(ExperimentSpec(
        name="cycle_limit_2", province="shandong",
        K=200, deviation_bound=0.10, deg_cost=2.0, cycle_limit=2.0,
    ))

    return specs


def build_multiprovince_baselines() -> list[ExperimentSpec]:
    """其他 3 省 baseline"""
    return [
        ExperimentSpec(name="baseline", province=p, K=200)
        for p in ["shanxi", "guangdong", "gansu"]
    ]


def build_cross_province_ablations() -> list[ExperimentSpec]:
    """跨省 × deviation 消融"""
    specs = []
    for prov in ["shanxi", "guangdong", "gansu"]:
        for dev in [0.0, 0.20]:
            specs.append(ExperimentSpec(
                name=f"dev_{int(dev*100):02d}", province=prov,
                K=200, deviation_bound=dev,
            ))
    return specs


# ============================================================
# 批量执行
# ============================================================


def run_all(
    specs: list[ExperimentSpec],
    output_dir: Path = Path("runs/milp_experiments"),
    n_workers: int = 4,
    solver_threads: int = 2,
    time_limit: float = 120.0,
    solver_backend: str = "appsi_highs",
    test_days_override: list[int] | None = None,
) -> list[BacktestResult]:
    """依次执行多个实验"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 记录执行元信息
    meta = {
        "started_at": datetime.now().isoformat(),
        "n_experiments": len(specs),
        "solver_backend": solver_backend,
        "n_workers": n_workers,
        "solver_threads": solver_threads,
        "specs": [asdict(s) for s in specs],
    }
    with open(output_dir / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    results = []
    for i, spec in enumerate(specs):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"#  Experiment {i+1}/{len(specs)}: {spec.province}_{spec.name}")
        logger.info(f"{'#' * 70}")

        try:
            r = run_experiment(
                spec,
                solver_backend=solver_backend,
                n_workers=n_workers,
                solver_threads=solver_threads,
                time_limit=time_limit,
                output_dir=output_dir,
                test_days_override=test_days_override,
            )
            results.append(r)
        except Exception as e:
            logger.exception(f"Experiment {spec.name} failed: {e}")

    # 汇总
    summary = [{
        "province": r.province,
        "name": r.config_name,
        "total_revenue": r.total_revenue,
        "n_failed": r.n_failed,
        "solve_time_min": r.total_solve_time / 60,
        "wall_time_min": r.total_wall_time / 60,
    } for r in results]

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"All {len(results)} experiments done. See {output_dir}/")
    for row in summary:
        logger.info(f"  {row['province']}_{row['name']}: ¥{row['total_revenue']:,.0f} "
                    f"(failed: {row['n_failed']}, {row['wall_time_min']:.1f} min)")

    return results


if __name__ == "__main__":
    # 快速展示：山东所有 15 个消融 + 3 省 baseline + 6 个跨省消融
    specs = build_shandong_ablation_matrix()
    specs += build_multiprovince_baselines()
    specs += build_cross_province_ablations()
    logger.info(f"Total experiments in matrix: {len(specs)}")
    for s in specs:
        logger.info(f"  {s.province}_{s.name}: K={s.K}, dev={s.deviation_bound:.0%}, "
                    f"deg={s.deg_cost:.0f}")
