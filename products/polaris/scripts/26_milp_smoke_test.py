"""
MILP Smoke Test: 山东 15 天，K=200

目的：
  - 验证 MILP 在真实数据上可行
  - 估算全年耗时
  - 初步看 MILP vs Regime V3 的方向

跑完大约 15 分钟（4 并发）。

Usage:
    caffeinate -i python3 -u scripts/26_milp_smoke_test.py 2>&1 | tee runs/milp_smoke.log
"""
from pathlib import Path
from loguru import logger

from optimization.milp.experiment_runner import run_experiment, ExperimentSpec


def main():
    # 山东 Q4 2024 中段 15 天
    test_days = list(range(1650, 1665))

    spec = ExperimentSpec(
        name="smoke",
        province="shandong",
        K=200,
        deviation_bound=0.10,
        deg_cost=2.0,
    )

    logger.info("=== MILP Smoke Test ===")
    logger.info(f"Province: shandong")
    logger.info(f"Days: {test_days[0]}..{test_days[-1]} ({len(test_days)} days)")
    logger.info(f"K: 200 scenarios")
    logger.info(f"Solver: HiGHS (auto)")
    logger.info(f"Workers: 4")

    result = run_experiment(
        spec,
        solver_backend="appsi_highs",
        n_workers=4,
        solver_threads=2,
        time_limit=180,
        output_dir=Path("runs/milp_smoke"),
        test_days_override=test_days,
    )

    logger.info(f"\n完成。总收入 ¥{result.total_revenue:,.0f}")
    logger.info(f"平均 ¥{result.total_revenue/len(test_days):,.0f}/天")
    logger.info(f"失败 {result.n_failed}/{len(test_days)}")
    logger.info(f"Wall time {result.total_wall_time:.1f}s")
    logger.info(f"  → 估算全年 (365 days): {result.total_wall_time/len(test_days)*365/60:.1f} min")


if __name__ == "__main__":
    main()
