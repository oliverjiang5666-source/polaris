"""
MILP 完整山东消融实验（15 个）

预计耗时：~4-8 小时（4 并发），每个实验 15-30 min。

分配策略：
  - 这台 Mac (Mac #1) 跑此脚本
  - 另一台 Mac 跑 scripts/28_milp_other_provinces.py

Usage:
    caffeinate -i python3 -u scripts/27_milp_shandong_all.py 2>&1 | tee runs/milp_shandong.log &
"""
from pathlib import Path
from loguru import logger

from optimization.milp.experiment_runner import (
    build_shandong_ablation_matrix, run_all,
)


def main():
    specs = build_shandong_ablation_matrix()

    logger.info(f"将运行 {len(specs)} 个山东消融实验")
    for i, s in enumerate(specs):
        logger.info(f"  {i+1:2d}. {s.name}: K={s.K}, dev={s.deviation_bound:.0%}, "
                    f"deg=¥{s.deg_cost:.0f}")

    results = run_all(
        specs,
        output_dir=Path("runs/milp_experiments"),
        n_workers=4,
        solver_threads=2,
        time_limit=180,
        solver_backend="appsi_highs",
    )

    logger.info(f"\n{'=' * 70}")
    logger.info(f"完成 {len(results)} / {len(specs)} 个山东实验")


if __name__ == "__main__":
    main()
