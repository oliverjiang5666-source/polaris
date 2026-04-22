"""
MILP 跨省实验：山西 + 广东 + 甘肃，各 3 个实验（baseline + dev=0% + dev=20%）

预计耗时：~3-5 小时（4 并发）

这个脚本适合另一台 Mac 跑，和 27 并行。

Usage (on Mac #2):
    caffeinate -i python3 -u scripts/28_milp_other_provinces.py 2>&1 | tee runs/milp_other.log &
"""
from pathlib import Path
from loguru import logger

from optimization.milp.experiment_runner import (
    build_multiprovince_baselines, build_cross_province_ablations, run_all,
)


def main():
    specs = build_multiprovince_baselines()  # 3 个
    specs += build_cross_province_ablations()  # 6 个

    logger.info(f"将运行 {len(specs)} 个跨省实验")
    for i, s in enumerate(specs):
        logger.info(f"  {i+1:2d}. {s.province}_{s.name}: K={s.K}, "
                    f"dev={s.deviation_bound:.0%}")

    results = run_all(
        specs,
        output_dir=Path("runs/milp_experiments"),
        n_workers=4,
        solver_threads=2,
        time_limit=180,
        solver_backend="appsi_highs",
    )

    logger.info(f"\n完成 {len(results)} / {len(specs)} 个跨省实验")


if __name__ == "__main__":
    main()
