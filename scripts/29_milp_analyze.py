"""
分析所有 MILP 实验结果，生成对比报告

Usage:
    python3 scripts/29_milp_analyze.py
"""
from pathlib import Path
from loguru import logger

from optimization.milp.analyze_results import (
    load_all_results, make_comparison_table, print_table,
    analyze_ablation_dev, analyze_ablation_deg, analyze_ablation_K,
    analyze_multiprovince, write_report_md,
)


def main():
    runs_dir = Path("runs/milp_experiments")
    if not runs_dir.exists():
        logger.error(f"未找到 {runs_dir}，请先跑 27/28 脚本")
        return

    results = load_all_results(runs_dir)
    logger.info(f"加载 {len(results)} 个实验")

    df = make_comparison_table(results)
    print_table(df)
    analyze_ablation_dev(results)
    analyze_ablation_deg(results)
    analyze_ablation_K(results)
    analyze_multiprovince(results)

    df.to_csv(runs_dir / "summary.csv", index=False)
    write_report_md(results, runs_dir / "report.md")
    logger.info(f"\n汇总 CSV: {runs_dir}/summary.csv")
    logger.info(f"报告 MD:  {runs_dir}/report.md")


if __name__ == "__main__":
    main()
