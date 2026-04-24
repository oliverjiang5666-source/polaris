"""
MILP 实验结果分析 + Regime V3 对比
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from optimization.milp.parallel_backtest import BacktestResult


# Regime V3 历史数据（从 HANDOFF_REGIME.md 摘出）
REGIME_V3_REFERENCE = {
    "shandong": {
        "capture": 64.8,
        "revenue": 53_810_000,
        "lgbm": 44_160_000,
        "oracle": 83_080_000,
    },
    "shanxi": {
        "capture": 63.2,
        "revenue": 62_440_000,
        "lgbm": 54_650_000,
        "oracle": 98_800_000,  # 估计
    },
    "guangdong": {
        "capture": 56.2,
        "revenue": 35_590_000,
        "lgbm": 22_340_000,
        "oracle": 63_300_000,  # 估计
    },
    "gansu": {
        "capture": 41.1,
        "revenue": 32_420_000,
        "lgbm": 31_030_000,
        "oracle": 78_900_000,  # 估计
    },
}


def load_all_results(runs_dir: Path) -> dict[str, BacktestResult]:
    """加载目录下所有 .pkl 结果"""
    out = {}
    for p in runs_dir.glob("*.pkl"):
        with open(p, "rb") as f:
            r = pickle.load(f)
        key = f"{r.province}_{r.config_name}"
        out[key] = r
    return out


def make_comparison_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    生成完整对比表：每个 MILP 实验 vs Regime V3 vs LightGBM 基线
    """
    rows = []
    for key, r in results.items():
        ref = REGIME_V3_REFERENCE.get(r.province, {})
        rev_r = ref.get("revenue", 0)
        rev_l = ref.get("lgbm", 0)
        rev_o = ref.get("oracle", 0)

        vs_regime = (r.total_revenue / rev_r - 1) * 100 if rev_r else 0
        vs_lgbm = (r.total_revenue / rev_l - 1) * 100 if rev_l else 0
        capture = r.total_revenue / rev_o * 100 if rev_o else 0

        rows.append({
            "key": key,
            "province": r.province,
            "config": r.config_name,
            "milp_revenue": r.total_revenue,
            "regime_v3_revenue": rev_r,
            "lgbm_revenue": rev_l,
            "oracle_revenue": rev_o,
            "vs_regime_v3_pct": vs_regime,
            "vs_lgbm_pct": vs_lgbm,
            "milp_capture_pct": capture,
            "n_failed": r.n_failed,
            "wall_time_min": r.total_wall_time / 60,
        })
    return pd.DataFrame(rows).sort_values(["province", "config"])


def print_table(df: pd.DataFrame):
    """终端友好格式"""
    logger.info("\n" + "=" * 110)
    logger.info(f"{'Config':<35}{'MILP 收入':>12}{'Regime V3':>12}{'LGBM':>12}"
                f"{'vs Regime':>10}{'vs LGBM':>10}{'Capture':>9}{'Time':>7}")
    logger.info("-" * 110)
    for _, r in df.iterrows():
        logger.info(
            f"{r['key']:<35}"
            f"{r['milp_revenue']/1e4:>11,.0f}万"
            f"{r['regime_v3_revenue']/1e4:>11,.0f}万"
            f"{r['lgbm_revenue']/1e4:>11,.0f}万"
            f"{r['vs_regime_v3_pct']:>+9.1f}%"
            f"{r['vs_lgbm_pct']:>+9.1f}%"
            f"{r['milp_capture_pct']:>8.1f}%"
            f"{r['wall_time_min']:>6.1f}m"
        )
    logger.info("=" * 110)


def analyze_ablation_dev(results: dict[str, BacktestResult]):
    """分析 deviation bound 消融"""
    logger.info("\n【偏差界消融】山东")
    logger.info(f"{'Dev':<8}{'Revenue':>14}{'vs Baseline':>14}")
    baseline = results.get("shandong_baseline")
    if not baseline:
        logger.warning("No baseline found")
        return

    for key in ["shandong_dev_00", "shandong_baseline", "shandong_dev_20", "shandong_dev_50"]:
        r = results.get(key)
        if not r:
            logger.info(f"  {key}: [missing]")
            continue
        dev = {"shandong_dev_00": "0%", "shandong_baseline": "10%",
               "shandong_dev_20": "20%", "shandong_dev_50": "50%"}[key]
        diff = (r.total_revenue / baseline.total_revenue - 1) * 100
        logger.info(f"  {dev:<8}{r.total_revenue/1e4:>13,.0f}万{diff:>+13.1f}%")


def analyze_ablation_deg(results: dict[str, BacktestResult]):
    """分析 degradation cost 消融"""
    logger.info("\n【降解成本消融】山东")
    logger.info(f"{'Deg':<8}{'Revenue':>14}{'vs Baseline':>14}")
    baseline = results.get("shandong_baseline")
    if not baseline:
        return

    items = [
        ("¥2", "shandong_baseline"),
        ("¥8", "shandong_deg_08"),
        ("¥20", "shandong_deg_20"),
        ("¥50", "shandong_deg_50"),
    ]
    for deg, key in items:
        r = results.get(key)
        if not r:
            logger.info(f"  {deg:<8}[missing]")
            continue
        diff = (r.total_revenue / baseline.total_revenue - 1) * 100
        logger.info(f"  {deg:<8}{r.total_revenue/1e4:>13,.0f}万{diff:>+13.1f}%")


def analyze_ablation_K(results: dict[str, BacktestResult]):
    """分析 K 消融"""
    logger.info("\n【K 消融】山东")
    logger.info(f"{'K':<8}{'Revenue':>14}{'Wall Time':>12}{'Edge':>10}")
    items = [
        (50,  "shandong_K_050"),
        (100, "shandong_K_100"),
        (200, "shandong_baseline"),
        (500, "shandong_K_500"),
    ]
    for K, key in items:
        r = results.get(key)
        if not r:
            logger.info(f"  K={K}: [missing]")
            continue
        # 边际收益
        marginal = ""
        if K > 50:
            prev = results.get(items[items.index((K, key)) - 1][1])
            if prev:
                delta = r.total_revenue - prev.total_revenue
                pct = delta / prev.total_revenue * 100
                marginal = f"{pct:+.1f}%"
        logger.info(f"  {K:<8}{r.total_revenue/1e4:>13,.0f}万{r.total_wall_time/60:>11.1f}m{marginal:>10}")


def analyze_multiprovince(results: dict[str, BacktestResult]):
    """跨省对比"""
    logger.info("\n【跨省泛化】")
    logger.info(f"{'Province':<12}{'MILP':>14}{'Regime V3':>14}{'vs Regime':>12}{'Capture':>10}")
    for prov in ["shandong", "shanxi", "guangdong", "gansu"]:
        r = results.get(f"{prov}_baseline")
        if not r:
            logger.info(f"  {prov}: [missing]")
            continue
        ref = REGIME_V3_REFERENCE[prov]
        diff = (r.total_revenue / ref["revenue"] - 1) * 100 if ref["revenue"] else 0
        cap = r.total_revenue / ref["oracle"] * 100 if ref["oracle"] else 0
        logger.info(f"  {prov:<12}{r.total_revenue/1e4:>13,.0f}万{ref['revenue']/1e4:>13,.0f}万"
                    f"{diff:>+11.1f}%{cap:>9.1f}%")


def write_report_md(results: dict[str, BacktestResult], output_path: Path):
    """写完整 Markdown 报告"""
    df = make_comparison_table(results)

    lines = ["# MILP 实验报告\n\n"]
    lines.append("## 汇总表\n\n")
    lines.append(df.to_markdown(index=False, floatfmt=",.1f"))
    lines.append("\n\n")

    # 各类消融
    if "shandong_baseline" in results:
        baseline = results["shandong_baseline"]
        ref = REGIME_V3_REFERENCE["shandong"]
        diff = (baseline.total_revenue / ref["revenue"] - 1) * 100
        lines.append(f"## 核心结论\n\n")
        lines.append(f"**山东 Baseline MILP vs Regime V3**: "
                     f"¥{baseline.total_revenue/1e4:.0f}万 vs ¥{ref['revenue']/1e4:.0f}万 "
                     f"(**{diff:+.1f}%**)\n\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(lines)
    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    import sys
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/milp_experiments")

    if not runs_dir.exists():
        logger.error(f"Runs dir not found: {runs_dir}")
        sys.exit(1)

    results = load_all_results(runs_dir)
    logger.info(f"加载 {len(results)} 个实验结果")

    df = make_comparison_table(results)
    print_table(df)
    analyze_ablation_dev(results)
    analyze_ablation_deg(results)
    analyze_ablation_K(results)
    analyze_multiprovince(results)

    df.to_csv(runs_dir / "summary.csv", index=False)
    write_report_md(results, runs_dir / "report.md")
