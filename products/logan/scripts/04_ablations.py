"""
Logan · Script 04 — Ablations
==============================

逐个关掉 Logan 的组件，看每个组件的边际价值。

变体：
    Naive             零偏差申报（基线）
    A0 · Full         完整 Logan
    A1 · -Spread      关掉 spread direction（offset_ratio = 0）
    A2 · -SysDev      关掉 system deviation damp
    A3 · -Ladder      单档报价
    A4 · -Flat        所有档用 P50 (价格分档但无差异化)

用法：
    PYTHONPATH=. python3 products/logan/scripts/04_ablations.py --days 90
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass, replace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.calendar_features import add_calendar_features
from products.logan.bid_curve_generator import BidCurveGenerator, BidCurveConfig
from products.logan.evaluator import LoganEvaluator, SettlementConfig


PROCESSED_DIR = ROOT / "data" / "china" / "processed"
MODELS_DIR = ROOT / "models" / "logan"
RUNS_DIR = ROOT / "runs" / "logan"


def aggregate_15min_to_hour(arr_96: np.ndarray) -> np.ndarray:
    return arr_96.reshape(24, 4).mean(axis=1)


def load_heads(province: str) -> dict:
    model_dir = MODELS_DIR / province
    heads = {}
    for name in ["regime_classifier", "da_forecaster", "spread_direction",
                 "system_deviation", "rt_forecaster"]:
        with open(model_dir / f"{name}.pkl", "rb") as f:
            heads[name] = pickle.load(f)
    return heads


def run_backtest(
    province: str,
    bid_cfg: BidCurveConfig,
    capacity_mw: float,
    test_days: int,
    forecast_noise_std: float,
    rng_seed: int,
    heads: dict,
    df: pd.DataFrame,
    scaled_actual_power: np.ndarray,
) -> dict:
    """
    核心 backtest 循环。给定 BidCurveConfig，跑 test_days 天，返回总统计。
    """
    da_fcst = heads["da_forecaster"]
    spread_clf = heads["spread_direction"]
    sys_dev = heads["system_deviation"]

    split = int(len(df) * 0.8)
    test_start_day = split // 96
    n_days_total = len(df) // 96
    test_end_day = min(test_start_day + test_days, n_days_total - 1)

    rng = np.random.default_rng(rng_seed)
    generator = BidCurveGenerator(capacity_mw=capacity_mw, config=bid_cfg)
    evaluator = LoganEvaluator(SettlementConfig())

    total_rev = 0.0
    total_naive_rev = 0.0
    total_pen = 0.0
    total_rev_pen_hrs = 0
    total_logan_da = 0.0
    total_logan_rt = 0.0
    total_logan_clear_mwh = 0.0
    total_logan_actual_mwh = 0.0
    total_naive_da = 0.0
    total_naive_rt = 0.0
    n_days = 0

    for d in range(test_start_day, test_end_day):
        try:
            da_q_96 = da_fcst.predict_day_all_quantiles(d, df)
            da_q_hr = da_q_96.reshape(24, 4, -1).mean(axis=1)[:, :4]

            spread_dir = spread_clf.predict_proba_day(d, df)
            sys_risk = sys_dev.predict_proba_day(d, df)

            start_idx = d * 96
            end_idx = start_idx + 96
            actual_power_96 = scaled_actual_power[start_idx:end_idx]
            actual_power_hr = aggregate_15min_to_hour(actual_power_96)
            noise = rng.normal(0, forecast_noise_std, 24)
            power_forecast_hr = np.clip(actual_power_hr * (1 + noise), 0.0, capacity_mw)

            bids = generator.generate(
                power_forecast_hourly=power_forecast_hr,
                da_quantiles_hourly=da_q_hr,
                spread_dir_prob_hourly=spread_dir,
                system_shortage_prob_hourly=sys_risk["prob_shortage"],
                system_surplus_prob_hourly=sys_risk["prob_surplus"],
            )

            rt_96 = df["rt_price"].iloc[start_idx:end_idx].fillna(0).values
            da_96 = df["da_price"].iloc[start_idx:end_idx].ffill().fillna(0).values
            rt_hr = aggregate_15min_to_hour(rt_96)
            da_hr = aggregate_15min_to_hour(da_96)

            logan_res = evaluator.settle_bids(bids, actual_power_hr, da_hr, rt_hr, dt_hours=1.0)
            naive_res = evaluator.settle_naive_with_forecast(
                power_forecast_hr, actual_power_hr, da_hr, rt_hr, dt_hours=1.0
            )

            # Track Logan's cleared quantity for analysis
            logan_cleared_hr = BidCurveGenerator.bids_to_cleared_array(bids, da_hr)

            total_rev += logan_res.total_revenue
            total_naive_rev += naive_res.total_revenue
            total_pen += logan_res.standard_penalty + logan_res.reverse_penalty
            total_rev_pen_hrs += logan_res.reverse_penalty_hours
            total_logan_da += logan_res.da_revenue
            total_logan_rt += logan_res.rt_deviation_revenue
            total_logan_clear_mwh += float(logan_cleared_hr.sum())
            total_logan_actual_mwh += float(actual_power_hr.sum())
            total_naive_da += naive_res.da_revenue
            total_naive_rt += naive_res.rt_deviation_revenue
            n_days += 1
        except Exception as e:
            logger.warning(f"Day {d} skipped: {e}")

    clear_ratio = total_logan_clear_mwh / max(total_logan_actual_mwh, 1.0)
    return {
        "total_revenue": total_rev,
        "naive_revenue": total_naive_rev,
        "total_penalty": total_pen,
        "reverse_pen_hrs": total_rev_pen_hrs,
        "n_days": n_days,
        "gain_abs": total_rev - total_naive_rev,
        "gain_pct": (total_rev - total_naive_rev) / max(abs(total_naive_rev), 1.0) * 100,
        "logan_da_revenue": total_logan_da,
        "logan_rt_revenue": total_logan_rt,
        "logan_clear_ratio": clear_ratio,
        "naive_da_revenue": total_naive_da,
        "naive_rt_revenue": total_naive_rt,
    }


@dataclass
class Variant:
    name: str
    description: str
    config: BidCurveConfig


def define_variants() -> list[Variant]:
    base = BidCurveConfig()
    return [
        Variant("A0_Full", "完整 Logan",
                base),
        Variant("A1_NoSpread", "关掉 spread 方向 (offset=0)",
                replace(base, disable_spread_direction=True)),
        Variant("A2_NoSysDev", "关掉 system deviation damp",
                replace(base, disable_system_deviation=True)),
        Variant("A3_SingleStep", "单档报价（无阶梯）",
                replace(base, use_single_step=True)),
        Variant("A4_FlatPrice", "所有档用 P50（无价差）",
                replace(base, use_flat_price=True)),
        # 组合测试
        Variant("A5_OnlyLadder", "只保留阶梯（其他全关）",
                replace(base, disable_spread_direction=True, disable_system_deviation=True)),
    ]


def main(province: str = "gansu", capacity_mw: float = 100.0,
         test_days: int = 90, forecast_noise_std: float = 0.08,
         rng_seed: int = 42):
    logger.info(f"=== Logan Ablation Study ({province}, {test_days} days, {capacity_mw} MW) ===")

    # 一次性加载数据 + 模型
    heads = load_heads(province)
    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")
    df = add_calendar_features(df)
    df = df[df["da_price"].notna() & df["load_mw"].notna()]

    if "renewable_mw" in df.columns and df["renewable_mw"].notna().sum() > len(df) * 0.3:
        raw = df["renewable_mw"].ffill().fillna(0).values
    elif "solar_mw" in df.columns and df["solar_mw"].notna().sum() > len(df) * 0.3:
        raw = df["solar_mw"].ffill().fillna(0).values
    else:
        raw = df["load_mw"].fillna(0).values * 0.1
    raw_max = float(np.percentile(raw, 99))
    scaled_actual_power = raw / raw_max * capacity_mw if raw_max > 0 else np.zeros_like(raw)

    variants = define_variants()
    results = []

    for v in variants:
        logger.info(f"\n--- Running {v.name}: {v.description} ---")
        r = run_backtest(
            province, v.config, capacity_mw, test_days,
            forecast_noise_std, rng_seed, heads, df, scaled_actual_power
        )
        logger.info(
            f"  Revenue: ¥{r['total_revenue']:>12,.0f}  Naive: ¥{r['naive_revenue']:>12,.0f}"
            f"  Gain: ¥{r['gain_abs']:>+10,.0f} ({r['gain_pct']:+.2f}%)"
            f"  RevPen: {r['reverse_pen_hrs']}h"
        )
        logger.info(
            f"  Logan: DA=¥{r['logan_da_revenue']:>12,.0f}  "
            f"RT偏差=¥{r['logan_rt_revenue']:>+10,.0f}  "
            f"中标率={r['logan_clear_ratio']*100:.1f}%"
        )
        logger.info(
            f"  Naive: DA=¥{r['naive_da_revenue']:>12,.0f}  "
            f"RT偏差=¥{r['naive_rt_revenue']:>+10,.0f}"
        )
        results.append({
            "name": v.name,
            "description": v.description,
            **r,
        })

    # Ablation summary table
    logger.info(f"\n{'=' * 100}")
    logger.info(f"  ABLATION SUMMARY ({province}, {test_days} days)")
    logger.info(f"{'=' * 100}")
    # Use full for reference
    full = next(r for r in results if r["name"] == "A0_Full")
    logger.info(
        f"{'Variant':<18} {'Description':<35} {'Revenue':>14} {'Gain vs Naive':>14} {'Δ vs Full':>12}"
    )
    logger.info("-" * 100)
    for r in results:
        delta_vs_full = r["total_revenue"] - full["total_revenue"]
        logger.info(
            f"{r['name']:<18} {r['description']:<35} "
            f"¥{r['total_revenue']:>12,.0f} "
            f"{r['gain_pct']:>+10.2f}% "
            f"¥{delta_vs_full:>+10,.0f}"
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RUNS_DIR / "ablations.csv", index=False)
    logger.info(f"\nSaved: {RUNS_DIR / 'ablations.csv'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="gansu")
    parser.add_argument("--capacity", type=float, default=100.0)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--noise", type=float, default=0.08)
    args = parser.parse_args()
    main(args.province, args.capacity, args.days, args.noise)
