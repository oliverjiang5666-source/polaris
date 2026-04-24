"""
Logan · Script 06 — DFL Backtest
=================================

对比 Heuristic Bid vs DFL Bid 在甘肃回测上的表现。

差异：
    Heuristic : offset_ratio 由 if-else 规则决定，阶梯分档固定
    DFL       : 每小时对 35 个候选 bid 做 SAA，选期望收益最大

用法：
    PYTHONPATH=. python3 products/logan/scripts/06_dfl_backtest.py --days 90
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.calendar_features import add_calendar_features
from core.joint_distribution import EmpiricalJointDistribution, JointDistConfig
from products.logan.bid_curve_generator import BidCurveGenerator, BidCurveConfig
from products.logan.dfl_bid_curve import DFLBidCurveGenerator, DFLConfig
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


def _estimate_spread_mag_std(df: pd.DataFrame, train_end_idx: int, spread_mag_hourly=None) -> np.ndarray:
    """从训练集估计每小时的 |spread| 标准差"""
    rt = df["rt_price"].iloc[:train_end_idx].fillna(0).values
    da = df["da_price"].iloc[:train_end_idx].ffill().fillna(0).values
    spread = rt - da
    n_days = len(spread) // 96
    spread_days = spread[:n_days * 96].reshape(n_days, 96)
    # 聚合到小时
    spread_hr = spread_days.reshape(n_days, 24, 4).mean(axis=2)  # (n_days, 24)
    mag_std = np.std(spread_hr, axis=0)  # (24,)
    return mag_std


def run_one_strategy(
    strategy_name: str,
    strategy_kind: str,      # "heuristic" or "dfl"
    province: str,
    capacity_mw: float,
    test_days: int,
    forecast_noise_std: float,
    rng_seed: int,
    heads: dict,
    df: pd.DataFrame,
    scaled_actual_power: np.ndarray,
    spread_mag_std_hourly: np.ndarray,
    joint_dist: EmpiricalJointDistribution | None = None,
) -> dict:
    da_fcst = heads["da_forecaster"]
    spread_clf = heads["spread_direction"]
    sys_dev = heads["system_deviation"]

    split = int(len(df) * 0.8)
    test_start_day = split // 96
    n_days_total = len(df) // 96
    test_end_day = min(test_start_day + test_days, n_days_total - 1)

    if strategy_kind == "heuristic":
        generator = BidCurveGenerator(capacity_mw=capacity_mw, config=BidCurveConfig())
    elif strategy_kind == "dfl":
        generator = DFLBidCurveGenerator(
            capacity_mw=capacity_mw,
            config=DFLConfig(n_scenarios=200),
        )
    elif strategy_kind == "dfl_joint":
        generator = DFLBidCurveGenerator(
            capacity_mw=capacity_mw,
            config=DFLConfig(n_scenarios=200),
            joint_dist=joint_dist,
        )
    else:
        raise ValueError(strategy_kind)

    evaluator = LoganEvaluator(SettlementConfig())
    rng = np.random.default_rng(rng_seed)

    total_rev = 0.0
    total_naive_rev = 0.0
    total_pen = 0.0
    total_da = 0.0
    total_rt = 0.0
    total_clear = 0.0
    total_actual = 0.0
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

            if strategy_kind == "heuristic":
                bids = generator.generate(
                    power_forecast_hourly=power_forecast_hr,
                    da_quantiles_hourly=da_q_hr,
                    spread_dir_prob_hourly=spread_dir,
                    system_shortage_prob_hourly=sys_risk["prob_shortage"],
                    system_surplus_prob_hourly=sys_risk["prob_surplus"],
                )
            else:
                bids = generator.generate(
                    power_forecast_hourly=power_forecast_hr,
                    da_quantiles_hourly=da_q_hr,
                    spread_dir_prob_hourly=spread_dir,
                    spread_mag_std_hourly=spread_mag_std_hourly,
                )

            rt_96 = df["rt_price"].iloc[start_idx:end_idx].fillna(0).values
            da_96 = df["da_price"].iloc[start_idx:end_idx].ffill().fillna(0).values
            rt_hr = aggregate_15min_to_hour(rt_96)
            da_hr = aggregate_15min_to_hour(da_96)

            logan_res = evaluator.settle_bids(bids, actual_power_hr, da_hr, rt_hr, dt_hours=1.0)
            naive_res = evaluator.settle_naive_with_forecast(
                power_forecast_hr, actual_power_hr, da_hr, rt_hr, dt_hours=1.0
            )

            cleared_hr = BidCurveGenerator.bids_to_cleared_array(bids, da_hr)

            total_rev += logan_res.total_revenue
            total_naive_rev += naive_res.total_revenue
            total_pen += logan_res.standard_penalty + logan_res.reverse_penalty
            total_da += logan_res.da_revenue
            total_rt += logan_res.rt_deviation_revenue
            total_clear += float(cleared_hr.sum())
            total_actual += float(actual_power_hr.sum())
            n_days += 1
        except Exception as e:
            logger.warning(f"Day {d} skipped: {e}")

    return {
        "strategy": strategy_name,
        "total_revenue": total_rev,
        "naive_revenue": total_naive_rev,
        "da_revenue": total_da,
        "rt_revenue": total_rt,
        "penalty": total_pen,
        "clear_ratio": total_clear / max(total_actual, 1.0),
        "n_days": n_days,
        "gain_pct": (total_rev - total_naive_rev) / max(abs(total_naive_rev), 1.0) * 100,
    }


def main(province: str = "gansu", capacity_mw: float = 100.0,
         test_days: int = 90, forecast_noise_std: float = 0.08,
         rng_seed: int = 42):
    logger.info(f"=== DFL vs Heuristic Backtest ({province}, {test_days} days, {capacity_mw} MW) ===")

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

    train_end_idx = int(len(df) * 0.8)
    spread_mag_std_hourly = _estimate_spread_mag_std(df, train_end_idx)

    # 拟合 joint distribution（用 recent 540 天对齐 head training）
    recent_days = 540
    joint_train_start = max(0, train_end_idx - recent_days * 96)
    df_joint_train = df.iloc[joint_train_start:train_end_idx]
    logger.info(f"Fitting EmpiricalJointDistribution on {len(df_joint_train):,} rows (recent {recent_days}d)")
    joint_dist = EmpiricalJointDistribution(JointDistConfig(n_da_buckets=10))
    joint_dist.fit(df_joint_train)
    rhos = joint_dist.compute_rank_correlation()
    logger.info(f"Spearman ρ(DA, RT) per hour: min={rhos.min():.2f}, mean={rhos.mean():.2f}, max={rhos.max():.2f}")

    results = []
    for name, kind in [("Heuristic", "heuristic"), ("DFL-indep", "dfl"), ("DFL-joint", "dfl_joint")]:
        logger.info(f"\n--- Running {name} ---")
        r = run_one_strategy(
            name, kind, province, capacity_mw, test_days,
            forecast_noise_std, rng_seed, heads, df, scaled_actual_power,
            spread_mag_std_hourly,
            joint_dist=joint_dist if kind == "dfl_joint" else None,
        )
        logger.info(
            f"  Revenue: ¥{r['total_revenue']:>12,.0f}  Gain vs Naive: {r['gain_pct']:+.2f}%"
        )
        logger.info(
            f"  DA=¥{r['da_revenue']:>12,.0f}  RT=¥{r['rt_revenue']:>+10,.0f}  "
            f"Penalty=¥{r['penalty']:>10,.0f}  Clear={r['clear_ratio']*100:.1f}%"
        )
        results.append(r)

    # Comparison
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  Strategy Comparison")
    logger.info(f"{'=' * 80}")
    for r in results:
        logger.info(
            f"{r['strategy']:<14} ¥{r['total_revenue']:>12,.0f}  "
            f"({r['gain_pct']:+.2f}% vs naive)  clear={r['clear_ratio']*100:.1f}%"
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RUNS_DIR / "dfl_vs_heuristic.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="gansu")
    parser.add_argument("--capacity", type=float, default=100.0)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    main(args.province, args.capacity, args.days)
