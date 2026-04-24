"""
Logan · Script 10 — Daily Bid Backtest (甘肃合规架构)
======================================================

对比 3 种策略在甘肃真实规则下的 production-ready 表现：

  Naive             老实报：cleared = forecast (每时段)
  DailyBid          全天一条合规曲线（新架构）
  LegacyOptimal     旧的 per-hour 架构（对照，用于看损失）

结算：全部按 15 分钟粒度（dt=0.25），甘肃方式二结算。

用法：
    PYTHONPATH=. python3 products/logan/scripts/10_daily_bid_backtest.py --days 90
"""
from __future__ import annotations

import sys
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.calendar_features import add_calendar_features
from core.joint_distribution import EmpiricalJointDistribution, JointDistConfig
from products.logan.daily_bid import DailyBidGenerator, DailyBidConfig
from products.logan.optimal_bid import OptimalBidGenerator, OptimalBidConfig
from products.logan.evaluator import LoganEvaluator, SettlementConfig
from products.logan.compliance import ComplianceRules, validate_daily


PROCESSED_DIR = ROOT / "data" / "china" / "processed"
MODELS_DIR = ROOT / "models" / "logan"
RUNS_DIR = ROOT / "runs" / "logan" / "experiments"
RULES_PATH = ROOT / "products" / "logan" / "settlement_rules" / "gansu.yaml"


def load_heads(province: str) -> dict:
    model_dir = MODELS_DIR / province
    heads = {}
    for name in ["regime_classifier", "da_forecaster", "spread_direction",
                 "system_deviation", "rt_forecaster"]:
        with open(model_dir / f"{name}.pkl", "rb") as f:
            heads[name] = pickle.load(f)
    return heads


def load_data_and_scale(province: str, capacity_mw: float):
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
    scaled = raw / raw_max * capacity_mw if raw_max > 0 else np.zeros_like(raw)
    return df, scaled


def run_strategy(
    strategy: str,
    heads: dict,
    joint_dist,
    df: pd.DataFrame,
    scaled_actual_power: np.ndarray,
    capacity_mw: float,
    rules: ComplianceRules,
    test_start_day: int,
    test_end_day: int,
    rng_seed: int = 42,
    forecast_noise_std: float = 0.08,
) -> dict:
    da_fcst = heads["da_forecaster"]

    daily_gen = DailyBidGenerator(
        capacity_mw=capacity_mw,
        rules=rules,
        config=DailyBidConfig(n_scenarios=120),
        joint_dist=joint_dist,
    )
    optimal_gen = OptimalBidGenerator(
        capacity_mw=capacity_mw,
        rules=rules,
        config=OptimalBidConfig(n_scenarios=120),
        joint_dist=joint_dist,
    )

    evaluator = LoganEvaluator(SettlementConfig())
    rng = np.random.default_rng(rng_seed)
    quantile_levels_all = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    total_rev = 0.0
    total_da = 0.0
    total_rt = 0.0
    total_clear_mwh = 0.0
    total_actual_mwh = 0.0
    compliance_ok_days = 0
    compliance_fail_days = 0
    n_days = 0
    daily_bid_segment_counts = []

    for d in range(test_start_day, test_end_day):
        try:
            start_idx = d * 96
            end_idx = start_idx + 96

            # ---- Inputs ----
            da_q_96 = da_fcst.predict_day_all_quantiles(d, df)  # (96, 5)

            actual_power_96 = scaled_actual_power[start_idx:end_idx]
            # Forecast 每 15 分钟版本（actual + 噪声）
            # 注意 noise 应该是每 15 分钟独立的，但为了简化（也更现实：功率预测通常小时级粗糙）
            # 这里 24 小时独立 noise，每 4 个 15-min 共享（更接近现实）
            noise_24 = rng.normal(0, forecast_noise_std, 24)
            noise_96 = np.repeat(noise_24, 4)
            power_forecast_96 = np.clip(actual_power_96 * (1 + noise_96), 0.0, capacity_mw)

            rt_96 = df["rt_price"].iloc[start_idx:end_idx].fillna(0).values
            da_96 = df["da_price"].iloc[start_idx:end_idx].ffill().fillna(0).values

            # ---- Strategy ----
            if strategy == "naive":
                # 老实报：每时段 cleared = forecast
                cleared_96 = power_forecast_96.copy()
                res = evaluator.settle_hourly(
                    cleared_qty=cleared_96,
                    actual_power=actual_power_96,
                    da_prices_node=da_96,
                    rt_prices_node=rt_96,
                    dt_hours=0.25,
                )
                n_segs = 1  # 近似：naive 相当于 1 段
            elif strategy == "daily":
                bid = daily_gen.generate_day(
                    power_forecast_96=power_forecast_96,
                    da_quantiles_96=da_q_96,
                    quantile_levels=quantile_levels_all,
                )
                v = validate_daily(bid, rules)
                if v.ok:
                    compliance_ok_days += 1
                else:
                    compliance_fail_days += 1
                n_segs = bid.n_segments
                res = evaluator.settle_daily_bid(
                    daily_bid=bid,
                    actual_power_96=actual_power_96,
                    forecast_96=power_forecast_96,
                    da_prices_96=da_96,
                    rt_prices_96=rt_96,
                )
            elif strategy == "legacy_optimal":
                # 旧 per-hour 架构（24 条曲线）
                # Agg to hour for legacy interface
                actual_hr = actual_power_96.reshape(24, 4).mean(axis=1)
                forecast_hr = power_forecast_96.reshape(24, 4).mean(axis=1)
                da_hr = da_96.reshape(24, 4).mean(axis=1)
                rt_hr = rt_96.reshape(24, 4).mean(axis=1)
                da_q_hr = da_q_96.reshape(24, 4, -1).mean(axis=1)

                bids = optimal_gen.generate(
                    power_forecast_hourly=forecast_hr,
                    da_quantiles_hourly=da_q_hr,
                    quantile_levels=quantile_levels_all,
                )
                res = evaluator.settle_bids(
                    bids=bids, actual_power=actual_hr,
                    da_prices_node=da_hr, rt_prices_node=rt_hr,
                    dt_hours=1.0,
                )
                n_segs = int(np.mean([len(b.steps) for b in bids if b.steps] or [0]))
            else:
                raise ValueError(strategy)

            total_rev += res.total_revenue
            total_da += res.da_spot_revenue
            total_rt += res.rt_spot_revenue
            total_clear_mwh += res.cleared_mwh
            total_actual_mwh += res.actual_mwh
            daily_bid_segment_counts.append(n_segs)
            n_days += 1
        except Exception as e:
            logger.warning(f"Day {d} skipped ({strategy}): {e}")

    return {
        "strategy": strategy,
        "total_revenue": total_rev,
        "da_revenue": total_da,
        "rt_revenue": total_rt,
        "clear_ratio": total_clear_mwh / max(total_actual_mwh, 1.0),
        "avg_segments": float(np.mean(daily_bid_segment_counts)) if daily_bid_segment_counts else 0,
        "compliance_ok_days": compliance_ok_days,
        "compliance_fail_days": compliance_fail_days,
        "n_days": n_days,
    }


def main(province: str = "gansu", capacity_mw: float = 100.0,
         test_days: int = 90, rng_seed: int = 42):
    logger.info(f"=== Daily Bid Backtest ({province}, {test_days}d, {capacity_mw} MW) ===")

    rules = ComplianceRules.from_yaml(RULES_PATH, capacity_mw=capacity_mw)
    logger.info(f"Rules: {rules.min_steps}-{rules.max_steps} 段, ≥ {rules.min_step_mw:.1f} MW, "
                f"{rules.price_precision} 元精度")

    heads = load_heads(province)
    df, scaled_actual_power = load_data_and_scale(province, capacity_mw)

    split = int(len(df) * 0.8)
    test_start = split // 96
    n_days_total = len(df) // 96
    test_end = min(test_start + test_days, n_days_total - 1)

    # Joint dist
    joint_start = max(0, split - 540 * 96)
    df_joint = df.iloc[joint_start:split]
    joint_dist = EmpiricalJointDistribution(JointDistConfig(n_da_buckets=10))
    joint_dist.fit(df_joint)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for strat in ["naive", "legacy_optimal", "daily"]:
        logger.info(f"\n--- Running {strat} ---")
        t0 = time.time()
        r = run_strategy(
            strategy=strat, heads=heads, joint_dist=joint_dist,
            df=df, scaled_actual_power=scaled_actual_power,
            capacity_mw=capacity_mw, rules=rules,
            test_start_day=test_start, test_end_day=test_end,
            rng_seed=rng_seed,
        )
        elapsed = time.time() - t0
        logger.info(
            f"  Revenue: ¥{r['total_revenue']:>12,.0f}  "
            f"DA=¥{r['da_revenue']:>12,.0f}  RT=¥{r['rt_revenue']:>+10,.0f}  "
            f"Clear={r['clear_ratio']*100:.1f}%  "
            f"avg_segs={r['avg_segments']:.1f}  "
            f"({elapsed:.1f}s)"
        )
        if strat == "daily":
            logger.info(
                f"  Compliance: {r['compliance_ok_days']}/{r['n_days']} days OK, "
                f"{r['compliance_fail_days']} fails"
            )
        results.append(r)

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  Summary")
    logger.info(f"{'=' * 80}")
    naive = next(r for r in results if r["strategy"] == "naive")
    for r in results:
        gain = r["total_revenue"] - naive["total_revenue"]
        pct = gain / max(abs(naive["total_revenue"]), 1.0) * 100
        logger.info(
            f"{r['strategy']:<18} ¥{r['total_revenue']:>12,.0f}  "
            f"gain=¥{gain:>+10,.0f} ({pct:+.2f}%)  clear={r['clear_ratio']*100:.1f}%"
        )

    pd.DataFrame(results).to_csv(RUNS_DIR / "daily_bid.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="gansu")
    parser.add_argument("--capacity", type=float, default=100.0)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    main(args.province, args.capacity, args.days)
