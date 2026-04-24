"""
Logan · Script 03 — End-to-end backtest on Gansu
=================================================

端到端回测：
  1. 加载训好的 4 个 head
  2. 在测试集上逐日：
     - 预测 DA（5 个分位数）
     - 预测 spread 方向（24 小时概率）
     - 预测系统偏差（shortage/surplus 概率）
     - 读取功率预测（proxy：从 parquet 的新能源总出力 + noise，模拟"客户提供的预测"）
     - 生成 bid curve
  3. 用真实 DA/RT 结算，对比 Logan vs 老实报

注意："客户功率预测"我们现在没有。脚本用两个代理：
  - perfect_forecast: 用实际出力作为"完美功率预测"（上限）
  - noisy_forecast:   实际 + 5-10% 高斯噪声（更真实）

用法：
    PYTHONPATH=. python3 products/logan/scripts/03_backtest_gansu.py
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
from products.logan.bid_curve_generator import BidCurveGenerator, BidCurveConfig
from products.logan.evaluator import LoganEvaluator, SettlementConfig


PROCESSED_DIR = ROOT / "data" / "china" / "processed"
MODELS_DIR = ROOT / "models" / "logan"
RUNS_DIR = ROOT / "runs" / "logan"


def aggregate_15min_to_hour(arr_96: np.ndarray) -> np.ndarray:
    """(96,) 15 分钟 → (24,) 小时（均值聚合）"""
    return arr_96.reshape(24, 4).mean(axis=1)


def load_heads(province: str) -> dict:
    model_dir = MODELS_DIR / province
    heads = {}
    for name in ["regime_classifier", "da_forecaster", "spread_direction", "system_deviation", "rt_forecaster"]:
        p = model_dir / f"{name}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"Missing model: {p}. Run script 02 first.")
        with open(p, "rb") as f:
            heads[name] = pickle.load(f)
    return heads


def main(province: str = "gansu",
         capacity_mw: float = 100.0,
         test_days: int = 90,
         forecast_noise_std: float = 0.08,
         rng_seed: int = 42):

    logger.info(f"=== Logan end-to-end backtest: {province} ===")
    logger.info(f"Virtual plant: {capacity_mw} MW new-energy")

    heads = load_heads(province)
    rc = heads["regime_classifier"]
    da_fcst = heads["da_forecaster"]
    spread_clf = heads["spread_direction"]
    sys_dev = heads["system_deviation"]

    # Load data
    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")
    df = add_calendar_features(df)
    df = df[df["da_price"].notna() & df["load_mw"].notna()]

    # 客户"功率预测"代理：用新能源总出力的一个比例 scale 到 capacity_mw
    # 目的：模拟一个 capacity_mw 级光伏/风电场站
    if "renewable_mw" in df.columns and df["renewable_mw"].notna().sum() > len(df) * 0.3:
        raw = df["renewable_mw"].fillna(method="ffill").fillna(0).values
    elif "solar_mw" in df.columns and df["solar_mw"].notna().sum() > len(df) * 0.3:
        raw = df["solar_mw"].fillna(method="ffill").fillna(0).values
    else:
        raw = df["load_mw"].fillna(0).values * 0.1  # 最后 fallback

    # Scale 到 capacity_mw 量级
    raw_max = float(np.percentile(raw, 99))
    if raw_max > 0:
        scaled_actual_power = raw / raw_max * capacity_mw
    else:
        scaled_actual_power = np.zeros_like(raw)

    # Split train/test
    split = int(len(df) * 0.8)
    test_start_day = split // 96
    n_days_total = len(df) // 96
    test_end_day = min(test_start_day + test_days, n_days_total - 1)
    logger.info(f"Test days: {test_start_day} → {test_end_day} ({test_end_day - test_start_day} days)")

    # 生成 bid curves
    rng = np.random.default_rng(rng_seed)
    bid_cfg = BidCurveConfig(deviation_bound=0.10)
    generator = BidCurveGenerator(capacity_mw=capacity_mw, config=bid_cfg)
    settlement_cfg = SettlementConfig(
        deviation_bound=0.10,
        deviation_penalty_ratio=0.20,
        reverse_penalty_spread_threshold=50.0,
        reverse_penalty_ratio=0.30,
    )
    evaluator = LoganEvaluator(settlement_cfg)

    all_logan_revenue = 0.0
    all_naive_revenue = 0.0
    all_logan_penalty = 0.0
    all_naive_penalty = 0.0
    all_rev_pen_hours = 0
    n_eval_days = 0

    daily_records = []

    for d in range(test_start_day, test_end_day):
        try:
            # Step 1: DA 预测（5 分位，15 分钟 → 聚合成小时）
            da_quantiles_96 = da_fcst.predict_day_all_quantiles(d, df)  # (96, 5)
            # 15 分钟聚合到小时
            da_quantiles_hr = da_quantiles_96.reshape(24, 4, -1).mean(axis=1)  # (24, 5)

            # Step 2: spread direction （24 小时概率，因为 aggregate_to_hour=True）
            spread_dir = spread_clf.predict_proba_day(d, df)  # (24,)

            # Step 3: system deviation
            sys_risk = sys_dev.predict_proba_day(d, df)  # {"prob_shortage", "prob_surplus"}: (24,)

            # Step 4: "功率预测"（用实际 + 噪声作代理）
            start_idx = d * 96
            end_idx = start_idx + 96
            actual_power_96 = scaled_actual_power[start_idx:end_idx]
            actual_power_hr = aggregate_15min_to_hour(actual_power_96)
            noise = rng.normal(0, forecast_noise_std, 24)
            power_forecast_hr = np.maximum(actual_power_hr * (1 + noise), 0.0)
            power_forecast_hr = np.minimum(power_forecast_hr, capacity_mw)

            # Step 5: 生成 bid curve
            bids = generator.generate(
                power_forecast_hourly=power_forecast_hr,
                da_quantiles_hourly=da_quantiles_hr[:, :4],  # 用 P5/P25/P50/P75 作阶梯
                spread_dir_prob_hourly=spread_dir,
                system_shortage_prob_hourly=sys_risk["prob_shortage"],
                system_surplus_prob_hourly=sys_risk["prob_surplus"],
            )

            # Step 6: 真实市场结算
            rt_prices_96 = df["rt_price"].iloc[start_idx:end_idx].fillna(0).values
            da_prices_96 = df["da_price"].iloc[start_idx:end_idx].fillna(method="ffill").fillna(0).values
            rt_prices_hr = aggregate_15min_to_hour(rt_prices_96)
            da_prices_hr = aggregate_15min_to_hour(da_prices_96)

            logan_res = evaluator.settle_bids(
                bids=bids,
                actual_power=actual_power_hr,
                da_prices=da_prices_hr,
                rt_prices=rt_prices_hr,
                dt_hours=1.0,
            )
            naive_res = evaluator.settle_naive_with_forecast(
                forecast_power=power_forecast_hr,
                actual_power=actual_power_hr,
                da_prices=da_prices_hr,
                rt_prices=rt_prices_hr,
                dt_hours=1.0,
            )

            all_logan_revenue += logan_res.total_revenue
            all_naive_revenue += naive_res.total_revenue
            all_logan_penalty += logan_res.standard_penalty + logan_res.reverse_penalty
            all_naive_penalty += naive_res.standard_penalty + naive_res.reverse_penalty
            all_rev_pen_hours += logan_res.reverse_penalty_hours

            daily_records.append({
                "day": d,
                "date": df.index[start_idx].date(),
                "logan_revenue": logan_res.total_revenue,
                "naive_revenue": naive_res.total_revenue,
                "gain": logan_res.total_revenue - naive_res.total_revenue,
                "logan_penalty": logan_res.standard_penalty + logan_res.reverse_penalty,
                "naive_penalty": naive_res.standard_penalty + naive_res.reverse_penalty,
                "avg_deviation": logan_res.avg_deviation_ratio,
            })
            n_eval_days += 1

        except Exception as e:
            logger.warning(f"Day {d} skipped: {e}")
            continue

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Logan Backtest Summary ({province}, {n_eval_days} days)")
    logger.info(f"{'=' * 60}")
    logger.info(f"Logan total revenue:  ¥{all_logan_revenue:,.0f}")
    logger.info(f"Naive total revenue:  ¥{all_naive_revenue:,.0f}")
    abs_gain = all_logan_revenue - all_naive_revenue
    pct_gain = abs_gain / max(abs(all_naive_revenue), 1.0) * 100
    logger.info(f"Absolute gain:        ¥{abs_gain:+,.0f}")
    logger.info(f"Percent gain:         {pct_gain:+.2f}%")
    logger.info(f"Logan total penalty:  ¥{all_logan_penalty:,.0f}")
    logger.info(f"Naive total penalty:  ¥{all_naive_penalty:,.0f}")
    logger.info(f"Reverse penalty hrs:  {all_rev_pen_hours}/{n_eval_days * 24}")

    # Save daily records
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    df_daily = pd.DataFrame(daily_records)
    out_csv = RUNS_DIR / f"backtest_{province}.csv"
    df_daily.to_csv(out_csv, index=False)
    logger.info(f"\nPer-day records: {out_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="gansu")
    parser.add_argument("--capacity", type=float, default=100.0)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--noise", type=float, default=0.08)
    args = parser.parse_args()
    main(args.province, args.capacity, args.days, args.noise)
