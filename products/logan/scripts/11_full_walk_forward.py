"""
Logan · Script 11 — Strict Walk-Forward Backtest (无作弊)
===========================================================

在甘肃**全部历史数据**（5 年）上做严格 walk-forward 回测。

# 严格性（无作弊）的 3 个关键
# -------------------------------
# 1. **滚动重训**：每 30 天重训所有模型（DA forecaster + joint distribution）
#    模型永远只见到"当前时点之前"的数据
# 2. **Climatology forecast**：功率预测用 "过去 N 天同 15-min 步均值" + 8% 噪声
#    不用 "actual × (1 + noise)"（那会让 forecaster 偷看未来）
# 3. **Test 集覆盖全部 test 期**（test_start 到 n_days_total）—— 不 cherry-pick 好 window

# 数据结构
# -------------------------------
# 甘肃 parquet: 2021-04-02 ~ 2026-04-03, 约 1383 valid days
# Train burn-in: 首次训练需 ≥540 天历史数据
# Test period: 从 day 540 开始到结束 = 约 843 天（约 2.3 年）

# Rolling 方案
# -------------------------------
# for checkpoint in [540, 570, 600, ..., n_days - 30]:
#     train on [checkpoint - 540, checkpoint]
#     test on  [checkpoint, checkpoint + 30]
# 每 30 天重训一次，总计约 28 次 retrain，test 总天数 ≈ 843

用法:
    PYTHONPATH=. python3 products/logan/scripts/11_full_walk_forward.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.calendar_features import add_calendar_features
from core.joint_distribution import EmpiricalJointDistribution, JointDistConfig
from products.logan.da_forecaster import DAForecaster
from products.logan.daily_bid import DailyBidGenerator, DailyBidConfig
from products.logan.evaluator import LoganEvaluator, SettlementConfig
from products.logan.compliance import ComplianceRules, validate_daily


PROCESSED_DIR = ROOT / "data" / "china" / "processed"
RUNS_DIR = ROOT / "runs" / "logan" / "experiments"
RULES_PATH = ROOT / "products" / "logan" / "settlement_rules" / "gansu.yaml"


# ============================================================
# Data + climatology forecast
# ============================================================

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


def climatology_forecast(
    actual_history_96: np.ndarray,   # (d × 96,) actual power for past d days
    target_day_start_idx: int,        # 目标日的起点 index (= d × 96)
    lookback_days: int = 7,
    rng: np.random.Generator | None = None,
    noise_std: float = 0.08,
    capacity_mw: float = 100.0,
) -> np.ndarray:
    """
    用"过去 lookback_days 天同 15-min 步的均值"作为当日功率预测 baseline.
    再加高斯噪声模拟 forecaster 的随机误差。

    这是新能源场站真实交易员常用的 persistence 预测 baseline。
    绝对不使用 target_day 本身的 actual（无作弊）。
    """
    # 取 target_day 前 lookback_days 天（每天 96 步）
    start = target_day_start_idx - lookback_days * 96
    if start < 0:
        # 不够回看，用现有最远数据
        start = 0
        lookback_days = target_day_start_idx // 96
        if lookback_days < 1:
            # 最早的几天完全无历史，返回 0 forecast
            return np.zeros(96)

    history = actual_history_96[start:target_day_start_idx]
    # reshape 成 (lookback_days, 96)
    n_days = len(history) // 96
    if n_days < 1:
        return np.zeros(96)
    days = history[: n_days * 96].reshape(n_days, 96)
    baseline = days.mean(axis=0)  # (96,)

    if rng is None:
        return baseline
    noise_24 = rng.normal(0, noise_std, 24)
    noise_96 = np.repeat(noise_24, 4)
    forecast = np.clip(baseline * (1 + noise_96), 0.0, capacity_mw)
    return forecast


# ============================================================
# Rolling walk-forward loop
# ============================================================

def run_full_walk_forward(
    province: str = "gansu",
    capacity_mw: float = 100.0,
    train_window_days: int = 540,
    min_train_days: int = 180,               # 首次 burn-in 最少 180 天
    retrain_every_days: int = 30,
    forecast_noise_std: float = 0.08,
    forecast_lookback_days: int = 7,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    严格 walk-forward 回测。

    Train window 策略：
      - 首次训练（day min_train_days 起）：用 [0, min_train_days) 的数据（不足 540 天也可）
      - 后续（有足够数据时）：滑动窗口 [checkpoint - train_window, checkpoint]
    """
    logger.info(f"=== Strict Walk-Forward ({province}, {capacity_mw} MW) ===")
    logger.info(f"min_train={min_train_days}d, train_window={train_window_days}d, "
                f"retrain_every={retrain_every_days}d")
    logger.info(f"forecast = climatology (last {forecast_lookback_days} days) × (1 + N(0, {forecast_noise_std}))")

    rules = ComplianceRules.from_yaml(RULES_PATH, capacity_mw=capacity_mw)
    df, scaled_actual_power = load_data_and_scale(province, capacity_mw)
    n_days_total = len(df) // 96
    logger.info(f"Total valid days: {n_days_total}")

    test_start_day = min_train_days         # 首次 test 从 day 180 开始
    test_end_day = n_days_total - 1
    logger.info(f"Test period: day {test_start_day} → {test_end_day} ({test_end_day - test_start_day} days)")

    # Retrain checkpoints
    checkpoints = list(range(test_start_day, test_end_day, retrain_every_days))
    # Last checkpoint 的 test 窗口截到 test_end_day
    logger.info(f"Retrain checkpoints: {len(checkpoints)} times\n")

    rng = np.random.default_rng(rng_seed)
    evaluator = LoganEvaluator(SettlementConfig())
    quantile_levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    daily_results: list[dict] = []

    for ck_idx, ck in enumerate(checkpoints):
        # Training range:
        #   首次（ck = min_train_days）：用 [0, ck) 的全部早期数据（不足 540 也用）
        #   后续：滑动窗口 [ck - train_window_days, ck)
        train_start = max(0, ck - train_window_days)
        train_end = ck
        df_train = df.iloc[train_start * 96 : train_end * 96]

        t0 = time.time()

        # 1) Train DA forecaster
        da_fcst = DAForecaster()
        da_fcst.fit(df_train)

        # 2) Fit joint distribution
        joint_dist = EmpiricalJointDistribution(JointDistConfig(n_da_buckets=10))
        joint_dist.fit(df_train)

        # 3) Build generator
        daily_gen = DailyBidGenerator(
            capacity_mw=capacity_mw,
            rules=rules,
            config=DailyBidConfig(n_scenarios=120, random_seed=rng_seed + ck_idx),
            joint_dist=joint_dist,
        )

        train_elapsed = time.time() - t0

        # Test window: [ck, ck + retrain_every_days) 或到 test_end_day
        test_start = ck
        test_end = min(ck + retrain_every_days, test_end_day)

        naive_rev_sum = 0.0
        daily_rev_sum = 0.0
        daily_compliance_ok = 0
        daily_compliance_fail = 0
        n_test_days = 0

        tb0 = time.time()
        for d in range(test_start, test_end):
            try:
                start_idx = d * 96
                end_idx = start_idx + 96

                # Climatology forecast (NO leak of future)
                forecast_96 = climatology_forecast(
                    actual_history_96=scaled_actual_power,
                    target_day_start_idx=start_idx,
                    lookback_days=forecast_lookback_days,
                    rng=rng,
                    noise_std=forecast_noise_std,
                    capacity_mw=capacity_mw,
                )
                actual_power_96 = scaled_actual_power[start_idx:end_idx]
                da_96 = df["da_price"].iloc[start_idx:end_idx].ffill().fillna(0).values
                rt_96 = df["rt_price"].iloc[start_idx:end_idx].fillna(0).values

                # DA quantile prediction (walk-forward compliant: da_fcst only trained on data < ck)
                da_q_96 = da_fcst.predict_day_all_quantiles(d, df)

                # --- Naive strategy (baseline) ---
                naive_res = evaluator.settle_hourly(
                    cleared_qty=forecast_96.copy(),
                    actual_power=actual_power_96,
                    da_prices_node=da_96,
                    rt_prices_node=rt_96,
                    dt_hours=0.25,
                )

                # --- Daily Bid strategy ---
                bid = daily_gen.generate_day(
                    power_forecast_96=forecast_96,
                    da_quantiles_96=da_q_96,
                    quantile_levels=quantile_levels,
                )
                v = validate_daily(bid, rules)
                if v.ok:
                    daily_compliance_ok += 1
                else:
                    daily_compliance_fail += 1
                daily_res = evaluator.settle_daily_bid(
                    daily_bid=bid,
                    actual_power_96=actual_power_96,
                    forecast_96=forecast_96,
                    da_prices_96=da_96,
                    rt_prices_96=rt_96,
                )

                daily_results.append({
                    "checkpoint": ck,
                    "day": d,
                    "date": str(df.index[start_idx].date()),
                    "naive_rev": float(naive_res.total_revenue),
                    "daily_rev": float(daily_res.total_revenue),
                    "naive_da": float(naive_res.da_spot_revenue),
                    "naive_rt": float(naive_res.rt_spot_revenue),
                    "daily_da": float(daily_res.da_spot_revenue),
                    "daily_rt": float(daily_res.rt_spot_revenue),
                    "daily_segments": bid.n_segments,
                    "compliant": bool(v.ok),
                    "da_mean": float(da_96.mean()),
                    "rt_mean": float(rt_96.mean()),
                    "forecast_total_mwh": float(forecast_96.sum() * 0.25),
                    "actual_total_mwh": float(actual_power_96.sum() * 0.25),
                })
                naive_rev_sum += naive_res.total_revenue
                daily_rev_sum += daily_res.total_revenue
                n_test_days += 1
            except Exception as e:
                logger.warning(f"Day {d} skipped: {e}")

        test_elapsed = time.time() - tb0
        gain = daily_rev_sum - naive_rev_sum
        pct = gain / max(abs(naive_rev_sum), 1.0) * 100 if n_test_days else 0

        logger.info(
            f"Ck {ck_idx+1}/{len(checkpoints)} day={ck}: "
            f"train={train_elapsed:.1f}s, test {n_test_days}d={test_elapsed:.1f}s, "
            f"naive=¥{naive_rev_sum:>10,.0f}, daily=¥{daily_rev_sum:>10,.0f}, "
            f"gain={pct:+.2f}%, comply {daily_compliance_ok}/{n_test_days}"
        )

    df_results = pd.DataFrame(daily_results)
    return df_results


# ============================================================
# Summary
# ============================================================

def print_summary(df: pd.DataFrame):
    logger.info(f"\n{'=' * 90}")
    logger.info("  STRICT WALK-FORWARD SUMMARY (全数据, 无作弊)")
    logger.info(f"{'=' * 90}")

    n = len(df)
    logger.info(f"Total test days: {n}")
    logger.info(f"Date range: {df['date'].min()} → {df['date'].max()}")
    logger.info(f"DA price range: {df['da_mean'].min():.0f} → {df['da_mean'].max():.0f} 元/MWh (daily mean)")
    logger.info(f"Compliance: {df['compliant'].sum()}/{n} days ({df['compliant'].mean()*100:.1f}%)")
    logger.info(f"Daily bid avg segments: {df['daily_segments'].mean():.2f}")

    logger.info("")
    logger.info("Total revenues:")
    total_naive = df["naive_rev"].sum()
    total_daily = df["daily_rev"].sum()
    gain = total_daily - total_naive
    pct = gain / abs(total_naive) * 100
    logger.info(f"  Naive: ¥{total_naive:>14,.0f}")
    logger.info(f"  Daily: ¥{total_daily:>14,.0f}")
    logger.info(f"  Gain:  ¥{gain:>+14,.0f} ({pct:+.2f}%)")

    # Annualized
    days = n
    naive_annual = total_naive / days * 365
    daily_annual = total_daily / days * 365
    logger.info("")
    logger.info("Annualized (extrapolated):")
    logger.info(f"  Naive: ¥{naive_annual:>14,.0f}/年")
    logger.info(f"  Daily: ¥{daily_annual:>14,.0f}/年")
    logger.info(f"  Gain:  ¥{daily_annual - naive_annual:>+14,.0f}/年")

    # Quarterly breakdown
    logger.info("")
    logger.info("Quarterly breakdown (first date in window):")
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("Q")
    q_agg = df.groupby("month").agg(
        naive_rev=("naive_rev", "sum"),
        daily_rev=("daily_rev", "sum"),
        n_days=("day", "count"),
    ).reset_index()
    q_agg["gain_pct"] = (q_agg["daily_rev"] - q_agg["naive_rev"]) / q_agg["naive_rev"].abs() * 100
    for _, r in q_agg.iterrows():
        logger.info(
            f"  {str(r['month']):<10} n={int(r['n_days']):>3}  "
            f"naive=¥{r['naive_rev']:>12,.0f}  daily=¥{r['daily_rev']:>12,.0f}  "
            f"gain={r['gain_pct']:+.2f}%"
        )

    # Win-rate
    df["daily_wins"] = df["daily_rev"] > df["naive_rev"]
    win_rate = df["daily_wins"].mean() * 100
    logger.info("")
    logger.info(f"Daily win rate (Daily > Naive): {win_rate:.1f}% of days")

    # Drawdown analysis
    df["gain"] = df["daily_rev"] - df["naive_rev"]
    logger.info("")
    logger.info(f"Per-day gain distribution:")
    logger.info(f"  min: ¥{df['gain'].min():+,.0f}  (worst day Daily underperforms Naive)")
    logger.info(f"  25%: ¥{df['gain'].quantile(0.25):+,.0f}")
    logger.info(f"  50%: ¥{df['gain'].median():+,.0f}")
    logger.info(f"  75%: ¥{df['gain'].quantile(0.75):+,.0f}")
    logger.info(f"  max: ¥{df['gain'].max():+,.0f}  (best day)")


def main(province: str = "gansu", capacity_mw: float = 100.0):
    df_results = run_full_walk_forward(
        province=province,
        capacity_mw=capacity_mw,
    )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RUNS_DIR / "full_walk_forward.csv", index=False)
    logger.info(f"\nSaved per-day results: {RUNS_DIR / 'full_walk_forward.csv'}")
    print_summary(df_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="gansu")
    parser.add_argument("--capacity", type=float, default=100.0)
    args = parser.parse_args()
    main(args.province, args.capacity)
