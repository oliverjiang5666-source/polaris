"""
Logan · Script 05 — Rolling Window Calibration Test
=====================================================

诊断：Supply Curve 的分位数预测为什么偏低？
测试：用"最近 N 个月"训练是否改善 calibration？

假设（来自 P0 元思考）：
    甘肃 DA 价有上涨趋势（distribution shift）
    → 用全量历史训练的模型预测偏低
    → P50 empirical coverage 71% 而非 50%

方法：
    固定 test 窗口（最后 20%），换不同 train 窗口：
      full:     train = 全部历史
      recent24: train = 最近 24 个月
      recent18: train = 最近 18 个月
      recent12: train = 最近 12 个月
      recent6:  train = 最近 6 个月

    对每种：在 test 上测 RMSE + quantile coverage。
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.calendar_features import add_calendar_features
from core.supply_curve import SupplyCurve, SupplyCurveConfig


PROCESSED_DIR = ROOT / "data" / "china" / "processed"


def test_window(province: str, df: pd.DataFrame, train_days: int | None, df_test: pd.DataFrame) -> dict:
    """给定 train 窗口大小（None = 全量），评估 test 表现"""
    if train_days is None:
        df_train = df.iloc[:int(len(df) * 0.8)]
        win_name = "full"
    else:
        # 从 test 起点往前 取 train_days × 96 行
        test_start = int(len(df) * 0.8)
        train_start = max(0, test_start - train_days * 96)
        df_train = df.iloc[train_start:test_start]
        win_name = f"recent{train_days}d"

    sc = SupplyCurve(SupplyCurveConfig(
        seasonal=True,
        time_of_day_split=True,
        residual_model=True,
        quantile_regression=True,
    ))
    sc.fit(df_train)

    # Test prediction
    pred = sc.predict(
        net_load=df_test["net_load"].values,
        season=df_test["season"].values,
        hour_bucket=df_test["hour_bucket"].values,
        extra_df=df_test,
    )
    actual = df_test["da_price"].values
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    mae = np.mean(np.abs(pred - actual))
    # mean bias (预测 - 实际)
    bias = np.mean(pred - actual)

    # Quantile coverage
    coverages = {}
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        pred_q = sc.predict_quantile(
            net_load=df_test["net_load"].values,
            season=df_test["season"].values,
            hour_bucket=df_test["hour_bucket"].values,
            quantile=q,
            extra_df=df_test,
        )
        coverages[q] = float(np.mean(actual <= pred_q))

    return {
        "window": win_name,
        "train_rows": len(df_train),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "test_bias": float(bias),
        "coverage_p05": coverages[0.05],
        "coverage_p25": coverages[0.25],
        "coverage_p50": coverages[0.5],
        "coverage_p75": coverages[0.75],
        "coverage_p95": coverages[0.95],
    }


def main(province: str = "gansu"):
    logger.info(f"=== Calibration Test ({province}) ===")

    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")
    df = add_calendar_features(df)
    wind = df["wind_mw"].fillna(0) if "wind_mw" in df.columns else 0
    solar = df["solar_mw"].fillna(0) if "solar_mw" in df.columns else 0
    df["net_load"] = df["load_mw"].fillna(0) - wind - solar
    df = df[df["da_price"].notna() & df["load_mw"].notna()]

    # Common test set: last 20%
    test_start = int(len(df) * 0.8)
    df_test = df.iloc[test_start:].copy()
    logger.info(f"Test: {df_test.index.min()} → {df_test.index.max()} ({len(df_test):,} rows)")
    logger.info(f"Test DA price mean: {df_test['da_price'].mean():.1f}, std: {df_test['da_price'].std():.1f}")

    results = []
    for train_days in [None, 730, 540, 365, 180]:
        r = test_window(province, df, train_days, df_test)
        logger.info(
            f"  {r['window']:<10} rows={r['train_rows']:>7,}  "
            f"RMSE={r['test_rmse']:>6.1f}  MAE={r['test_mae']:>6.1f}  bias={r['test_bias']:>+7.1f}"
        )
        logger.info(
            f"            coverage: P05={r['coverage_p05']*100:.0f}%  "
            f"P25={r['coverage_p25']*100:.0f}%  P50={r['coverage_p50']*100:.0f}%  "
            f"P75={r['coverage_p75']*100:.0f}%  P95={r['coverage_p95']*100:.0f}% "
            f"(ideal 5/25/50/75/95)"
        )
        results.append(r)

    # Summary: which window is best?
    logger.info("")
    logger.info("Summary:")
    for r in results:
        p50_err = abs(r["coverage_p50"] - 0.5) * 100
        logger.info(
            f"  {r['window']:<10}  RMSE={r['test_rmse']:>5.1f}  "
            f"P50 miscalibration={p50_err:>4.1f}pp  bias={r['test_bias']:>+7.1f}"
        )


if __name__ == "__main__":
    main()
