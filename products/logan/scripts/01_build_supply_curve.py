"""
Logan · Script 01 — Build & visualize Supply Curve (Gansu)
==========================================================

拟合甘肃的供给曲线（DA 价 = f(净负荷)），输出：
  - 文字摘要：每个 bucket 的样本量 + 残差 RMSE
  - 甘肃全年的 "净负荷 → DA 价" 散点 + isotonic 曲线图（可选 matplotlib）
  - 分位数预测 sample check

用法：
    PYTHONPATH=. python3 products/logan/scripts/01_build_supply_curve.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# 让 core/products 可 import
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.calendar_features import add_calendar_features
from core.supply_curve import SupplyCurve, SupplyCurveConfig


PROCESSED_DIR = ROOT / "data" / "china" / "processed"


def main(province: str = "gansu"):
    logger.info(f"=== Logan Script 01: Supply Curve ({province}) ===")

    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")
    logger.info(f"Loaded {len(df):,} rows, {df.index.min()} → {df.index.max()}")

    # 加日历特征
    df = add_calendar_features(df)

    # 提前加 net_load 列，train/test 都能用（避免 plot 时缺列）
    wind = df["wind_mw"].fillna(0) if "wind_mw" in df.columns else 0
    solar = df["solar_mw"].fillna(0) if "solar_mw" in df.columns else 0
    df["net_load"] = df["load_mw"].fillna(0) - wind - solar

    # 只用有 DA 价格的行训练
    df = df[df["da_price"].notna() & df["load_mw"].notna()]
    logger.info(f"Valid rows: {len(df):,}")

    # 用前 80% 训练（留 20% 做 out-of-sample 评估）
    split = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]
    logger.info(f"Train: {len(df_train):,}  Test: {len(df_test):,}")

    # Fit
    sc = SupplyCurve(SupplyCurveConfig(
        seasonal=True,
        time_of_day_split=True,
        residual_model=True,
        quantile_regression=True,
    ))
    sc.fit(df_train)

    logger.info("")
    logger.info("Summary:")
    desc = sc.describe()
    for k, v in desc.items():
        logger.info(f"  {k}: {v}")

    # Out-of-sample test
    logger.info("")
    logger.info("=== Out-of-sample evaluation ===")
    df_test = df_test.copy()
    if "net_load" not in df_test.columns:
        wind = df_test.get("wind_mw", 0).fillna(0) if "wind_mw" in df_test.columns else 0
        solar = df_test.get("solar_mw", 0).fillna(0) if "solar_mw" in df_test.columns else 0
        df_test["net_load"] = df_test["load_mw"].fillna(0) - wind - solar

    pred = sc.predict(
        net_load=df_test["net_load"].values,
        season=df_test["season"].values,
        hour_bucket=df_test["hour_bucket"].values,
        extra_df=df_test,
    )
    actual = df_test["da_price"].values
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    mae = np.mean(np.abs(pred - actual))
    logger.info(f"Test RMSE: {rmse:.1f} 元/MWh")
    logger.info(f"Test MAE:  {mae:.1f} 元/MWh")
    logger.info(f"Test actual mean: {actual.mean():.1f}, std: {actual.std():.1f}")

    # Base-only (无残差) vs Full 对比
    base_only_mask = np.zeros(len(df_test), dtype=bool)  # 不用 extra
    pred_base = np.array([
        sc._predict_base_scalar(
            float(df_test["net_load"].iloc[i]),
            int(df_test["season"].iloc[i]),
            int(df_test["hour_bucket"].iloc[i]),
        )
        for i in range(len(df_test))
    ])
    rmse_base = np.sqrt(np.mean((pred_base - actual) ** 2))
    logger.info(f"Test RMSE (供给曲线 base only, 无残差): {rmse_base:.1f} 元/MWh")
    logger.info(f"残差层带来 RMSE 改进: {rmse_base - rmse:+.1f} 元/MWh ({(rmse_base - rmse) / rmse_base * 100:+.1f}%)")

    # Quantile coverage
    logger.info("")
    logger.info("=== Quantile calibration ===")
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        pred_q = sc.predict_quantile(
            net_load=df_test["net_load"].values,
            season=df_test["season"].values,
            hour_bucket=df_test["hour_bucket"].values,
            quantile=q,
            extra_df=df_test,
        )
        coverage = float(np.mean(actual <= pred_q))
        logger.info(f"  P{int(q*100):02d}: empirical coverage = {coverage*100:5.1f}% (理想 {q*100:.0f}%)")

    # 尝试画图（可选）
    try:
        import matplotlib.pyplot as plt
        out_path = ROOT / "runs" / "logan" / "supply_curve_gansu.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, (season_name, season_id) in enumerate([("春", 0), ("夏", 1), ("秋", 2), ("冬", 3)]):
            ax = axes[i // 2, i % 2]
            mask = df_train["season"] == season_id
            ax.scatter(
                df_train.loc[mask, "net_load"],
                df_train.loc[mask, "da_price"],
                s=1, alpha=0.1, c="gray",
            )
            # Isotonic curves for 4 hour buckets
            for hb, color in zip(range(4), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
                key = (season_id, hb)
                iso = sc.iso_by_key.get(key)
                if iso is None:
                    continue
                nl_range = np.linspace(
                    df_train.loc[mask, "net_load"].quantile(0.02),
                    df_train.loc[mask, "net_load"].quantile(0.98),
                    200,
                )
                da_pred = iso.predict(nl_range)
                hb_names = ["0-6h", "6-12h", "12-18h", "18-24h"]
                ax.plot(nl_range, da_pred, color=color, linewidth=2, label=hb_names[hb])
            ax.set_title(f"{season_name} (season={season_id})")
            ax.set_xlabel("Net Load (MW)")
            ax.set_ylabel("DA Price (元/MWh)")
            ax.legend()
            ax.grid(alpha=0.3)
        plt.suptitle(f"{province.capitalize()} Supply Curves by Season × Hour Bucket")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        logger.info(f"Saved figure: {out_path}")
    except Exception as e:
        logger.warning(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
