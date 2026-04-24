"""
Logan · Script 02 — Train all 4 heads on Gansu
===============================================

训练 + 保存 4 个 head：
  - Regime Classifier
  - DAForecaster (含 supply curve + net load)
  - SpreadDirectionClassifier
  - SystemDeviationProxy
  - RTForecaster

保存到 `models/logan/{province}/` pickle 文件。

用法：
    PYTHONPATH=. python3 products/logan/scripts/02_train_all_heads.py
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
from core.regime_classifier import RegimeClassifier
from products.logan.da_forecaster import DAForecaster
from products.logan.spread_direction import SpreadDirectionClassifier
from products.logan.system_deviation import SystemDeviationProxy
from products.logan.rt_forecaster import RTForecaster


PROCESSED_DIR = ROOT / "data" / "china" / "processed"
MODELS_DIR = ROOT / "models" / "logan"


def train_province(province: str, train_frac: float = 0.8, recent_days: int | None = None):
    logger.info(f"=== Training Logan heads for {province} ===")

    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")
    logger.info(f"Loaded {len(df):,} rows")

    df = add_calendar_features(df)
    df = df[df["da_price"].notna() & df["load_mw"].notna()]
    logger.info(f"Valid rows (DA & load): {len(df):,}")

    split = int(len(df) * train_frac)
    if recent_days is not None:
        train_start = max(0, split - recent_days * 96)
        df_train = df.iloc[train_start:split].copy()
        logger.info(f"Using recent {recent_days} days window (rolling calibration)")
    else:
        df_train = df.iloc[:split].copy()
    logger.info(f"Training on {len(df_train):,} rows ({df_train.index.min()} → {df_train.index.max()})")

    # -------- Head (prerequisite): Regime --------
    logger.info("\n[1/5] Training Regime Classifier...")
    rc = RegimeClassifier(n_regimes=12)
    rc.fit(df_train)

    # -------- Head 1: DA Forecaster --------
    logger.info("\n[2/5] Training DA Forecaster...")
    da_fcst = DAForecaster()
    da_fcst.fit(df_train)

    # -------- Head 3: Spread Direction --------
    logger.info("\n[3/5] Training Spread Direction Classifier...")
    spread_clf = SpreadDirectionClassifier(regime_classifier=rc)
    spread_clf.fit(df_train)

    # -------- Head 4: System Deviation Proxy --------
    logger.info("\n[4/5] Training System Deviation Proxy...")
    sys_dev = SystemDeviationProxy(regime_classifier=rc)
    sys_dev.fit(df_train)

    # -------- Head 2: RT Forecaster (Kalman) --------
    logger.info("\n[5/5] Training RT Forecaster (Kalman)...")
    rt_fcst = RTForecaster()
    rt_fcst.fit(df_train)

    # Save
    out_dir = MODELS_DIR / province
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "regime_classifier.pkl", "wb") as f:
        pickle.dump(rc, f)
    with open(out_dir / "da_forecaster.pkl", "wb") as f:
        pickle.dump(da_fcst, f)
    with open(out_dir / "spread_direction.pkl", "wb") as f:
        pickle.dump(spread_clf, f)
    with open(out_dir / "system_deviation.pkl", "wb") as f:
        pickle.dump(sys_dev, f)
    with open(out_dir / "rt_forecaster.pkl", "wb") as f:
        pickle.dump(rt_fcst, f)

    logger.info(f"\nAll 5 heads saved to {out_dir}/")
    logger.info(f"Train end index: {split}  (→ first test day idx = {split // 96})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="gansu", choices=["gansu", "shandong", "shanxi", "guangdong"])
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--recent-days", type=int, default=None,
                        help="Use only most recent N days before test start (rolling window calibration)")
    args = parser.parse_args()
    train_province(args.province, args.train_frac, args.recent_days)
