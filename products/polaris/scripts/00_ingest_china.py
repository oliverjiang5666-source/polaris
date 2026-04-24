#!/usr/bin/env python3
"""
Step 0: 中国市场数据接入

Excel → 清洗 → 特征工程 → parquet
"""

import sys
from pathlib import Path

# 添加项目根目录到path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger
from data.china.ingest import ingest_all
from data.china.features import build_all_features


def main():
    logger.info("=" * 60)
    logger.info("Step 0: Ingesting China market data")
    logger.info("=" * 60)

    # Step 1: Excel → clean parquet
    logger.info("\n[1/2] Ingesting raw data...")
    results = ingest_all()

    if not results:
        logger.error("No data ingested! Check data directory.")
        return

    # Step 2: Clean parquet → features parquet
    logger.info("\n[2/2] Building features...")
    features = build_all_features()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    for province, df in features.items():
        n_days = len(df) / 96
        logger.info(
            f"  {province:<12s}: {len(df):>8,} rows "
            f"({n_days:.0f} days), "
            f"{df.index.min().date()} → {df.index.max().date()}"
        )


if __name__ == "__main__":
    main()
