"""
数据加载器：复用现有 parquet，拆分 walk-forward 训练/测试集。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "china" / "processed"


@dataclass
class ProvinceData:
    """省级数据容器"""
    province: str
    df: pd.DataFrame
    rt_prices: np.ndarray       # [n_days, 96]
    dam_prices: np.ndarray      # [n_days, 96]
    n_days: int

    def get_day(self, d: int, price_type: str = "rt") -> np.ndarray:
        """取第 d 天的 96 点价格"""
        return (self.rt_prices[d] if price_type == "rt" else self.dam_prices[d]).copy()

    def get_features_for_day(self, d: int, features_cols: list[str]) -> pd.DataFrame:
        """取第 d 天的特征（96 行）"""
        start, end = d * 96, (d + 1) * 96
        return self.df[features_cols].iloc[start:end]


def load_province(province: str, processed_dir: Path | None = None) -> ProvinceData:
    """
    加载省级数据并 reshape 成每日 96 点格式。

    Args:
        province: "shandong" / "shanxi" / "guangdong" / "gansu"
        processed_dir: parquet 目录

    Returns:
        ProvinceData 容器
    """
    pdir = processed_dir or PROCESSED_DIR
    path = pdir / f"{province}_oracle.parquet"
    assert path.exists(), f"Missing {path}"

    df = pd.read_parquet(path)
    df = df.dropna(subset=["rt_price", "da_price"])

    n_total = len(df)
    n_days = n_total // 96

    rt_flat = df["rt_price"].values[:n_days * 96].astype(np.float64)
    dam_flat = df["da_price"].values[:n_days * 96].astype(np.float64)

    rt_prices = rt_flat.reshape(n_days, 96)
    dam_prices = dam_flat.reshape(n_days, 96)

    return ProvinceData(
        province=province,
        df=df.iloc[:n_days * 96],
        rt_prices=rt_prices,
        dam_prices=dam_prices,
        n_days=n_days,
    )


def split_walkforward(
    data: ProvinceData,
    test_days: int = 365,
    n_quarters: int = 4,
) -> list[Tuple[int, int]]:
    """
    Walk-forward 季度切分，与 scripts/22_regime_v3_allprov.py 保持一致。

    Returns:
        list of (test_start_day, test_end_day)
    """
    n = data.n_days
    start = n - test_days
    if start < 400:  # 训练数据不足 400 天，缩短测试期
        start = n // 2
        test_days = n - start

    quarter_size = test_days // n_quarters
    quarters = []
    for q in range(n_quarters):
        qs = start + q * quarter_size
        qe = start + (q + 1) * quarter_size if q < n_quarters - 1 else n
        quarters.append((qs, qe))
    return quarters


if __name__ == "__main__":
    from loguru import logger
    for prov in ["shandong", "shanxi", "guangdong", "gansu"]:
        data = load_province(prov)
        logger.info(f"{prov}: {data.n_days} days")
        logger.info(f"  DAM: mean={data.dam_prices.mean():.1f}, "
                    f"min={data.dam_prices.min():.1f}, "
                    f"max={data.dam_prices.max():.1f}")
        logger.info(f"  RT:  mean={data.rt_prices.mean():.1f}, "
                    f"min={data.rt_prices.min():.1f}, "
                    f"max={data.rt_prices.max():.1f}")
        quarters = split_walkforward(data)
        logger.info(f"  Quarters: {quarters}")
