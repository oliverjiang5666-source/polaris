"""
特征工程 — 从RTM电价构建训练特征

输入：ercot_rtm_spp.parquet（15分钟粒度）
输出：单个hub的特征DataFrame，每行=一个15分钟时段
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def build_features(hub: str = "HB_WEST") -> pd.DataFrame:
    """为指定hub构建特征"""
    raw_dir = Path(__file__).parent / "raw"
    rtm = pd.read_parquet(raw_dir / "ercot_rtm_spp.parquet")

    # 筛选hub
    df = rtm[rtm["Settlement Point Name"] == hub].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.rename(columns={"Settlement Point Price": "price"})
    logger.info(f"{hub}: {len(df):,} rows, {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # === 价格特征 ===
    df["price_lag_1"] = df["price"].shift(1)
    df["price_lag_2"] = df["price"].shift(2)
    df["price_lag_3"] = df["price"].shift(3)
    df["price_lag_4"] = df["price"].shift(4)

    # 滚动统计
    df["price_ma_4"] = df["price"].rolling(4).mean()       # 1小时
    df["price_ma_16"] = df["price"].rolling(16).mean()      # 4小时
    df["price_ma_96"] = df["price"].rolling(96).mean()      # 24小时
    df["price_std_16"] = df["price"].rolling(16).std()      # 4小时波动
    df["price_std_96"] = df["price"].rolling(96).std()      # 24小时波动

    # 趋势
    df["price_trend"] = df["price_ma_4"] - df["price_ma_16"]

    # 百分位（当前价格在24小时内的位置）
    df["price_percentile"] = df["price"].rolling(96).rank(pct=True)

    # 价格比率（Threshold策略的核心特征）
    df["price_ma_ratio"] = (df["price"] / df["price_ma_96"].clip(lower=1)).clip(upper=10)
    df["price_ma4_ratio"] = (df["price"] / df["price_ma_4"].clip(lower=1)).clip(upper=10)

    # === 日前市场(DAM)价格特征 ===
    dam_path = raw_dir / "ercot_dam_spp.parquet"
    if dam_path.exists():
        dam = pd.read_parquet(dam_path)
        dam_hub = dam[dam["Settlement Point"] == hub][["timestamp", "Settlement Point Price"]].copy()
        dam_hub = dam_hub.rename(columns={"Settlement Point Price": "dam_price"})
        dam_hub = dam_hub.sort_values("timestamp").drop_duplicates("timestamp")
        # DAM是小时级，RTM是15分钟级 → 用floor对齐到小时，然后merge
        df["_hour_ts"] = df["timestamp"].dt.floor("h")
        df = df.merge(dam_hub, left_on="_hour_ts", right_on="timestamp",
                       how="left", suffixes=("", "_dam"))
        df = df.drop(columns=["_hour_ts", "timestamp_dam"], errors="ignore")
        # DAM-RTM价差（核心套利信号）
        df["dam_rtm_spread"] = df["dam_price"] - df["price"]
        # DAM价格比率
        df["dam_ratio"] = (df["price"] / df["dam_price"].clip(lower=1)).clip(-5, 10)
        df["dam_price"] = df["dam_price"].fillna(df["price"])
        df["dam_rtm_spread"] = df["dam_rtm_spread"].fillna(0)
        df["dam_ratio"] = df["dam_ratio"].fillna(1)

        # === DAM前瞻特征（合法：DAM在前一天下午就公布了全天价格）===
        # 构建每日DAM统计（向量化，不用iterrows）
        dam_hub["date"] = dam_hub["timestamp"].dt.date
        dam_hub["dam_hour"] = dam_hub["timestamp"].dt.hour

        # 每日DAM最高/最低价
        daily_stats = dam_hub.groupby("date")["dam_price"].agg(["max", "min"]).reset_index()
        daily_stats.columns = ["date", "dam_day_max", "dam_day_min"]

        # 每日每小时的DAM价格 pivot
        dam_pivot = dam_hub.pivot_table(index="date", columns="dam_hour",
                                         values="dam_price", aggfunc="first")

        # 计算"从当前小时到23点"的剩余最高/最低DAM
        # 预计算每个(date, hour)的remaining max/min
        remaining_max = {}
        remaining_min = {}
        for d in dam_pivot.index:
            row = dam_pivot.loc[d].dropna()
            # 从后往前扫描（reverse cumulative max/min）
            hours = sorted(row.index)
            rmax, rmin = row[hours[-1]], row[hours[-1]]
            for h in reversed(hours):
                rmax = max(rmax, row[h])
                rmin = min(rmin, row[h])
                remaining_max[(d, h)] = rmax
                remaining_min[(d, h)] = rmin

        # 构建lookup表用merge（比apply快100x）
        lookup_rows = []
        for d in dam_pivot.index:
            row = dam_pivot.loc[d]
            for h in range(24):
                lookup_rows.append({
                    "date": d, "hour_int": h,
                    "dam_remaining_max": remaining_max.get((d, h), np.nan),
                    "dam_remaining_min": remaining_min.get((d, h), np.nan),
                    "dam_next_4h": row.get(min(h + 4, 23), np.nan),
                    "dam_next_8h": row.get(min(h + 8, 23), np.nan),
                })
        lookup = pd.DataFrame(lookup_rows)

        df["_date"] = df["timestamp"].dt.date
        df["_hour_int"] = df["timestamp"].dt.hour
        df = df.merge(lookup, left_on=["_date", "_hour_int"],
                       right_on=["date", "hour_int"], how="left")
        df = df.drop(columns=["_date", "_hour_int", "date", "hour_int"])

        # dam_position: 当前DAM在今日剩余区间的位置
        rng = (df["dam_remaining_max"] - df["dam_remaining_min"]).clip(lower=1)
        df["dam_position"] = ((df["dam_price"] - df["dam_remaining_min"]) / rng).clip(0, 1)

        # fillna
        for col in ["dam_remaining_max", "dam_remaining_min", "dam_next_4h", "dam_next_8h"]:
            df[col] = df[col].fillna(df["price"])
        df["dam_position"] = df["dam_position"].fillna(0.5)

        logger.info(f"DAM features merged: {dam_hub.shape[0]} hourly prices + lookahead")
    else:
        logger.warning("DAM data not found, skipping DAM features")
        df["dam_price"] = df["price"]
        df["dam_rtm_spread"] = 0.0
        df["dam_ratio"] = 1.0

    # === 时间特征（周期编码）===
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month"] = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 丢弃warmup期
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Features built: {len(df):,} rows, {len(df.columns)} columns")

    return df


FEATURE_COLS = [
    "price", "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_4",
    "price_ma_4", "price_ma_16", "price_ma_96",
    "price_std_16", "price_std_96",
    "price_trend", "price_percentile",
    "price_ma_ratio", "price_ma4_ratio",
    "dam_price", "dam_rtm_spread", "dam_ratio",
    "dam_remaining_max", "dam_remaining_min", "dam_next_4h", "dam_next_8h", "dam_position",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
]
