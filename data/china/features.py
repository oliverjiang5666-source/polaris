"""
中国电力市场特征工程

31维特征 = 20维ERCOT兼容 + 11维中国独有

输入：省份清洗后的parquet（ingest.py产出）
输出：带完整特征的parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path(__file__).parent / "processed"


# ============================================================
# 特征列定义
# ============================================================

# 与ERCOT兼容的20个特征（用rt_price替代price）
PRICE_FEATURES = [
    "rt_price",
    "rt_price_lag_1", "rt_price_lag_2", "rt_price_lag_3", "rt_price_lag_4",
    "rt_price_ma_4", "rt_price_ma_16", "rt_price_ma_96",
    "rt_price_std_16", "rt_price_std_96",
    "rt_price_trend", "rt_price_percentile",
    "rt_price_ma_ratio", "rt_price_ma4_ratio",
]

TIME_FEATURES = [
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
]

# 中国市场独有的11个特征
CHINA_FEATURES = [
    "da_price",
    "da_rt_spread",
    "da_price_ma_ratio",
    "load_norm",
    "load_change",
    "renewable_penetration",
    "wind_ratio",
    "solar_ratio",
    "net_load_norm",
    "tie_line_norm",
    "temperature_norm",
]

# V4因果驱动特征: 天气→供需→电价
CAUSAL_FEATURES = [
    "wind_speed_norm",           # 风速（因→风电出力）
    "solar_radiation_norm",      # 光照（因→光伏出力）
    "temp_load_interaction",     # 温度×负荷交互（极端天气→需求激增）
    "wind_ramp",                 # 风电爬坡率（急变→价格spike）
    "solar_ramp",                # 光伏变化率（日落骤降→晚高峰）
    "net_load_ramp",             # 净负荷爬坡（供需平衡变化速度）
    "supply_demand_tightness",   # 供需紧张度（净负荷/总负荷，接近1=紧张）
    "renewable_forecast_proxy",  # 新能源出力4h MA作为预测代理
]

FEATURE_COLS = PRICE_FEATURES + TIME_FEATURES + CHINA_FEATURES

# V4: 31 base + 8 causal = 39 features
FEATURE_COLS_V4 = FEATURE_COLS + CAUSAL_FEATURES

N_FEATURES = len(FEATURE_COLS)  # 31
N_FEATURES_V4 = len(FEATURE_COLS_V4)  # 39


# ============================================================
# 特征构建
# ============================================================

def build_features(df: pd.DataFrame, province: str = "") -> pd.DataFrame:
    """
    为一个省的清洗数据构建31维特征

    Args:
        df: ingest.py产出的清洗数据（index=timestamp, columns=标准列名）
        province: 省份名（用于日志）

    Returns:
        带有FEATURE_COLS所有列的DataFrame
    """
    df = df.copy()
    df = df.sort_index()

    logger.info(f"Building features for {province}: {len(df):,} rows, {len(df.columns)} columns")

    # --- 实时价格特征（20维中的14维） ---
    _build_price_features(df, "rt_price")

    # --- 日前价格特征（3维） ---
    _build_da_features(df)

    # --- 基本面特征（5维） ---
    _build_fundamental_features(df)

    # --- 联络线特征（1维） ---
    _build_tie_line_features(df)

    # --- 天气/温度特征（1维） ---
    _build_temperature_features(df)

    # --- V4因果特征（8维） ---
    _build_causal_features(df)

    # --- 时间编码（6维） ---
    _build_time_features(df)

    # --- 清理 ---
    # 去掉warmup期（需要96步MA）
    warmup_col = "rt_price_ma_96"
    if warmup_col in df.columns:
        first_valid = df[warmup_col].first_valid_index()
        if first_valid is not None:
            before = len(df)
            df = df.loc[first_valid:]
            logger.info(f"  Dropped {before - len(df)} warmup rows")

    # 确保所有特征列存在（包括V4因果特征）
    all_cols = FEATURE_COLS_V4
    for col in all_cols:
        if col not in df.columns:
            logger.warning(f"  Feature '{col}' missing, filling with 0.0")
            df[col] = 0.0

    # 前向填充剩余NaN
    for col in all_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            df[col] = df[col].ffill().fillna(0.0)

    logger.info(f"  Features complete: {len(df):,} rows × {N_FEATURES} features")

    return df


def _build_price_features(df: pd.DataFrame, price_col: str = "rt_price"):
    """构建价格相关的14个特征"""
    if price_col not in df.columns:
        logger.error(f"  Price column '{price_col}' not found!")
        return

    p = df[price_col]

    # Lags
    for i in range(1, 5):
        df[f"{price_col}_lag_{i}"] = p.shift(i)

    # Moving averages
    df[f"{price_col}_ma_4"] = p.rolling(4, min_periods=1).mean()
    df[f"{price_col}_ma_16"] = p.rolling(16, min_periods=1).mean()
    df[f"{price_col}_ma_96"] = p.rolling(96, min_periods=1).mean()

    # Volatility
    df[f"{price_col}_std_16"] = p.rolling(16, min_periods=1).std().fillna(0)
    df[f"{price_col}_std_96"] = p.rolling(96, min_periods=1).std().fillna(0)

    # Trend
    df[f"{price_col}_trend"] = df[f"{price_col}_ma_4"] - df[f"{price_col}_ma_16"]

    # Percentile in 24h window
    df[f"{price_col}_percentile"] = p.rolling(96, min_periods=1).rank(pct=True)

    # 关键特征：价格/均值比
    ma96 = df[f"{price_col}_ma_96"].replace(0, np.nan)
    df[f"{price_col}_ma_ratio"] = (p / ma96).clip(0.1, 10.0).fillna(1.0)

    ma4 = df[f"{price_col}_ma_4"].replace(0, np.nan)
    df[f"{price_col}_ma4_ratio"] = (p / ma4).clip(0.1, 10.0).fillna(1.0)


def _build_da_features(df: pd.DataFrame):
    """构建日前价格相关的3个特征"""
    if "da_price" not in df.columns:
        df["da_price"] = 0.0
        df["da_rt_spread"] = 0.0
        df["da_price_ma_ratio"] = 1.0
        return

    df["da_rt_spread"] = df["da_price"] - df.get("rt_price", 0)

    da_ma96 = df["da_price"].rolling(96, min_periods=1).mean().replace(0, np.nan)
    df["da_price_ma_ratio"] = (df["da_price"] / da_ma96).clip(0.1, 10.0).fillna(1.0)


def _build_fundamental_features(df: pd.DataFrame):
    """构建负荷+新能源的5个特征"""
    # 负荷
    if "load_mw" in df.columns:
        load_ma = df["load_mw"].rolling(96, min_periods=1).mean().clip(lower=1)
        df["load_norm"] = df["load_mw"] / load_ma
        df["load_change"] = df["load_mw"].pct_change(4).clip(-1, 1).fillna(0)
    else:
        df["load_norm"] = 1.0
        df["load_change"] = 0.0

    # 新能源渗透率
    if "renewable_mw" in df.columns and "load_mw" in df.columns:
        load_safe = df["load_mw"].clip(lower=1)
        df["renewable_penetration"] = (df["renewable_mw"] / load_safe).clip(0, 1).fillna(0)
    else:
        df["renewable_penetration"] = 0.0

    # 风/光占比
    if "wind_mw" in df.columns and "renewable_mw" in df.columns:
        ren_safe = df["renewable_mw"].clip(lower=1)
        df["wind_ratio"] = (df["wind_mw"] / ren_safe).clip(0, 1).fillna(0.5)
    else:
        df["wind_ratio"] = 0.5

    if "solar_mw" in df.columns and "renewable_mw" in df.columns:
        ren_safe = df["renewable_mw"].clip(lower=1)
        df["solar_ratio"] = (df["solar_mw"] / ren_safe).clip(0, 1).fillna(0.5)
    else:
        df["solar_ratio"] = 0.5

    # 净负荷
    if "load_mw" in df.columns and "renewable_mw" in df.columns:
        load_safe = df["load_mw"].clip(lower=1)
        net_load = df["load_mw"] - df["renewable_mw"]
        df["net_load_norm"] = (net_load / load_safe).clip(-1, 2).fillna(1.0)
    else:
        df["net_load_norm"] = 1.0


def _build_tie_line_features(df: pd.DataFrame):
    """构建联络线特征"""
    if "tie_line_mw" in df.columns:
        tie_ma = df["tie_line_mw"].rolling(96, min_periods=1).mean()
        tie_ma_abs = tie_ma.abs().clip(lower=1)
        df["tie_line_norm"] = df["tie_line_mw"] / tie_ma_abs
    else:
        df["tie_line_norm"] = 0.0


def _build_temperature_features(df: pd.DataFrame):
    """构建温度特征（如果有温度数据）"""
    if "temperature" in df.columns:
        temp_ma = df["temperature"].rolling(96 * 7, min_periods=1).mean()
        temp_std = df["temperature"].rolling(96 * 7, min_periods=1).std().clip(lower=1)
        df["temperature_norm"] = ((df["temperature"] - temp_ma) / temp_std).fillna(0)
    else:
        df["temperature_norm"] = 0.0


def _build_causal_features(df: pd.DataFrame):
    """
    V4因果驱动特征：建模 天气→供需→电价 的因果链。

    核心思想：不只看"供需是什么"，而是看"供需为什么变化、变化多快"。
    """
    # --- 天气因果特征（天气→新能源出力） ---
    if "wind_speed_10m" in df.columns:
        # 风速归一化（用7天滑动窗口，捕捉季节性变化）
        ws = df["wind_speed_10m"]
        ws_ma = ws.rolling(96 * 7, min_periods=1).mean().clip(lower=0.1)
        df["wind_speed_norm"] = (ws / ws_ma).clip(0, 5).fillna(1.0)
    else:
        df["wind_speed_norm"] = 1.0

    if "shortwave_radiation" in df.columns:
        # 光照归一化（用日间均值避免夜间=0干扰）
        sr = df["shortwave_radiation"]
        sr_ma = sr.rolling(96 * 7, min_periods=1).mean().clip(lower=1.0)
        df["solar_radiation_norm"] = (sr / sr_ma).clip(0, 5).fillna(0.0)
    else:
        df["solar_radiation_norm"] = 0.0

    # --- 温度×负荷交互（极端天气→需求激增） ---
    if "temperature_2m" in df.columns and "load_mw" in df.columns:
        temp = df["temperature_2m"]
        # 极端温度偏离度（|T - 20°C|/10，夏冬极端时>2）
        temp_extreme = ((temp - 20).abs() / 10).clip(0, 4)
        load_norm = df.get("load_norm", pd.Series(1.0, index=df.index))
        df["temp_load_interaction"] = (temp_extreme * load_norm).fillna(0)
    else:
        df["temp_load_interaction"] = 0.0

    # --- 供给侧爬坡率（急变→价格spike的直接原因） ---
    if "wind_mw" in df.columns:
        # 风电1小时变化率（4步=1小时，归一化到负荷）
        wind_change = df["wind_mw"].diff(4)
        load_safe = df.get("load_mw", pd.Series(1.0, index=df.index)).clip(lower=100)
        df["wind_ramp"] = (wind_change / load_safe).clip(-0.5, 0.5).fillna(0)
    else:
        df["wind_ramp"] = 0.0

    if "solar_mw" in df.columns:
        # 光伏1小时变化率（日落时为大负值→晚高峰价格上涨）
        solar_change = df["solar_mw"].diff(4)
        load_safe = df.get("load_mw", pd.Series(1.0, index=df.index)).clip(lower=100)
        df["solar_ramp"] = (solar_change / load_safe).clip(-0.5, 0.5).fillna(0)
    elif "renewable_mw" in df.columns:
        # 广东没有wind/solar分拆，用renewable整体
        ren_change = df["renewable_mw"].diff(4)
        load_safe = df.get("load_mw", pd.Series(1.0, index=df.index)).clip(lower=100)
        df["solar_ramp"] = (ren_change / load_safe).clip(-0.5, 0.5).fillna(0)
    else:
        df["solar_ramp"] = 0.0

    # --- 净负荷爬坡率（供需平衡变化速度→边际机组调度的触发信号） ---
    if "load_mw" in df.columns and "renewable_mw" in df.columns:
        net_load = df["load_mw"] - df["renewable_mw"]
        net_load_change = net_load.diff(4)
        load_safe = df["load_mw"].clip(lower=100)
        df["net_load_ramp"] = (net_load_change / load_safe).clip(-0.5, 0.5).fillna(0)
    else:
        df["net_load_ramp"] = 0.0

    # --- 供需紧张度（净负荷占总负荷比例，越高越紧张→高电价） ---
    if "load_mw" in df.columns and "renewable_mw" in df.columns:
        load_safe = df["load_mw"].clip(lower=100)
        tightness = (df["load_mw"] - df["renewable_mw"]) / load_safe
        df["supply_demand_tightness"] = tightness.clip(-0.5, 2.0).fillna(1.0)
    else:
        df["supply_demand_tightness"] = 1.0

    # --- 新能源出力4h MA（作为"预测"代理，平滑的趋势比瞬时值更接近forecaster能看到的） ---
    if "renewable_mw" in df.columns and "load_mw" in df.columns:
        ren_pen = df["renewable_mw"] / df["load_mw"].clip(lower=100)
        df["renewable_forecast_proxy"] = ren_pen.rolling(16, min_periods=1).mean().clip(0, 1).fillna(0)
    else:
        df["renewable_forecast_proxy"] = 0.0


def _build_time_features(df: pd.DataFrame):
    """构建时间编码（正弦/余弦）"""
    ts = df.index
    df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * ts.weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * ts.weekday / 7)
    df["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * ts.month / 12)


# ============================================================
# 批量处理
# ============================================================

def build_all_features() -> dict[str, pd.DataFrame]:
    """为所有已清洗的省份构建特征"""
    results = {}
    for parquet_file in sorted(PROCESSED_DIR.glob("*_clean.parquet")):
        province = parquet_file.stem.replace("_clean", "")
        logger.info(f"\n{'='*60}")
        df = pd.read_parquet(parquet_file)
        df_feat = build_features(df, province=province)

        out_path = PROCESSED_DIR / f"{province}_features.parquet"
        df_feat.to_parquet(out_path)
        logger.info(f"  Saved: {out_path}")

        results[province] = df_feat

    return results


if __name__ == "__main__":
    build_all_features()
