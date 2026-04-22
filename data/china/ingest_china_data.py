"""
中国电力现货市场真实数据接入管线

数据源：4省（广东、山东、山西、甘肃）Excel文件
字段：15分钟现价、日前电价、负荷、实时电价、联络线、
      新能源出力、风电出力、光伏出力、天气

目标：对齐到与ERCOT pipeline相同的接口格式，
      同时保留中国市场独有的丰富特征
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# ============================================================
# Step 1: 数据探测 —— 自动识别Excel结构
# ============================================================

def probe_excel(filepath: str | Path) -> dict:
    """
    探测Excel文件结构，返回所有sheet及其列名、行数、样本数据
    """
    filepath = Path(filepath)
    logger.info(f"\n{'='*60}")
    logger.info(f"Probing: {filepath.name}")
    logger.info(f"Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

    xls = pd.ExcelFile(filepath)
    info = {
        "file": str(filepath),
        "size_mb": filepath.stat().st_size / 1024 / 1024,
        "sheets": {},
    }

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, nrows=10)
        info["sheets"][sheet_name] = {
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "shape_preview": f"(rows=?, cols={df.shape[1]})",
            "sample": df.head(3).to_dict(),
        }
        logger.info(f"\n  Sheet: {sheet_name}")
        logger.info(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        logger.info(f"  Sample:\n{df.head(3).to_string()}")

    return info


def probe_all():
    """探测raw目录下所有Excel文件"""
    excel_files = list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.xls"))
    if not excel_files:
        logger.warning(
            f"No Excel files found in {RAW_DIR}\n"
            f"请先将数据文件拷贝到: {RAW_DIR}"
        )
        return []

    all_info = []
    for f in sorted(excel_files):
        info = probe_excel(f)
        all_info.append(info)

    return all_info


# ============================================================
# Step 2: 数据清洗与标准化
# ============================================================

# 预期的中文列名 → 标准英文列名映射
# 根据实际Excel列名调整（probe后确认）
COLUMN_MAP_CANDIDATES = {
    # 时间
    "时间": "timestamp",
    "日期": "date",
    "时段": "period",
    "日期时间": "timestamp",

    # 价格
    "日前电价": "da_price",
    "日前价格": "da_price",
    "日前出清价格": "da_price",
    "日前出清电价": "da_price",
    "实时电价": "rt_price",
    "实时价格": "rt_price",
    "实时出清价格": "rt_price",
    "实时出清电价": "rt_price",
    "现价": "rt_price",
    "结算价格": "settlement_price",

    # 负荷
    "负荷": "load_mw",
    "系统负荷": "load_mw",
    "统调负荷": "load_mw",
    "全省负荷": "load_mw",
    "网供负荷": "load_mw",

    # 联络线
    "联络线": "tie_line_mw",
    "联络线功率": "tie_line_mw",
    "省间联络线": "tie_line_mw",
    "外送功率": "tie_line_mw",
    "外来电": "tie_line_mw",

    # 新能源
    "新能源出力": "renewable_mw",
    "新能源发电": "renewable_mw",
    "新能源总出力": "renewable_mw",
    "风电出力": "wind_mw",
    "风电发电": "wind_mw",
    "风力发电": "wind_mw",
    "光伏出力": "solar_mw",
    "光伏发电": "solar_mw",
    "太阳能出力": "solar_mw",

    # 天气
    "温度": "temperature",
    "气温": "temperature",
    "天气": "weather_desc",
    "风速": "wind_speed",
    "辐照": "solar_irradiance",
    "湿度": "humidity",
}


def auto_map_columns(df: pd.DataFrame) -> dict:
    """自动匹配列名"""
    mapping = {}
    unmapped = []

    for col in df.columns:
        col_str = str(col).strip()
        if col_str in COLUMN_MAP_CANDIDATES:
            mapping[col_str] = COLUMN_MAP_CANDIDATES[col_str]
        else:
            # 模糊匹配
            matched = False
            for cn_key, en_val in COLUMN_MAP_CANDIDATES.items():
                if cn_key in col_str or col_str in cn_key:
                    mapping[col_str] = en_val
                    matched = True
                    break
            if not matched:
                unmapped.append(col_str)

    if unmapped:
        logger.warning(f"Unmapped columns: {unmapped}")

    return mapping


def clean_and_standardize(
    df: pd.DataFrame,
    province: str,
    column_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """
    清洗并标准化一个省的数据

    输出格式:
    - timestamp: datetime64[ns, Asia/Shanghai]
    - da_price: float (元/MWh)
    - rt_price: float (元/MWh)
    - load_mw: float
    - tie_line_mw: float
    - renewable_mw: float
    - wind_mw: float
    - solar_mw: float
    - temperature: float (°C)
    - province: str
    """
    df = df.copy()

    # 列名映射
    if column_mapping is None:
        column_mapping = auto_map_columns(df)

    logger.info(f"Column mapping: {column_mapping}")
    df = df.rename(columns=column_mapping)

    # 处理时间列
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns and "period" in df.columns:
        # 有些数据用 日期+时段号 表示
        df["timestamp"] = pd.to_datetime(df["date"]) + pd.to_timedelta(
            (df["period"].astype(int) - 1) * 15, unit="min"
        )
    else:
        logger.error("Cannot identify timestamp column!")
        return df

    # 本地化时区
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Shanghai")

    # 单位标准化
    # 检测电价单位：如果均值 < 2，可能是 元/kWh，需要 ×1000 转为 元/MWh
    for price_col in ["da_price", "rt_price"]:
        if price_col in df.columns:
            mean_price = df[price_col].dropna().mean()
            if 0 < mean_price < 5:
                logger.info(f"{price_col} mean={mean_price:.3f}, likely 元/kWh → ×1000 to 元/MWh")
                df[price_col] = df[price_col] * 1000
            elif mean_price > 50:
                logger.info(f"{price_col} mean={mean_price:.1f}, likely already 元/MWh")

    # 添加省份标签
    df["province"] = province

    # 排序去重
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.reset_index(drop=True)

    # 数据质量报告
    logger.info(f"\n--- {province.upper()} Data Quality ---")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Date range: {df.timestamp.min()} → {df.timestamp.max()}")
    logger.info(f"Freq check: {pd.infer_freq(df.timestamp[:100])}")

    for col in df.select_dtypes(include=[np.number]).columns:
        nans = df[col].isna().sum()
        pct = nans / len(df) * 100
        logger.info(f"  {col:<20s}: mean={df[col].mean():>10.2f}, "
                     f"min={df[col].min():>10.2f}, max={df[col].max():>10.2f}, "
                     f"NaN={nans} ({pct:.1f}%)")

    return df


# ============================================================
# Step 3: 特征工程（扩展版，利用所有新数据）
# ============================================================

# 标准特征列（对齐ERCOT的20个 + 中国市场独有的新特征）
ERCOT_COMPATIBLE_FEATURES = [
    # 原有ERCOT特征（用rt_price替代price）
    "rt_price",
    "rt_price_lag_1", "rt_price_lag_2", "rt_price_lag_3", "rt_price_lag_4",
    "rt_price_ma_4", "rt_price_ma_16", "rt_price_ma_96",
    "rt_price_std_16", "rt_price_std_96",
    "rt_price_trend", "rt_price_percentile",
    "rt_price_ma_ratio", "rt_price_ma4_ratio",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
]

CHINA_EXTRA_FEATURES = [
    # 日前价格特征（ERCOT没有的核心信号！）
    "da_price",
    "da_rt_spread",           # DA-RT价差（套利空间直接度量）
    "da_price_ma_ratio",      # DA价格/24h均值

    # 供需基本面（ERCOT没有的因果特征）
    "load_norm",              # 归一化负荷
    "load_change",            # 负荷变化率
    "renewable_penetration",  # 新能源出力/总负荷
    "wind_ratio",             # 风电/新能源
    "solar_ratio",            # 光伏/新能源
    "net_load_norm",          # (负荷-新能源)/负荷，净负荷占比
    "tie_line_norm",          # 归一化联络线

    # 天气
    "temperature_norm",       # 归一化温度
]

ALL_FEATURE_COLS = ERCOT_COMPATIBLE_FEATURES + CHINA_EXTRA_FEATURES


def build_features_china(df: pd.DataFrame) -> pd.DataFrame:
    """
    为中国市场数据构建完整特征集

    输入: clean_and_standardize()的输出
    输出: 带有ALL_FEATURE_COLS的DataFrame
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- 实时价格特征（对齐ERCOT） ---
    rt = df["rt_price"]

    df["rt_price_lag_1"] = rt.shift(1)
    df["rt_price_lag_2"] = rt.shift(2)
    df["rt_price_lag_3"] = rt.shift(3)
    df["rt_price_lag_4"] = rt.shift(4)

    df["rt_price_ma_4"] = rt.rolling(4).mean()       # 1h MA
    df["rt_price_ma_16"] = rt.rolling(16).mean()      # 4h MA
    df["rt_price_ma_96"] = rt.rolling(96).mean()      # 24h MA

    df["rt_price_std_16"] = rt.rolling(16).std()
    df["rt_price_std_96"] = rt.rolling(96).std()

    df["rt_price_trend"] = df["rt_price_ma_4"] - df["rt_price_ma_16"]

    df["rt_price_percentile"] = rt.rolling(96).rank(pct=True)

    # 关键特征：价格/24h均值比（BC成功的核心）
    df["rt_price_ma_ratio"] = (rt / df["rt_price_ma_96"]).clip(0.1, 10)
    df["rt_price_ma4_ratio"] = (rt / df["rt_price_ma_4"]).clip(0.1, 10)

    # --- 日前价格特征（中国独有！）---
    if "da_price" in df.columns:
        df["da_rt_spread"] = df["da_price"] - df["rt_price"]
        da_ma96 = df["da_price"].rolling(96).mean()
        df["da_price_ma_ratio"] = (df["da_price"] / da_ma96).clip(0.1, 10)
    else:
        df["da_price"] = np.nan
        df["da_rt_spread"] = 0
        df["da_price_ma_ratio"] = 1.0

    # --- 负荷特征 ---
    if "load_mw" in df.columns:
        load_ma96 = df["load_mw"].rolling(96).mean()
        df["load_norm"] = df["load_mw"] / load_ma96.clip(lower=1)
        df["load_change"] = df["load_mw"].pct_change(4)  # 1h变化率
    else:
        df["load_norm"] = 1.0
        df["load_change"] = 0.0

    # --- 新能源特征 ---
    if "renewable_mw" in df.columns and "load_mw" in df.columns:
        df["renewable_penetration"] = (
            df["renewable_mw"] / df["load_mw"].clip(lower=1)
        ).clip(0, 1)
    else:
        df["renewable_penetration"] = 0.0

    if "wind_mw" in df.columns and "renewable_mw" in df.columns:
        df["wind_ratio"] = (
            df["wind_mw"] / df["renewable_mw"].clip(lower=1)
        ).clip(0, 1)
    else:
        df["wind_ratio"] = 0.5

    if "solar_mw" in df.columns and "renewable_mw" in df.columns:
        df["solar_ratio"] = (
            df["solar_mw"] / df["renewable_mw"].clip(lower=1)
        ).clip(0, 1)
    else:
        df["solar_ratio"] = 0.5

    # 净负荷
    if "load_mw" in df.columns and "renewable_mw" in df.columns:
        net_load = df["load_mw"] - df["renewable_mw"]
        df["net_load_norm"] = net_load / df["load_mw"].clip(lower=1)
    else:
        df["net_load_norm"] = 1.0

    # --- 联络线特征 ---
    if "tie_line_mw" in df.columns:
        tie_ma96 = df["tie_line_mw"].rolling(96).mean()
        df["tie_line_norm"] = df["tie_line_mw"] / tie_ma96.abs().clip(lower=1)
    else:
        df["tie_line_norm"] = 0.0

    # --- 天气特征 ---
    if "temperature" in df.columns:
        temp_mean = df["temperature"].rolling(96*7).mean()  # 7天均温
        temp_std = df["temperature"].rolling(96*7).std().clip(lower=1)
        df["temperature_norm"] = (df["temperature"] - temp_mean) / temp_std
    else:
        df["temperature_norm"] = 0.0

    # --- 时间编码（与ERCOT一致）---
    ts = df["timestamp"]
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * ts.dt.weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * ts.dt.weekday / 7)
    df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)

    # 去掉warmup期的NaN
    df = df.dropna(subset=["rt_price_ma_96"]).reset_index(drop=True)

    # 填充剩余NaN
    for col in ALL_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(method="ffill").fillna(0)

    logger.info(f"Feature engineering done: {len(df)} rows, {len(ALL_FEATURE_COLS)} features")

    return df


# ============================================================
# Step 4: 主管线 —— 从Excel到Parquet
# ============================================================

def ingest_province(
    filepath: str | Path,
    province: str,
    sheet_name: str | int = 0,
    column_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """
    完整管线：Excel → 清洗 → 特征 → Parquet

    Args:
        filepath: Excel文件路径
        province: 省份标识 (shandong/guangdong/shanxi/gansu)
        sheet_name: Excel sheet名称或索引
        column_mapping: 自定义列名映射（None则自动匹配）

    Returns:
        特征化后的DataFrame
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Ingesting {province} from {filepath}")
    logger.info(f"{'='*60}")

    # 读取
    logger.info("Reading Excel...")
    df_raw = pd.read_excel(filepath, sheet_name=sheet_name)
    logger.info(f"Raw shape: {df_raw.shape}")
    logger.info(f"Raw columns: {list(df_raw.columns)}")

    # 清洗
    logger.info("Cleaning...")
    df_clean = clean_and_standardize(df_raw, province, column_mapping)

    # 特征工程
    logger.info("Building features...")
    df_feat = build_features_china(df_clean)

    # 保存
    province_dir = Path(__file__).parent / province
    province_dir.mkdir(exist_ok=True)

    parquet_path = province_dir / f"{province}_spot_real.parquet"
    df_feat.to_parquet(parquet_path, index=False)
    logger.info(f"Saved: {parquet_path} ({len(df_feat)} rows)")

    # 同时保存到processed合集
    return df_feat


def ingest_all():
    """
    接入所有省份数据

    使用前请确认：
    1. Excel文件已放入 data/china/raw/ 目录
    2. 运行 probe_all() 确认列名映射正确
    3. 根据probe结果调整本函数中的参数
    """
    excel_files = sorted(RAW_DIR.glob("*.xlsx")) + sorted(RAW_DIR.glob("*.xls"))

    if not excel_files:
        logger.error(f"No Excel files in {RAW_DIR}!")
        logger.info("请将数据文件放入此目录后重试")
        return

    logger.info(f"Found {len(excel_files)} Excel files:")
    for f in excel_files:
        logger.info(f"  {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")

    # Step 1: 先探测所有文件结构
    logger.info("\n\n" + "="*60)
    logger.info("STEP 1: Probing file structures")
    logger.info("="*60)

    for f in excel_files:
        probe_excel(f)

    logger.info("\n\n" + "="*60)
    logger.info("STEP 2: Please review column mappings above")
    logger.info("Then call ingest_province() for each file")
    logger.info("="*60)
    logger.info("""
Example usage:

  # 如果文件名包含省份信息
  ingest_province("raw/山东_现货数据.xlsx", "shandong")
  ingest_province("raw/广东_现货数据.xlsx", "guangdong")
  ingest_province("raw/山西_现货数据.xlsx", "shanxi")
  ingest_province("raw/甘肃_现货数据.xlsx", "gansu")

  # 如果列名需要自定义映射
  ingest_province(
      "raw/山东.xlsx", "shandong",
      column_mapping={"时间戳": "timestamp", "日前价": "da_price", ...}
  )
""")


# ============================================================
# CLI入口
# ============================================================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"

    if cmd == "probe":
        probe_all()
    elif cmd == "ingest":
        ingest_all()
    elif cmd == "features":
        # 对已清洗数据重新构建特征
        for province in ["shandong", "guangdong", "shanxi", "gansu"]:
            p_dir = Path(__file__).parent / province
            clean_file = p_dir / f"{province}_spot_real.parquet"
            if clean_file.exists():
                df = pd.read_parquet(clean_file)
                df_feat = build_features_china(df)
                df_feat.to_parquet(clean_file, index=False)
                logger.info(f"Rebuilt features for {province}: {len(df_feat)} rows")
    else:
        print(f"Usage: python ingest_china_data.py [probe|ingest|features]")
