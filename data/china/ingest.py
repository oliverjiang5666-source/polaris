"""
中国电力现货数据接入管线

Excel长表 → pivot宽表 → 省份特殊清洗 → 标准列名 → 15min规整 → parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from data.china.province_registry import get_province, list_provinces, ProvinceSpec

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


# ============================================================
# 核心接入函数
# ============================================================

def ingest_province(excel_path: str | Path, province_name: str) -> pd.DataFrame:
    """
    完整管线：Excel → pivot → clean → standardize → parquet

    Args:
        excel_path: Excel文件路径
        province_name: 省份标识 (shandong/shanxi/guangdong/gansu)

    Returns:
        清洗后的宽表DataFrame
    """
    spec = get_province(province_name)
    excel_path = Path(excel_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"Ingesting {spec.name_cn} ({spec.name})")
    logger.info(f"File: {excel_path.name} ({excel_path.stat().st_size / 1024 / 1024:.1f} MB)")
    logger.info(f"{'='*60}")

    # --- Step 1: 读取所有sheet，合并 ---
    xls = pd.ExcelFile(excel_path)
    dfs = []
    for sheet in xls.sheet_names:
        logger.info(f"  Reading sheet: {sheet}")
        df = pd.read_excel(xls, sheet_name=sheet)
        dfs.append(df)
    df_long = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Total rows (long): {len(df_long):,}")

    # --- Step 2: 清洗长表 ---
    df_long["时间"] = pd.to_datetime(df_long["时间"])
    df_long["值"] = pd.to_numeric(df_long["值"], errors="coerce")

    # 移除sentinel值
    sentinel_mask = df_long["值"] <= spec.sentinel_value + 1
    if sentinel_mask.sum() > 0:
        logger.warning(f"  Removing {sentinel_mask.sum()} sentinel values (<= {spec.sentinel_value})")
        df_long.loc[sentinel_mask, "值"] = np.nan

    # --- Step 3: pivot长表→宽表 ---
    df_wide = df_long.pivot_table(
        index="时间",
        columns="指标名称",
        values="值",
        aggfunc="first",
    )
    df_wide = df_wide.sort_index()
    logger.info(f"  Pivoted to wide: {df_wide.shape} (rows × indicators)")
    logger.info(f"  Indicators: {list(df_wide.columns)}")

    # --- Step 4: 省份特殊处理 ---
    df_wide = _apply_province_transforms(df_wide, spec)

    # --- Step 5: 列名标准化 ---
    rename_map = {}
    for cn_name, en_name in spec.indicator_map.items():
        if cn_name in df_wide.columns:
            rename_map[cn_name] = en_name
        else:
            logger.warning(f"  Indicator '{cn_name}' not found in data")
    df_wide = df_wide.rename(columns=rename_map)

    # 只保留映射过的列
    known_cols = [v for v in spec.indicator_map.values() if v in df_wide.columns]
    extra_cols = [c for c in df_wide.columns if c not in known_cols]
    if extra_cols:
        logger.info(f"  Dropping unmapped columns: {extra_cols}")
    df_wide = df_wide[known_cols]

    # --- Step 6: 添加省份标签 ---
    df_wide["province"] = spec.name

    # --- Step 7: 广东特殊列映射 ---
    if spec.name == "guangdong":
        df_wide = _map_guangdong_proxies(df_wide)

    # --- Step 8: 重采样到规整15min网格 ---
    df_wide = _resample_to_regular_grid(df_wide)

    # --- Step 9: 数据质量报告 ---
    _print_quality_report(df_wide, spec)

    # --- Step 10: 保存 ---
    out_path = PROCESSED_DIR / f"{spec.name}_clean.parquet"
    df_wide.to_parquet(out_path)
    logger.info(f"  Saved: {out_path} ({len(df_wide):,} rows)")

    return df_wide


# ============================================================
# 省份特殊处理
# ============================================================

def _apply_province_transforms(df: pd.DataFrame, spec: ProvinceSpec) -> pd.DataFrame:
    """省份特定的数据转换"""

    # 山东：光伏负值clip到0
    if spec.clip_solar_negative and "光伏出力" in df.columns:
        neg_count = (df["光伏出力"] < 0).sum()
        if neg_count > 0:
            logger.info(f"  Clipping {neg_count:,} negative solar values to 0")
            df["光伏出力"] = df["光伏出力"].clip(lower=0)

    # 甘肃：合并河东/河西价格为统一价格
    if spec.has_regional_prices:
        df = _merge_gansu_prices(df)

    return df


def _merge_gansu_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    甘肃价格体系：
    - 2021.4 ~ 2024.7: 河东/河西分别定价
    - 2024.7 ~ 至今: 统一价格

    策略：优先用统一价格，缺失时用河东/河西均值填补
    """
    for price_type in ["日前价格", "实时价格"]:
        hedong_col = f"河东{price_type}"
        hexi_col = f"河西{price_type}"

        if price_type not in df.columns:
            # 统一价格列不存在，需要从区域价格创建
            if hedong_col in df.columns and hexi_col in df.columns:
                logger.info(f"  Creating unified {price_type} from 河东/河西 average")
                df[price_type] = df[[hedong_col, hexi_col]].mean(axis=1)
            elif hedong_col in df.columns:
                df[price_type] = df[hedong_col]
        else:
            # 统一价格列存在但可能有NaN（早期数据）
            if hedong_col in df.columns and hexi_col in df.columns:
                regional_avg = df[[hedong_col, hexi_col]].mean(axis=1)
                na_mask = df[price_type].isna()
                filled = na_mask.sum()
                if filled > 0:
                    logger.info(f"  Filling {filled:,} NaN in unified {price_type} with 河东/河西 avg")
                    df.loc[na_mask, price_type] = regional_avg[na_mask]

    return df


def _map_guangdong_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    广东缺风/光/新能源数据，用替代指标：
    - 省内B类电源 / 负荷 → renewable_penetration proxy
    - 西电东送 → tie_line_mw proxy
    """
    # B类电源作为renewable proxy (燃气+新能源)
    if "gen_class_b_mw" in df.columns and "load_mw" in df.columns:
        df["renewable_mw"] = df["gen_class_b_mw"]  # 粗糙代理
        logger.info("  Guangdong: using B-class generation as renewable proxy")

    # 西电东送作为联络线
    if "west_east_mw" in df.columns:
        df["tie_line_mw"] = df["west_east_mw"]
        logger.info("  Guangdong: using west-east power as tie_line proxy")

    return df


def _resample_to_regular_grid(df: pd.DataFrame) -> pd.DataFrame:
    """重采样到规整的15分钟网格，前向填充短间隔缺失"""
    # 推断当前频率
    if len(df) < 2:
        return df

    # 生成规整网格
    start = df.index.min().floor("15min")
    end = df.index.max().ceil("15min")
    regular_idx = pd.date_range(start=start, end=end, freq="15min")

    # reindex到规整网格
    original_len = len(df)
    df = df.reindex(regular_idx)

    # 前向填充（最多4个时段=1小时的gap）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill(limit=4)

    # province列填充
    if "province" in df.columns:
        df["province"] = df["province"].ffill().bfill()

    new_len = len(df)
    if new_len != original_len:
        logger.info(f"  Resampled: {original_len:,} → {new_len:,} rows (15min grid)")

    return df


# ============================================================
# 数据质量报告
# ============================================================

def _print_quality_report(df: pd.DataFrame, spec: ProvinceSpec):
    """打印数据质量摘要"""
    logger.info(f"\n  --- {spec.name_cn} Quality Report ---")
    logger.info(f"  Rows: {len(df):,}")
    logger.info(f"  Time: {df.index.min()} → {df.index.max()}")
    logger.info(f"  Days: {(df.index.max() - df.index.min()).days}")

    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        nans = df[col].isna().sum()
        pct_nan = nans / len(df) * 100
        if len(vals) == 0:
            logger.warning(f"    {col:<25s}: ALL NaN!")
            continue
        logger.info(
            f"    {col:<25s}: "
            f"mean={vals.mean():>10.1f}  "
            f"min={vals.min():>10.1f}  "
            f"max={vals.max():>10.1f}  "
            f"NaN={nans}({pct_nan:.1f}%)"
        )


# ============================================================
# 批量接入
# ============================================================

def ingest_all(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """接入所有省份的数据"""
    if data_dir is None:
        data_dir = Path.home() / "Desktop" / "中国电价现货交易数据"
    data_dir = Path(data_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {}

    # 文件名 → 省份的映射
    file_province_map = {
        "山东": "shandong",
        "山西": "shanxi",
        "广东": "guangdong",
        "甘肃": "gansu",
    }

    results = {}
    for fname in sorted(data_dir.glob("*.xlsx")):
        # 从文件名提取省份
        province_name = None
        for cn, en in file_province_map.items():
            if cn in fname.name:
                province_name = en
                break

        if province_name is None:
            logger.warning(f"  Skipping unknown file: {fname.name}")
            continue

        df = ingest_province(fname, province_name)
        results[province_name] = df

    logger.info(f"\n{'='*60}")
    logger.info(f"Ingestion complete: {len(results)} provinces")
    for name, df in results.items():
        logger.info(f"  {name}: {len(df):,} rows")
    logger.info(f"{'='*60}")

    return results


if __name__ == "__main__":
    ingest_all()
