"""RawRecord → 标准化 DataFrame"""

from __future__ import annotations
import pandas as pd
from datetime import datetime
from loguru import logger

from crawlers.sources.base import RawRecord
from crawlers.config.provinces import ProvinceSpec


def records_to_long_df(records: list[RawRecord]) -> pd.DataFrame:
    """RawRecord列表 → 长表DataFrame（与现有xlsx格式对齐）"""
    if not records:
        return pd.DataFrame(columns=[
            "指标名称", "地区", "时间", "值", "日期", "时刻", "数据单位",
            "source", "fetched_at",
        ])
    rows = []
    for r in records:
        rows.append({
            "指标名称": r.indicator,
            "地区": r.province,
            "时间": r.timestamp,
            "值": r.value,
            "日期": r.timestamp.date(),
            "时刻": r.timestamp.strftime("%H:%M:%S"),
            "数据单位": r.unit,
            "source": r.source,
            "fetched_at": r.fetched_at,
        })
    return pd.DataFrame(rows)


def long_to_wide(df: pd.DataFrame, spec: ProvinceSpec) -> pd.DataFrame:
    """
    长表 → 宽表（pivot），应用indicator_map映射标准列名。
    输出格式与ingest.py的输出兼容。
    """
    if df.empty:
        return df

    # 只保留已注册的指标
    known = set(spec.indicator_map.keys())
    df_known = df[df["指标名称"].isin(known)].copy()
    dropped = set(df["指标名称"].unique()) - known
    if dropped:
        logger.debug("Dropped unknown indicators for {}: {}", spec.name_cn, dropped)

    # pivot
    wide = df_known.pivot_table(
        index="时间",
        columns="指标名称",
        values="值",
        aggfunc="first",
    )

    # 重命名列
    wide.columns = [spec.indicator_map.get(c, c) for c in wide.columns]
    wide = wide.reset_index().rename(columns={"时间": "timestamp"})
    wide["timestamp"] = pd.to_datetime(wide["timestamp"])

    # 特殊处理
    if spec.clip_solar_negative and "solar_mw" in wide.columns:
        n_neg = (wide["solar_mw"] < 0).sum()
        if n_neg > 0:
            wide["solar_mw"] = wide["solar_mw"].clip(lower=0)
            logger.debug("Clipped {} negative solar values for {}", n_neg, spec.name_cn)

    if spec.sentinel_value != 0:
        for col in wide.select_dtypes(include="number").columns:
            n_sentinel = (wide[col] == spec.sentinel_value).sum()
            if n_sentinel > 0:
                wide.loc[wide[col] == spec.sentinel_value, col] = None
                logger.debug("Replaced {} sentinel values in {}.{}", n_sentinel, spec.name_cn, col)

    return wide.sort_values("timestamp").reset_index(drop=True)
