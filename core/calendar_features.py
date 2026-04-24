"""
Calendar feature utilities
===========================

Logan 所有 head 都要用的 "日历" 基础特征：小时 sin/cos、周末、节假日、季节。
这些东西不需要深学习，但如果漏了会导致模型在节假日系统性偏。

节假日列表按需补：这里给个近 3 年中国法定节假日的硬编码表，便于 offline 训练；
上线时可换成 chinese_calendar 包。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# 中国法定节假日（硬编码近 3 年，避免依赖外部包）
# ============================================================
CHINA_HOLIDAYS = {
    # 2024
    "2024-01-01",
    "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14",
    "2024-02-15", "2024-02-16", "2024-02-17",
    "2024-04-04", "2024-04-05", "2024-04-06",
    "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
    "2024-06-10",
    "2024-09-15", "2024-09-16", "2024-09-17",
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04",
    "2024-10-05", "2024-10-06", "2024-10-07",
    # 2025
    "2025-01-01",
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",
    "2025-04-04", "2025-04-05", "2025-04-06",
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",
    "2025-05-31", "2025-06-01", "2025-06-02",
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04",
    "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",
    # 2026
    "2026-01-01", "2026-01-02", "2026-01-03",
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19",
    "2026-02-20", "2026-02-21", "2026-02-22",
    "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-05-01", "2026-05-02", "2026-05-03",
}
_HOLIDAY_DATES = {pd.Timestamp(d).date() for d in CHINA_HOLIDAYS}


def is_weekend(ts: pd.Timestamp) -> bool:
    return ts.weekday() >= 5


def is_holiday(ts: pd.Timestamp) -> bool:
    return ts.date() in _HOLIDAY_DATES


def is_workday(ts: pd.Timestamp) -> bool:
    return (not is_weekend(ts)) and (not is_holiday(ts))


def season_id(month: int) -> int:
    """0=春(3-5), 1=夏(6-8), 2=秋(9-11), 3=冬(12,1,2)"""
    if 3 <= month <= 5:
        return 0
    if 6 <= month <= 8:
        return 1
    if 9 <= month <= 11:
        return 2
    return 3


def hour_bucket_4(hour: int) -> int:
    """把一天划成 4 档：0=夜(0-6), 1=上午(6-12), 2=下午(12-18), 3=晚(18-24)"""
    if hour < 6:
        return 0
    if hour < 12:
        return 1
    if hour < 18:
        return 2
    return 3


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    给 DataFrame（index 必须是 DatetimeIndex）加上标准日历特征。

    Added columns:
        hour, minute_of_day (0..95), hour_sin, hour_cos,
        weekday, weekday_sin, weekday_cos,
        month, month_sin, month_cos,
        season (0-3), hour_bucket (0-3),
        is_weekend, is_holiday, is_workday
    """
    out = df.copy()
    idx = out.index
    assert isinstance(idx, pd.DatetimeIndex), "Index must be DatetimeIndex"

    hours = idx.hour
    minutes = idx.minute
    out["hour"] = hours
    out["minute_of_day"] = hours * 4 + minutes // 15  # 0..95

    # 时段 sin/cos（周期 24h）
    angle_h = 2 * np.pi * (hours + minutes / 60) / 24
    out["hour_sin"] = np.sin(angle_h)
    out["hour_cos"] = np.cos(angle_h)

    # 周内
    wd = idx.weekday
    out["weekday"] = wd
    angle_w = 2 * np.pi * wd / 7
    out["weekday_sin"] = np.sin(angle_w)
    out["weekday_cos"] = np.cos(angle_w)

    # 月份
    mo = idx.month
    out["month"] = mo
    angle_m = 2 * np.pi * mo / 12
    out["month_sin"] = np.sin(angle_m)
    out["month_cos"] = np.cos(angle_m)

    # 季节与时段
    out["season"] = np.array([season_id(m) for m in mo])
    out["hour_bucket"] = np.array([hour_bucket_4(h) for h in hours])

    # 节假日
    out["is_weekend"] = np.array([w >= 5 for w in wd], dtype=np.int8)
    out["is_holiday"] = np.array([d in _HOLIDAY_DATES for d in idx.date], dtype=np.int8)
    out["is_workday"] = ((out["is_weekend"] == 0) & (out["is_holiday"] == 0)).astype(np.int8)

    return out


CALENDAR_COLS = [
    "hour", "minute_of_day",
    "hour_sin", "hour_cos",
    "weekday", "weekday_sin", "weekday_cos",
    "month", "month_sin", "month_cos",
    "season", "hour_bucket",
    "is_weekend", "is_holiday", "is_workday",
]
