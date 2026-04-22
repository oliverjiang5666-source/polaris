"""
Walk-Forward回测引擎

扩展窗口：训练用所有历史数据，测试用下一个季度
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger


@dataclass
class WFWindow:
    """一个walk-forward窗口"""
    train_start: int  # 训练起始index
    train_end: int    # 训练结束index（不含）
    test_start: int   # 测试起始index
    test_end: int     # 测试结束index（不含）
    window_id: int


def generate_windows(
    n_total: int,
    steps_per_day: int = 96,
    min_train_days: int = 180,
    test_days: int = 90,
    stride_days: int = 90,
    expanding: bool = True,
) -> list[WFWindow]:
    """
    生成walk-forward窗口列表

    Args:
        n_total: 总数据步数
        steps_per_day: 每天步数
        min_train_days: 最少训练天数
        test_days: 每个测试窗口天数
        stride_days: 步进天数
        expanding: True=扩展窗口（训练始终从0开始），False=滚动窗口

    Returns:
        WFWindow列表
    """
    min_train_steps = min_train_days * steps_per_day
    test_steps = test_days * steps_per_day
    stride_steps = stride_days * steps_per_day

    windows = []
    test_start = min_train_steps
    win_id = 0

    while test_start + test_steps <= n_total:
        if expanding:
            train_start = 0
        else:
            train_start = max(0, test_start - min_train_steps)

        w = WFWindow(
            train_start=train_start,
            train_end=test_start,
            test_start=test_start,
            test_end=test_start + test_steps,
            window_id=win_id,
        )
        windows.append(w)
        test_start += stride_steps
        win_id += 1

    if windows:
        logger.info(
            f"  Generated {len(windows)} walk-forward windows "
            f"(train≥{min_train_days}d, test={test_days}d, stride={stride_days}d)"
        )

    return windows
