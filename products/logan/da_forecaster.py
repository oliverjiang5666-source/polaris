"""
Logan · Head 1: DA Price Forecaster
====================================

在 D-1 上午（10:00 前）预测 D 日 96 点日前电价（+ 分位数）。

架构（第一性原理）：
    DA[t] = supply_curve( net_load[t], season, hour_bucket ) + residual
    residual ∈ LGBM-quantile-regression

    net_load[t] 来自 NetLoadForecaster 的 D 日预测。
    supply_curve 把净负荷 map 成"典型 DA 价"。
    residual LGBM 学剩余因素。

输入信息集（D-1 10:00 时点）：
    - 历史 RT / DA / 负荷 / 新能源
    - D 日日历
    - D 日天气预报（NWP）
    - D-1 已出清的 DA（可能用作 lag）

输出：
    predict(d, df) → (96,) 点估计
    predict_quantile(d, df, q=0.9) → (96,) 分位数
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger

from core.calendar_features import add_calendar_features
from core.supply_curve import SupplyCurve
from core.net_load_forecaster import NetLoadForecaster


STEPS_PER_DAY = 96


@dataclass
class DAForecasterConfig:
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)


class DAForecaster:
    """
    Layer 1 + Layer 2 两层结构的 DA 预测器。

    用法：
        fcst = DAForecaster()
        fcst.fit(df_train)
        da_96 = fcst.predict_day(target_day_idx=d, df=df)
        da_96_q05 = fcst.predict_day_quantile(d, df, q=0.05)
    """

    def __init__(
        self,
        supply_curve: SupplyCurve | None = None,
        net_load_forecaster: NetLoadForecaster | None = None,
        config: DAForecasterConfig | None = None,
    ):
        self.config = config or DAForecasterConfig()
        self.supply_curve = supply_curve or SupplyCurve()
        self.net_load = net_load_forecaster or NetLoadForecaster()

    def fit(self, df_train: pd.DataFrame) -> "DAForecaster":
        """训练供给曲线 + 净负荷预测器"""
        df = df_train.copy()
        if "hour" not in df.columns:
            df = add_calendar_features(df)

        logger.info("=== DAForecaster.fit ===")
        logger.info("  [1/2] Fitting SupplyCurve...")
        self.supply_curve.fit(df)

        logger.info("  [2/2] Fitting NetLoadForecaster...")
        self.net_load.fit(df)

        logger.info("DAForecaster fit complete.")
        logger.info(f"  SupplyCurve: {self.supply_curve.describe()}")
        return self

    def _prepare_day_df(
        self,
        target_day_idx: int,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """切出 D 日的 96 行 df，补上日历特征"""
        df = df.copy()
        if "hour" not in df.columns:
            df = add_calendar_features(df)
        start = target_day_idx * STEPS_PER_DAY
        end = start + STEPS_PER_DAY
        if end > len(df):
            raise ValueError(f"target_day_idx {target_day_idx} 超出 df 长度 {len(df)}")
        return df.iloc[start:end].copy()

    def predict_day(
        self,
        target_day_idx: int,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        预测 D 日 96 点 DA 价。

        Returns:
            np.ndarray shape (96,)
        """
        day_df = self._prepare_day_df(target_day_idx, df)

        # Step 1: 预测净负荷
        net_load_pred = self.net_load.predict_day(target_day_idx, df)
        day_df = day_df.assign(net_load=net_load_pred)

        # Step 2: 供给曲线（中位数点估计 = base + median residual）
        da_pred = self.supply_curve.predict(
            net_load=net_load_pred,
            season=day_df["season"].values,
            hour_bucket=day_df["hour_bucket"].values,
            extra_df=day_df,
        )
        return da_pred

    def predict_day_quantile(
        self,
        target_day_idx: int,
        df: pd.DataFrame,
        q: float,
    ) -> np.ndarray:
        """
        预测 D 日 96 点 DA 价的 q 分位。

        注意：这里的不确定性来自两部分 ——
          (a) 残差分位数（supply_curve 内部）
          (b) 净负荷预测的误差（暂未显式建模，后续可加）
        """
        day_df = self._prepare_day_df(target_day_idx, df)
        net_load_pred = self.net_load.predict_day(target_day_idx, df)
        day_df = day_df.assign(net_load=net_load_pred)

        return self.supply_curve.predict_quantile(
            net_load=net_load_pred,
            season=day_df["season"].values,
            hour_bucket=day_df["hour_bucket"].values,
            quantile=q,
            extra_df=day_df,
        )

    def predict_day_all_quantiles(
        self,
        target_day_idx: int,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        返回 shape (96, len(quantiles)) 的分位数矩阵（按 config.quantiles 顺序）。
        """
        arr = []
        for q in self.config.quantiles:
            arr.append(self.predict_day_quantile(target_day_idx, df, q))
        return np.stack(arr, axis=1)
