"""
Logan · Head 4: System Deviation Risk Proxy
=============================================

目标：预测 "反向偏差加罚" 在 D 日每个时段的触发概率。

真实版需要全省级数据（所有机组实际 vs 计划、跨省联络线实际 vs 计划）——
一般客户拿不到。所以这里实现 **代理版 (proxy)**，等客户给全省数据再做升级版。

代理思路（基于市场 signal 反推系统偏差）：
  系统整体缺电 ⇔ RT 显著偏高 DA（大 positive spread）
  系统整体过剩 ⇔ RT 显著偏低 DA（大 negative spread）

  反向加罚触发条件（各省规则，此处参数化）：
    - 条件 A：系统缺电（e.g. RT > DA + threshold）
    - 条件 B：你少发（负偏差）

    → P(triggered | hour) ≈ P(大 positive spread | hour, regime)
                            + P(大 negative spread | hour, regime) × P(你多发 | hour)

因为"你多发/少发"还没发生（D-1 预测），我们只估 P(系统方向 ∈ {缺, 过剩})。
上层 BidCurveGenerator 再结合自己的申报方向推出加罚概率。

实现：
  - 用训练集 spread 的极端分位数（|spread| > q95）标注 "显著缺电/过剩"
  - 用 Regime 条件化，和 SpreadDirectionClassifier 逻辑一致
  - 输出两个概率：P(显著正 spread | hour)、P(显著负 spread | hour)

接口留空给真实版：
    fit_full(loads_forecast, loads_actual, ...) —— 占位，等全省数据
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger

from core.regime_classifier import RegimeClassifier


STEPS_PER_DAY = 96


@dataclass
class SystemDeviationConfig:
    positive_threshold_pct: float = 0.85  # top 15% 正 spread 视为"显著缺电"
    negative_threshold_pct: float = 0.15  # bottom 15% 负 spread 视为"显著过剩"
    smoothing_alpha: float = 5.0
    smoothing_prior: float = 0.15
    aggregate_to_hour: bool = True


class SystemDeviationProxy:
    """
    P(系统显著偏差方向 | hour) via regime conditioning.

    Output:
        prob_shortage[t]:   P(系统缺电 | t)  → 你少发时会被反向加罚
        prob_surplus[t]:    P(系统过剩 | t)  → 你多发时会被反向加罚
    """

    def __init__(
        self,
        regime_classifier: RegimeClassifier,
        config: SystemDeviationConfig | None = None,
    ):
        self.regime = regime_classifier
        self.config = config or SystemDeviationConfig()

        self.shortage_table: np.ndarray | None = None  # (K, 24) or (K, 96)
        self.surplus_table: np.ndarray | None = None
        self.pos_threshold: float | None = None
        self.neg_threshold: float | None = None

    def fit(
        self,
        df_train: pd.DataFrame,
        rt_col: str = "rt_price",
        da_col: str = "da_price",
    ) -> "SystemDeviationProxy":
        if self.regime.kmeans is None:
            raise RuntimeError("Underlying RegimeClassifier must be fitted first.")

        rt_values = df_train[rt_col].fillna(0).values.astype(np.float64)
        da_values = df_train[da_col].fillna(method="ffill").fillna(0).values.astype(np.float64)

        n_days = len(rt_values) // STEPS_PER_DAY
        rt_days = rt_values[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        da_days = da_values[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        spread = rt_days - da_days

        labels = self.regime.train_labels
        if len(labels) < n_days:
            n_days = len(labels)
            spread = spread[:n_days]

        # 全局阈值（正侧和负侧独立）
        flat = spread.flatten()
        self.pos_threshold = float(np.quantile(flat, self.config.positive_threshold_pct))
        self.neg_threshold = float(np.quantile(flat, self.config.negative_threshold_pct))

        is_shortage = (spread > self.pos_threshold).astype(np.float64)  # (n_days, 96)
        is_surplus = (spread < self.neg_threshold).astype(np.float64)

        if self.config.aggregate_to_hour:
            is_shortage = is_shortage.reshape(-1, 24, 4).mean(axis=2)
            is_surplus = is_surplus.reshape(-1, 24, 4).mean(axis=2)
            cols = 24
        else:
            cols = 96

        K = self.regime.n_regimes
        short_table = np.zeros((K, cols))
        surp_table = np.zeros((K, cols))
        alpha = self.config.smoothing_alpha
        prior = self.config.smoothing_prior

        for k in range(K):
            mask = labels == k
            if mask.sum() == 0:
                short_table[k] = prior
                surp_table[k] = prior
                continue
            n_k = int(mask.sum())
            short_freq = is_shortage[mask].mean(axis=0)
            surp_freq = is_surplus[mask].mean(axis=0)
            short_table[k] = (n_k * short_freq + alpha * prior) / (n_k + alpha)
            surp_table[k] = (n_k * surp_freq + alpha * prior) / (n_k + alpha)

        self.shortage_table = short_table
        self.surplus_table = surp_table

        logger.info(
            f"SystemDeviationProxy fit: "
            f"pos_threshold={self.pos_threshold:.1f}, neg_threshold={self.neg_threshold:.1f}, "
            f"avg P(shortage)={short_table.mean():.3f}, avg P(surplus)={surp_table.mean():.3f}"
        )
        return self

    def predict_proba_day(
        self,
        target_day_idx: int,
        df: pd.DataFrame,
        labels_so_far: np.ndarray | None = None,
    ) -> dict:
        assert self.shortage_table is not None
        regime_probs = self.regime.predict_proba(
            day_idx=target_day_idx, df=df, labels_so_far=labels_so_far
        )
        return {
            "prob_shortage": regime_probs @ self.shortage_table,
            "prob_surplus": regime_probs @ self.surplus_table,
        }

    # 真实版接口占位（等全省数据）
    def fit_full(self, *args, **kwargs):
        raise NotImplementedError(
            "Full version requires province-level generation/load actual-vs-plan data. "
            "Waiting for client data handover."
        )
