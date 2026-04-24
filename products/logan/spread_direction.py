"""
Logan · Head 3: Spread Direction Classifier
============================================

预测明天每小时 P( sign(RT - DA) > 0 ) —— 即"明天各时段 RT 会不会高于 DA"。

**为什么这是最关键的 head**：
  - 申报量决策只要方向信息，不要绝对值
  - 方向分类比回归样本效率高 2-3 倍
  - 下游 bid curve 生成只需要 P(spread>0) 这一个条件概率

架构（Regime-Conditioned）：
    P(spread>0 | t) = Σ_k P(regime=k | today_features) × P(spread>0 | regime=k, t)

    Step 1: Regime Classifier 输出 12 维概率
    Step 2: 从训练集统计每个 (regime, hour_of_day) 的 spread>0 经验频率
    Step 3: 线性组合得到今日每小时方向概率

优点：
  - 样本效率高：每个 (regime, hour) 只需几十个样本稳概率估计
  - 自动校准：输出就是概率
  - 可解释：能拆出"因为今天是 regime 5 所以中午 spread<0 的概率 78%"

可选增强：
  - Laplace smoothing（小样本 bucket 加先验）
  - Bayesian shrinkage（全局 mean 拉近）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger

from core.calendar_features import add_calendar_features
from core.regime_classifier import RegimeClassifier


STEPS_PER_DAY = 96


@dataclass
class SpreadDirectionConfig:
    smoothing_alpha: float = 5.0   # Laplace smoothing（等效先验样本量）
    smoothing_prior: float = 0.5   # 先验 P(spread>0) = 0.5
    aggregate_to_hour: bool = True # True: 24 小时表；False: 96 步表


class SpreadDirectionClassifier:
    """
    Probabilistic spread direction classifier via regime conditioning.

    Usage:
        clf = SpreadDirectionClassifier(regime_classifier=rc)
        clf.fit(df_train)
        p96 = clf.predict_proba_day(target_day_idx=d, df=df)  # (96,) or (24,)
    """

    def __init__(
        self,
        regime_classifier: RegimeClassifier,
        config: SpreadDirectionConfig | None = None,
    ):
        self.regime = regime_classifier
        self.config = config or SpreadDirectionConfig()

        # Main conditional table
        # shape (n_regimes, 96) if aggregate=False
        # shape (n_regimes, 24) if aggregate=True
        self.prob_table: np.ndarray | None = None
        self.n_samples_table: np.ndarray | None = None  # per bucket

    def _fit_table(
        self,
        rt_days: np.ndarray,        # (n_days, 96)
        da_days: np.ndarray,        # (n_days, 96)
        regime_labels: np.ndarray,  # (n_days,)
    ) -> None:
        K = self.regime.n_regimes
        spread_sign = (rt_days > da_days).astype(np.float64)  # (n_days, 96)

        if self.config.aggregate_to_hour:
            # Aggregate to 24 hours
            spread_sign_hr = spread_sign.reshape(-1, 24, 4).mean(axis=2)  # (n_days, 24)
            cols = 24
        else:
            spread_sign_hr = spread_sign
            cols = 96

        prob_table = np.zeros((K, cols))
        count_table = np.zeros((K, cols), dtype=np.int64)
        alpha = self.config.smoothing_alpha
        prior = self.config.smoothing_prior

        for k in range(K):
            mask = regime_labels == k
            if mask.sum() == 0:
                prob_table[k, :] = prior
                continue
            # Empirical frequency
            freq = spread_sign_hr[mask].mean(axis=0)
            n_k = int(mask.sum())
            # Laplace smoothing: pushes probs toward prior if sample small
            smoothed = (n_k * freq + alpha * prior) / (n_k + alpha)
            prob_table[k, :] = smoothed
            count_table[k, :] = n_k

        self.prob_table = prob_table
        self.n_samples_table = count_table

    def fit(
        self,
        df_train: pd.DataFrame,
        rt_col: str = "rt_price",
        da_col: str = "da_price",
    ) -> "SpreadDirectionClassifier":
        """
        训练方向表。
        要求：RegimeClassifier 已经 fit（self.regime.train_labels 可用）。
        """
        if self.regime.kmeans is None:
            raise RuntimeError("Underlying RegimeClassifier must be fitted first.")

        df = df_train.copy()
        rt_values = df[rt_col].fillna(0).values.astype(np.float64)
        da_values = df[da_col].fillna(method="ffill").fillna(0).values.astype(np.float64)

        n_days = len(rt_values) // STEPS_PER_DAY
        rt_days = rt_values[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        da_days = da_values[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)

        # 用 regime classifier 已经聚好的 labels（训练集）
        labels = self.regime.train_labels
        if len(labels) < n_days:
            # 对齐截断
            n_days = len(labels)
            rt_days = rt_days[:n_days]
            da_days = da_days[:n_days]

        self._fit_table(rt_days, da_days, labels)

        # 统计
        mean_prob = float(self.prob_table.mean())
        logger.info(
            f"SpreadDirectionClassifier fit: "
            f"table shape={self.prob_table.shape}, "
            f"avg P(spread>0)={mean_prob:.3f}, "
            f"min_bucket_samples={int(self.n_samples_table.min())}"
        )
        return self

    def predict_proba_day(
        self,
        target_day_idx: int,
        df: pd.DataFrame,
        labels_so_far: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        返回 D 日各时段 P(spread>0)。

        Returns:
            shape (24,) if aggregate_to_hour else (96,)
        """
        assert self.prob_table is not None, "Classifier not fitted"

        regime_probs = self.regime.predict_proba(
            day_idx=target_day_idx,
            df=df,
            labels_so_far=labels_so_far,
        )  # (K,)
        # P(spread>0 | hour) = Σ_k P(regime=k) × P(spread>0 | k, hour)
        return regime_probs @ self.prob_table

    def explain(
        self,
        target_day_idx: int,
        df: pd.DataFrame,
        top_k: int = 3,
    ) -> dict:
        """
        返回可解释信息：top-k regime + 对应方向概率曲线 + 样本量。
        """
        regime_probs = self.regime.predict_proba(day_idx=target_day_idx, df=df)
        top_idx = np.argsort(regime_probs)[::-1][:top_k]
        return {
            "regime_probs": regime_probs.tolist(),
            "top_regimes": [
                {
                    "regime": int(k),
                    "probability": float(regime_probs[k]),
                    "direction_curve": self.prob_table[k].tolist(),
                    "train_samples": int(self.n_samples_table[k][0]),
                }
                for k in top_idx
            ],
            "predicted_direction_curve": (regime_probs @ self.prob_table).tolist(),
        }
