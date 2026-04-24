"""
Empirical Joint Distribution for (DA, RT)
============================================

非参数的 (DA, RT) 联合分布采样器。按小时分桶。

第一性原理：
    我们需要 P(RT | DA, hour) 的条件分布，因为：
    - DA 和 RT 在现实市场中**正相关**（同一供需驱动）
    - 晚高峰：DA 高，RT 更高（spread 正）
    - 中午：DA 低，RT 更低（spread 负）

    P1.1 的 DFL 独立采样 DA 和 RT 忽略这个相关性 → 错误决策。

方法：
    1. 按小时分桶（24 个）
    2. 每小时把训练 DA 分成 Q 个分位 bucket（Q=10）
    3. 每 bucket 记录对应的 RT 值集合
    4. 推理时：
       - 采样 u_da ~ Uniform(0, 1)
       - da_value = 插值 DA 分位数预测（来自 DAForecaster）
       - 找 da_value 落在哪个 bucket
       - 从 bucket 里随机 bootstrap 一个 RT

    这样"DA 高 → RT 也高"的相关性天然保留（因为 RT 从同 bucket 的训练天采样）。

vs Gaussian copula：
    Gaussian copula 假设 rank-correlation 线性 → 可能漏掉非线性相关（晚高峰的极端 spread）
    empirical 无参数假设，**样本够多时更准**

vs parametric：
    - KDE: 过平滑
    - Mixture model: 过参数
    - 这个: 简单 robust
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger


STEPS_PER_DAY = 96


@dataclass
class JointDistConfig:
    n_da_buckets: int = 10
    min_samples_per_bucket: int = 5
    aggregate_to_hour: bool = True  # True: work on 24 hours; False: 96 steps


class EmpiricalJointDistribution:
    """
    Non-parametric sampler for (DA, RT) joint distribution, hour-conditional.
    """

    def __init__(self, config: JointDistConfig | None = None):
        self.config = config or JointDistConfig()
        # hour -> bucket edges (Q+1,)
        self.bucket_edges: dict[int, np.ndarray] = {}
        # hour -> list of Q arrays (rt values in each bucket)
        self.bucket_rts: dict[int, list[np.ndarray]] = {}
        # hour -> training DA range for safety
        self.hour_da_range: dict[int, tuple[float, float]] = {}
        # hour -> global rt mean (fallback)
        self.hour_rt_mean: dict[int, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        da_col: str = "da_price",
        rt_col: str = "rt_price",
    ) -> "EmpiricalJointDistribution":
        """
        df: 15 分钟粒度 DataFrame，要求含 da_col + rt_col。
        """
        da = df[da_col].ffill().fillna(0).values.astype(np.float64)
        rt = df[rt_col].fillna(0).values.astype(np.float64)
        n_days = min(len(da), len(rt)) // STEPS_PER_DAY
        da = da[: n_days * STEPS_PER_DAY].reshape(n_days, 24, 4)
        rt = rt[: n_days * STEPS_PER_DAY].reshape(n_days, 24, 4)

        # Aggregate to hour
        da_hr = da.mean(axis=2)     # (n_days, 24)
        rt_hr = rt.mean(axis=2)

        n_buckets = self.config.n_da_buckets
        min_samples = self.config.min_samples_per_bucket

        for h in range(24):
            da_h = da_hr[:, h]
            rt_h = rt_hr[:, h]
            # 过滤无效
            valid = ~(np.isnan(da_h) | np.isnan(rt_h))
            da_h = da_h[valid]
            rt_h = rt_h[valid]
            if len(da_h) < n_buckets * min_samples:
                logger.warning(f"Hour {h}: only {len(da_h)} valid samples, using fewer buckets")

            edges = np.quantile(da_h, np.linspace(0, 1, n_buckets + 1))
            # 确保唯一（如果 DA 有很多相同值可能退化）
            edges = np.maximum.accumulate(edges)
            # 如果后面的值相等，加 epsilon
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-6

            self.bucket_edges[h] = edges
            buckets = []
            for b in range(n_buckets):
                lo = edges[b]
                hi = edges[b + 1]
                if b == n_buckets - 1:
                    mask = (da_h >= lo) & (da_h <= hi)
                else:
                    mask = (da_h >= lo) & (da_h < hi)
                buckets.append(rt_h[mask])
            self.bucket_rts[h] = buckets

            self.hour_da_range[h] = (float(da_h.min()), float(da_h.max()))
            self.hour_rt_mean[h] = float(rt_h.mean())

        # Diagnostics
        total_samples = sum(len(b) for h in range(24) for b in self.bucket_rts[h])
        logger.info(
            f"EmpiricalJointDistribution.fit: {n_days} days × 24 hours × {n_buckets} buckets = "
            f"{total_samples} bucket samples total"
        )
        # Avg samples per bucket
        avg_per_bucket = total_samples / (24 * n_buckets)
        logger.info(f"  Avg samples per bucket: {avg_per_bucket:.1f}")
        return self

    # ------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------

    def sample(
        self,
        hour: int,
        da_quantiles: np.ndarray,         # (K,) predicted DA quantile values
        quantile_levels: np.ndarray,      # (K,) corresponding levels, e.g. [0.05, 0.25, 0.5, 0.75]
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (da_samples, rt_samples), both (n_scenarios,).

        流程：
            1. u ~ Uniform(0,1)
            2. da = interp_1d(u, quantile_levels, da_quantiles)
            3. for each da_s: find bucket, sample rt from bucket
        """
        N = n_scenarios
        # Pad quantile levels with 0 and 1 for interpolation robustness
        # Extrapolate low end: 50% of P05; high end: 150% of max given quantile
        q_levels = np.asarray(quantile_levels)
        q_vals = np.asarray(da_quantiles)
        full_levels = np.concatenate([[0.0], q_levels, [1.0]])
        full_vals = np.concatenate([[q_vals[0] * 0.5], q_vals, [q_vals[-1] * 1.5]])

        u = rng.uniform(0, 1, N)
        da_samples = np.interp(u, full_levels, full_vals)

        # For each da_sample: find bucket
        edges = self.bucket_edges.get(hour)
        buckets = self.bucket_rts.get(hour)
        if edges is None or buckets is None:
            # Hour not in training data (shouldn't happen)
            return da_samples, np.full(N, self.hour_rt_mean.get(hour, float(np.mean(da_samples))))

        # Vectorize bucket lookup
        # np.searchsorted: returns index where da_samples would be inserted
        bucket_idx = np.searchsorted(edges, da_samples, side="right") - 1
        bucket_idx = np.clip(bucket_idx, 0, len(buckets) - 1)

        rt_samples = np.zeros(N)
        for i in range(N):
            bkt = buckets[int(bucket_idx[i])]
            if len(bkt) == 0:
                # Empty bucket; fallback to hour mean
                rt_samples[i] = self.hour_rt_mean.get(hour, da_samples[i])
            else:
                rt_samples[i] = bkt[rng.integers(0, len(bkt))]

        return da_samples, rt_samples

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def compute_rank_correlation(self) -> np.ndarray:
        """每小时的 Spearman rank correlation(DA, RT)"""
        rhos = np.zeros(24)
        for h in range(24):
            all_da = []
            all_rt = []
            edges = self.bucket_edges.get(h)
            buckets = self.bucket_rts.get(h)
            if edges is None:
                continue
            for b_idx, rts in enumerate(buckets):
                if len(rts) > 0:
                    # center DA values within bucket
                    center = (edges[b_idx] + edges[b_idx + 1]) / 2
                    all_da.extend([center] * len(rts))
                    all_rt.extend(rts)
            if len(all_da) > 10:
                try:
                    from scipy.stats import spearmanr
                    rho, _ = spearmanr(all_da, all_rt)
                    rhos[h] = rho if not np.isnan(rho) else 0.0
                except ImportError:
                    rhos[h] = np.corrcoef(all_da, all_rt)[0, 1]
        return rhos
