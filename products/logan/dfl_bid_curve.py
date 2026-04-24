"""
Logan · Decision-Focused Bid Curve Generator (SAA)
====================================================

真正的 "优化期望收益" bid 生成器，取代 heuristic 版本 BidCurveGenerator。

思想（第一性原理）：
    每个小时的决策变量 = bid curve = 一组 (quantity, price) 对
    目标 = argmax_{bid} E_{DA, RT, power}[ revenue(bid, DA, RT, power) ]

    revenue = cleared × DA + (power - cleared) × RT − penalty

    其中 cleared = Σ{ q_i : p_i ≤ DA_cleared }

实现：Sample Average Approximation (SAA)
    1. 从 DA/RT/power 分布采样 N=200 条联合场景
    2. 候选 bid curves 池 = 35 个参数化变体（7 总量 × 5 量分配模式）
    3. 每个候选在 N 条场景上算 revenue，取均值
    4. argmax

关键改进 vs heuristic BidCurveGenerator：
    1. 目标函数**就是收益**，不是 heuristic 规则
    2. Spread direction 作为 **条件分布参数** 进入 RT 采样，不是 if-else 阈值
    3. 中标率不再是"副作用"，是**决策变量**（由总量 + 分档自然决定）

候选空间设计：
    total_offset ∈ {-0.15, -0.10, -0.05, 0, +0.05, +0.10, +0.15}
    allocation pattern ∈ {
        [1.0, 0, 0, 0],         "single_P05"
        [0.5, 0.3, 0.15, 0.05], "heavy_low"
        [0.25, 0.25, 0.25, 0.25], "uniform"
        [0.05, 0.15, 0.3, 0.5], "heavy_high"
        [0, 0, 1.0, 0],         "single_P50"
    }
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from products.logan.bid_curve_generator import BidStep, HourlyBid
from core.joint_distribution import EmpiricalJointDistribution


# ============================================================
# Config
# ============================================================

@dataclass
class DFLConfig:
    # SAA
    n_scenarios: int = 200
    random_seed: int = 42

    # Settlement rules (应该和 evaluator 一致)
    deviation_bound: float = 0.10
    deviation_penalty_ratio: float = 0.20
    reverse_penalty_spread_threshold: float = 50.0
    reverse_penalty_ratio: float = 0.30

    # Candidate search（更密的 grid + 更多模式）
    offset_grid: tuple[float, ...] = (
        -0.15, -0.12, -0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09
    )
    allocation_patterns: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0, 0.0),           # single_P05
        (0.7, 0.3, 0.0, 0.0),           # heavy_low_2
        (0.5, 0.3, 0.15, 0.05),         # heavy_low
        (0.25, 0.25, 0.25, 0.25),       # uniform
        (0.4, 0.3, 0.2, 0.1),           # gradient_low
        (0.1, 0.2, 0.3, 0.4),           # gradient_high
        (0.05, 0.15, 0.3, 0.5),         # heavy_high
        (0.0, 0.0, 1.0, 0.0),           # single_P50
        (0.0, 1.0, 0.0, 0.0),           # single_P25
    )
    allocation_pattern_names: tuple[str, ...] = (
        "single_P05", "heavy_low_2", "heavy_low", "uniform",
        "gradient_low", "gradient_high", "heavy_high",
        "single_P50", "single_P25",
    )

    # Sampling distributions
    power_noise_std: float = 0.05      # actual_power std as fraction of forecast
    spread_magnitude_std_fallback: float = 50.0  # if not provided


# ============================================================
# Main generator
# ============================================================

class DFLBidCurveGenerator:

    def __init__(
        self,
        capacity_mw: float,
        config: DFLConfig | None = None,
        joint_dist: EmpiricalJointDistribution | None = None,
    ):
        self.capacity_mw = capacity_mw
        self.config = config or DFLConfig()
        self.joint_dist = joint_dist  # 可选：如果提供则用 joint sampling
        self.rng = np.random.default_rng(self.config.random_seed)

    # ------------------------------------------------------------
    # Scenario sampling
    # ------------------------------------------------------------

    def _sample_da(
        self,
        da_quantiles: np.ndarray,   # (n_q,) e.g. [P05, P25, P50, P75, P95]
        quantile_levels: np.ndarray = np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
    ) -> np.ndarray:
        """从 DA 分位数插值采样 n_scenarios 个 DA 价"""
        N = self.config.n_scenarios
        u = self.rng.uniform(0, 1, N)
        # 用分位数在 CDF 上做线性插值
        return np.interp(u, quantile_levels, da_quantiles)

    def _sample_rt_given_da(
        self,
        da_samples: np.ndarray,
        spread_dir_prob: float,
        spread_magnitude_std: float,
    ) -> np.ndarray:
        """
        从 P(RT > DA | hour) 和 spread 幅值分布采样 RT。

        sign ~ Bernoulli(spread_dir_prob)
        |spread| ~ HalfNormal(0, std)  （简化）
        RT = DA + sign × |spread|

        注：这里每个 DA sample 采一个 RT，所以 correlation 结构只来自 sign 的概率
        """
        N = len(da_samples)
        signs = np.where(self.rng.random(N) < spread_dir_prob, 1.0, -1.0)
        magnitudes = np.abs(self.rng.normal(0, spread_magnitude_std, N))
        return da_samples + signs * magnitudes

    def _sample_power(self, forecast: float, std_frac: float) -> np.ndarray:
        """采样实际出力 = forecast × (1 + N(0, std_frac))，clip 到 [0, capacity]"""
        N = self.config.n_scenarios
        noise = self.rng.normal(0, std_frac, N)
        return np.clip(forecast * (1 + noise), 0.0, self.capacity_mw)

    # ------------------------------------------------------------
    # Bid curve evaluation (vectorized SAA)
    # ------------------------------------------------------------

    def _evaluate_bid(
        self,
        bid_quantities: np.ndarray,   # (4,)
        bid_prices: np.ndarray,       # (4,)
        da_samples: np.ndarray,       # (N,)
        rt_samples: np.ndarray,       # (N,)
        power_samples: np.ndarray,    # (N,)
    ) -> float:
        """E[revenue] over N scenarios, vectorized"""
        cfg = self.config

        # cleared[s, k] = I(bid_prices[k] <= da_samples[s]) × bid_quantities[k]
        mask = bid_prices[None, :] <= da_samples[:, None]  # (N, K)
        cleared = (bid_quantities[None, :] * mask).sum(axis=1)  # (N,)

        da_rev = cleared * da_samples                      # (N,)
        deviation = power_samples - cleared                # (N,)
        rt_rev = deviation * rt_samples

        # Standard penalty
        denom = np.maximum(cleared, 1.0)
        dev_ratio = np.abs(deviation) / denom
        excess = np.maximum(dev_ratio - cfg.deviation_bound, 0.0) * denom
        std_pen = excess * np.abs(rt_samples) * cfg.deviation_penalty_ratio

        # Reverse penalty
        # 系统缺电 → RT 显著 > DA: sample RT - da > threshold
        # 此时如果 deviation < 0 (少发), 触发反向
        # 系统过剩 → RT 显著 < DA, deviation > 0 触发
        spread = rt_samples - da_samples
        sys_shortage = spread > cfg.reverse_penalty_spread_threshold
        sys_surplus = spread < -cfg.reverse_penalty_spread_threshold
        rev_trigger = (sys_shortage & (deviation < 0)) | (sys_surplus & (deviation > 0))
        rev_pen = np.where(
            rev_trigger,
            np.abs(deviation) * np.abs(rt_samples) * cfg.reverse_penalty_ratio,
            0.0,
        )

        revenues = da_rev + rt_rev - std_pen - rev_pen
        return float(np.mean(revenues))

    # ------------------------------------------------------------
    # Generate candidate bids
    # ------------------------------------------------------------

    def _candidates_for_hour(
        self,
        power_forecast: float,
        da_quantiles_4: np.ndarray,   # (4,) e.g. [P05, P25, P50, P75]
    ) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
        """
        Returns list of (name, quantities, prices, total_q).
        """
        candidates = []
        cfg = self.config
        for offset in cfg.offset_grid:
            total_q = np.clip(power_forecast * (1 + offset), 0.0, self.capacity_mw)
            if total_q < 1e-3:
                continue
            for pattern, pname in zip(cfg.allocation_patterns, cfg.allocation_pattern_names):
                quantities = total_q * np.array(pattern)
                prices = np.array(da_quantiles_4)
                candidates.append((
                    f"ofs{offset:+.2f}_{pname}",
                    quantities,
                    prices,
                    total_q,
                ))
        return candidates

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def generate(
        self,
        power_forecast_hourly: np.ndarray,         # (24,) MW
        da_quantiles_hourly: np.ndarray,            # (24, 4) 4 分位
        spread_dir_prob_hourly: np.ndarray,         # (24,)
        rt_mean_hourly: np.ndarray | None = None,   # (24,) 可选
        rt_std_hourly: np.ndarray | None = None,    # (24,)
        spread_mag_std_hourly: np.ndarray | None = None,  # (24,)
    ) -> list[HourlyBid]:
        """
        返回 24 个 HourlyBid。

        实现：对每小时独立做 SAA（忽略 24 维相关性；相关性建模留给 P1.2 copula）
        """
        cfg = self.config
        H = len(power_forecast_hourly)
        bids: list[HourlyBid] = []

        for t in range(H):
            pf = float(power_forecast_hourly[t])
            da_q = da_quantiles_hourly[t]        # (4,)
            sd = float(spread_dir_prob_hourly[t])

            if pf < 1e-3:
                bids.append(HourlyBid(
                    hour=t, power_forecast=0.0,
                    rationale="功率预测 = 0"
                ))
                continue

            # Sampling: 优先用 joint distribution（如果提供）
            if self.joint_dist is not None:
                da_samples, rt_samples = self.joint_dist.sample(
                    hour=t,
                    da_quantiles=da_q,
                    quantile_levels=np.array([0.05, 0.25, 0.5, 0.75]),
                    n_scenarios=cfg.n_scenarios,
                    rng=self.rng,
                )
            else:
                # Fallback: 独立 DA + conditional RT via spread_dir_prob
                full_quantiles = np.concatenate([[da_q[0] * 0.5], da_q, [da_q[3] * 1.5]])
                full_levels = np.array([0.0, 0.05, 0.25, 0.5, 0.75, 1.0])
                da_samples = self._sample_da(full_quantiles, full_levels)
                mag_std = (
                    float(spread_mag_std_hourly[t])
                    if spread_mag_std_hourly is not None
                    else cfg.spread_magnitude_std_fallback
                )
                rt_samples = self._sample_rt_given_da(da_samples, sd, mag_std)

            # Power: 正态扰动
            power_samples = self._sample_power(pf, cfg.power_noise_std)

            # 候选 bid curves
            candidates = self._candidates_for_hour(pf, da_q)

            best_rev = -np.inf
            best_bid_info = None
            for (name, quantities, prices, total_q) in candidates:
                rev = self._evaluate_bid(quantities, prices, da_samples, rt_samples, power_samples)
                if rev > best_rev:
                    best_rev = rev
                    best_bid_info = (name, quantities, prices, total_q)

            # 构造 HourlyBid
            name, quantities, prices, total_q = best_bid_info
            steps = [
                BidStep(quantity_mw=float(q), price_yuan_mwh=float(p))
                for q, p in zip(quantities, prices)
                if q > 1e-6
            ]
            bids.append(HourlyBid(
                hour=t,
                steps=steps,
                intended_quantity=float(total_q),
                power_forecast=pf,
                offset_ratio=float(total_q / pf - 1.0) if pf > 0 else 0.0,
                rationale=f"DFL best={name}, E[rev]=¥{best_rev:,.0f}",
            ))

        return bids
