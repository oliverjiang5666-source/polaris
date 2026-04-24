"""
Logan · Optimal Bid Generator (理论最优，合规)
=================================================

第一性原理推导（见 session 元思考）：

    新能源场站期望收益 over bid curve：
        E[R'] = Σ_k q_k × w_k  +  const

    其中  w_k = E[ 1{p_k ≤ P^DA} × (P^DA - P^RT) ]

         = P(P^DA ≥ p_k) × E[P^DA - P^RT | P^DA ≥ p_k]

    即每档的期望边际贡献。

算法：
    1. 从 (P^DA, P^RT) 联合分布采 N 个场景
    2. 对一批候选 breakpoint 集合（K ∈ [3, 10] × 若干 alpha pattern）：
       a. 计算每档 w_k（向量化）
       b. 解量分配优化（greedy ILP，保合规）
       c. 记录 expected revenue
    3. 取最优

合规约束（从 compliance.py / gansu.yaml 读）：
    - K ∈ [3, 10]
    - 每段 ≥ 10% × capacity
    - 总量 ≤ min(功率预测, capacity)
    - 价格 round 到 10 元/MWh
    - 单调非递减

相比 P1.2 DFL：
    - DFL 枚举 fixed patterns（违规候选多），只在小搜索空间里做 SAA
    - 本版：breakpoint 搜索 + 量分配 LP，所有候选合规，更优
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from products.logan.bid_curve_generator import BidStep, HourlyBid
from products.logan.compliance import ComplianceRules, round_to_multiple, round_quantity_array
from core.joint_distribution import EmpiricalJointDistribution


# ============================================================
# Config
# ============================================================

@dataclass
class OptimalBidConfig:
    n_scenarios: int = 300
    random_seed: int = 42

    # Breakpoint 搜索：每组 alpha vector 定义一组 K 个 breakpoint
    # 每个 alpha 对应 P^DA 的分位数
    # 搜索空间覆盖不同 K 和不同"分布重心"
    alpha_grid: tuple[tuple[float, ...], ...] = (
        # K=3
        (0.10, 0.50, 0.90),
        (0.05, 0.30, 0.70),
        (0.30, 0.60, 0.90),
        # K=4
        (0.10, 0.30, 0.60, 0.90),
        (0.05, 0.25, 0.50, 0.75),
        (0.25, 0.50, 0.75, 0.95),
        # K=5
        (0.05, 0.25, 0.50, 0.75, 0.95),
        (0.10, 0.30, 0.50, 0.70, 0.90),
        (0.05, 0.15, 0.35, 0.60, 0.85),
        (0.15, 0.35, 0.55, 0.75, 0.95),
        # K=6
        (0.05, 0.20, 0.40, 0.60, 0.80, 0.95),
        (0.10, 0.25, 0.40, 0.55, 0.70, 0.90),
        # K=8
        (0.05, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95),
        (0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95),
        # K=10
        (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95),
    )


# ============================================================
# Main class
# ============================================================

class OptimalBidGenerator:
    """
    Breakpoint + quantity 优化的合规 bid 生成器。
    """

    def __init__(
        self,
        capacity_mw: float,
        rules: ComplianceRules,
        config: OptimalBidConfig | None = None,
        joint_dist: EmpiricalJointDistribution | None = None,
    ):
        self.capacity_mw = capacity_mw
        self.rules = rules
        self.config = config or OptimalBidConfig()
        self.joint_dist = joint_dist
        self.rng = np.random.default_rng(self.config.random_seed)

    # ============================================================
    # Quantity allocation (given breakpoints)
    # ============================================================

    def _allocate_quantities(
        self,
        prices: np.ndarray,    # (K,) sorted
        w: np.ndarray,          # (K,) weights
        forecast: float,
    ) -> np.ndarray | None:
        """
        求解：
            max   Σ_k w_k × q_k
            s.t.  Σ_k q_k ≤ Q_total_max = min(capacity, forecast)
                  q_k ∈ {0} ∪ [min_step, cap_remaining]
                  #_active ∈ [min_steps, max_steps] where active = (q_k > 0)
                  Σ q_k = Q_total_max (若 coverage="full" 且强制)

        为简化，使用 greedy + 每段默认 min_step、剩余量加到 w_k 最大的档。
        这在段约束下近似最优（全局最优需要 ILP，但 K ≤ 10 搜索空间小可接受）。

        Returns None if infeasible.
        """
        rules = self.rules
        K = len(prices)
        min_step = rules.min_step_mw
        total_cap = min(self.capacity_mw, forecast)

        # 强制所有 K 段激活（覆盖全区间）
        # 最少先给每段 min_step
        base = min_step * K
        if base > total_cap + 1e-6:
            # 总预算不够，退化到激活更少的段
            max_active = int(total_cap // min_step)
            if max_active < rules.min_steps:
                return None
            # 选 w 最大的 max_active 段激活
            # 但价格需要保持单调：选 top-w 中索引构成连续段可能不行
            # 简化：激活 w 最大的 max_active 个档（可能价格不连续）
            sorted_idx = np.argsort(-w)[:max_active]
            sorted_idx = np.sort(sorted_idx)  # 按 price index 排
            q = np.zeros(K)
            remaining = total_cap
            # 分配 min_step 到每个激活档
            for i in sorted_idx:
                q[i] = min_step
                remaining -= min_step
            # 剩余给 w 最大的激活档
            if remaining > 0:
                best = sorted_idx[np.argmax(w[sorted_idx])]
                q[best] += remaining
            return q

        # 正常情况：激活全部 K 段
        q = np.full(K, min_step, dtype=np.float64)
        remaining = total_cap - base

        if remaining > 0:
            # 给 w_k 最大的档加剩余量
            # 但要考虑：每档上限 = capacity（物理）
            # 用 greedy：从 w 最大的档开始分配
            sorted_idx = np.argsort(-w)
            for i in sorted_idx:
                headroom = self.capacity_mw - q[i]  # 单档上限 = 装机
                add = min(remaining, headroom)
                if add > 0:
                    q[i] += add
                    remaining -= add
                if remaining <= 1e-6:
                    break

        return q

    # ============================================================
    # Evaluation
    # ============================================================

    def _evaluate_breakpoints(
        self,
        prices: np.ndarray,      # (K,) sorted, rounded
        da_samples: np.ndarray,  # (N,)
        rt_samples: np.ndarray,  # (N,)
        forecast: float,
    ) -> tuple[float, np.ndarray]:
        """
        给定 breakpoints，计算最优量分配 + 期望收入。
        Returns (expected_revenue, quantities) or (-inf, None) if infeasible.
        """
        # 对每档计算 w_k = E[ 1{P^DA ≥ p_k} × (P^DA - P^RT) ]
        spread = da_samples - rt_samples       # (N,)
        # 对每个 p_k 计算 mask
        mask = da_samples[:, None] >= prices[None, :]  # (N, K)
        contrib = mask * spread[:, None]                # (N, K)
        w = contrib.mean(axis=0)                        # (K,)

        q = self._allocate_quantities(prices, w, forecast)
        if q is None:
            return -np.inf, None

        # 期望收入（仅可变部分）= Σ q_k × w_k
        revenue = float(np.sum(q * w))
        return revenue, q

    # ============================================================
    # Main generation loop
    # ============================================================

    def generate_hourly(
        self,
        hour: int,
        power_forecast: float,
        da_quantiles: np.ndarray,          # (n_q,) for sampling DA
        quantile_levels: np.ndarray,       # (n_q,) e.g. [0.05, 0.25, 0.5, 0.75, 0.95]
        spread_dir_prob: float | None = None,
        spread_mag_std: float | None = None,
    ) -> HourlyBid:
        """生成单小时的最优 bid"""
        rules = self.rules

        if power_forecast < rules.min_step_mw - 1e-6:
            # 预测量 < 10% 装机，无法凑出一段
            return HourlyBid(
                hour=hour, power_forecast=power_forecast,
                rationale=f"功率预测 {power_forecast:.2f} < 最小段 {rules.min_step_mw:.2f} MW",
            )

        # ---- Step 1: 采样 (DA, RT) ----
        if self.joint_dist is not None:
            da_samples, rt_samples = self.joint_dist.sample(
                hour=hour,
                da_quantiles=da_quantiles,
                quantile_levels=quantile_levels,
                n_scenarios=self.config.n_scenarios,
                rng=self.rng,
            )
        else:
            # Fallback：独立采样
            u = self.rng.uniform(0, 1, self.config.n_scenarios)
            da_samples = np.interp(u, quantile_levels, da_quantiles)
            mag_std = spread_mag_std or 50.0
            sd = spread_dir_prob if spread_dir_prob is not None else 0.5
            signs = np.where(self.rng.random(self.config.n_scenarios) < sd, 1.0, -1.0)
            mags = np.abs(self.rng.normal(0, mag_std, self.config.n_scenarios))
            rt_samples = da_samples + signs * mags

        # ---- Step 2: 枚举候选 breakpoints ----
        best_revenue = -np.inf
        best_q = None
        best_prices = None
        best_alpha = None

        for alpha_vec in self.config.alpha_grid:
            alpha = np.asarray(alpha_vec)
            K = len(alpha)
            if K < rules.min_steps or K > rules.max_steps:
                continue

            # 把 alpha 映射到预测分位数
            prices_raw = np.interp(alpha, quantile_levels, da_quantiles)
            # Round 到价格精度
            prices_r = np.array([round_to_multiple(p, rules.price_precision) for p in prices_raw])

            # 去重（round 后可能有重复）
            unique_prices = np.unique(prices_r)
            if len(unique_prices) < rules.min_steps:
                # round 后段数不够，跳过
                continue

            # Clip 到价格范围
            if rules.bid_price_lower is not None:
                unique_prices = np.maximum(unique_prices, rules.bid_price_lower)
            if rules.bid_price_upper is not None:
                unique_prices = np.minimum(unique_prices, rules.bid_price_upper)
            unique_prices = np.unique(unique_prices)  # 再去重

            if len(unique_prices) < rules.min_steps or len(unique_prices) > rules.max_steps:
                continue

            revenue, q = self._evaluate_breakpoints(
                prices=unique_prices,
                da_samples=da_samples,
                rt_samples=rt_samples,
                forecast=power_forecast,
            )
            if q is None:
                continue

            if revenue > best_revenue:
                best_revenue = revenue
                best_q = q
                best_prices = unique_prices
                best_alpha = alpha_vec

        if best_q is None:
            # 所有候选都 infeasible
            return HourlyBid(
                hour=hour, power_forecast=power_forecast,
                rationale="所有 breakpoint 候选 infeasible",
            )

        # ---- Step 3: Round quantities ----
        best_q = round_quantity_array(best_q, rules.quantity_precision)

        # ---- Step 4: 构造合规 bid ----
        steps = [
            BidStep(quantity_mw=float(q), price_yuan_mwh=float(p))
            for q, p in zip(best_q, best_prices)
            if q > 1e-9
        ]

        return HourlyBid(
            hour=hour,
            steps=steps,
            intended_quantity=float(best_q.sum()),
            power_forecast=power_forecast,
            offset_ratio=float(best_q.sum() / power_forecast - 1.0) if power_forecast > 0 else 0.0,
            rationale=f"OptimalBid α={best_alpha}, E[rev]=¥{best_revenue:,.1f}",
        )

    def generate(
        self,
        power_forecast_hourly: np.ndarray,         # (24,)
        da_quantiles_hourly: np.ndarray,            # (24, n_q)
        quantile_levels: np.ndarray,                # (n_q,)
        spread_dir_prob_hourly: np.ndarray | None = None,
        spread_mag_std_hourly: np.ndarray | None = None,
    ) -> list[HourlyBid]:
        H = len(power_forecast_hourly)
        bids: list[HourlyBid] = []
        for t in range(H):
            bid = self.generate_hourly(
                hour=t,
                power_forecast=float(power_forecast_hourly[t]),
                da_quantiles=da_quantiles_hourly[t],
                quantile_levels=quantile_levels,
                spread_dir_prob=(
                    float(spread_dir_prob_hourly[t]) if spread_dir_prob_hourly is not None else None
                ),
                spread_mag_std=(
                    float(spread_mag_std_hourly[t]) if spread_mag_std_hourly is not None else None
                ),
            )
            bids.append(bid)
        return bids
