"""
Logan · Daily Bid Generator (合规, 全天一条曲线)
=================================================

根据《甘肃电力现货市场管理实施细则 V3.2》附件2 第 32 条：

    "发电侧从 0 至交易单元最大技术出力之间可最多申报 10 段，
     每段电力不小于装机容量的 10%，
     量价曲线必须为出力区间和报价单调非递减。"

新能源场站的申报结构：
    一条 (power, price) 阶梯曲线，全天 96 个时段共用。
    每 15 分钟时段按当时的 DA 出清价 + 该时段功率预测上限 → 截断计算中标量。

对比 per-hour 版本（optimal_bid.py，已 deprecated）：
    per-hour: 24 条独立曲线, 总共可能 24×10=240 段 → **违规**
    daily:    1 条曲线, 3-10 段, 单调非递减, Σ q = capacity → **合规**

目标函数（单天期望收益最大化）：
    max over (p_1..K, q_1..K)
        E[ Σ_t Q^DA_t × P^DA_t + (Q^actual_t - Q^DA_t) × P^RT_t ] × dt

    其中 Q^DA_t = min( forecast_t, Σ_{k: p_k ≤ P^DA_t} q_k )

约束：
    - K ∈ [min_steps, max_steps] (甘肃 3~10)
    - p_k round to price_precision (10 元/MWh)
    - q_k round to quantity_precision (0.001 MW)
    - p_1 ≤ p_2 ≤ ... ≤ p_K (单调非递减)
    - q_k ≥ min_step_mw (10% capacity)
    - Σ q_k = capacity (覆盖全区间)

算法：
    1. 采样 (DA, RT, actual_power) × 96 时段 × N 场景
    2. 把全天全场景的 DA 价合成一个 flat 分布
    3. 对每个候选 (K, α)：
        a. breakpoints p_k = DA_flat.quantile(α_k), round
        b. greedy 量分配: 先每段 min_step，剩余给 W_k 最大的档
        c. 用完整 cap 逻辑 evaluate E[Revenue]
    4. 取最优

其中 W_k = E[ Σ_t [1{p_k ≤ P^DA_t} × (P^DA_t - P^RT_t)] ]  (简化版忽略 cap)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from products.logan.bid_curve_generator import BidStep
from products.logan.compliance import ComplianceRules, round_to_multiple, round_quantity_array
from core.joint_distribution import EmpiricalJointDistribution


STEPS_PER_DAY = 96


# ============================================================
# Data classes
# ============================================================

@dataclass
class DailyBid:
    """全天一条 bid curve（适用所有 96 时段）"""
    steps: list[BidStep]                    # K 段 (quantity, price)
    capacity_mw: float                       # 装机容量
    expected_revenue: float = 0.0
    rationale: str = ""

    @property
    def n_segments(self) -> int:
        return len(self.steps)

    @property
    def total_quantity(self) -> float:
        return sum(s.quantity_mw for s in self.steps)

    def cleared_at(self, da_price: float, forecast_mw: float) -> float:
        """给定该时段的 DA 出清价 + 功率预测上限，返回中标量"""
        accumulated = 0.0
        for step in self.steps:
            if step.price_yuan_mwh <= da_price:
                accumulated += step.quantity_mw
        # 功率预测上限截断（新能源不能超报 = 中标量 ≤ 该时段功率预测）
        return min(accumulated, forecast_mw)

    def cleared_series(self, da_prices_96: np.ndarray, forecast_96: np.ndarray) -> np.ndarray:
        """返回 (96,) 每时段的中标量"""
        out = np.zeros(len(da_prices_96))
        for t in range(len(da_prices_96)):
            out[t] = self.cleared_at(float(da_prices_96[t]), float(forecast_96[t]))
        return out


@dataclass
class DailyBidConfig:
    n_scenarios: int = 120                 # 采样场景数
    random_seed: int = 42
    actual_noise_std: float = 0.05         # 实际出力 ~ forecast × (1 + N(0, std))

    # 候选搜索空间
    alpha_grid: tuple[tuple[float, ...], ...] = (
        # K=3
        (0.20, 0.50, 0.85),
        (0.30, 0.60, 0.90),
        (0.10, 0.40, 0.75),
        (0.05, 0.35, 0.70),
        # K=4
        (0.15, 0.40, 0.65, 0.90),
        (0.10, 0.30, 0.55, 0.85),
        (0.20, 0.45, 0.70, 0.92),
        # K=5
        (0.10, 0.25, 0.45, 0.70, 0.90),
        (0.05, 0.25, 0.50, 0.75, 0.95),
        (0.15, 0.35, 0.55, 0.75, 0.92),
        # K=6
        (0.05, 0.20, 0.40, 0.60, 0.80, 0.95),
        (0.10, 0.25, 0.40, 0.55, 0.70, 0.88),
        # K=8
        (0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95),
        # K=10（用满所有段）
        (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95),
    )


# ============================================================
# Main class
# ============================================================

class DailyBidGenerator:
    """
    生成合规的全天 bid 曲线（甘肃 V3.2 规则）。
    """

    def __init__(
        self,
        capacity_mw: float,
        rules: ComplianceRules,
        config: DailyBidConfig | None = None,
        joint_dist: EmpiricalJointDistribution | None = None,
    ):
        self.capacity_mw = capacity_mw
        self.rules = rules
        self.config = config or DailyBidConfig()
        self.joint_dist = joint_dist
        self.rng = np.random.default_rng(self.config.random_seed)

    # ============================================================
    # Scenario sampling (DA + RT + actual per 15-min step)
    # ============================================================

    def _sample_scenarios(
        self,
        da_quantiles_96: np.ndarray,          # (96, n_levels)
        quantile_levels: np.ndarray,           # (n_levels,)
        forecast_96: np.ndarray,               # (96,) MW
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (da_samples, rt_samples, actual_samples), all shape (96, N).
        """
        N = self.config.n_scenarios
        H = STEPS_PER_DAY
        da = np.zeros((H, N))
        rt = np.zeros((H, N))
        actual = np.zeros((H, N))

        for t in range(H):
            hour = t // 4
            if self.joint_dist is not None:
                da_t, rt_t = self.joint_dist.sample(
                    hour=hour,
                    da_quantiles=da_quantiles_96[t],
                    quantile_levels=quantile_levels,
                    n_scenarios=N,
                    rng=self.rng,
                )
            else:
                # Fallback: 独立采样
                u = self.rng.uniform(0, 1, N)
                full_levels = np.concatenate([[0.0], quantile_levels, [1.0]])
                full_vals = np.concatenate([
                    [da_quantiles_96[t, 0] * 0.5],
                    da_quantiles_96[t],
                    [da_quantiles_96[t, -1] * 1.5],
                ])
                da_t = np.interp(u, full_levels, full_vals)
                rt_t = da_t + self.rng.normal(0, 50, N)

            # actual 服从 forecast × (1 + ε)
            fc = float(forecast_96[t])
            noise = self.rng.normal(0, self.config.actual_noise_std, N)
            actual_t = np.clip(fc * (1 + noise), 0.0, self.capacity_mw)

            da[t] = da_t
            rt[t] = rt_t
            actual[t] = actual_t

        return da, rt, actual

    # ============================================================
    # Greedy quantity allocation (given breakpoints)
    # ============================================================

    def _allocate_quantities(
        self,
        prices: np.ndarray,          # (K,) sorted + rounded
        da_flat: np.ndarray,          # (96 × N,) flat DA samples
        rt_flat: np.ndarray,          # (96 × N,)
    ) -> np.ndarray | None:
        """
        解量分配（忽略 forecast cap 简化）：
            max Σ_k q_k × W_k
            s.t. Σ q_k = capacity, q_k ≥ min_step

        W_k = E[ 1{p_k ≤ P^DA} × (P^DA - P^RT) ]（忽略 cap）
        """
        rules = self.rules
        K = len(prices)
        min_step = rules.min_step_mw

        # 每档权重（按段累积模型）
        # 注意：当 p_k ≤ DA 时第 k 段激活，则"段 k 的量 × (DA - RT)"计入
        # 但多段同时激活时每段独立贡献
        mask = da_flat[:, None] >= prices[None, :]     # (M, K), M = 96*N
        spread = (da_flat - rt_flat)[:, None]           # (M, 1)
        w = (mask * spread).mean(axis=0)                # (K,)

        # 总额约束
        total = self.capacity_mw
        if K * min_step > total + 1e-6:
            # 段数太多装不下
            return None

        # 初始分配：每段 min_step
        q = np.full(K, min_step, dtype=np.float64)
        remaining = total - K * min_step

        if remaining > 1e-6:
            # Greedy: 按 w 从大到小分配剩余量，单档上限 = capacity
            order = np.argsort(-w)
            for i in order:
                headroom = self.capacity_mw - q[i]
                add = min(remaining, headroom)
                if add > 0:
                    q[i] += add
                    remaining -= add
                if remaining <= 1e-6:
                    break

        return q

    # ============================================================
    # Evaluation with forecast cap
    # ============================================================

    def _evaluate_bid(
        self,
        prices: np.ndarray,          # (K,) sorted
        quantities: np.ndarray,       # (K,)
        da: np.ndarray,               # (96, N)
        rt: np.ndarray,               # (96, N)
        actual: np.ndarray,           # (96, N)
        forecast_96: np.ndarray,      # (96,)
    ) -> float:
        """
        E[ Σ_t Q^DA_t × P^DA_t + (Q^actual_t - Q^DA_t) × P^RT_t ] × dt
        dt = 0.25 h (15-min step)
        """
        # For each (t, s): Q^DA = min(forecast_t, Σ_{p_k ≤ da[t,s]} q_k)
        # Vectorized:
        # cumq[k] = Σ_{j ≤ k} q_j (prefix sum of quantities, length K+1 with cumq[0]=0)
        K = len(prices)
        cumq = np.concatenate([[0.0], np.cumsum(quantities)])   # (K+1,)

        # For each (t, s): find largest k such that prices[k-1] ≤ da[t, s]
        # → activated cumq
        # Use searchsorted on 'prices' which is sorted ascending
        # indices = number of prices ≤ da[t, s]
        idx = np.searchsorted(prices, da, side="right")         # (96, N), values in [0, K]
        activated = cumq[idx]                                    # (96, N), Σ_{p_j ≤ da} q_j

        # 截断到 forecast cap
        forecast_bcast = forecast_96[:, None]                    # (96, 1)
        q_da = np.minimum(activated, forecast_bcast)            # (96, N)

        # 每 (t, s) 收入
        dt = 0.25
        revenue_per_step = (q_da * da + (actual - q_da) * rt) * dt  # (96, N)

        # 平均 N 场景
        expected_revenue = float(revenue_per_step.sum(axis=0).mean())
        return expected_revenue

    # ============================================================
    # Main generation
    # ============================================================

    def generate_day(
        self,
        power_forecast_96: np.ndarray,       # (96,) MW
        da_quantiles_96: np.ndarray,          # (96, n_levels)
        quantile_levels: np.ndarray,          # (n_levels,)
    ) -> DailyBid:
        """
        生成全天一条合规 bid curve。
        """
        assert power_forecast_96.shape == (STEPS_PER_DAY,)
        assert da_quantiles_96.shape[0] == STEPS_PER_DAY

        # Step 1: Sample scenarios
        da, rt, actual = self._sample_scenarios(
            da_quantiles_96=da_quantiles_96,
            quantile_levels=quantile_levels,
            forecast_96=power_forecast_96,
        )

        # Step 2: Flat DA distribution for breakpoint selection
        da_flat = da.flatten()
        rt_flat = rt.flatten()

        # Step 3: Search candidates
        best_rev = -np.inf
        best_prices = None
        best_quantities = None
        best_alpha = None

        rules = self.rules
        for alpha_vec in self.config.alpha_grid:
            alpha = np.asarray(alpha_vec)
            K = len(alpha)
            if K < rules.min_steps or K > rules.max_steps:
                continue

            # Breakpoints from flat DA distribution
            prices_raw = np.quantile(da_flat, alpha)
            prices_r = np.array([round_to_multiple(float(p), rules.price_precision) for p in prices_raw])

            # Clip to price bounds
            if rules.bid_price_lower is not None:
                prices_r = np.maximum(prices_r, rules.bid_price_lower)
            if rules.bid_price_upper is not None:
                prices_r = np.minimum(prices_r, rules.bid_price_upper)

            # 去重 + 保持升序
            prices_r = np.unique(prices_r)
            if len(prices_r) < rules.min_steps or len(prices_r) > rules.max_steps:
                continue

            # 单调性是自动满足的（np.unique 排序）
            # Allocate quantities
            q = self._allocate_quantities(prices_r, da_flat, rt_flat)
            if q is None:
                continue

            # Round quantities
            q = round_quantity_array(q, rules.quantity_precision)
            # Sum constraint after rounding: adjust last segment
            diff = self.capacity_mw - q.sum()
            if abs(diff) > rules.quantity_precision:
                q[-1] += diff
                q[-1] = max(q[-1], rules.min_step_mw)

            # Evaluate
            expected = self._evaluate_bid(
                prices=prices_r,
                quantities=q,
                da=da, rt=rt, actual=actual,
                forecast_96=power_forecast_96,
            )

            if expected > best_rev:
                best_rev = expected
                best_prices = prices_r
                best_quantities = q
                best_alpha = alpha_vec

        if best_prices is None:
            # Fallback: 3 段均分，全部报最低价
            K = 3
            min_step = rules.min_step_mw
            prices_r = np.array([
                rules.bid_price_lower if rules.bid_price_lower is not None else 0,
                rules.bid_price_lower + rules.price_precision if rules.bid_price_lower is not None else rules.price_precision,
                rules.bid_price_lower + 2 * rules.price_precision if rules.bid_price_lower is not None else 2 * rules.price_precision,
            ])
            q = np.array([self.capacity_mw / 3] * 3)
            best_prices = prices_r
            best_quantities = q
            best_alpha = "fallback"
            best_rev = 0.0

        # Construct bid
        steps = [
            BidStep(quantity_mw=float(q), price_yuan_mwh=float(p))
            for q, p in zip(best_quantities, best_prices)
            if q > 1e-9
        ]

        return DailyBid(
            steps=steps,
            capacity_mw=self.capacity_mw,
            expected_revenue=float(best_rev),
            rationale=f"DailyBid K={len(steps)}, α={best_alpha}, E[rev]=¥{best_rev:,.0f}",
        )
