"""
Logan · Regime-Aware Bid Generator
====================================

动机（来自实验 B 的发现）：
    在 3 个 90 天 window 上，Logan 相对 Naive 的增益分布极不均匀：
      Window 1（低电价季）：Optimal 比 Naive 多 ¥2.2M
      Window 3（高电价季）：Naive 比 Optimal 多 ¥60k

    原因：Optimal 的"低中标率 + RT 套利"在 RT < DA 时亏钱。
         Naive 的"全量中标 DA"在 DA > RT 时最优。

策略：
    用 spread_direction head 在 D-1 预测每小时 P(RT > DA)。
    - 若 P(RT > DA) 高（e.g., > 0.55）→ Optimal（不中标推 RT）
    - 若 P(RT > DA) 低（e.g., < 0.45）→ Naive-style（全量低价报）
    - 中间区间 → Optimal（默认）

每小时独立选择，而非日级。更精细，对 per-hour regime 变化敏感。

实现：
    两个子 generator 的包装器。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from loguru import logger

from products.logan.bid_curve_generator import BidStep, HourlyBid
from products.logan.optimal_bid import OptimalBidGenerator
from products.logan.compliance import ComplianceRules, round_to_multiple, round_quantity_array
from core.joint_distribution import EmpiricalJointDistribution


@dataclass
class RegimeAwareConfig:
    # 切换到 naive-style 的阈值（P(RT > DA) 低于此）
    naive_threshold: float = 0.45

    # Naive-style bid 的报价水平（相对 DA P05 的倍数）
    naive_bid_price_ratio: float = 0.5       # 报 P05 × 0.5（低价保中标）


class RegimeAwareBidGenerator:
    """
    Per-hour regime selector:
      - spread_dir_prob < threshold → Naive-style (全量低价)
      - spread_dir_prob >= threshold → Optimal
    """

    def __init__(
        self,
        capacity_mw: float,
        rules: ComplianceRules,
        optimal_gen: OptimalBidGenerator,
        config: RegimeAwareConfig | None = None,
    ):
        self.capacity_mw = capacity_mw
        self.rules = rules
        self.optimal_gen = optimal_gen
        self.config = config or RegimeAwareConfig()

    def _make_naive_style_bid(
        self,
        hour: int,
        power_forecast: float,
        da_q05: float,
    ) -> HourlyBid:
        """
        生成合规的 naive-style bid：3 段（或更少），所有段低价保全量中标。
        返回空 bid 如 forecast < min_step。
        """
        rules = self.rules
        min_step = rules.min_step_mw
        if power_forecast < min_step:
            return HourlyBid(
                hour=hour, power_forecast=power_forecast,
                rationale="naive-style: forecast < min_step"
            )

        # 尝试 K 段（K=3 是下限；更多段可以但无必要）
        K = rules.min_steps
        # 每段尽量 min_step 以上且 Σ ≤ forecast 且 Σ = min(capacity, forecast)
        total = min(self.capacity_mw, power_forecast)

        # 如果 K × min_step > total，退化到更少段
        while K * min_step > total + 1e-6 and K > 1:
            K -= 1

        if K < rules.min_steps:
            # 退到 1 段（违反最少 3 段，但 enforce 时会处理）
            # 直接 fallback 给 Optimal
            return self.optimal_gen.generate_hourly(
                hour=hour,
                power_forecast=power_forecast,
                da_quantiles=np.array([da_q05]),
                quantile_levels=np.array([0.05]),
            )

        # 分配量
        base_q = total / K
        quantities = np.full(K, base_q)
        # 每段 clip 到 min_step 以上
        quantities = np.maximum(quantities, min_step)
        # 若 Σ > total，按比例压
        s = quantities.sum()
        if s > total + 1e-6:
            quantities = quantities * total / s
        quantities = round_quantity_array(quantities, rules.quantity_precision)

        # 低价：每段价相差 price_precision，起始价 = max(0, q05 × ratio) rounded
        base_price = max(
            round_to_multiple(da_q05 * self.config.naive_bid_price_ratio, rules.price_precision),
            round_to_multiple(rules.bid_price_lower if rules.bid_price_lower is not None else 0,
                              rules.price_precision),
        )
        # 保证都 clip 到 lower bound 以上
        if rules.bid_price_lower is not None and base_price < rules.bid_price_lower:
            base_price = rules.bid_price_lower
        prices = np.array([base_price + i * rules.price_precision for i in range(K)])

        # Clip 到上限
        if rules.bid_price_upper is not None:
            prices = np.minimum(prices, rules.bid_price_upper)

        # 确保单调递增（去重）
        prices = np.unique(prices)
        while len(prices) < K:
            # 如果去重后不够，最后一段价 + precision
            prices = np.append(prices, prices[-1] + rules.price_precision)
        prices = prices[:K]

        steps = [
            BidStep(quantity_mw=float(q), price_yuan_mwh=float(p))
            for q, p in zip(quantities, prices)
            if q > 1e-9
        ]
        return HourlyBid(
            hour=hour,
            steps=steps,
            intended_quantity=float(quantities.sum()),
            power_forecast=power_forecast,
            offset_ratio=float(quantities.sum() / power_forecast - 1.0) if power_forecast > 0 else 0.0,
            rationale=f"naive-style (spread_dir low), K={K}, base_price={base_price}",
        )

    def generate(
        self,
        power_forecast_hourly: np.ndarray,
        da_quantiles_hourly: np.ndarray,       # (24, n_q)
        quantile_levels: np.ndarray,
        spread_dir_prob_hourly: np.ndarray,    # (24,)
    ) -> list[HourlyBid]:
        H = len(power_forecast_hourly)
        threshold = self.config.naive_threshold

        # 先全部跑 Optimal
        bids = self.optimal_gen.generate(
            power_forecast_hourly=power_forecast_hourly,
            da_quantiles_hourly=da_quantiles_hourly,
            quantile_levels=quantile_levels,
        )

        # 对 spread_dir < threshold 的小时切换 naive
        switches = 0
        for t in range(H):
            if spread_dir_prob_hourly[t] < threshold:
                # DA P05 index in quantile_levels
                idx_05 = int(np.argmin(np.abs(quantile_levels - 0.05)))
                da_q05 = float(da_quantiles_hourly[t, idx_05])
                naive_bid = self._make_naive_style_bid(
                    hour=t,
                    power_forecast=float(power_forecast_hourly[t]),
                    da_q05=da_q05,
                )
                bids[t] = naive_bid
                switches += 1

        # 更新 rationale 增加 regime 信息
        for b in bids:
            if b.steps and "naive-style" not in b.rationale:
                b.rationale = f"[regime=opt] {b.rationale}"
        return bids
