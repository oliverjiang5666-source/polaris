"""
Logan · Bid Curve Generator
============================

把四个预测（DA、RT、spread 方向、系统偏差）+ 客户功率预测，
组合成 D 日 24/96 个时段的阶梯报价曲线。

**决策逻辑（来自前面第一性原理讨论）**：

对每个小时 t：

1. 申报量（总量决策）：
   offset_ratio = target_offset_fn(spread_direction_prob[t], system_deviation_risk[t])
     - spread_dir_prob > 0.65 (RT 很可能 > DA): offset 负（少报，偏差朝上）
     - spread_dir_prob < 0.35 (RT 很可能 < DA): offset 正（多报，偏差朝下）
     - 反向加罚风险高：offset 幅度降低
   申报量 = power_forecast × (1 + offset_ratio)
   偏差幅度受 deviation_bound 约束（典型 ±10%）

2. 申报价阶梯（价分档决策）：
   用 DA 分位数切档：
     - 第 1 档（占量 40%）：报 DA P05（保证中标）
     - 第 2 档（占量 30%）：报 DA P25
     - 第 3 档（占量 20%）：报 DA P50
     - 第 4 档（占量 10%）：报 DA P75
     - （不报 P95 段，作"期权"）
   量按申报总量分配。

3. 输出：每个时段的阶梯 (quantity, price) 列表。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class BidStep:
    quantity_mw: float
    price_yuan_mwh: float


@dataclass
class HourlyBid:
    hour: int                           # 0..23 (或时段索引 0..95 取决于粒度)
    steps: list[BidStep] = field(default_factory=list)
    intended_quantity: float = 0.0      # 我们希望申报的总量
    power_forecast: float = 0.0         # 原始功率预测（MW）
    offset_ratio: float = 0.0           # 相对 forecast 的偏移比例
    rationale: str = ""                 # 决策说明（for 可解释性）

    @property
    def total_quantity(self) -> float:
        return sum(s.quantity_mw for s in self.steps)


@dataclass
class BidCurveConfig:
    deviation_bound: float = 0.10          # 偏差率硬约束（±10% 触发加罚）
    aggressive_threshold_up: float = 0.65  # P(spread>0) > 此值 → 少报
    aggressive_threshold_down: float = 0.35 # P(spread>0) < 此值 → 多报
    max_offset_ratio: float = 0.09         # 卡在加罚阈值内（±10% 的 90%）
    system_risk_penalty: float = 1.0        # 反向加罚风险每增加 1，offset 折半
    step_fractions: tuple[float, ...] = (0.40, 0.30, 0.20, 0.10)  # 阶梯量占比
    step_quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75)  # 阶梯价分位
    # ---- Ablation switches ----
    disable_spread_direction: bool = False  # True: offset_ratio 恒为 0
    disable_system_deviation: bool = False  # True: damp 恒为 1
    use_single_step: bool = False           # True: 单档报价（所有量在 P05）
    use_flat_price: bool = False            # True: 所有档都用 P50


class BidCurveGenerator:
    """
    生成申报曲线。对外只有一个 generate() 接口。
    """

    def __init__(self, capacity_mw: float, config: BidCurveConfig | None = None):
        self.capacity_mw = capacity_mw
        self.config = config or BidCurveConfig()
        assert len(self.config.step_fractions) == len(self.config.step_quantiles)
        assert abs(sum(self.config.step_fractions) - 1.0) < 1e-6

    # ============================================================
    # Core decision logic
    # ============================================================

    def _compute_offset_ratio(
        self,
        spread_dir_prob: float,
        system_shortage_prob: float,
        system_surplus_prob: float,
    ) -> tuple[float, str]:
        """
        返回 (offset_ratio, rationale)。

        逻辑：
        - 先根据 spread 方向决定 base offset
        - 再用 system deviation 风险 damp
        """
        cfg = self.config
        base = 0.0
        rationale = []

        # Ablation: disable spread direction completely
        if cfg.disable_spread_direction:
            return 0.0, "spread_direction disabled (ablation)"

        if spread_dir_prob > cfg.aggressive_threshold_up:
            # RT 很可能 > DA → 少报（偏差朝上）
            # offset < 0: 申报量 < 功率预测
            intensity = (spread_dir_prob - cfg.aggressive_threshold_up) / (1.0 - cfg.aggressive_threshold_up)
            base = -cfg.max_offset_ratio * intensity
            rationale.append(f"spread上行概率 {spread_dir_prob:.2f} → 少报")
        elif spread_dir_prob < cfg.aggressive_threshold_down:
            # RT 很可能 < DA → 多报（偏差朝下）
            intensity = (cfg.aggressive_threshold_down - spread_dir_prob) / cfg.aggressive_threshold_down
            base = +cfg.max_offset_ratio * intensity
            rationale.append(f"spread下行概率 {1-spread_dir_prob:.2f} → 多报")
        else:
            rationale.append(f"spread方向不清（prob={spread_dir_prob:.2f}）→ 按实报")

        # System risk damp — 可 ablation 关掉
        if not cfg.disable_system_deviation:
            if base > 0 and system_surplus_prob > 0.3:
                damp = 1.0 / (1.0 + cfg.system_risk_penalty * (system_surplus_prob - 0.3))
                base *= damp
                rationale.append(f"系统过剩风险{system_surplus_prob:.2f} 压缩多报幅度")
            elif base < 0 and system_shortage_prob > 0.3:
                damp = 1.0 / (1.0 + cfg.system_risk_penalty * (system_shortage_prob - 0.3))
                base *= damp
                rationale.append(f"系统缺电风险{system_shortage_prob:.2f} 压缩少报幅度")

        # Hard cap 在 deviation_bound 以内
        base = float(np.clip(base, -cfg.deviation_bound * 0.95, cfg.deviation_bound * 0.95))
        return base, "; ".join(rationale)

    def _build_step_ladder(
        self,
        intended_quantity: float,
        da_quantiles: np.ndarray,  # shape (n_q,) 对应 config.step_quantiles 的价格
    ) -> list[BidStep]:
        """
        把 intended_quantity 切成阶梯。

        设计：
          - 前档（P05/P25）报低价 → 稳中标
          - 后档（P50/P75）报高价 → 只有 DA 出清价更高时才中
        """
        cfg = self.config
        # Ablation: single step（所有量报最低分位，保中标）
        if cfg.use_single_step:
            return [BidStep(quantity_mw=intended_quantity, price_yuan_mwh=float(da_quantiles[0]))]

        # Ablation: flat price（所有档都用 P50）
        if cfg.use_flat_price:
            # 找 P50（step_quantiles 里找 0.5，找不到用 index=len//2）
            try:
                p50_idx = list(cfg.step_quantiles).index(0.5)
            except ValueError:
                p50_idx = len(da_quantiles) // 2
            flat_price = float(da_quantiles[p50_idx])
            return [BidStep(quantity_mw=intended_quantity, price_yuan_mwh=flat_price)]

        steps = []
        for frac, price in zip(cfg.step_fractions, da_quantiles):
            q = frac * intended_quantity
            if q > 1e-6:
                steps.append(BidStep(quantity_mw=q, price_yuan_mwh=float(price)))
        return steps

    # ============================================================
    # Public API
    # ============================================================

    def generate(
        self,
        power_forecast_hourly: np.ndarray,             # (H,) MW, H=24 或 96
        da_quantiles_hourly: np.ndarray,               # (H, n_quantiles) 对应 step_quantiles
        spread_dir_prob_hourly: np.ndarray,            # (H,)
        system_shortage_prob_hourly: np.ndarray,       # (H,)
        system_surplus_prob_hourly: np.ndarray,        # (H,)
    ) -> list[HourlyBid]:
        """
        返回长度 H 的 HourlyBid 列表。

        所有输入长度必须一致（24 或 96，自洽即可）。
        da_quantiles_hourly 的第二维长度必须等于 len(config.step_quantiles)。
        """
        H = len(power_forecast_hourly)
        assert da_quantiles_hourly.shape == (H, len(self.config.step_quantiles)), \
            f"DA 分位数形状应为 ({H}, {len(self.config.step_quantiles)})"
        assert len(spread_dir_prob_hourly) == H
        assert len(system_shortage_prob_hourly) == H
        assert len(system_surplus_prob_hourly) == H

        bids: list[HourlyBid] = []
        for t in range(H):
            pf = float(power_forecast_hourly[t])
            if pf < 1e-3:
                # 功率预测为 0（如夜间光伏），直接不报
                bids.append(HourlyBid(hour=t, power_forecast=0.0, rationale="功率预测为 0"))
                continue

            offset, rationale = self._compute_offset_ratio(
                spread_dir_prob=float(spread_dir_prob_hourly[t]),
                system_shortage_prob=float(system_shortage_prob_hourly[t]),
                system_surplus_prob=float(system_surplus_prob_hourly[t]),
            )

            intended = pf * (1 + offset)
            # 物理上不超过装机
            intended = float(np.clip(intended, 0.0, self.capacity_mw))

            steps = self._build_step_ladder(
                intended_quantity=intended,
                da_quantiles=da_quantiles_hourly[t],
            )

            bids.append(HourlyBid(
                hour=t,
                steps=steps,
                intended_quantity=intended,
                power_forecast=pf,
                offset_ratio=offset,
                rationale=rationale,
            ))

        return bids

    # ============================================================
    # Helpers for evaluator
    # ============================================================

    @staticmethod
    def cleared_quantity(bid: HourlyBid, da_clearing_price: float) -> float:
        """给定实际 DA 出清价，这个小时中标多少 MW"""
        cleared = 0.0
        for step in bid.steps:
            if step.price_yuan_mwh <= da_clearing_price:
                cleared += step.quantity_mw
        return cleared

    @staticmethod
    def total_intended(bid: HourlyBid) -> float:
        return bid.intended_quantity

    @staticmethod
    def bids_to_cleared_array(
        bids: list[HourlyBid],
        da_clearing_prices: np.ndarray,  # (H,)
    ) -> np.ndarray:
        """批量：每个时段中标量"""
        H = len(bids)
        out = np.zeros(H)
        for i, b in enumerate(bids):
            out[i] = BidCurveGenerator.cleared_quantity(b, float(da_clearing_prices[i]))
        return out
