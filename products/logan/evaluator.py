"""
Logan · Evaluator (v2, 甘肃真实规则)
====================================

按甘肃《电力现货市场结算实施细则 V3.2》附件3 第 20-24 条的四段结算：

    R = Q^MLT × P^MLT                               [中长期电量]
      + Q^MLT × (P^DA,node − P^ref)                 [中长期阻塞]
      + (Q^DA − Q^MLT) × P^DA,node                  [日前现货]
      + (Q^actual − Q^DA) × P^RT,node               [实时偏差]

⚠️  和 v1 的差异（v1 是我编的）：
    - 去掉 deviation_penalty（甘肃无 ±10% 加罚）
    - 去掉 reverse_penalty（甘肃无反向加罚）
    - 加上 MLT 电量 + MLT 阻塞对冲（可选）

对 Logan 的 Bid 的关系：
    Q^DA = Σ_k q_k × 1{p_k ≤ P^DA,node}   （bid curve 中标量）
    Q^actual = 实际上网电量
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from products.logan.bid_curve_generator import HourlyBid, BidCurveGenerator


# ============================================================
# Config
# ============================================================

@dataclass
class SettlementConfig:
    """甘肃真实规则的结算配置"""
    use_mlt: bool = False                      # 是否启用中长期合同项
    mlt_quantity_mw: float = 0.0               # 中长期合同量（每小时）
    mlt_price_yuan_mwh: float = 0.0            # 中长期合同价
    mlt_reference_price_default: float = 0.0   # 参考点价（缺省用统一结算点价）

    # 节点 vs 统一结算点（简化：默认节点价 = 统一价，无阻塞）
    use_node_lmp: bool = True

    # 价格限值（可选）
    clearing_price_lower: float | None = None
    clearing_price_upper: float | None = None


# ============================================================
# Result
# ============================================================

@dataclass
class EvalResult:
    total_revenue: float
    mlt_energy_revenue: float = 0.0
    mlt_congestion_revenue: float = 0.0
    da_spot_revenue: float = 0.0
    rt_spot_revenue: float = 0.0

    # 统计
    n_hours: int = 0
    cleared_mwh: float = 0.0
    actual_mwh: float = 0.0
    deviation_ratios: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary(self) -> str:
        return (
            f"Total revenue: ¥{self.total_revenue:,.0f}\n"
            f"  MLT energy:   ¥{self.mlt_energy_revenue:,.0f}\n"
            f"  MLT congest:  ¥{self.mlt_congestion_revenue:+,.0f}\n"
            f"  DA spot:      ¥{self.da_spot_revenue:,.0f}\n"
            f"  RT spot:      ¥{self.rt_spot_revenue:+,.0f}\n"
            f"  Hours: {self.n_hours}  Cleared: {self.cleared_mwh:.1f} MWh  "
            f"Actual: {self.actual_mwh:.1f} MWh"
        )

    @property
    def clear_ratio(self) -> float:
        return self.cleared_mwh / max(self.actual_mwh, 1e-9)


# ============================================================
# Evaluator
# ============================================================

class LoganEvaluator:
    """
    Settle bids against actual market + power, return revenue breakdown.
    甘肃规则：无偏差加罚，纯 DA + RT 差价结算。
    """

    def __init__(self, config: SettlementConfig | None = None):
        self.config = config or SettlementConfig()

    def _clip_prices(self, prices: np.ndarray) -> np.ndarray:
        """限值处理"""
        if self.config.clearing_price_lower is not None:
            prices = np.maximum(prices, self.config.clearing_price_lower)
        if self.config.clearing_price_upper is not None:
            prices = np.minimum(prices, self.config.clearing_price_upper)
        return prices

    def settle_hourly(
        self,
        cleared_qty: np.ndarray,      # (H,) 日前中标量 MW
        actual_power: np.ndarray,     # (H,) 实际出力 MW
        da_prices_node: np.ndarray,   # (H,) 节点日前价
        rt_prices_node: np.ndarray,   # (H,) 节点实时价
        da_prices_unified: np.ndarray | None = None,  # (H,) 统一结算点日前价（用于 MLT 阻塞）
        dt_hours: float = 1.0,
    ) -> EvalResult:
        """
        按附件3 第 20-24 条结算。
        """
        cfg = self.config
        H = len(cleared_qty)
        assert all(len(x) == H for x in (actual_power, da_prices_node, rt_prices_node))

        da_prices_node = self._clip_prices(np.asarray(da_prices_node, dtype=np.float64))
        rt_prices_node = self._clip_prices(np.asarray(rt_prices_node, dtype=np.float64))
        if da_prices_unified is None:
            da_prices_unified = da_prices_node
        else:
            da_prices_unified = self._clip_prices(np.asarray(da_prices_unified, dtype=np.float64))

        # ---- 1. 中长期电量电费 ----
        mlt_qty = cfg.mlt_quantity_mw if cfg.use_mlt else 0.0
        mlt_energy = mlt_qty * cfg.mlt_price_yuan_mwh * dt_hours * H

        # ---- 2. 中长期阻塞对冲 ----
        # R^MLT_congestion = Σ Q^MLT × (P^DA_node − P^ref)
        # 参考点价默认 = 统一结算点价（简化）
        ref_price = da_prices_unified
        mlt_congestion = float(mlt_qty * (da_prices_node - ref_price).sum() * dt_hours)

        # ---- 3. 日前现货 ----
        # R^DA = (Q^DA - Q^MLT) × P^DA_node
        da_spot_arr = (cleared_qty - mlt_qty) * da_prices_node * dt_hours
        da_spot = float(da_spot_arr.sum())

        # ---- 4. 实时偏差 ----
        # R^RT = (Q^actual - Q^DA) × P^RT_node
        deviation = actual_power - cleared_qty
        rt_spot_arr = deviation * rt_prices_node * dt_hours
        rt_spot = float(rt_spot_arr.sum())

        # ---- 统计 ----
        eps = 1e-6
        denom = np.maximum(np.abs(cleared_qty), 1.0)
        deviation_ratios = deviation / denom

        total = mlt_energy + mlt_congestion + da_spot + rt_spot

        return EvalResult(
            total_revenue=total,
            mlt_energy_revenue=mlt_energy,
            mlt_congestion_revenue=mlt_congestion,
            da_spot_revenue=da_spot,
            rt_spot_revenue=rt_spot,
            n_hours=H,
            cleared_mwh=float((cleared_qty * dt_hours).sum()),
            actual_mwh=float((actual_power * dt_hours).sum()),
            deviation_ratios=deviation_ratios,
        )

    # ============================================================
    # Convenience
    # ============================================================

    def settle_bids(
        self,
        bids: list[HourlyBid],
        actual_power: np.ndarray,
        da_prices_node: np.ndarray,
        rt_prices_node: np.ndarray,
        da_prices_unified: np.ndarray | None = None,
        dt_hours: float = 1.0,
    ) -> EvalResult:
        cleared = BidCurveGenerator.bids_to_cleared_array(bids, da_prices_node)
        return self.settle_hourly(
            cleared_qty=cleared,
            actual_power=actual_power,
            da_prices_node=da_prices_node,
            rt_prices_node=rt_prices_node,
            da_prices_unified=da_prices_unified,
            dt_hours=dt_hours,
        )

    def settle_daily_bid(
        self,
        daily_bid,                           # DailyBid
        actual_power_96: np.ndarray,         # (96,) MW per 15-min
        forecast_96: np.ndarray,             # (96,) MW per 15-min（用于 cap）
        da_prices_96: np.ndarray,            # (96,) DA 节点价 per 15-min
        rt_prices_96: np.ndarray,            # (96,) RT 节点价 per 15-min
        da_prices_unified_96: np.ndarray | None = None,
    ) -> EvalResult:
        """
        按甘肃合规规则结算全天 bid（一条曲线，每 15 分钟用该曲线 + forecast cap 计算中标）。
        """
        assert actual_power_96.shape == (96,)
        cleared_96 = daily_bid.cleared_series(
            da_prices_96=da_prices_96,
            forecast_96=forecast_96,
        )
        return self.settle_hourly(
            cleared_qty=cleared_96,
            actual_power=actual_power_96,
            da_prices_node=da_prices_96,
            rt_prices_node=rt_prices_96,
            da_prices_unified=da_prices_unified_96,
            dt_hours=0.25,                  # 15 min
        )

    def settle_naive_full_clear(
        self,
        forecast_power: np.ndarray,
        actual_power: np.ndarray,
        da_prices_node: np.ndarray,
        rt_prices_node: np.ndarray,
        da_prices_unified: np.ndarray | None = None,
        dt_hours: float = 1.0,
    ) -> EvalResult:
        """
        基线 1 · "报 0 元保中标"：
        cleared = forecast（新能源报 0 元理论上保证全量中标）
        甘肃实际中此场景下新能源按节点 LMP 结算，不加额外罚金。
        """
        return self.settle_hourly(
            cleared_qty=forecast_power,
            actual_power=actual_power,
            da_prices_node=da_prices_node,
            rt_prices_node=rt_prices_node,
            da_prices_unified=da_prices_unified,
            dt_hours=dt_hours,
        )

    def compare(self, logan: EvalResult, naive: EvalResult) -> dict:
        abs_gain = logan.total_revenue - naive.total_revenue
        pct = abs_gain / max(abs(naive.total_revenue), 1.0) * 100
        return {
            "logan_revenue": logan.total_revenue,
            "naive_revenue": naive.total_revenue,
            "absolute_gain": abs_gain,
            "percent_gain": pct,
        }
