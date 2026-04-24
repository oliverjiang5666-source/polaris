"""
Polaris 山东 · Settlement Evaluator
====================================

实现 §14.3.4 / §14.3.5 / §14.5.5-7 / §14.6.3-5 的 Two-Settlement + CfD 结算公式。

发电侧（储能放电收入）:
    R_发电_电能量 = R_实时 + R_日前 + R_中长期
    R_实时  = Σ Q^actual_放_t × P^实时节点_t               (§14.5.5)
    R_日前  = Σ Q^DA_放_t × (P^日前节点_t − P^实时统一_t)  (§14.5.6)
    R_中长期 = Σ Q^MLT_t × (P^MLT − P^参考点现货_t)         (§14.5.7)

用户侧（储能充电支出）:
    C_用户_电能量 = C_实时 + C_日前 + C_中长期
    C_实时  = Σ Q^actual_充_t × P^实时节点_t              (§14.6.3 独立储能特例)
    C_日前  = Σ Q^DA_充_t × (P^日前节点_t − P^实时统一_t) (§14.6.4)
    C_中长期 = Σ Q^MLT_t × (P^MLT − P^参考点现货_t)        (§14.6.5)

净储能收入 = R_发电 − C_用户

⚠️  LMP proxy:
    现阶段真实节点 LMP 数据未接入，evaluator 接口传
    (da_lmp_96, rt_lmp_96, rt_unified_96, mlt_reference_96)
    实际调用时可传 rt_price 作所有四个的 proxy，等 LMP 数据到位后替换。

可选组件:
    - 容量补偿（月度，由 capacity_compensation.py 单独算）
    - AGC 收入（按调频里程 × 性能 × 出清价，§14.8.3）
    - 调度强制调用补偿（§14.10.3，本模块只结算，不触发）
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from products.polaris_shandong.bid_curve import StorageBidCurve


# ============================================================
# Config
# ============================================================

@dataclass
class SettlementConfig:
    """山东结算配置"""
    dt_hours: float = 0.25                  # 15 分钟步长

    # 中长期合约（可选，无则全 0）
    mlt_contract_mw_96: np.ndarray | None = None    # 96 步中长期合约量 (正=储能作为卖方/放电)
    mlt_contract_price: float = 0.0                 # 合约价 元/MWh

    # 中长期结算参考点选择（§4.3.7）
    #   "unified":  实时市场用户侧统一结算点（集中竞价/滚动撮合/挂牌，默认）
    #   "node":     物理节点（双边协商可选）
    mlt_reference_point: str = "unified"

    # AGC 调频参数（§14.8.3）
    agc_enabled: bool = False
    agc_clearing_price_yuan_per_mw: float = 6.0     # 出清价
    agc_performance_coeff: float = 0.95             # 性能系数 K_pd
    agc_mileage_per_mw_per_h: float = 3.0           # 期望调频里程/MW/h

    # 调度强制调用补偿（§14.10.3）
    forced_dispatch_loss_rate: float = 0.10         # 充放电损耗率 R_核定损耗
    forced_dispatch_compensation_enabled: bool = False

    # 价格限值（clip 用）
    clearing_price_lower: float | None = None
    clearing_price_upper: float | None = None


# ============================================================
# Result
# ============================================================

@dataclass
class EvalResult:
    # 电能量电费分解（发电侧）
    gen_realtime_rev: float = 0.0          # R_实时
    gen_da_cfd_rev: float = 0.0            # R_日前 CfD
    gen_mlt_cfd_rev: float = 0.0           # R_中长期 CfD
    gen_total_energy_rev: float = 0.0      # 发电侧电能量合计

    # 电能量电费分解（用户侧）
    user_realtime_cost: float = 0.0
    user_da_cfd_cost: float = 0.0
    user_mlt_cfd_cost: float = 0.0
    user_total_energy_cost: float = 0.0

    # 辅助
    agc_revenue: float = 0.0
    forced_dispatch_compensation: float = 0.0

    # 净
    net_energy_revenue: float = 0.0        # 发电 − 用户（电能量部分）
    total_revenue: float = 0.0             # 电能量 + AGC + 强制调用补偿（容量补偿另算）

    # 统计
    n_steps: int = 0
    discharge_mwh: float = 0.0
    charge_mwh: float = 0.0

    # 诊断
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"总收入:       ¥{self.total_revenue:,.0f}",
            f"  净电能量:   ¥{self.net_energy_revenue:+,.0f}",
            f"    发电电能量: ¥{self.gen_total_energy_rev:+,.0f}",
            f"      实时:    ¥{self.gen_realtime_rev:+,.0f}",
            f"      日前:    ¥{self.gen_da_cfd_rev:+,.0f}",
            f"      中长期:  ¥{self.gen_mlt_cfd_rev:+,.0f}",
            f"    用户电能量: ¥{self.user_total_energy_cost:+,.0f}",
            f"      实时:    ¥{self.user_realtime_cost:+,.0f}",
            f"      日前:    ¥{self.user_da_cfd_cost:+,.0f}",
            f"      中长期:  ¥{self.user_mlt_cfd_cost:+,.0f}",
            f"  AGC:        ¥{self.agc_revenue:+,.0f}",
            f"  强制调用:   ¥{self.forced_dispatch_compensation:+,.0f}",
            f"  放电:       {self.discharge_mwh:.1f} MWh",
            f"  充电:       {self.charge_mwh:.1f} MWh",
        ]
        return "\n".join(lines)


# ============================================================
# Evaluator
# ============================================================

class ShandongEvaluator:
    """
    山东独立储能结算引擎（Two-Settlement + CfD + AGC + 强制调用补偿）
    """

    def __init__(self, config: SettlementConfig | None = None):
        self.config = config or SettlementConfig()

    def _clip(self, prices: np.ndarray) -> np.ndarray:
        cfg = self.config
        prices = np.asarray(prices, dtype=np.float64).copy()
        if cfg.clearing_price_lower is not None:
            prices = np.maximum(prices, cfg.clearing_price_lower)
        if cfg.clearing_price_upper is not None:
            prices = np.minimum(prices, cfg.clearing_price_upper)
        return prices

    def settle(
        self,
        actual_power_96: np.ndarray,           # (96,) 实际净功率, 正=放电, 负=充电
        da_cleared_96: np.ndarray,             # (96,) 日前出清净功率 (同上符号)
        #
        da_lmp_gen_96: np.ndarray,             # (96,) 日前发电侧节点电价
        rt_lmp_gen_96: np.ndarray,             # (96,) 实时发电侧节点电价
        da_lmp_user_96: np.ndarray,            # (96,) 日前用户侧节点电价 (独立储能)
        rt_lmp_user_96: np.ndarray,            # (96,) 实时用户侧节点电价
        rt_unified_96: np.ndarray,             # (96,) 实时市场统一结算点电价（日前 CfD 参考点）
        #
        mlt_reference_96: np.ndarray | None = None,  # (96,) 中长期结算参考点现货价
        # AGC
        actual_agc_up_mw_96: np.ndarray | None = None,     # (96,) 实际中标 AGC 上调容量
        actual_agc_down_mw_96: np.ndarray | None = None,
    ) -> EvalResult:
        """
        按山东规则结算 96 个时段。
        """
        cfg = self.config
        dt = cfg.dt_hours

        assert len(actual_power_96) == 96, "必须 96 点"
        da_lmp_gen = self._clip(da_lmp_gen_96)
        rt_lmp_gen = self._clip(rt_lmp_gen_96)
        da_lmp_user = self._clip(da_lmp_user_96)
        rt_lmp_user = self._clip(rt_lmp_user_96)
        rt_unified = self._clip(rt_unified_96)

        # 拆分放电和充电（正=放, 负=充）
        actual_discharge = np.maximum(actual_power_96, 0.0)        # (96,)
        actual_charge = np.maximum(-actual_power_96, 0.0)          # (96,) magnitude

        da_discharge = np.maximum(da_cleared_96, 0.0)
        da_charge = np.maximum(-da_cleared_96, 0.0)

        # ============================================================
        # 发电侧（放电）结算 §14.5.5-7
        # ============================================================
        # R_实时 = Σ Q^actual_放_t × P^实时节点_t × dt
        gen_rt = float((actual_discharge * rt_lmp_gen * dt).sum())

        # R_日前 = Σ Q^DA_放_t × (P^日前节点_t − P^实时统一_t) × dt
        gen_da = float((da_discharge * (da_lmp_gen - rt_unified) * dt).sum())

        # R_中长期 = Σ Q^MLT × (P^MLT − P^参考点现货) × dt
        gen_mlt = 0.0
        if cfg.mlt_contract_mw_96 is not None:
            mlt_q = np.asarray(cfg.mlt_contract_mw_96, dtype=np.float64)
            # MLT 参考点：默认用 rt_unified，双边协商可传 mlt_reference_96
            ref = mlt_reference_96 if mlt_reference_96 is not None else rt_unified
            ref = self._clip(ref)
            gen_mlt = float((mlt_q * (cfg.mlt_contract_price - ref) * dt).sum())

        gen_total = gen_rt + gen_da + gen_mlt

        # ============================================================
        # 用户侧（充电）结算 §14.6.3-5
        # ============================================================
        # C_实时 = Σ Q^actual_充_t × P^实时节点_t × dt   (独立储能特例：用节点价)
        user_rt = float((actual_charge * rt_lmp_user * dt).sum())

        # C_日前 = Σ Q^DA_充_t × (P^日前节点_t − P^实时统一_t) × dt
        user_da = float((da_charge * (da_lmp_user - rt_unified) * dt).sum())

        # 中长期：用户侧一般 MLT = 0（储能不作为用户买长期合约），留接口
        user_mlt = 0.0

        user_total = user_rt + user_da + user_mlt

        # ============================================================
        # AGC 调频 §14.8.3
        # ============================================================
        agc_rev = 0.0
        if cfg.agc_enabled and (actual_agc_up_mw_96 is not None or actual_agc_down_mw_96 is not None):
            up = np.zeros(96) if actual_agc_up_mw_96 is None else np.asarray(actual_agc_up_mw_96)
            dn = np.zeros(96) if actual_agc_down_mw_96 is None else np.asarray(actual_agc_down_mw_96)

            # C_AGC = D × K_pd × Y_AGC
            # D (小时调频里程) = AGC容量 × 期望里程/MW/h × dt
            mileage = (up + dn) * cfg.agc_mileage_per_mw_per_h * dt  # MW·mileage/step
            agc_rev = float((mileage * cfg.agc_performance_coeff * cfg.agc_clearing_price_yuan_per_mw).sum())

        # ============================================================
        # 汇总
        # ============================================================
        net_energy = gen_total - user_total
        total = net_energy + agc_rev  # 容量补偿另月度结算

        notes = []
        if cfg.mlt_contract_mw_96 is None:
            notes.append("MLT = 0 (无中长期合约)")
        if not cfg.agc_enabled:
            notes.append("AGC 未启用")

        return EvalResult(
            gen_realtime_rev=gen_rt,
            gen_da_cfd_rev=gen_da,
            gen_mlt_cfd_rev=gen_mlt,
            gen_total_energy_rev=gen_total,
            user_realtime_cost=user_rt,
            user_da_cfd_cost=user_da,
            user_mlt_cfd_cost=user_mlt,
            user_total_energy_cost=user_total,
            agc_revenue=agc_rev,
            forced_dispatch_compensation=0.0,
            net_energy_revenue=net_energy,
            total_revenue=total,
            n_steps=96,
            discharge_mwh=float(actual_discharge.sum() * dt),
            charge_mwh=float(actual_charge.sum() * dt),
            notes=notes,
        )

    # ============================================================
    # 便利方法：从 bid curve + 实际数据结算
    # ============================================================

    def settle_from_bid_curve(
        self,
        bid: StorageBidCurve,
        actual_power_96: np.ndarray,
        da_lmp_gen_96: np.ndarray,
        rt_lmp_gen_96: np.ndarray,
        da_lmp_user_96: np.ndarray | None = None,
        rt_lmp_user_96: np.ndarray | None = None,
        rt_unified_96: np.ndarray | None = None,
        da_charge_upper_96: np.ndarray | None = None,
        da_discharge_upper_96: np.ndarray | None = None,
        **kwargs,
    ) -> EvalResult:
        """
        方便路径：给定 bid curve 和实际功率，自动推日前出清量。

        LMP proxy 默认用 rt_lmp_gen_96（如未显式传用户侧/统一价）。
        """
        # Defaults (LMP proxy)
        if da_lmp_user_96 is None:
            da_lmp_user_96 = da_lmp_gen_96
        if rt_lmp_user_96 is None:
            rt_lmp_user_96 = rt_lmp_gen_96
        if rt_unified_96 is None:
            rt_unified_96 = rt_lmp_gen_96    # ⚠️ TODO 真实统一结算点价从省聚合数据来

        # 日前出清 = bid curve 在日前 LMP 下的 cleared power
        da_cleared_96 = bid.cleared_series_96(
            lmp_96=da_lmp_gen_96,
            charge_upper_96=da_charge_upper_96,
            discharge_upper_96=da_discharge_upper_96,
        )

        return self.settle(
            actual_power_96=actual_power_96,
            da_cleared_96=da_cleared_96,
            da_lmp_gen_96=da_lmp_gen_96,
            rt_lmp_gen_96=rt_lmp_gen_96,
            da_lmp_user_96=da_lmp_user_96,
            rt_lmp_user_96=rt_lmp_user_96,
            rt_unified_96=rt_unified_96,
            **kwargs,
        )
