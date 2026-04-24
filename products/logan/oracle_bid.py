"""
Logan · Oracle Bid (Upper Bound)
==================================

"上帝视角" bid 生成器：知道明天 actual DA / RT 价，求合规下的最大 revenue。

推导（第一性原理）：
    单小时 revenue = Q_cleared × DA + (Q_actual - Q_cleared) × RT
                  = Q_cleared × (DA - RT) + Q_actual × RT

    斜率 = DA - RT：
        DA > RT → Q_cleared 越大越好 → max (forecast)
        DA ≤ RT → Q_cleared 越小越好 → min (compliance: 10% capacity)

合规约束下：
    Q_cleared ∈ [min_step, forecast]  (若 forecast ≥ min_step)
    Q_cleared = 0  (若 forecast < min_step，退出本时段)

输出两个版本：
    - **Oracle_compliance**：受"每段 ≥ 10% capacity"限制，DA<RT 时至少 min_step 中标
    - **Oracle_unconstrained**：理想化，DA<RT 时 cleared = 0（不合规但作参考）
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class OracleResult:
    revenue: float
    da_revenue: float
    rt_revenue: float
    cleared_mwh: float
    actual_mwh: float
    clear_ratio: float


def compute_oracle_revenue(
    actual_power_hr: np.ndarray,
    actual_da_hr: np.ndarray,
    actual_rt_hr: np.ndarray,
    forecast_hr: np.ndarray,
    capacity_mw: float,
    min_step_fraction: float = 0.10,
    compliance_constrained: bool = True,
    dt_hours: float = 1.0,
) -> OracleResult:
    """
    Oracle (perfect foresight) revenue given actual DA / RT.

    Args:
        actual_power_hr: 实际出力 (H,)
        actual_da_hr: 真实 DA 出清价 (H,)
        actual_rt_hr: 真实 RT 价 (H,)
        forecast_hr: 功率预测上限 (H,)（不能超报）
        capacity_mw: 装机容量
        min_step_fraction: 最小段占装机比例（默认 10%）
        compliance_constrained: True=合规版（DA<RT 时 cleared=min_step），
                                False=理想版（DA<RT 时 cleared=0）
    """
    H = len(actual_power_hr)
    min_step = min_step_fraction * capacity_mw

    cleared = np.zeros(H)
    for t in range(H):
        fc = float(forecast_hr[t])
        if fc < min_step and compliance_constrained:
            # 无法合规报价（段数约束），视为退出该时段市场
            cleared[t] = 0.0
            continue

        if actual_da_hr[t] > actual_rt_hr[t]:
            # DA 更高 → 全量中标最优
            cleared[t] = min(fc, capacity_mw)
        else:
            # RT 更高 → 最小中标（或 0）
            if compliance_constrained:
                cleared[t] = min(min_step, fc)  # 最少 min_step，但不超过 forecast
            else:
                cleared[t] = 0.0

    da_rev = float((cleared * actual_da_hr * dt_hours).sum())
    rt_rev = float(((actual_power_hr - cleared) * actual_rt_hr * dt_hours).sum())
    revenue = da_rev + rt_rev

    return OracleResult(
        revenue=revenue,
        da_revenue=da_rev,
        rt_revenue=rt_rev,
        cleared_mwh=float((cleared * dt_hours).sum()),
        actual_mwh=float((actual_power_hr * dt_hours).sum()),
        clear_ratio=float(cleared.sum()) / max(float(actual_power_hr.sum()), 1e-9),
    )


def compute_oracle_revenue_choice(
    actual_power_hr: np.ndarray,
    actual_da_hr: np.ndarray,
    actual_rt_hr: np.ndarray,
    forecast_hr: np.ndarray,
    capacity_mw: float,
    min_step_fraction: float = 0.10,
    dt_hours: float = 1.0,
) -> dict:
    """
    返回 compliance + unconstrained 两个版本的 Oracle revenue。
    """
    comp = compute_oracle_revenue(
        actual_power_hr, actual_da_hr, actual_rt_hr, forecast_hr,
        capacity_mw, min_step_fraction, compliance_constrained=True, dt_hours=dt_hours,
    )
    uncon = compute_oracle_revenue(
        actual_power_hr, actual_da_hr, actual_rt_hr, forecast_hr,
        capacity_mw, min_step_fraction, compliance_constrained=False, dt_hours=dt_hours,
    )
    return {
        "compliance": comp,
        "unconstrained": uncon,
        "compliance_tax": uncon.revenue - comp.revenue,  # 合规成本
    }
