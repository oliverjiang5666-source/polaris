"""
Polaris 山东 · 容量补偿 (§3.4.3, §3.4.6 第四项)
==================================================

独立新型储能日可用容量公式（§3.4.6 第四项 1）:

    daily_available_capacity = rated_discharge_power × K × H / 24

    K = 日可用系数 = 电站当日运行及备用状态下的小时数 / 24
                    （计划检修、临故修不计入）
    H = 日可用等效小时数 = 核定放电功率下的最大连续放电小时数

月度发电侧容量补偿费分配（§3.4.4）:

    发电侧主体月度容量补偿费
        = 全网发电侧容量补偿费
          × 主体月度市场化可用容量
          / 全网月度市场化可用容量

其中:
    全网发电侧市场化容量补偿费用
        = 市场化容量补偿电价
          × (省内发电侧市场化电量 − 新能源机制电量)

    全网发电侧月度市场化可用容量
        = Σ 当月发电侧主体日市场化可用容量 / 当月总天数

⚠️  核定充放电功率及充放电小时数由电力调度机构测试认定（§3.4.6 第四项 1 末句）。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from loguru import logger


@dataclass
class CapacityCompensationConfig:
    # 核定参数（由电力调度机构测试认定）
    certified_discharge_power_mw: float = 200.0
    certified_duration_hours: float = 2.0          # H, 最大连续放电小时数

    # 省发改委核定电价（月度）
    capacity_price_yuan_per_kw_month: float = 0.0  # 待填省政策


def daily_available_capacity_mw(
    rated_discharge_power_mw: float,
    available_hours_on_day: float,                  # 电站当日运行及备用小时数
    certified_duration_hours: float,                # H
) -> float:
    """
    §3.4.6 第四项 1 公式:
        日市场化可用容量 = P_放电_核定 × K × H / 24

    Returns: MW（日均等效可用容量）
    """
    K = np.clip(available_hours_on_day, 0.0, 24.0) / 24.0
    H = certified_duration_hours
    return rated_discharge_power_mw * K * H / 24.0


def monthly_available_capacity_mw(
    daily_available_mw_list: list[float] | np.ndarray,
) -> float:
    """
    §3.4.5 公式:
        月度市场化可用容量 = Σ 日市场化可用容量 / 当月总天数
    """
    arr = np.asarray(daily_available_mw_list, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return float(arr.sum() / len(arr))


def monthly_capacity_compensation_fee(
    entity_monthly_available_mw: float,             # 本主体月度可用容量
    grid_total_monthly_available_mw: float,         # 全网发电侧月度可用容量
    grid_total_capacity_fee_yuan: float,            # 全网发电侧容量补偿费用
) -> float:
    """
    §3.4.4 分配公式:
        本主体月度容量补偿费用
            = 全网费用 × 本主体月度可用容量 / 全网月度可用容量
    """
    if grid_total_monthly_available_mw <= 1e-9:
        return 0.0
    return grid_total_capacity_fee_yuan * entity_monthly_available_mw / grid_total_monthly_available_mw


def estimate_monthly_fee_from_standalone_price(
    rated_discharge_power_mw: float,
    certified_duration_hours: float,
    capacity_price_yuan_per_kw_month: float,
    typical_K: float = 1.0,
) -> float:
    """
    单机粗算（不依赖全网数据）:
        月度容量补偿费 ≈ 月度可用容量 (MW) × 容量补偿电价 × 月份天数等效
                     ≈ P_额 × K × H / 24  (MW)
                       × price (元/kW/月)
                       × 1000 kW/MW

    这是一个**参考估算**，真实分配按 §3.4.4 全网法。
    """
    daily_mw = daily_available_capacity_mw(
        rated_discharge_power_mw=rated_discharge_power_mw,
        available_hours_on_day=24.0 * typical_K,
        certified_duration_hours=certified_duration_hours,
    )
    # 单位换算: MW × 元/kW/月 = MW × 1000 kW × 元/kW/月 = 1000 × MW × 元/月
    monthly_fee_yuan = daily_mw * 1000.0 * capacity_price_yuan_per_kw_month
    return float(monthly_fee_yuan)


if __name__ == "__main__":
    # 典型案例: 200 MW / 400 MWh 储能
    # H = 400/200 = 2 小时
    # K = 1.0 (全月在线)
    # 假设省容量补偿电价 = 100 元/kW/月 (示意值，真实需省发改委核定)

    daily = daily_available_capacity_mw(
        rated_discharge_power_mw=200.0,
        available_hours_on_day=24.0,
        certified_duration_hours=2.0,
    )
    logger.info(f"200MW/400MWh 日可用容量: {daily:.2f} MW/day")
    # = 200 × 1 × 2 / 24 ≈ 16.67 MW/day

    # 若容量电价 = 100 元/kW/月（示意）:
    est_monthly = estimate_monthly_fee_from_standalone_price(
        rated_discharge_power_mw=200.0,
        certified_duration_hours=2.0,
        capacity_price_yuan_per_kw_month=100.0,
        typical_K=1.0,
    )
    logger.info(f"粗估月度容量补偿费: ¥{est_monthly:,.0f}")
    # = 16.67 × 1000 × 100 = ¥1,666,667 /月 （示意，真实值需核定）
