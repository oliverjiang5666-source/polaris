"""
MLT 约束与分配器 (Modification #2)

问题：真实客户有月度/年度 MLT 合同。部分电量已锁定，不能完全自由优化。

建模：
- 月度 MLT 总量 Q_month（MWh）
- 分摊到 30 天 → 每日 Q_day MWh
- 日内必须在指定时段（peak/flat/custom）交付 Q_day
- 剩余功率才能做现货套利

两层优化：
- 外层：月度 MLT 量 → 按各日预期价差分配（非均匀分摊，好日子多放）
- 内层：给定当天 MLT 必交付，优化剩余容量

本文件提供：
1. MLTContract: 合同描述
2. MLTConfig: 运行配置
3. allocate_monthly_to_daily: 月度→日度分配
4. build_mlt_daily_profile: 日内 96 点 MLT 分布
5. solve_day_with_mlt: 带 MLT 约束的日内 LP
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import linprog
from config import BatteryConfig


# ============================================================
# 合同描述
# ============================================================


@dataclass
class MLTContract:
    """一份 MLT 合同"""
    monthly_volume_mwh: float = 0.0       # 月度合同总量
    contract_price: float = 400.0         # 合同价（元/MWh）
    delivery_profile: str = "peak"         # "peak" / "flat" / "offpeak" / "custom"
    custom_daily_profile: np.ndarray | None = None  # 96 点自定义分布

    def __post_init__(self):
        assert self.delivery_profile in ["peak", "flat", "offpeak", "custom"]
        if self.delivery_profile == "custom":
            assert self.custom_daily_profile is not None


@dataclass
class MLTConfig:
    """MLT 整体配置"""
    contracts: list = field(default_factory=list)  # 多份合同

    @property
    def total_monthly_volume(self) -> float:
        return sum(c.monthly_volume_mwh for c in self.contracts)


# ============================================================
# 分配逻辑
# ============================================================


def get_delivery_profile_96(profile_name: str, daily_mwh: float) -> np.ndarray:
    """
    生成 96 点的 MLT 日内交付分布（MWh/step）

    返回：长度 96 的数组，和为 daily_mwh。正数=放电必交付。
    """
    if profile_name == "peak":
        # 18:00-22:00 峰段放电（idx 72-88）
        profile = np.zeros(96)
        profile[72:88] = 1.0
    elif profile_name == "flat":
        # 白天（8:00-20:00）均匀放电
        profile = np.zeros(96)
        profile[32:80] = 1.0
    elif profile_name == "offpeak":
        # 低谷充电（0:00-7:00 充，12:00-14:00 补充）
        # 注：MLT 通常是购售合同，负数表示买电
        profile = np.zeros(96)
        profile[0:28] = -1.0 / 2  # 夜间充
        profile[48:56] = -1.0 / 2  # 中午
    else:
        raise ValueError(f"unknown profile: {profile_name}")

    profile = profile / (np.abs(profile).sum() + 1e-8) * daily_mwh
    return profile


def allocate_monthly_to_daily(
    monthly_mwh: float,
    n_days: int,
    daily_price_estimates: np.ndarray | None = None,
    equal_distribution: bool = True,
) -> np.ndarray:
    """
    月度总量分配到各日。

    Args:
        monthly_mwh: 月度总量
        n_days: 月天数
        daily_price_estimates: 每日价差估计（可选）。用于非均匀分配。
        equal_distribution: True=均匀分摊

    Returns:
        每日配额数组（长度 n_days）
    """
    if equal_distribution or daily_price_estimates is None:
        return np.ones(n_days) * monthly_mwh / n_days

    # 非均匀：价差大的日子多分配
    weights = np.maximum(daily_price_estimates, 1.0)  # 避免 0
    weights = weights / weights.sum()
    return weights * monthly_mwh


def build_mlt_daily_profile(
    mlt_config: MLTConfig,
    day_idx_in_month: int,
    n_days_in_month: int,
    daily_price_estimates: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    给定 MLT 配置 + 今天是月度第几天，返回今天的 96 点 MLT 交付曲线 + 加权合同价。

    Returns:
        (96 点 MWh，加权合同价)
    """
    total_profile = np.zeros(96)
    total_mwh = 0.0
    weighted_price_sum = 0.0

    for contract in mlt_config.contracts:
        daily_allocation = allocate_monthly_to_daily(
            contract.monthly_volume_mwh,
            n_days_in_month,
            daily_price_estimates,
            equal_distribution=True,
        )
        today_mwh = daily_allocation[day_idx_in_month]
        profile_96 = get_delivery_profile_96(contract.delivery_profile, today_mwh)
        total_profile += profile_96
        total_mwh += today_mwh
        weighted_price_sum += contract.contract_price * today_mwh

    weighted_price = weighted_price_sum / (total_mwh + 1e-8)
    return total_profile, weighted_price


# ============================================================
# 带 MLT 约束的 LP
# ============================================================


def solve_day_with_mlt(
    dam_prices: np.ndarray,
    battery: BatteryConfig,
    mlt_profile_96: np.ndarray,
    mlt_contract_price: float,
    init_soc: float = 0.5,
    deg: float = 2.0,
) -> dict:
    """
    带 MLT 约束的单日 LP：

    决策变量：
        free_discharge[t], free_charge[t]: 现货上的自由放/充电
        (MLT 部分已通过 mlt_profile_96 给定)

    物理约束：
        |mlt_profile[t] + free_discharge[t] - free_charge[t]| <= P_max
        SoC 演化考虑 MLT 交付 + 自由操作
        SoC ∈ [min, max]

    目标：
        max Σ [mlt_profile[t] × mlt_price + (free_discharge[t] - free_charge[t]) × dam_prices[t]]
             - degradation × Σ(|total_energy[t]|)

    其中 mlt_profile[t] > 0 = 必放电给 MLT；mlt_profile[t] < 0 = 必买电给 MLT

    Returns:
        dict with net_power, revenue, mlt_revenue, spot_revenue, soc
    """
    n = len(dam_prices)
    dt = battery.interval_hours
    cap_mw = battery.capacity_mw
    cap_mwh = battery.capacity_mwh
    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency

    # 决策变量：x = [free_charge_0..n-1, free_discharge_0..n-1]，共 2n 个
    # 目标最小化： -Σ free_discharge × dam_p × dt  + Σ free_charge × dam_p × dt + deg成本

    c_obj = np.zeros(2 * n)
    for t in range(n):
        c_obj[t] = dam_prices[t] * dt + deg * dt  # free_charge cost
        c_obj[n + t] = -dam_prices[t] * dt + deg * dt  # free_discharge (negative because maximize)

    # 约束：功率总量 |mlt[t] + free_dis[t] - free_char[t]| <= cap_mw
    # 即：
    #   mlt[t] + free_dis[t] - free_char[t] <= cap_mw     (上界)
    #   -(mlt[t] + free_dis[t] - free_char[t]) <= cap_mw  (下界)
    # 二元变换：
    #   free_dis[t] - free_char[t] <= cap_mw - mlt[t]
    #   -free_dis[t] + free_char[t] <= cap_mw + mlt[t]

    A_power_upper = np.zeros((n, 2 * n))
    b_power_upper = np.zeros(n)
    A_power_lower = np.zeros((n, 2 * n))
    b_power_lower = np.zeros(n)

    for t in range(n):
        A_power_upper[t, t] = -1  # -free_charge
        A_power_upper[t, n + t] = 1  # +free_discharge
        b_power_upper[t] = cap_mw - mlt_profile_96[t]

        A_power_lower[t, t] = 1
        A_power_lower[t, n + t] = -1
        b_power_lower[t] = cap_mw + mlt_profile_96[t]

    # SoC 约束：SoC[t] 由充放电累积决定
    # 每一步 SoC 变化量（考虑 MLT + 自由）：
    # mlt > 0 (放) → SoC 减 mlt × dt / cap_mwh / eta_d
    # mlt < 0 (充) → SoC 加 -mlt × dt × eta_c / cap_mwh
    # free_charge > 0 → SoC 加 free_char × dt × eta_c / cap_mwh
    # free_discharge > 0 → SoC 减 free_dis × dt / cap_mwh / eta_d

    charge_coeff = dt * eta_c / cap_mwh
    discharge_coeff = dt / (cap_mwh * eta_d)

    # MLT 对每步 SoC 的预定影响
    mlt_soc_effect = np.zeros(n)
    for t in range(n):
        if mlt_profile_96[t] > 0:
            mlt_soc_effect[t] = -mlt_profile_96[t] * dt / (cap_mwh * eta_d)
        elif mlt_profile_96[t] < 0:
            mlt_soc_effect[t] = -mlt_profile_96[t] * dt * eta_c / cap_mwh

    mlt_cumsum = np.cumsum(mlt_soc_effect)

    # 每个 t 的 SoC 上下界约束
    A_soc_upper = np.zeros((n, 2 * n))
    b_soc_upper = np.zeros(n)
    A_soc_lower = np.zeros((n, 2 * n))
    b_soc_lower = np.zeros(n)

    for t in range(1, n + 1):
        # SoC[t] = init_soc + mlt_cumsum[t-1] + Σ_{k<t} (free_char[k]*charge_coeff - free_dis[k]*discharge_coeff)
        row_u = t - 1
        for k in range(t):
            A_soc_upper[row_u, k] = charge_coeff
            A_soc_upper[row_u, n + k] = -discharge_coeff
        b_soc_upper[row_u] = battery.max_soc - init_soc - mlt_cumsum[t - 1]

        for k in range(t):
            A_soc_lower[row_u, k] = -charge_coeff
            A_soc_lower[row_u, n + k] = discharge_coeff
        b_soc_lower[row_u] = init_soc + mlt_cumsum[t - 1] - battery.min_soc

    # 组合
    A_ub = np.vstack([A_power_upper, A_power_lower, A_soc_upper, A_soc_lower])
    b_ub = np.concatenate([b_power_upper, b_power_lower, b_soc_upper, b_soc_lower])

    # 变量边界：0 <= free_charge, free_discharge <= cap_mw
    bounds = [(0, cap_mw)] * (2 * n)

    # 求解
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                     method="highs", options={"time_limit": 10.0})

    if not result.success:
        # MLT 约束可能不可行：返回纯 MLT 结果
        return {
            "free_charge": np.zeros(n),
            "free_discharge": np.zeros(n),
            "net_power": mlt_profile_96 / dt,  # 只有 MLT
            "soc": np.full(n + 1, init_soc),
            "mlt_revenue": float(np.sum(mlt_profile_96) * mlt_contract_price),
            "spot_revenue": 0.0,
            "revenue": float(np.sum(mlt_profile_96) * mlt_contract_price),
            "feasible": False,
        }

    free_charge = result.x[:n]
    free_discharge = result.x[n:]
    net_power = (mlt_profile_96 / dt) + free_discharge - free_charge

    # SoC 轨迹
    soc = np.zeros(n + 1)
    soc[0] = init_soc
    for t in range(n):
        delta = mlt_soc_effect[t] + free_charge[t] * charge_coeff - free_discharge[t] * discharge_coeff
        soc[t + 1] = np.clip(soc[t] + delta, battery.min_soc, battery.max_soc)

    mlt_revenue = float(np.sum(mlt_profile_96) * mlt_contract_price)
    spot_revenue = float(np.sum((free_discharge - free_charge) * dam_prices) * dt)
    degradation = float(np.sum(np.abs((mlt_profile_96 / dt) + free_discharge - free_charge)) * deg * dt)

    return {
        "free_charge": free_charge,
        "free_discharge": free_discharge,
        "net_power": net_power,
        "soc": soc,
        "mlt_revenue": mlt_revenue,
        "spot_revenue": spot_revenue,
        "degradation": degradation,
        "revenue": mlt_revenue + spot_revenue - degradation,
        "feasible": True,
    }


# ============================================================
# 快速测试
# ============================================================


if __name__ == "__main__":
    from loguru import logger
    np.random.seed(42)

    battery = BatteryConfig()

    # 模拟一天电价（山东式）
    hours = np.arange(96) / 4
    dam_prices = 320 - 80 * np.cos(2 * np.pi * hours / 24) + \
                 40 * np.sin(4 * np.pi * hours / 24) + \
                 np.random.normal(0, 20, 96)

    # 测试 1: MLT = 0（应该等于纯 LP）
    from oracle.lp_oracle import solve_day
    baseline = solve_day(dam_prices, battery)
    logger.info(f"Test 1 (无MLT): baseline revenue = ¥{baseline['revenue']:,.0f}")

    mlt_profile_zero = np.zeros(96)
    with_zero_mlt = solve_day_with_mlt(dam_prices, battery, mlt_profile_zero, 400.0)
    logger.info(f"Test 1 (MLT=0): with_mlt revenue = ¥{with_zero_mlt['revenue']:,.0f}, "
                f"feasible={with_zero_mlt['feasible']}")

    diff_pct = abs(baseline['revenue'] - with_zero_mlt['revenue']) / abs(baseline['revenue']) * 100
    logger.info(f"  差异 {diff_pct:.1f}% {'✅' if diff_pct < 2 else '❌'}")

    # 测试 2: MLT = 200 MWh/day peak delivery
    mlt_200 = get_delivery_profile_96("peak", 200.0)
    logger.info(f"\nTest 2 (MLT=200 MWh peak): MLT total energy = {mlt_200.sum():.1f} MWh")
    with_200_mlt = solve_day_with_mlt(dam_prices, battery, mlt_200, 400.0)
    logger.info(f"  feasible = {with_200_mlt['feasible']}")
    logger.info(f"  MLT 收入 = ¥{with_200_mlt['mlt_revenue']:,.0f}")
    logger.info(f"  Spot 收入 = ¥{with_200_mlt['spot_revenue']:,.0f}")
    logger.info(f"  总收入 = ¥{with_200_mlt['revenue']:,.0f}")

    # 测试 3: MLT 很大（接近满）
    mlt_400 = get_delivery_profile_96("peak", 400.0)  # 很接近电池容量
    with_400_mlt = solve_day_with_mlt(dam_prices, battery, mlt_400, 400.0)
    logger.info(f"\nTest 3 (MLT=400 MWh): feasible = {with_400_mlt['feasible']}")
    logger.info(f"  总收入 = ¥{with_400_mlt['revenue']:,.0f}")
