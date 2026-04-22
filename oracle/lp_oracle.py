"""
LP Oracle — 线性规划求解储能最优调度

给定一天96个时段的真实电价，用LP求解数学最优的充放电方案。
这是理论上限（上帝视角），用于：
1. 作为BC训练的完美老师
2. 作为backtest的上界参考

求解规模：192变量 × ~385约束 → scipy.linprog <1ms/天
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from pathlib import Path
from loguru import logger
from config import BatteryConfig, OracleConfig, ACTIONS


PROCESSED_DIR = Path(__file__).parent.parent / "data" / "china" / "processed"


def solve_day(
    prices: np.ndarray,
    battery: BatteryConfig | None = None,
    init_soc: float = 0.5,
    end_soc_min: float | None = None,
) -> dict:
    """
    求解一天(96步)的最优充放电方案

    Args:
        prices: 长度96的电价数组 (元/MWh)
        battery: 电池参数
        init_soc: 初始SOC (0-1)

    Returns:
        dict with keys:
            charge: np.ndarray[96] 充电功率 (MW, ≥0)
            discharge: np.ndarray[96] 放电功率 (MW, ≥0)
            net_power: np.ndarray[96] 净功率 (MW, 正=放电)
            soc: np.ndarray[97] SOC轨迹
            revenue: float 总收入 (元)
            actions: np.ndarray[96] 量化后的离散动作 (0-4)
    """
    if battery is None:
        battery = BatteryConfig()

    n = len(prices)
    dt = battery.interval_hours  # 0.25h
    cap_mw = battery.capacity_mw  # 200 MW
    cap_mwh = battery.capacity_mwh  # 400 MWh
    eta_c = battery.charge_efficiency  # 0.9487
    eta_d = battery.discharge_efficiency  # 0.9487
    deg = battery.degradation_cost_per_mwh  # 2.0 元/MWh
    soc_min = battery.min_soc  # 0.05
    soc_max = battery.max_soc  # 0.95

    # ================================================================
    # 决策变量: x = [charge_0, ..., charge_95, discharge_0, ..., discharge_95]
    # 共 2n = 192 变量
    # ================================================================

    # 目标函数: minimize c^T x
    # 原始目标是maximize，取负号变minimize
    # revenue = Σ (discharge[t] - charge[t]) × price[t] × dt
    # degradation = Σ (charge[t] + discharge[t]) × deg × dt
    # maximize: revenue - degradation
    # minimize: -revenue + degradation

    c = np.zeros(2 * n)
    for t in range(n):
        # charge[t] 的系数: +price[t]*dt + deg*dt (花钱买电 + 降解)
        c[t] = prices[t] * dt + deg * dt
        # discharge[t] 的系数: -price[t]*dt + deg*dt (卖电收入取负 + 降解)
        c[n + t] = -prices[t] * dt + deg * dt

    # ================================================================
    # 不等式约束: A_ub @ x <= b_ub
    # SOC约束：SOC[t] = init_soc + Σ_{k<t} [charge[k]*dt*eta_c/cap_mwh - discharge[k]*dt/(cap_mwh*eta_d)]
    # 需要 soc_min <= SOC[t] <= soc_max
    # 即:
    #   SOC[t] <= soc_max → 上界约束
    #   -SOC[t] <= -soc_min → 下界约束 (取负)
    # ================================================================

    # SOC上界和下界约束: 对 t=1..n (共2n个约束)
    A_ub = np.zeros((2 * n, 2 * n))
    b_ub = np.zeros(2 * n)

    # 累积系数
    charge_coeff = dt * eta_c / cap_mwh    # 充电对SOC的贡献（正）
    discharge_coeff = dt / (cap_mwh * eta_d)  # 放电对SOC的消耗（正）

    for t in range(1, n + 1):
        # SOC[t] = init_soc + Σ_{k=0}^{t-1} [charge[k]*charge_coeff - discharge[k]*discharge_coeff]

        # 上界: SOC[t] <= soc_max
        # Σ charge[k]*charge_coeff - Σ discharge[k]*discharge_coeff <= soc_max - init_soc
        row_upper = t - 1  # 行索引 0..n-1
        for k in range(t):
            A_ub[row_upper, k] = charge_coeff        # charge[k]
            A_ub[row_upper, n + k] = -discharge_coeff  # discharge[k]
        b_ub[row_upper] = soc_max - init_soc

        # 下界: SOC[t] >= soc_min → -SOC[t] <= -soc_min
        # -Σ charge[k]*charge_coeff + Σ discharge[k]*discharge_coeff <= init_soc - soc_min
        row_lower = n + t - 1  # 行索引 n..2n-1
        for k in range(t):
            A_ub[row_lower, k] = -charge_coeff
            A_ub[row_lower, n + k] = discharge_coeff
        b_ub[row_lower] = init_soc - soc_min

    # ================================================================
    # Optional: end-of-horizon SOC minimum constraint
    # SOC[T] >= end_soc_min
    # → -Σ charge[k]*coeff_c + Σ discharge[k]*coeff_d <= init_soc - end_soc_min
    # ================================================================
    if end_soc_min is not None and end_soc_min > soc_min:
        extra_row = np.zeros((1, 2 * n))
        for k in range(n):
            extra_row[0, k] = -charge_coeff
            extra_row[0, n + k] = discharge_coeff
        extra_b = np.array([init_soc - end_soc_min])
        A_ub = np.vstack([A_ub, extra_row])
        b_ub = np.concatenate([b_ub, extra_b])

    # ================================================================
    # 变量边界: 0 <= charge[t], discharge[t] <= cap_mw
    # ================================================================
    bounds = [(0, cap_mw)] * (2 * n)

    # ================================================================
    # 求解
    # ================================================================
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 10.0},
    )

    if not result.success:
        logger.warning(f"LP solve failed: {result.message}")
        return _empty_result(n)

    # ================================================================
    # 提取结果
    # ================================================================
    charge = result.x[:n]
    discharge = result.x[n:]
    net_power = discharge - charge

    # 计算SOC轨迹
    soc = np.zeros(n + 1)
    soc[0] = init_soc
    for t in range(n):
        soc[t + 1] = soc[t] + charge[t] * charge_coeff - discharge[t] * discharge_coeff
        soc[t + 1] = np.clip(soc[t + 1], soc_min, soc_max)

    # 计算收入
    revenue = float(np.sum(net_power * prices * dt) - np.sum((charge + discharge) * deg * dt))

    # 量化到离散动作
    actions = quantize_to_discrete(net_power, cap_mw)

    return {
        "charge": charge,
        "discharge": discharge,
        "net_power": net_power,
        "soc": soc,
        "revenue": revenue,
        "actions": actions,
    }


def quantize_to_discrete(
    net_power: np.ndarray,
    capacity_mw: float,
    oracle_config: OracleConfig | None = None,
) -> np.ndarray:
    """
    连续功率 → 5个离散动作

    Args:
        net_power: 净功率数组 (正=放电, 负=充电)
        capacity_mw: 额定功率
        oracle_config: 量化阈值配置

    Returns:
        动作数组 (0=wait, 1=slow_charge, 2=fast_charge, 3=slow_discharge, 4=fast_discharge)
    """
    if oracle_config is None:
        oracle_config = OracleConfig()

    ratio = net_power / capacity_mw  # 归一化到 [-1, 1]
    actions = np.zeros(len(ratio), dtype=np.int64)

    wait_th = oracle_config.wait_threshold
    slow_th = oracle_config.slow_threshold

    for i, r in enumerate(ratio):
        if abs(r) < wait_th:
            actions[i] = 0  # wait
        elif r < -slow_th:
            actions[i] = 2  # fast_charge
        elif r < -wait_th:
            actions[i] = 1  # slow_charge
        elif r > slow_th:
            actions[i] = 4  # fast_discharge
        else:
            actions[i] = 3  # slow_discharge

    return actions


def solve_dataset(
    df: pd.DataFrame,
    battery: BatteryConfig | None = None,
    price_col: str = "rt_price",
    init_soc: float = 0.5,
) -> pd.DataFrame:
    """
    对整个数据集按天求解LP Oracle

    Args:
        df: 带有price_col列的DataFrame, index=timestamp
        battery: 电池参数
        price_col: 价格列名
        init_soc: 每天起始SOC

    Returns:
        DataFrame with columns: oracle_action, oracle_revenue, oracle_soc
    """
    if battery is None:
        battery = BatteryConfig()

    prices = df[price_col].values
    n_total = len(prices)
    steps_per_day = 96  # 15min × 96 = 24h

    # 输出列
    oracle_actions = np.zeros(n_total, dtype=np.int64)
    oracle_soc = np.full(n_total, init_soc)
    oracle_net_power = np.zeros(n_total)

    n_days = n_total // steps_per_day
    total_revenue = 0.0
    solved = 0
    failed = 0

    for d in range(n_days):
        start = d * steps_per_day
        end = start + steps_per_day
        day_prices = prices[start:end]

        # 跳过全NaN的天
        if np.isnan(day_prices).all():
            failed += 1
            continue

        # 用前向填充处理NaN价格
        day_prices_clean = pd.Series(day_prices).ffill().bfill().values
        if np.isnan(day_prices_clean).any():
            failed += 1
            continue

        result = solve_day(day_prices_clean, battery, init_soc)
        oracle_actions[start:end] = result["actions"]
        oracle_soc[start:end] = result["soc"][:steps_per_day]
        oracle_net_power[start:end] = result["net_power"]
        total_revenue += result["revenue"]
        solved += 1

    # 处理末尾不完整的天
    remainder = n_total % steps_per_day
    if remainder > 0:
        # 不完整的天用wait填充
        pass

    logger.info(
        f"  Oracle solved: {solved}/{n_days} days, "
        f"failed: {failed}, "
        f"total revenue: {total_revenue:,.0f} 元"
    )

    # 添加到DataFrame
    df_out = df.copy()
    df_out["oracle_action"] = oracle_actions
    df_out["oracle_soc"] = oracle_soc
    df_out["oracle_net_power"] = oracle_net_power

    return df_out, total_revenue


def _empty_result(n: int) -> dict:
    """LP求解失败时的空结果"""
    return {
        "charge": np.zeros(n),
        "discharge": np.zeros(n),
        "net_power": np.zeros(n),
        "soc": np.full(n + 1, 0.5),
        "revenue": 0.0,
        "actions": np.zeros(n, dtype=np.int64),
    }


# ============================================================
# 双结算模型（Modification #1）
# DAM + RTM 两结算制：日前报价+实时偏差
# ============================================================


def solve_day_dual(
    dam_prices: np.ndarray,
    rt_prices: np.ndarray,
    battery: BatteryConfig | None = None,
    init_soc: float = 0.5,
    end_soc_min: float | None = None,
    deviation_bound: float = 0.10,
    deviation_penalty_ratio: float = 0.0,
) -> dict:
    """
    求解一天(96步)的双结算最优方案：DAM 承诺 + RT 偏差
    （正经 LP 解法，替代旧的启发式版本）

    ⚠️  重要 caveat：
    当 deviation_bound 较大（>20%）时，LP 会发现"虚拟套利"：
      - 承诺 dam=-P_max（名义全力充电）
      - 实际 actual≈0（物理不动）
      - 偏差=+P_max，按 RT 价结算
      - 当 RT > DAM 时净赚 (RT-DAM) × P_max
    这在现实中会被视为"不物理交付的财务投机"，监管会处罚
    （副总警告的灰色行为就是这种）。
    推荐 deviation_bound ∈ [0.05, 0.15]，此时数字贴近真实监管边界。

    经济逻辑：
        revenue = DAM_commitment × DAM_price × dt
                + deviation × RT_price × dt
                - degradation × |actual| × dt
                - deviation_penalty

    LP 形式（单场景 Oracle，知道 DAM 和 RT 两条真实价格）：
        决策变量（每时段 t=0..95）：
            charge[t]     ∈ [0, P_max]
            discharge[t]  ∈ [0, P_max]
            dam[t]        ∈ [-P_max, P_max]
        辅助关系：
            actual[t] = discharge[t] - charge[t]
            deviation[t] = actual[t] - dam[t]
        约束：
            |deviation[t]| ≤ deviation_bound × P_max   （偏差上限）
            SoC dynamics + 边界
        目标（最大化）：
            Σ_t [dam[t] × dam_price[t] + deviation[t] × rt_price[t]] × dt
               - deg × Σ_t (charge[t] + discharge[t]) × dt

    Args:
        dam_prices: 96点日前出清价
        rt_prices: 96点实时价
        battery: 电池参数
        init_soc: 初始 SoC
        end_soc_min: 结束 SoC 下限（可选）
        deviation_bound: 允许执行偏离 DAM 的比例（默认 10%）
        deviation_penalty_ratio: 偏差罚金占偏差金额的比例（默认 0）

    Returns:
        dict:
            dam_commitment: [96] DAM 承诺功率
            actual_power: [96] 实际执行功率
            deviation: [96] 偏差
            soc: [97] SoC 轨迹
            revenue_dam / revenue_rt / revenue_degradation / revenue_penalty / revenue_total
    """
    if battery is None:
        battery = BatteryConfig()
    n = len(dam_prices)
    assert len(rt_prices) == n, "DAM and RT must have same length"

    dt = battery.interval_hours
    cap_mw = battery.capacity_mw
    cap_mwh = battery.capacity_mwh
    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    deg = battery.degradation_cost_per_mwh
    soc_min_val = battery.min_soc
    soc_max_val = battery.max_soc
    dev_max = deviation_bound * cap_mw

    # ========= 变量布局 =========
    # x = [charge_0..n-1, discharge_0..n-1, dam_0..n-1]  共 3n
    idx_charge = lambda t: t
    idx_discharge = lambda t: n + t
    idx_dam = lambda t: 2 * n + t
    n_vars = 3 * n

    # ========= 目标函数（最小化 -revenue）=========
    # -max Σ[dam*dam_price + (dis-ch-dam)*rt_price - deg*(ch+dis)] * dt
    # = min Σ[-dam*dam_price - (dis-ch-dam)*rt_price + deg*(ch+dis)] * dt
    # charge coef:     (rt_price + deg) * dt
    # discharge coef:  (-rt_price + deg) * dt
    # dam coef:        (-dam_price + rt_price) * dt = (rt_price - dam_price) * dt
    c = np.zeros(n_vars)
    for t in range(n):
        c[idx_charge(t)] = (rt_prices[t] + deg) * dt
        c[idx_discharge(t)] = (-rt_prices[t] + deg) * dt
        c[idx_dam(t)] = (rt_prices[t] - dam_prices[t]) * dt

    # ========= 不等式约束 =========
    A_ub_rows = []
    b_ub = []

    # 1. 偏差界 |dis - ch - dam| ≤ dev_max
    # 上： dis - ch - dam ≤ dev_max
    # 下： -dis + ch + dam ≤ dev_max
    for t in range(n):
        row_up = np.zeros(n_vars)
        row_up[idx_charge(t)] = -1
        row_up[idx_discharge(t)] = 1
        row_up[idx_dam(t)] = -1
        A_ub_rows.append(row_up)
        b_ub.append(dev_max)

        row_lo = np.zeros(n_vars)
        row_lo[idx_charge(t)] = 1
        row_lo[idx_discharge(t)] = -1
        row_lo[idx_dam(t)] = 1
        A_ub_rows.append(row_lo)
        b_ub.append(dev_max)

    # 2. 总功率 |dis - ch| ≤ P_max （ch 和 dis 各自已 ≤ P_max，但同侧组合需约束）
    # 实际由于 LP 自然不会同时充放（降解惩罚），这条可略；保留以防数值问题
    for t in range(n):
        row_up = np.zeros(n_vars)
        row_up[idx_charge(t)] = -1
        row_up[idx_discharge(t)] = 1
        A_ub_rows.append(row_up)
        b_ub.append(cap_mw)

        row_lo = np.zeros(n_vars)
        row_lo[idx_charge(t)] = 1
        row_lo[idx_discharge(t)] = -1
        A_ub_rows.append(row_lo)
        b_ub.append(cap_mw)

    # 3. SoC 约束（每个时段）
    # SoC[t] = init + Σ_{k<t} (eta_c * charge[k] - discharge[k]/eta_d) * dt / cap_mwh
    # soc_min ≤ SoC[t] ≤ soc_max
    charge_coeff = eta_c * dt / cap_mwh
    discharge_coeff = dt / (cap_mwh * eta_d)

    for t in range(1, n + 1):
        # upper: Σ charge[k]*charge_coeff - discharge[k]*discharge_coeff ≤ soc_max - init_soc
        row_u = np.zeros(n_vars)
        for k in range(t):
            row_u[idx_charge(k)] = charge_coeff
            row_u[idx_discharge(k)] = -discharge_coeff
        A_ub_rows.append(row_u)
        b_ub.append(soc_max_val - init_soc)

        # lower: -Σ ... ≤ init_soc - soc_min
        row_l = np.zeros(n_vars)
        for k in range(t):
            row_l[idx_charge(k)] = -charge_coeff
            row_l[idx_discharge(k)] = discharge_coeff
        A_ub_rows.append(row_l)
        b_ub.append(init_soc - soc_min_val)

    # 4. 结束 SoC 下限（可选）
    if end_soc_min is not None and end_soc_min > soc_min_val:
        row = np.zeros(n_vars)
        for k in range(n):
            row[idx_charge(k)] = -charge_coeff
            row[idx_discharge(k)] = discharge_coeff
        A_ub_rows.append(row)
        b_ub.append(init_soc - end_soc_min)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub)

    # ========= 变量边界 =========
    bounds = (
        [(0, cap_mw)] * n +           # charge
        [(0, cap_mw)] * n +           # discharge
        [(-cap_mw, cap_mw)] * n       # dam（可正可负）
    )

    # ========= 求解 =========
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 10.0},
    )

    if not result.success:
        logger.warning(f"Dual LP failed: {result.message}")
        return {
            "dam_commitment": np.zeros(n),
            "actual_power": np.zeros(n),
            "deviation": np.zeros(n),
            "soc": np.full(n + 1, init_soc),
            "revenue_dam": 0.0,
            "revenue_rt": 0.0,
            "revenue_degradation": 0.0,
            "revenue_penalty": 0.0,
            "revenue_total": 0.0,
            "net_power": np.zeros(n),
            "revenue": 0.0,
        }

    x = result.x
    charge = x[:n]
    discharge = x[n:2 * n]
    dam_commitment = x[2 * n:3 * n]
    actual_power = discharge - charge
    deviation = actual_power - dam_commitment

    # ========= SoC 轨迹 =========
    soc_traj = np.zeros(n + 1)
    soc_traj[0] = init_soc
    for t in range(n):
        soc_traj[t + 1] = np.clip(
            soc_traj[t] + charge[t] * charge_coeff - discharge[t] * discharge_coeff,
            soc_min_val, soc_max_val,
        )

    # ========= 结算分解 =========
    revenue_dam = float(np.sum(dam_commitment * dam_prices) * dt)
    revenue_rt = float(np.sum(deviation * rt_prices) * dt)
    revenue_degradation = float(np.sum(charge + discharge) * deg * dt)

    rev_penalty = 0.0
    if deviation_penalty_ratio > 0:
        rev_penalty = float(np.sum(np.abs(deviation) * np.abs(rt_prices)) * deviation_penalty_ratio * dt)

    revenue_total = revenue_dam + revenue_rt - revenue_degradation - rev_penalty

    return {
        "dam_commitment": dam_commitment,
        "actual_power": actual_power,
        "deviation": deviation,
        "soc": soc_traj,
        "revenue_dam": float(revenue_dam),
        "revenue_rt": float(revenue_rt),
        "revenue_degradation": float(revenue_degradation),
        "revenue_penalty": float(rev_penalty),
        "revenue_total": float(revenue_total),
        # Backward compat
        "net_power": actual_power,
        "revenue": float(revenue_total),
    }


def compare_single_vs_dual(
    dam_prices: np.ndarray,
    rt_prices: np.ndarray,
    battery: BatteryConfig | None = None,
    init_soc: float = 0.5,
) -> dict:
    """
    对比单结算（只用 RT 价）vs 双结算的结果。

    用于验证 #1 修改的影响幅度。
    """
    single_rt = solve_day(rt_prices, battery, init_soc)
    single_dam = solve_day(dam_prices, battery, init_soc)
    dual = solve_day_dual(dam_prices, rt_prices, battery, init_soc)

    return {
        "single_rt_only": single_rt["revenue"],
        "single_dam_only": single_dam["revenue"],
        "dual_settlement": dual["revenue_total"],
        "dual_breakdown": {
            "dam_part": dual["revenue_dam"],
            "rt_part": dual["revenue_rt"],
            "degradation": dual["revenue_degradation"],
        },
    }


# ============================================================
# 验证工具
# ============================================================

def verify_oracle(prices: np.ndarray, result: dict, battery: BatteryConfig | None = None):
    """验证Oracle解的正确性"""
    if battery is None:
        battery = BatteryConfig()

    soc = result["soc"]
    n = len(prices)

    # 检查SOC边界
    assert soc.min() >= battery.min_soc - 1e-6, f"SOC below min: {soc.min()}"
    assert soc.max() <= battery.max_soc + 1e-6, f"SOC above max: {soc.max()}"

    # 检查功率边界
    charge = result["charge"]
    discharge = result["discharge"]
    assert charge.min() >= -1e-6, f"Negative charge: {charge.min()}"
    assert discharge.min() >= -1e-6, f"Negative discharge: {discharge.min()}"
    assert charge.max() <= battery.capacity_mw + 1e-6, f"Charge exceeds cap: {charge.max()}"
    assert discharge.max() <= battery.capacity_mw + 1e-6, f"Discharge exceeds cap: {discharge.max()}"

    # 检查收入计算
    dt = battery.interval_hours
    deg = battery.degradation_cost_per_mwh
    net = discharge - charge
    expected_rev = np.sum(net * prices * dt) - np.sum((charge + discharge) * deg * dt)
    assert abs(result["revenue"] - expected_rev) < 1.0, \
        f"Revenue mismatch: {result['revenue']:.2f} vs {expected_rev:.2f}"

    return True


if __name__ == "__main__":
    # 快速验证：用一天的合成数据测试
    np.random.seed(42)
    # 模拟山东典型日内电价（元/MWh）
    hours = np.arange(96) / 4  # 0-24h
    base = 320
    shape = -60 * np.cos(2 * np.pi * hours / 24) + 40 * np.sin(4 * np.pi * hours / 24)
    noise = np.random.normal(0, 20, 96)
    prices = base + shape + noise

    result = solve_day(prices)
    verify_oracle(prices, result)

    print(f"Revenue: {result['revenue']:,.0f} 元")
    print(f"SOC range: {result['soc'].min():.2f} - {result['soc'].max():.2f}")
    print(f"Action distribution: {np.bincount(result['actions'], minlength=5)}")
    print(f"Verification: PASSED ✅")
