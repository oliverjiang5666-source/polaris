"""
MPC (Model Predictive Control) controller for battery storage.

At each step:
1. Forecast future prices using a trained forecaster
2. Solve LP optimal schedule with forecasted prices + current SOC
3. Execute the first action (continuous or discrete), then replan next step
"""
from __future__ import annotations

import numpy as np
from config import BatteryConfig
from oracle.lp_oracle import solve_day


ACTION_POWER_RATIOS = [0.0, -0.3, -1.0, 0.3, 1.0]


class MPCController:

    def __init__(
        self,
        forecaster,
        battery: BatteryConfig | None = None,
        max_horizon: int = 96,
        continuous: bool = True,
        daily_profile: np.ndarray | None = None,
        end_soc_min: float | None = None,
    ):
        self.forecaster = forecaster
        self.battery = battery or BatteryConfig()
        self.max_horizon = max_horizon
        self.continuous = continuous
        self.daily_profile = daily_profile
        self.end_soc_min = end_soc_min

    def plan(
        self,
        features_t: np.ndarray,
        current_soc: float,
        idx: int,
        remaining_steps: int = 96,
    ) -> dict:
        """
        MPC planning: forecast → extend with daily profile → LP solve.
        """
        horizon = min(self.max_horizon, remaining_steps)
        if horizon < 2:
            return {"net_power": np.zeros(1), "actions": np.zeros(1, dtype=int)}

        price_forecast = self.forecaster.predict(features_t, idx, horizon=horizon)
        price_forecast = np.clip(price_forecast, -500, 50000)

        # Extend horizon with realistic daily price pattern.
        # The LP sees "there will be peaks and troughs tomorrow too",
        # so it holds SOC to trade against tomorrow's opportunities.
        if self.daily_profile is not None and horizon >= 16:
            # Align the profile to the correct time-of-day
            step_in_day = (idx + horizon) % 96
            extended = np.roll(self.daily_profile, -step_in_day)
            # Blend toward the forecast's mean level
            scale = np.mean(price_forecast) / (np.mean(self.daily_profile) + 1e-8)
            extended = extended * scale
            price_forecast = np.concatenate([price_forecast, extended])

        result = solve_day(price_forecast, self.battery, init_soc=current_soc,
                           end_soc_min=self.end_soc_min)
        return result

    def get_power(
        self,
        features_t: np.ndarray,
        current_soc: float,
        idx: int,
        remaining_steps: int = 96,
    ) -> float:
        """
        Returns first-step net power (MW). Positive = discharge, negative = charge.
        If continuous=True, uses LP's continuous solution.
        If continuous=False, returns discretized power.
        """
        result = self.plan(features_t, current_soc, idx, remaining_steps)
        if self.continuous:
            return float(result["net_power"][0])
        else:
            action = int(result["actions"][0])
            return ACTION_POWER_RATIOS[action] * self.battery.capacity_mw


# ============================================================
# Battery simulation (works with both continuous & discrete power)
# ============================================================

def _step_battery(
    power_mw: float,
    price: float,
    soc: float,
    battery: BatteryConfig,
    degradation_per_mwh: float = 2.0,
) -> tuple[float, float, float]:
    """
    Simulate one step of battery physics.
    Returns: (new_soc, net_revenue, actual_energy_mwh)
    """
    # Clamp power to battery limits
    power_mw = np.clip(power_mw, -battery.capacity_mw, battery.capacity_mw)
    energy_mwh = power_mw * battery.interval_hours

    if energy_mwh > 0:  # discharge
        soc_change = -energy_mwh / battery.capacity_mwh / battery.discharge_efficiency
    elif energy_mwh < 0:  # charge
        soc_change = -energy_mwh * battery.charge_efficiency / battery.capacity_mwh
    else:
        soc_change = 0.0

    new_soc = soc + soc_change
    if new_soc > battery.max_soc or new_soc < battery.min_soc:
        # Clip to valid range and recompute actual energy
        if new_soc > battery.max_soc:
            actual_soc_change = battery.max_soc - soc
        else:
            actual_soc_change = battery.min_soc - soc

        if energy_mwh > 0:  # was discharging
            energy_mwh = -actual_soc_change * battery.capacity_mwh * battery.discharge_efficiency
        elif energy_mwh < 0:  # was charging
            energy_mwh = -actual_soc_change * battery.capacity_mwh / battery.charge_efficiency
        else:
            energy_mwh = 0.0

        power_mw = energy_mwh / battery.interval_hours
        new_soc = soc + actual_soc_change

    revenue = energy_mwh * price
    degradation = abs(energy_mwh) * degradation_per_mwh
    return new_soc, revenue - degradation, energy_mwh


def simulate_mpc(
    controller: MPCController,
    features: np.ndarray,
    prices: np.ndarray,
    battery: BatteryConfig | None = None,
    init_soc: float = 0.5,
    degradation_per_mwh: float = 2.0,
    replan_every: int = 1,
    log_every: int = 0,
) -> dict:
    """
    Run MPC controller through a price series, simulating battery physics.
    Supports both continuous and discrete power control.
    """
    if battery is None:
        battery = BatteryConfig()

    n = len(prices)
    soc = init_soc
    total_revenue = 0.0
    soc_trajectory = [soc]
    powers = []
    daily_revenues = []
    day_revenue = 0.0
    cached_plan = None
    plan_step = 0

    for t in range(n):
        remaining = n - t

        # Replan if needed
        if t % replan_every == 0 or cached_plan is None:
            result = controller.plan(features[t], soc, t, remaining)
            cached_plan = result
            plan_step = 0

        # Get power for this step
        if controller.continuous:
            power_mw = float(cached_plan["net_power"][min(plan_step, len(cached_plan["net_power"]) - 1)])
        else:
            action = int(cached_plan["actions"][min(plan_step, len(cached_plan["actions"]) - 1)])
            power_mw = ACTION_POWER_RATIOS[action] * battery.capacity_mw

        plan_step += 1

        # Simulate battery
        soc, net_rev, energy = _step_battery(power_mw, prices[t], soc, battery, degradation_per_mwh)
        total_revenue += net_rev
        day_revenue += net_rev
        powers.append(power_mw)
        soc_trajectory.append(soc)

        if (t + 1) % 96 == 0:
            daily_revenues.append(day_revenue)
            if log_every > 0 and ((t + 1) // 96) % log_every == 0:
                from loguru import logger
                logger.info(f"  Day {(t+1)//96}: cumulative={total_revenue:,.0f}, SOC={soc:.2f}")
            day_revenue = 0.0

    if day_revenue != 0:
        daily_revenues.append(day_revenue)

    return {
        "revenue": total_revenue,
        "soc_trajectory": np.array(soc_trajectory),
        "powers": np.array(powers),
        "daily_revenues": np.array(daily_revenues),
    }


# ============================================================
# 双结算（Modification #1）
# DAM 计划 + RT 偏差双价格结算
# ============================================================


def _step_battery_dual(
    dam_commitment_mw: float,
    actual_power_mw: float,
    dam_price: float,
    rt_price: float,
    soc: float,
    battery: BatteryConfig,
    degradation_per_mwh: float = 2.0,
    deviation_penalty_ratio: float = 0.0,
) -> tuple[float, float, float, dict]:
    """
    双结算的一步模拟。

    Revenue = DAM_part + RT_deviation_part - degradation - penalty
        DAM_part = dam_commitment * dam_price * dt
        RT_part  = (actual - dam_commitment) * rt_price * dt
        degradation = |actual_energy| * degradation_per_mwh * dt
        penalty = |deviation| * |rt_price| * deviation_penalty_ratio * dt  (if triggered)

    Physical SoC 更新基于 actual_power_mw。

    Args:
        dam_commitment_mw: D-1 报的日前承诺功率（正=放，负=充）
        actual_power_mw:   D 实际执行功率
        dam_price:         DAM 出清价
        rt_price:          RT 出清价
        soc:               当前 SoC
        battery:           电池参数
        deviation_penalty_ratio: 超过 ±2% 偏差触发罚金的比例（典型 0.0-0.3）

    Returns:
        (new_soc, net_revenue, actual_energy_mwh, breakdown_dict)
    """
    dt = battery.interval_hours
    cap_mw = battery.capacity_mw

    # 物理 step：基于 actual 更新 SoC
    actual_power_mw = float(np.clip(actual_power_mw, -cap_mw, cap_mw))
    actual_energy_mwh = actual_power_mw * dt

    if actual_energy_mwh > 0:  # 放电
        soc_change = -actual_energy_mwh / battery.capacity_mwh / battery.discharge_efficiency
    elif actual_energy_mwh < 0:  # 充电
        soc_change = -actual_energy_mwh * battery.charge_efficiency / battery.capacity_mwh
    else:
        soc_change = 0.0

    new_soc = soc + soc_change

    # SoC 越界处理
    if new_soc > battery.max_soc or new_soc < battery.min_soc:
        if new_soc > battery.max_soc:
            actual_soc_change = battery.max_soc - soc
        else:
            actual_soc_change = battery.min_soc - soc

        if actual_energy_mwh > 0:  # 原本想放
            actual_energy_mwh = -actual_soc_change * battery.capacity_mwh * battery.discharge_efficiency
        elif actual_energy_mwh < 0:  # 原本想充
            actual_energy_mwh = -actual_soc_change * battery.capacity_mwh / battery.charge_efficiency
        else:
            actual_energy_mwh = 0.0
        actual_power_mw = actual_energy_mwh / dt
        new_soc = soc + actual_soc_change

    # 双结算
    dam_energy = dam_commitment_mw * dt
    deviation_energy = actual_energy_mwh - dam_energy

    rev_dam = dam_energy * dam_price
    rev_rt = deviation_energy * rt_price

    degradation = abs(actual_energy_mwh) * degradation_per_mwh

    # 偏差罚金
    penalty = 0.0
    if deviation_penalty_ratio > 0 and abs(dam_energy) > 1e-6:
        dev_ratio = abs(deviation_energy) / abs(dam_energy)
        if dev_ratio > 0.02:  # 触发阈值
            penalty = abs(deviation_energy) * abs(rt_price) * deviation_penalty_ratio

    net_revenue = rev_dam + rev_rt - degradation - penalty

    breakdown = {
        "dam_revenue": rev_dam,
        "rt_revenue": rev_rt,
        "degradation": degradation,
        "penalty": penalty,
    }

    return new_soc, net_revenue, actual_energy_mwh, breakdown


def simulate_dual_settlement(
    dam_commitment: np.ndarray,
    actual_power: np.ndarray,
    dam_prices: np.ndarray,
    rt_prices: np.ndarray,
    battery: BatteryConfig | None = None,
    init_soc: float = 0.5,
    degradation_per_mwh: float = 2.0,
    deviation_penalty_ratio: float = 0.0,
) -> dict:
    """
    对一段时间的双结算批量仿真。

    Args:
        dam_commitment: 96n 点 DAM 承诺功率序列
        actual_power:   96n 点实际执行功率序列
        dam_prices, rt_prices: 对应价格
        其他参数同 _step_battery_dual
    """
    if battery is None:
        battery = BatteryConfig()

    n = len(dam_prices)
    soc = init_soc
    total_rev = 0.0
    total_dam_rev = 0.0
    total_rt_rev = 0.0
    total_deg = 0.0
    total_penalty = 0.0
    soc_traj = [soc]

    for t in range(n):
        soc, rev, _, bd = _step_battery_dual(
            dam_commitment[t], actual_power[t],
            dam_prices[t], rt_prices[t],
            soc, battery, degradation_per_mwh, deviation_penalty_ratio,
        )
        total_rev += rev
        total_dam_rev += bd["dam_revenue"]
        total_rt_rev += bd["rt_revenue"]
        total_deg += bd["degradation"]
        total_penalty += bd["penalty"]
        soc_traj.append(soc)

    return {
        "revenue": total_rev,
        "revenue_dam": total_dam_rev,
        "revenue_rt": total_rt_rev,
        "degradation": total_deg,
        "penalty": total_penalty,
        "soc_trajectory": np.array(soc_traj),
    }


def simulate_threshold(
    prices: np.ndarray,
    ma96: np.ndarray,
    battery: BatteryConfig | None = None,
    charge_ratio: float = 0.65,
    discharge_ratio: float = 1.35,
    init_soc: float = 0.5,
    degradation_per_mwh: float = 2.0,
) -> dict:
    """Simulate threshold strategy (discrete actions)."""
    if battery is None:
        battery = BatteryConfig()

    n = len(prices)
    soc = init_soc
    total_revenue = 0.0
    actions_taken = []

    for t in range(n):
        ratio = prices[t] / (ma96[t] + 1e-8)
        if ratio < charge_ratio:
            action = 2  # fast_charge
        elif ratio > discharge_ratio:
            action = 4  # fast_discharge
        else:
            action = 0  # wait
        actions_taken.append(action)

        power_mw = ACTION_POWER_RATIOS[action] * battery.capacity_mw
        soc, net_rev, _ = _step_battery(power_mw, prices[t], soc, battery, degradation_per_mwh)
        total_revenue += net_rev

    return {"revenue": total_revenue, "actions": np.array(actions_taken)}


def simulate_oracle_continuous(
    prices: np.ndarray,
    battery: BatteryConfig | None = None,
    init_soc: float = 0.5,
) -> dict:
    """Oracle with daily reset, continuous power. This is the theoretical upper bound."""
    if battery is None:
        battery = BatteryConfig()

    n = len(prices)
    total_revenue = 0.0
    n_days = n // 96

    for d in range(n_days):
        start = d * 96
        day_prices = prices[start:start + 96]
        if np.isnan(day_prices).any():
            continue
        result = solve_day(day_prices, battery, init_soc)
        total_revenue += result["revenue"]

    return {"revenue": total_revenue}


def simulate_oracle_discrete(
    prices: np.ndarray,
    battery: BatteryConfig | None = None,
    init_soc: float = 0.5,
    degradation_per_mwh: float = 2.0,
) -> dict:
    """Oracle with daily reset, but using DISCRETE quantized actions.
    This is the achievable upper bound with 5-action space."""
    if battery is None:
        battery = BatteryConfig()

    n = len(prices)
    total_revenue = 0.0
    n_days = n // 96

    for d in range(n_days):
        start = d * 96
        day_prices = prices[start:start + 96]
        if np.isnan(day_prices).any():
            continue

        result = solve_day(day_prices, battery, init_soc)
        # Simulate with quantized actions
        soc = init_soc
        day_rev = 0.0
        for t in range(96):
            action = int(result["actions"][t])
            power_mw = ACTION_POWER_RATIOS[action] * battery.capacity_mw
            soc, net_rev, _ = _step_battery(power_mw, day_prices[t], soc, battery, degradation_per_mwh)
            day_rev += net_rev

        total_revenue += day_rev

    return {"revenue": total_revenue}
