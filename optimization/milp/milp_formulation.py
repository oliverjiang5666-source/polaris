"""
两阶段随机 LP 建模（Pyomo）

问题：
  max E[Revenue] - λ × degradation
       = ∑_s w_s [∑_t (p_dam[t] × DAM[t] + p_dev[t,s] × RT[t,s])] × dt
        - deg × ∑_t (p_charge[t,s] + p_discharge[t,s]) × dt

决策变量：
  第一阶段（DAM 承诺，此时此刻决定）：
    p_dam[t] ∈ [-P_max, P_max]   —— 96 个

  第二阶段（每场景下的 RT 调整）：
    p_dev[t, s] ∈ [-P_dev_max, P_dev_max]
    p_charge[t, s] ≥ 0
    p_discharge[t, s] ≥ 0
    soc[t, s] ∈ [soc_min, soc_max]

约束：
  p_dam[t] + p_dev[t, s] = p_discharge[t, s] - p_charge[t, s]
  |p_dam[t] + p_dev[t, s]| ≤ P_max
  soc[t+1, s] = soc[t, s] + (η_c × p_charge[t, s] - p_discharge[t, s] / η_d) × dt / E_max
  soc_min ≤ soc[t, s] ≤ soc_max
  soc[0, s] = init_soc
  soc[T, s] ≥ final_soc_min (可选)

规模（K=200, T=96）：
  第一阶段变量：96
  第二阶段变量：200 × 96 × 4 = 76,800
  约束：~250,000
"""
from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
from dataclasses import dataclass


@dataclass
class BatteryParams:
    """电池参数（和 config.BatteryConfig 一致但独立，避免耦合）"""
    P_max: float = 200.0
    E_max: float = 400.0
    soc_min: float = 0.05
    soc_max: float = 0.95
    eta_c: float = 0.9487
    eta_d: float = 0.9487
    dt: float = 0.25
    deg_cost: float = 2.0


@dataclass
class MILPConfig:
    """MILP 实验配置"""
    init_soc: float = 0.5
    final_soc_min: float = 0.3       # 强制一天结束 SoC 不能太低
    deviation_bound: float = 0.10    # RT 偏差允许的最大比例（相对 P_max）
    cycle_limit: float | None = None # 每日循环次数上限（可选）
    use_cvar: bool = False           # 是否用 CVaR 目标
    cvar_alpha: float = 0.05
    cvar_weight: float = 0.3


def build_two_stage_lp(
    dam_forecast: np.ndarray,      # [96]   DAM 预期价
    rt_scenarios: np.ndarray,      # [K, 96] RT 场景
    scenario_weights: np.ndarray,  # [K]
    battery: BatteryParams,
    config: MILPConfig,
) -> pyo.ConcreteModel:
    """构建两阶段随机 LP 模型。"""
    K, T = rt_scenarios.shape
    assert len(dam_forecast) == T
    assert len(scenario_weights) == K
    assert abs(scenario_weights.sum() - 1.0) < 1e-6, \
        f"权重和 {scenario_weights.sum()} 不等于 1"

    m = pyo.ConcreteModel(name="TwoStageBatteryLP")

    # 索引
    m.T = pyo.RangeSet(0, T - 1)
    m.T1 = pyo.RangeSet(0, T)  # 含 T+1 用于 SoC
    m.S = pyo.RangeSet(0, K - 1)

    # 参数
    m.dam_price = pyo.Param(m.T, initialize={t: float(dam_forecast[t]) for t in range(T)})
    m.rt_price = pyo.Param(m.T, m.S,
        initialize={(t, s): float(rt_scenarios[s, t]) for s in range(K) for t in range(T)})
    m.weight = pyo.Param(m.S, initialize={s: float(scenario_weights[s]) for s in range(K)})

    # 决策变量
    P_max = battery.P_max
    dev_max = config.deviation_bound * P_max

    m.p_dam = pyo.Var(m.T, domain=pyo.Reals, bounds=(-P_max, P_max))
    m.p_dev = pyo.Var(m.T, m.S, domain=pyo.Reals, bounds=(-dev_max, dev_max))
    m.p_charge = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.p_discharge = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.soc = pyo.Var(m.T1, m.S, domain=pyo.NonNegativeReals,
                    bounds=(battery.soc_min, battery.soc_max))

    # 约束 1: 功率分解
    def charge_discharge_rule(m, t, s):
        return m.p_dam[t] + m.p_dev[t, s] == m.p_discharge[t, s] - m.p_charge[t, s]
    m.c_power_split = pyo.Constraint(m.T, m.S, rule=charge_discharge_rule)

    # 约束 2: 总功率上下限
    def power_upper_rule(m, t, s):
        return m.p_dam[t] + m.p_dev[t, s] <= P_max
    def power_lower_rule(m, t, s):
        return m.p_dam[t] + m.p_dev[t, s] >= -P_max
    m.c_power_upper = pyo.Constraint(m.T, m.S, rule=power_upper_rule)
    m.c_power_lower = pyo.Constraint(m.T, m.S, rule=power_lower_rule)

    # 约束 3: SoC 动力学
    def soc_dynamics_rule(m, t, s):
        return m.soc[t + 1, s] == m.soc[t, s] + \
            battery.dt * (battery.eta_c * m.p_charge[t, s] - m.p_discharge[t, s] / battery.eta_d) / battery.E_max
    m.c_soc_dyn = pyo.Constraint(m.T, m.S, rule=soc_dynamics_rule)

    # 约束 4: 初始 SoC
    def soc_init_rule(m, s):
        return m.soc[0, s] == config.init_soc
    m.c_soc_init = pyo.Constraint(m.S, rule=soc_init_rule)

    # 约束 5: 终止 SoC 下限
    if config.final_soc_min is not None:
        def soc_final_rule(m, s):
            return m.soc[T, s] >= config.final_soc_min
        m.c_soc_final = pyo.Constraint(m.S, rule=soc_final_rule)

    # 约束 6: 循环次数（可选）
    if config.cycle_limit is not None:
        def cycle_rule(m, s):
            return sum((m.p_charge[t, s] + m.p_discharge[t, s]) for t in m.T) * battery.dt \
                <= config.cycle_limit * 2 * battery.E_max
        m.c_cycle = pyo.Constraint(m.S, rule=cycle_rule)

    # 目标函数
    def objective_rule(m):
        total = 0.0
        for s in m.S:
            rev_s = 0.0
            for t in m.T:
                rev_s += m.p_dam[t] * m.dam_price[t] * battery.dt
                rev_s += m.p_dev[t, s] * m.rt_price[t, s] * battery.dt
                rev_s -= battery.deg_cost * (m.p_charge[t, s] + m.p_discharge[t, s]) * battery.dt
            total += m.weight[s] * rev_s
        return total

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    return m


def extract_solution(model: pyo.ConcreteModel, K: int, T: int = 96) -> dict:
    """从已求解的模型中提取解"""
    p_dam = np.array([pyo.value(model.p_dam[t]) for t in range(T)])

    # 每场景下的实际功率
    p_actual_by_scenario = np.zeros((K, T))
    for s in range(K):
        for t in range(T):
            p_actual_by_scenario[s, t] = p_dam[t] + pyo.value(model.p_dev[t, s])

    # SoC 轨迹（每场景）
    soc_traj = np.zeros((K, T + 1))
    for s in range(K):
        for t in range(T + 1):
            soc_traj[s, t] = pyo.value(model.soc[t, s])

    return {
        "p_dam": p_dam,
        "p_actual_by_scenario": p_actual_by_scenario,
        "soc_trajectory": soc_traj,
        "objective": pyo.value(model.obj),
    }


def simulate_on_actual(
    p_dam: np.ndarray,              # [96] DAM 承诺
    actual_rt_price: np.ndarray,    # [96] 真实 RT 价
    actual_dam_price: np.ndarray,   # [96] 真实 DAM 价
    battery: BatteryParams,
    config: MILPConfig,
) -> dict:
    """
    用 DAM 承诺在真实价格上仿真。

    执行规则（简化）：
      - 物理按 p_dam 执行（不允许"后见之明"偏差）
      - 结算：收入 = p_dam × actual_dam_price × dt
              + 对于 SoC 自然偏差，按 actual_rt × 偏差量

    为什么这样简化：
      我们测试的是"MILP 做的 DAM 决策是否更好"，不是"事后偏差"。
      偏差策略是另一个维度（你可以叠加 MPC）。
    """
    T = len(p_dam)
    dt = battery.dt
    soc = config.init_soc
    revenue_dam = 0.0
    revenue_dev = 0.0
    total_charge_mwh = 0.0
    total_discharge_mwh = 0.0
    actual_power = np.zeros(T)

    for t in range(T):
        pw_commit = p_dam[t]
        # 功率约束检查
        pw_commit = float(np.clip(pw_commit, -battery.P_max, battery.P_max))

        if pw_commit > 0:  # 放电
            energy_out = pw_commit * dt
            soc_needed = energy_out / (battery.E_max * battery.eta_d)
            if soc - soc_needed < battery.soc_min:
                # SoC 不够，截断
                available = (soc - battery.soc_min) * battery.E_max * battery.eta_d
                energy_out = max(available, 0)
                pw_commit = energy_out / dt
                soc = battery.soc_min
            else:
                soc -= soc_needed
            total_discharge_mwh += energy_out
            actual_power[t] = pw_commit
        elif pw_commit < 0:  # 充电
            energy_in = -pw_commit * dt
            soc_added = energy_in * battery.eta_c / battery.E_max
            if soc + soc_added > battery.soc_max:
                available_room = (battery.soc_max - soc) * battery.E_max / battery.eta_c
                energy_in = max(available_room, 0)
                pw_commit = -energy_in / dt
                soc = battery.soc_max
            else:
                soc += soc_added
            total_charge_mwh += energy_in
            actual_power[t] = pw_commit
        else:
            actual_power[t] = 0.0

        # 结算：执行功率 × 实际 DAM 价（简化：执行 = 承诺；偏差 = 0）
        revenue_dam += actual_power[t] * actual_dam_price[t] * dt
        # 如果实际执行偏离承诺（因 SoC 约束），偏差部分按 RT 结算
        deviation = actual_power[t] - p_dam[t]
        revenue_dev += deviation * actual_rt_price[t] * dt

    degradation = battery.deg_cost * (total_charge_mwh + total_discharge_mwh)
    total = revenue_dam + revenue_dev - degradation

    return {
        "revenue_total": total,
        "revenue_dam": revenue_dam,
        "revenue_dev": revenue_dev,
        "degradation": degradation,
        "actual_power": actual_power,
        "final_soc": soc,
        "total_charge_mwh": total_charge_mwh,
        "total_discharge_mwh": total_discharge_mwh,
    }


if __name__ == "__main__":
    from loguru import logger

    battery = BatteryParams()
    config = MILPConfig(deviation_bound=0.10, cycle_limit=None)

    # 合成 5 个场景做测试
    np.random.seed(42)
    T = 96
    K = 5
    hours = np.arange(T) / 4
    base = 320 - 80 * np.cos(2 * np.pi * hours / 24)
    rt_scenarios = np.array([base + np.random.normal(0, 30, T) for _ in range(K)])
    dam_forecast = rt_scenarios.mean(axis=0)
    weights = np.ones(K) / K

    logger.info("构建模型...")
    m = build_two_stage_lp(dam_forecast, rt_scenarios, weights, battery, config)

    logger.info("调用 HiGHS...")
    solver = pyo.SolverFactory("appsi_highs")
    results = solver.solve(m, tee=False)
    logger.info(f"求解状态: {results.solver.termination_condition}")

    sol = extract_solution(m, K, T)
    logger.info(f"目标值: ¥{sol['objective']:,.0f}")
    logger.info(f"DAM 承诺: min={sol['p_dam'].min():.1f}, max={sol['p_dam'].max():.1f}")

    # 测试仿真
    actual_rt = base + np.random.normal(0, 30, T)
    actual_dam = dam_forecast.copy()
    sim = simulate_on_actual(sol['p_dam'], actual_rt, actual_dam, battery, config)
    logger.info(f"仿真真实价收入: ¥{sim['revenue_total']:,.0f}")
    logger.info(f"  DAM 部分: ¥{sim['revenue_dam']:,.0f}")
    logger.info(f"  偏差部分: ¥{sim['revenue_dev']:,.0f}")
    logger.info(f"  降解: ¥{sim['degradation']:,.0f}")
