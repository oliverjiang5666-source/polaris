"""
CVaR-约束 DP (Modification #5)

核心思想：从"期望收益最大化"升级到"尾部风险可控的收益最大化"。

原版（Regime V3）：
    max E[Revenue] = max Σ P(scenario) × Revenue(scenario)

CVaR 版：
    max [E[Revenue] - λ × CVaR_α(Loss)]
其中：
    CVaR_α(Loss) = 最坏 α 比例场景下的期望损失（典型 α=5%）

用法：
    from optimization.cvar_dp import solve_cvar_dp
    policy = solve_cvar_dp(scenarios, weights, battery, ..., alpha=0.05, cvar_weight=0.3)

理论：
    Rockafellar-Uryasev (2000): CVaR 可通过辅助变量 VaR 写成 LP
    minimize  VaR + (1/α) × E[max(Loss - VaR, 0)]

实现简化：
    对场景集 {s_k, w_k}，计算每个场景的收益 R_k（从基础 DP 解）
    按 R_k 排序，取最坏 α 比例场景作为 CVaR 估计
    在 DP 目标中：reward = E[R] - λ × CVaR_penalty

使用场景：
    - 需要给机构客户（银行、保险）展示风险控制能力
    - POC 合约要求"最坏情况不亏损"
    - 客户风险厌恶程度高
"""
from __future__ import annotations

import numpy as np
from config import BatteryConfig


def compute_cvar(values: np.ndarray, weights: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """
    计算 CVaR（Conditional Value at Risk）。

    Args:
        values: 场景收益数组
        weights: 场景概率
        alpha: 尾部比例（0.05 = 最坏 5%）

    Returns:
        (VaR, CVaR)
        VaR = alpha 分位数
        CVaR = 小于 VaR 的场景的条件期望
    """
    # 按收益排序
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # 累积概率
    cumprob = np.cumsum(sorted_weights)

    # 找到 alpha 分位数 (VaR)
    var_idx = np.searchsorted(cumprob, alpha)
    var_idx = min(var_idx, len(sorted_vals) - 1)
    var_value = sorted_vals[var_idx]

    # CVaR = 平均最坏场景
    tail_mask = cumprob <= alpha + 1e-6
    if tail_mask.sum() == 0:
        tail_mask[0] = True

    tail_vals = sorted_vals[tail_mask]
    tail_weights = sorted_weights[tail_mask]
    if tail_weights.sum() < 1e-6:
        return float(var_value), float(var_value)

    cvar = float(np.sum(tail_vals * tail_weights) / tail_weights.sum())
    return float(var_value), cvar


def solve_cvar_dp(
    scenarios_prices: np.ndarray,  # [n_scenarios, 96]
    weights: np.ndarray,            # [n_scenarios]
    battery: BatteryConfig,
    power_levels: np.ndarray,
    soc_levels: np.ndarray,
    alpha: float = 0.05,
    cvar_weight: float = 0.3,
    init_soc: float = 0.5,
    deg: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """
    CVaR-约束 DP。

    实现策略：
        在 DP 的每个状态-动作，计算场景加权"期望收益 + λ × tail_bonus"
        tail_bonus = 尾部场景（最坏α）下的平均收益（鼓励提升尾部）

    返回：
        (policy, info)
        policy[t, soc_idx] = 最优功率索引
        info 包含期望和 CVaR
    """
    n_scenarios, n_steps = scenarios_prices.shape
    n_soc = len(soc_levels)
    n_power = len(power_levels)

    dt = battery.interval_hours
    cap_mwh = battery.capacity_mwh
    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency

    # V_scenarios[t, soc, k]: 从 (t, soc) 开始在场景 k 下的最优未来价值
    # 由于场景独立 DP，可以分别算
    V_all = np.zeros((n_scenarios, n_steps + 1, n_soc))
    best_p_scenario = np.zeros((n_scenarios, n_steps, n_soc), dtype=np.int32)

    for k in range(n_scenarios):
        prices_k = scenarios_prices[k]
        V = V_all[k]

        for t in range(n_steps - 1, -1, -1):
            price = prices_k[t]
            for s_idx in range(n_soc):
                soc = soc_levels[s_idx]
                best_val, best_p = -1e18, n_power // 2
                for p_idx in range(n_power):
                    pw = power_levels[p_idx]
                    e = pw * dt
                    if e > 0:
                        sc = -e / cap_mwh / eta_d
                    elif e < 0:
                        sc = -e * eta_c / cap_mwh
                    else:
                        sc = 0.0
                    ns = soc + sc
                    if ns < battery.min_soc - 0.001 or ns > battery.max_soc + 0.001:
                        continue
                    ns = np.clip(ns, battery.min_soc, battery.max_soc)
                    ns_idx = int(np.clip(
                        round((ns - battery.min_soc) / (battery.max_soc - battery.min_soc) * (n_soc - 1)),
                        0, n_soc - 1))
                    reward = e * price - abs(e) * deg
                    total = reward + V[t + 1][ns_idx]
                    if total > best_val:
                        best_val = total
                        best_p = p_idx
                V[t][s_idx] = best_val
                best_p_scenario[k][t][s_idx] = best_p

    # 取每个 (t, soc) 的"鲁棒最优动作"：最大化 E - λ × CVaR_loss
    # 这里简化：对每个 (t, soc)，评估每个动作在所有场景下的期望价值和 CVaR
    policy = np.zeros((n_steps, n_soc), dtype=np.int32)

    for t in range(n_steps):
        for s_idx in range(n_soc):
            soc = soc_levels[s_idx]
            best_obj = -1e18
            best_p = n_power // 2

            for p_idx in range(n_power):
                pw = power_levels[p_idx]
                e = pw * dt
                if e > 0:
                    sc = -e / cap_mwh / eta_d
                elif e < 0:
                    sc = -e * eta_c / cap_mwh
                else:
                    sc = 0.0
                ns = soc + sc
                if ns < battery.min_soc - 0.001 or ns > battery.max_soc + 0.001:
                    continue
                ns = np.clip(ns, battery.min_soc, battery.max_soc)
                ns_idx = int(np.clip(
                    round((ns - battery.min_soc) / (battery.max_soc - battery.min_soc) * (n_soc - 1)),
                    0, n_soc - 1))

                # 当前动作在各场景下的价值
                scenario_values = np.zeros(n_scenarios)
                for k in range(n_scenarios):
                    price_k = scenarios_prices[k, t]
                    reward_k = e * price_k - abs(e) * deg + V_all[k, t + 1, ns_idx]
                    scenario_values[k] = reward_k

                # CVaR 惩罚：最坏 alpha 尾部
                _, cvar_val = compute_cvar(scenario_values, weights, alpha)
                expected = float(np.sum(scenario_values * weights))

                # 目标：max E - λ × (E - CVaR)（鼓励尾部接近期望）
                # 等价：max (1-λ) × E + λ × CVaR
                obj = (1 - cvar_weight) * expected + cvar_weight * cvar_val

                if obj > best_obj:
                    best_obj = obj
                    best_p = p_idx

            policy[t][s_idx] = best_p

    # 评估
    final_values = np.zeros(n_scenarios)
    for k in range(n_scenarios):
        # 用 policy 在场景 k 下前向仿真
        soc = init_soc
        rev = 0.0
        for t in range(n_steps):
            si = int(np.clip(
                round((soc - battery.min_soc) / (battery.max_soc - battery.min_soc) * (n_soc - 1)),
                0, n_soc - 1))
            pw = power_levels[policy[t][si]]
            e = pw * dt
            if e > 0:
                sc = -e / cap_mwh / eta_d
            elif e < 0:
                sc = -e * eta_c / cap_mwh
            else:
                sc = 0.0
            soc = np.clip(soc + sc, battery.min_soc, battery.max_soc)
            rev += e * scenarios_prices[k, t] - abs(e) * deg
        final_values[k] = rev

    var, cvar = compute_cvar(final_values, weights, alpha)
    expected = float(np.sum(final_values * weights))

    info = {
        "expected_revenue": expected,
        "var": var,
        "cvar": cvar,
        "scenario_values": final_values,
    }

    return policy, info


if __name__ == "__main__":
    from loguru import logger
    np.random.seed(42)

    battery = BatteryConfig()
    power_levels = np.arange(-200, 201, 10.0)
    soc_levels = np.linspace(0.05, 0.95, 15)

    # 生成 5 个场景（不同价格形态）
    n_scenarios = 5
    hours = np.arange(96) / 4
    base = 320 - 80 * np.cos(2 * np.pi * hours / 24)
    scenarios = np.zeros((n_scenarios, 96))
    for k in range(n_scenarios):
        volatility = 20 + k * 30  # 越来越波动
        scenarios[k] = base + np.random.normal(0, volatility, 96)

    # 均匀权重
    weights = np.ones(n_scenarios) / n_scenarios

    # 风险中性 (cvar_weight=0)
    policy_rn, info_rn = solve_cvar_dp(
        scenarios, weights, battery, power_levels, soc_levels,
        alpha=0.05, cvar_weight=0.0)
    logger.info(f"风险中性 (λ=0):")
    logger.info(f"  E[rev] = ¥{info_rn['expected_revenue']:,.0f}")
    logger.info(f"  CVaR = ¥{info_rn['cvar']:,.0f}")

    # CVaR 风险厌恶
    policy_cv, info_cv = solve_cvar_dp(
        scenarios, weights, battery, power_levels, soc_levels,
        alpha=0.05, cvar_weight=0.3)
    logger.info(f"\nCVaR 风险厌恶 (λ=0.3):")
    logger.info(f"  E[rev] = ¥{info_cv['expected_revenue']:,.0f}")
    logger.info(f"  CVaR = ¥{info_cv['cvar']:,.0f}")

    diff_e = (info_cv['expected_revenue'] - info_rn['expected_revenue']) / info_rn['expected_revenue'] * 100
    diff_c = (info_cv['cvar'] - info_rn['cvar']) / abs(info_rn['cvar']) * 100
    logger.info(f"\n对比：")
    logger.info(f"  期望收益：  {diff_e:+.1f}%  (可能略降)")
    logger.info(f"  CVaR：     {diff_c:+.1f}%  (应显著提升)")
