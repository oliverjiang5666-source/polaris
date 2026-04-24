"""
AGC 调频联合优化 (Modification #3)

决策维度：
- 能量套利（充/放/等待 × 功率）
- AGC 容量承诺（0/50/100 MW）—— 占用电池功率但产生容量费+里程费

关键约束：
- |能量套利功率| + AGC容量 ≤ 电池额定功率
- AGC 响应会影响 SoC（典型 ±5% 容量效应/时段）
- SoC 上下限同时考虑能量套利和 AGC 干扰

经济模型：
- 容量价（元/MW/h）：山东历史 5-15
- 里程价（元/MW·km）：5-12
- 期望里程（km/MW/h）：2-5

注：真实 AGC 数据需要从山东交易中心爬取。此处默认参数是典型值，
可在 config 中替换为实际值。

Usage:
    from optimization.agc_dp import solve_day_with_agc, AGCConfig
    agc = AGCConfig(capacity_price=8.0, mileage_price=6.0)
    result = solve_day_with_agc(dam_prices, battery, agc_config=agc)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from config import BatteryConfig


@dataclass
class AGCConfig:
    """AGC 市场参数（省际差异大，需替换为实际值）"""
    capacity_price_yuan_per_mw_h: float = 8.0   # 容量价：5-15 典型
    mileage_price_yuan_per_mw_km: float = 6.0   # 里程价：5-12 典型
    expected_mileage_per_mw_h: float = 3.0      # 期望里程量
    soc_impact_per_mw_h: float = 0.015          # SoC 扰动（相对容量的比例）
    min_agc_block_mw: float = 10.0              # 最小 AGC 竞标单位
    max_agc_mw: float = 100.0                   # 最大 AGC 容量（受电池能力限制）

    def per_step_revenue(self, agc_mw: float, dt: float) -> float:
        """每步 AGC 收入 = 容量费 + 里程费"""
        if agc_mw <= 0:
            return 0.0
        capacity_rev = agc_mw * self.capacity_price_yuan_per_mw_h * dt
        mileage_rev = agc_mw * self.expected_mileage_per_mw_h * self.mileage_price_yuan_per_mw_km * dt
        return capacity_rev + mileage_rev


def solve_day_with_agc(
    dam_prices: np.ndarray,
    battery: BatteryConfig,
    agc_config: AGCConfig | None = None,
    agc_levels_mw: list[float] | None = None,
    init_soc: float = 0.5,
    deg: float = 2.0,
) -> dict:
    """
    DP 求解：能量套利 + AGC 联合日调度。

    决策空间：
        每时段选 (power_level, agc_level)
        power_level: ±200 MW, 步长 20 MW（11 个值）
        agc_level: 0, 50, 100 MW（3 个值）
        总计 33 种组合（但约束过滤后实际更少）

    约束：
        |power| + agc_level ≤ capacity_mw
        SoC + agc_induced_soc_change ∈ [min, max]

    目标：
        max Σ [power × dam_price + agc_revenue] × dt - degradation
    """
    if agc_config is None:
        agc_config = AGCConfig()
    if agc_levels_mw is None:
        agc_levels_mw = [0.0, 50.0, 100.0]

    n = len(dam_prices)
    dt = battery.interval_hours
    cap_mw = battery.capacity_mw
    cap_mwh = battery.capacity_mwh
    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency

    # 功率级别（能量套利）
    power_levels = np.arange(-cap_mw, cap_mw + 1, 20.0)  # ±200 步 20
    n_soc = 20
    soc_levels = np.linspace(battery.min_soc, battery.max_soc, n_soc)

    n_power = len(power_levels)
    n_agc = len(agc_levels_mw)

    # DP：V[t, soc_idx] = 从 t 时刻开始的最优未来价值
    V = np.zeros((n + 1, n_soc))
    best_p_idx = np.zeros((n, n_soc), dtype=np.int32)
    best_agc_idx = np.zeros((n, n_soc), dtype=np.int32)

    for t in range(n - 1, -1, -1):
        price = dam_prices[t]
        for s_idx in range(n_soc):
            soc = soc_levels[s_idx]
            best_val = -1e18
            best_p, best_a = n_power // 2, 0

            for p_idx in range(n_power):
                pw = power_levels[p_idx]
                e = pw * dt

                if e > 0:  # 放电
                    soc_change_arb = -e / cap_mwh / eta_d
                elif e < 0:  # 充电
                    soc_change_arb = -e * eta_c / cap_mwh
                else:
                    soc_change_arb = 0.0

                for a_idx in range(n_agc):
                    agc_mw = agc_levels_mw[a_idx]

                    # 功率约束
                    if abs(pw) + agc_mw > cap_mw + 1e-6:
                        continue

                    # AGC 的 SoC 扰动（对称，中性）
                    # 保守建模：AGC 占用 SoC buffer（±soc_impact）
                    agc_soc_buffer = agc_mw * agc_config.soc_impact_per_mw_h * dt / cap_mwh

                    # 期望 SoC 变化（AGC 长期期望为 0，但需要保留 buffer）
                    new_soc = soc + soc_change_arb

                    # SoC 约束：必须留 buffer 给 AGC 动作
                    if new_soc < battery.min_soc + agc_soc_buffer - 1e-4:
                        continue
                    if new_soc > battery.max_soc - agc_soc_buffer + 1e-4:
                        continue

                    new_soc = np.clip(new_soc, battery.min_soc, battery.max_soc)
                    ns_idx = int(np.clip(
                        round((new_soc - battery.min_soc) / (battery.max_soc - battery.min_soc) * (n_soc - 1)),
                        0, n_soc - 1))

                    arb_reward = e * price - abs(e) * deg
                    agc_reward = agc_config.per_step_revenue(agc_mw, dt)

                    total = arb_reward + agc_reward + V[t + 1][ns_idx]

                    if total > best_val:
                        best_val = total
                        best_p = p_idx
                        best_a = a_idx

            V[t][s_idx] = best_val
            best_p_idx[t][s_idx] = best_p
            best_agc_idx[t][s_idx] = best_a

    # 前向执行
    soc = init_soc
    actual_powers = np.zeros(n)
    actual_agc = np.zeros(n)
    arb_rev = 0.0
    agc_rev = 0.0
    total_deg = 0.0

    for t in range(n):
        si = int(np.clip(
            round((soc - battery.min_soc) / (battery.max_soc - battery.min_soc) * (n_soc - 1)),
            0, n_soc - 1))
        p_idx = best_p_idx[t][si]
        a_idx = best_agc_idx[t][si]
        pw = power_levels[p_idx]
        agc_mw = agc_levels_mw[a_idx]

        e = pw * dt
        if e > 0:
            sc = -e / cap_mwh / eta_d
        elif e < 0:
            sc = -e * eta_c / cap_mwh
        else:
            sc = 0.0

        soc = np.clip(soc + sc, battery.min_soc, battery.max_soc)

        actual_powers[t] = pw
        actual_agc[t] = agc_mw
        arb_rev += e * dam_prices[t]
        agc_rev += agc_config.per_step_revenue(agc_mw, dt)
        total_deg += abs(e) * deg

    return {
        "net_power": actual_powers,
        "agc_capacity": actual_agc,
        "arb_revenue": float(arb_rev),
        "agc_revenue": float(agc_rev),
        "degradation": float(total_deg),
        "revenue": float(arb_rev + agc_rev - total_deg),
        "final_soc": soc,
    }


if __name__ == "__main__":
    from loguru import logger
    from oracle.lp_oracle import solve_day

    np.random.seed(42)
    battery = BatteryConfig()

    # Mock DAM 价
    hours = np.arange(96) / 4
    dam_prices = 320 - 80 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 20, 96)

    # Test 1: 无 AGC（设 capacity_price=0）
    agc_zero = AGCConfig(capacity_price_yuan_per_mw_h=0, mileage_price_yuan_per_mw_km=0)
    result_no_agc = solve_day_with_agc(dam_prices, battery, agc_config=agc_zero)
    baseline = solve_day(dam_prices, battery)
    logger.info(f"无 AGC DP: ¥{result_no_agc['revenue']:,.0f}")
    logger.info(f"LP baseline: ¥{baseline['revenue']:,.0f}")
    logger.info(f"  差异 {(result_no_agc['revenue'] / baseline['revenue'] - 1)*100:+.1f}% "
                f"(DP 粒度化损失)")

    # Test 2: 典型 AGC 参数
    agc_normal = AGCConfig()
    result_agc = solve_day_with_agc(dam_prices, battery, agc_config=agc_normal)
    logger.info(f"\n有 AGC DP: ¥{result_agc['revenue']:,.0f}")
    logger.info(f"  套利收入: ¥{result_agc['arb_revenue']:,.0f}")
    logger.info(f"  AGC 收入: ¥{result_agc['agc_revenue']:,.0f}")
    logger.info(f"  降解: ¥{result_agc['degradation']:,.0f}")
    logger.info(f"  AGC 占用时段: {(result_agc['agc_capacity'] > 0).sum()}/96")
    logger.info(f"  增量 vs 无 AGC: +{(result_agc['revenue'] - result_no_agc['revenue']):,.0f} "
                f"({(result_agc['revenue']/result_no_agc['revenue']-1)*100:+.1f}%)")
