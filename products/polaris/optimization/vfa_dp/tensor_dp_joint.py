"""
Tensor DP 联合优化: 电能量套利 + AGC 调频预留
=================================================

扩展 Lee & Sun 2025 Algorithm 1，在 action 空间上加 AGC 上调/下调容量预留。

State: SoC (1D, discretized to S points)
Action: (p^arb, c^up, c^down)
  p^arb ∈ action_grid (原 Tensor DP 的离散功率网格)
  c^up  ∈ agc_up_grid   (离散档, e.g. {0, 50, 100} MW)
  c^down ∈ agc_down_grid (离散档, e.g. {0, 50, 100} MW)

约束:
  |p^arb| + c^up + c^down ≤ P_max
  SoC transition 只由 p^arb 决定 (AGC 期望里程 ≈ 0, 对 SoC 期望影响 ≈ 0)
  但 AGC 需要预留 SoC buffer (见下)

Reward per step:
  R_arb = p^arb × λ_t × dt                          # 电能量套利（按节点 LMP）
  R_agc = (c^up + c^down) × m × K_pd × Y_AGC × dt   # 调频收入 (§14.8.3)
  Cost_deg = deg × (|p^arb| + c^up·m·σ_up + c^down·m·σ_down) × dt
  Q = R_arb + R_agc − Cost_deg + V^next(SoC + Δ)

其中:
  m = 期望调频里程系数 (MW·mileage/MW·h)
  K_pd = 调节性能指标
  Y_AGC = AGC 出清价
  σ = 里程→吞吐量损耗系数 (典型 0.5-1.0, 默认 1.0)

SoC buffer 约束 (§11.2.3):
  储能参与调频需预留 SoC 电量
  SoC - SoC_min ≥ c^up × T_reserve / cap_mwh   (上调频预留电量下限)
  SoC_max - SoC ≥ c^down × T_reserve / cap_mwh  (下调频预留电量上限)
  T_reserve ≈ 0.5-1 小时 (调频事件期望持续时间)

论文 Lee-Sun 2025 只做 1D action (p^arb)，本实现扩展到 3D。
复杂度: Tensor DP 每步 O(S × P × R) → 本版 O(S × P × |AGC_up| × |AGC_down| × R)

Usage:
    from optimization.vfa_dp.tensor_dp_joint import TensorDPJoint, JointDPConfig
    dp = TensorDPJoint(battery, JointDPConfig(delta_soc=0.01))
    V = dp.backward_induction(price_scen, price_prob, agc_price_scen, agc_price_prob)
    sim = dp.forward_simulate(V, actual_prices, actual_agc_prices, init_soc=0.5)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class JointDPConfig:
    """TensorDP 联合优化配置"""
    delta_soc: float = 0.01                         # SoC 网格粒度
    deg_cost: float = 2.0                           # 降解成本 ¥/MWh throughput

    # AGC 离散档位（上调 / 下调）
    agc_up_levels_mw: tuple[float, ...] = (0.0, 50.0, 100.0)
    agc_down_levels_mw: tuple[float, ...] = (0.0, 50.0, 100.0)

    # AGC 参数（期望，用于 DP 计算）
    agc_mileage_per_mw_h: float = 3.0               # 期望调频里程
    agc_performance_coeff: float = 0.95             # K_pd
    agc_soc_reserve_hours: float = 0.5              # 调频需预留的 SoC 折算小时
    agc_throughput_coeff: float = 1.0               # 里程→降解吞吐转换系数

    # 终态 SoC
    final_soc_penalty: float = 0.0
    final_soc_target: float = 0.5


class TensorDPJoint:
    """
    联合 AGC 的 Tensor DP。
    """

    def __init__(self, battery, config: Optional[JointDPConfig] = None, dt: float = 0.25):
        self.battery = battery
        self.config = config or JointDPConfig()
        self.dt = dt

        self.p_max = battery.capacity_mw
        self.cap_mwh = battery.capacity_mwh
        self.eta_c = battery.charge_efficiency
        self.eta_d = battery.discharge_efficiency
        self.soc_min = battery.min_soc
        self.soc_max = battery.max_soc

        self._build_state_grid()
        self._build_action_grid()
        self._precompute_transitions()

        logger.info(
            f"TensorDPJoint init: S={self.S}, P_arb={self.P}, "
            f"|AGC_up|={len(self.config.agc_up_levels_mw)}, "
            f"|AGC_down|={len(self.config.agc_down_levels_mw)}, "
            f"total action tuples = {self.P * len(self.config.agc_up_levels_mw) * len(self.config.agc_down_levels_mw)}"
        )

    def _build_state_grid(self):
        delta = self.config.delta_soc
        n = int(round((self.soc_max - self.soc_min) / delta)) + 1
        self.state_grid = np.linspace(self.soc_min, self.soc_max, n)
        self.S = n

    def _build_action_grid(self):
        """套利功率网格（和原 Tensor DP 一致）"""
        delta = self.config.delta_soc
        power_step_charge = delta * self.cap_mwh / (self.dt * self.eta_c)
        power_step_discharge = delta * self.cap_mwh * self.eta_d / self.dt

        n_dis = int(np.ceil(self.p_max / power_step_discharge))
        n_chg = int(np.ceil(self.p_max / power_step_charge))

        neg_powers = [-power_step_charge * i for i in range(1, n_chg)]
        neg_powers.append(-self.p_max)
        neg_powers.reverse()

        pos_powers = [power_step_discharge * j for j in range(1, n_dis)]
        pos_powers.append(self.p_max)

        self.action_grid = np.array(neg_powers + [0.0] + pos_powers)
        self.P = len(self.action_grid)

    def _precompute_transitions(self):
        delta_soc = np.where(
            self.action_grid >= 0,
            -self.action_grid * self.dt / (self.eta_d * self.cap_mwh),
            -self.action_grid * self.dt * self.eta_c / self.cap_mwh,
        )
        sigma = self.state_grid[:, None] + delta_soc[None, :]
        infeasible = (sigma < self.soc_min - 1e-9) | (sigma > self.soc_max + 1e-9)
        sigma_clipped = np.clip(sigma, self.soc_min, self.soc_max)
        z = (sigma_clipped - self.soc_min) / self.config.delta_soc
        z_low = np.floor(z).astype(np.int64)
        z_high = np.minimum(z_low + 1, self.S - 1)
        b = z - z_low
        b = np.where(z_low == z_high, 0.0, b)

        self._sigma = sigma
        self._z_low = z_low
        self._z_high = z_high
        self._b = b
        self._infeasible = infeasible

    # ============================================================
    # AGC feasibility: SoC buffer + 功率 + 额定 AGC
    # ============================================================

    def _agc_feasibility_mask(
        self,
        soc_vec: np.ndarray,                        # (S,)
        agc_up: float,
        agc_down: float,
        action_abs: np.ndarray,                     # (P,) |p^arb|
    ) -> np.ndarray:
        """
        返回 (S, P) boolean: 某状态下某 p^arb 动作在 (agc_up, agc_down) 预留下是否可行

        约束:
          1. |p^arb| + agc_up + agc_down ≤ P_max
          2. SoC - soc_min ≥ agc_up × T_reserve / cap_mwh
          3. soc_max - SoC ≥ agc_down × T_reserve / cap_mwh
        """
        cfg = self.config

        # 约束 1 (功率): action-dependent
        power_ok = (action_abs + agc_up + agc_down) <= self.p_max + 1e-6  # (P,)

        # 约束 2-3 (SoC buffer): state-dependent
        soc_buffer_up = agc_up * cfg.agc_soc_reserve_hours / self.cap_mwh
        soc_buffer_down = agc_down * cfg.agc_soc_reserve_hours / self.cap_mwh

        up_ok = (soc_vec - self.soc_min) >= soc_buffer_up - 1e-6        # (S,)
        down_ok = (self.soc_max - soc_vec) >= soc_buffer_down - 1e-6     # (S,)

        state_ok = up_ok & down_ok                                        # (S,)

        return state_ok[:, None] & power_ok[None, :]                      # (S, P)

    # ============================================================
    # Backward induction
    # ============================================================

    def backward_induction(
        self,
        price_scenarios: np.ndarray,                # (T, R) LMP 场景
        price_probs: np.ndarray,                    # (T, R)
        agc_clearing_price: float = 6.0,            # 标量或未来升级为 (T,) 场景
    ) -> np.ndarray:
        """
        联合 backward induction.

        Returns:
            V: (T+1, S)
        """
        cfg = self.config
        T, R = price_scenarios.shape
        probs = price_probs / price_probs.sum(axis=1, keepdims=True)

        V = np.zeros((T + 1, self.S), dtype=np.float64)
        if cfg.final_soc_penalty > 0:
            below = np.maximum(0.0, cfg.final_soc_target - self.state_grid)
            V[T, :] = -cfg.final_soc_penalty * below

        NEG_INF = -1e18
        abs_action = np.abs(self.action_grid)
        deg_arb = cfg.deg_cost * abs_action * self.dt       # (P,)

        agc_up_arr = np.asarray(cfg.agc_up_levels_mw)
        agc_dn_arr = np.asarray(cfg.agc_down_levels_mw)
        n_up, n_dn = len(agc_up_arr), len(agc_dn_arr)

        for t in range(T - 1, -1, -1):
            # V next interp (S, P)
            V_next_interp = (1 - self._b) * V[t + 1, self._z_low] + \
                            self._b * V[t + 1, self._z_high]
            V_next_interp = np.where(self._infeasible, NEG_INF, V_next_interp)

            # 套利 payoff per (j, r)
            payoff_arb = self.action_grid[:, None] * price_scenarios[t][None, :] * self.dt  # (P, R)
            payoff_arb = payoff_arb - deg_arb[:, None]                                      # (P, R)

            # 对每个 (up, down) AGC 组合评估
            best_V_per_state = np.full(self.S, NEG_INF)

            for iu, up in enumerate(agc_up_arr):
                for idn, dn in enumerate(agc_dn_arr):
                    # AGC 收入（不依赖场景 r, 固定值）
                    mileage = (up + dn) * cfg.agc_mileage_per_mw_h * self.dt
                    agc_rev = mileage * cfg.agc_performance_coeff * agc_clearing_price
                    agc_deg = cfg.deg_cost * (up + dn) * cfg.agc_mileage_per_mw_h * self.dt * cfg.agc_throughput_coeff

                    # Feasibility mask (S, P)
                    feas = self._agc_feasibility_mask(self.state_grid, up, dn, abs_action)

                    # Q_all = V_next + payoff_arb + (agc_rev - agc_deg)   for each (S, P, R)
                    payoff_adjusted = payoff_arb + (agc_rev - agc_deg)      # (P, R)
                    Q_all = V_next_interp[:, :, None] + payoff_adjusted[None, :, :]  # (S, P, R)

                    # Mask infeasible (S, P)
                    Q_all = np.where(feas[:, :, None], Q_all, NEG_INF)

                    # Q_ir = max over p^arb
                    Q_ir = Q_all.max(axis=1)                                 # (S, R)

                    # Expected over scenarios
                    V_this = Q_ir @ probs[t]                                 # (S,)

                    best_V_per_state = np.maximum(best_V_per_state, V_this)

            V[t, :] = best_V_per_state

        return V

    # ============================================================
    # Forward simulation
    # ============================================================

    def forward_simulate(
        self,
        V: np.ndarray,                              # (T+1, S)
        actual_prices: np.ndarray,                  # (T,) 真实 LMP
        actual_agc_prices: np.ndarray | None = None,  # (T,) 真实 AGC 出清价
        init_soc: float = 0.5,
    ) -> dict:
        """
        贪心 forward simulation. 每步基于 V[t+1] 选 (p^arb, c^up, c^down)。
        """
        cfg = self.config
        T = len(actual_prices)
        if actual_agc_prices is None:
            actual_agc_prices = np.full(T, 6.0)

        soc = init_soc
        powers = np.zeros(T)
        agc_up_taken = np.zeros(T)
        agc_down_taken = np.zeros(T)
        rewards = np.zeros(T)
        soc_traj = [soc]

        abs_action = np.abs(self.action_grid)
        deg_arb = cfg.deg_cost * abs_action * self.dt
        agc_up_arr = np.asarray(cfg.agc_up_levels_mw)
        agc_dn_arr = np.asarray(cfg.agc_down_levels_mw)

        for t in range(T):
            # 每个 (action, up, down) 的 Q
            immediate_arb = self.action_grid * actual_prices[t] * self.dt - deg_arb  # (P,)

            delta_soc = np.where(
                self.action_grid >= 0,
                -self.action_grid * self.dt / (self.eta_d * self.cap_mwh),
                -self.action_grid * self.dt * self.eta_c / self.cap_mwh,
            )
            next_soc = soc + delta_soc
            infeasible_arb = (next_soc < self.soc_min - 1e-9) | (next_soc > self.soc_max + 1e-9)

            next_soc_clipped = np.clip(next_soc, self.soc_min, self.soc_max)
            z = (next_soc_clipped - self.soc_min) / cfg.delta_soc
            z_low = np.floor(z).astype(np.int64)
            z_high = np.minimum(z_low + 1, self.S - 1)
            b = z - z_low
            b = np.where(z_low == z_high, 0.0, b)
            V_next = (1 - b) * V[t + 1, z_low] + b * V[t + 1, z_high]   # (P,)

            best_Q = -1e18
            best_j, best_up, best_dn = 0, 0.0, 0.0

            for up in agc_up_arr:
                for dn in agc_dn_arr:
                    # Buffer check
                    if (soc - self.soc_min) < up * cfg.agc_soc_reserve_hours / self.cap_mwh - 1e-6:
                        continue
                    if (self.soc_max - soc) < dn * cfg.agc_soc_reserve_hours / self.cap_mwh - 1e-6:
                        continue

                    mileage = (up + dn) * cfg.agc_mileage_per_mw_h * self.dt
                    agc_rev = mileage * cfg.agc_performance_coeff * actual_agc_prices[t]
                    agc_deg = cfg.deg_cost * (up + dn) * cfg.agc_mileage_per_mw_h * self.dt * cfg.agc_throughput_coeff

                    # 功率约束: |p^arb| + up + dn ≤ P_max
                    power_ok_vec = (abs_action + up + dn) <= self.p_max + 1e-6   # (P,)
                    feasible = power_ok_vec & ~infeasible_arb

                    if not feasible.any():
                        continue

                    Q_vec = immediate_arb + V_next + (agc_rev - agc_deg)
                    Q_vec = np.where(feasible, Q_vec, -1e18)

                    j = int(np.argmax(Q_vec))
                    if Q_vec[j] > best_Q:
                        best_Q = Q_vec[j]
                        best_j = j
                        best_up = float(up)
                        best_dn = float(dn)

            p_star = float(self.action_grid[best_j])

            # 物理更新 SoC（边界截断）
            if p_star >= 0:
                delta = -p_star * self.dt / (self.eta_d * self.cap_mwh)
            else:
                delta = -p_star * self.dt * self.eta_c / self.cap_mwh
            new_soc = soc + delta
            if new_soc > self.soc_max:
                actual_delta = self.soc_max - soc
                p_star = -actual_delta * self.cap_mwh / (self.dt * self.eta_c)
                new_soc = self.soc_max
            elif new_soc < self.soc_min:
                actual_delta = self.soc_min - soc
                p_star = -actual_delta * self.cap_mwh * self.eta_d / self.dt
                new_soc = self.soc_min

            # 结算
            arb_reward = p_star * actual_prices[t] * self.dt - cfg.deg_cost * abs(p_star) * self.dt
            mileage = (best_up + best_dn) * cfg.agc_mileage_per_mw_h * self.dt
            agc_reward = mileage * cfg.agc_performance_coeff * actual_agc_prices[t]
            agc_deg_cost = cfg.deg_cost * (best_up + best_dn) * cfg.agc_mileage_per_mw_h * self.dt * cfg.agc_throughput_coeff

            powers[t] = p_star
            agc_up_taken[t] = best_up
            agc_down_taken[t] = best_dn
            rewards[t] = arb_reward + agc_reward - agc_deg_cost
            soc = new_soc
            soc_traj.append(soc)

        return {
            "revenue_total": float(rewards.sum()),
            "revenue_arbitrage": float((powers * actual_prices * self.dt - cfg.deg_cost * np.abs(powers) * self.dt).sum()),
            "revenue_agc": float(rewards.sum() - (powers * actual_prices * self.dt - cfg.deg_cost * np.abs(powers) * self.dt).sum()),
            "powers": powers,
            "agc_up": agc_up_taken,
            "agc_down": agc_down_taken,
            "soc_trajectory": np.array(soc_traj),
            "rewards": rewards,
            "final_soc": soc,
        }


if __name__ == "__main__":
    """Sanity check: AGC=0 时应该和原 Tensor DP 结果一致"""
    from config import BatteryConfig
    from oracle.lp_oracle import solve_day
    import time

    battery = BatteryConfig()

    np.random.seed(42)
    hours = np.arange(96) / 4
    base = 320 - 80 * np.cos(2 * np.pi * hours / 24)
    prices = base + np.random.normal(0, 20, 96)

    lp_result = solve_day(prices, battery, init_soc=0.5)
    logger.info(f"LP Oracle: ¥{lp_result['revenue']:,.2f}")

    # AGC 关掉（levels = [0]）
    dp = TensorDPJoint(
        battery,
        JointDPConfig(
            delta_soc=0.01,
            agc_up_levels_mw=(0.0,),
            agc_down_levels_mw=(0.0,),
        ),
    )
    t0 = time.time()
    V = dp.backward_induction(prices[:, None], np.ones((96, 1)))
    sim = dp.forward_simulate(V, prices, init_soc=0.5)
    elapsed = time.time() - t0
    gap = (sim["revenue_total"] - lp_result["revenue"]) / lp_result["revenue"] * 100
    logger.info(f"Joint DP (AGC=0): ¥{sim['revenue_total']:,.2f}, gap {gap:+.3f}%, time {elapsed:.2f}s")

    # AGC 开启（典型 levels）
    dp2 = TensorDPJoint(
        battery,
        JointDPConfig(delta_soc=0.01),    # default AGC levels {0, 50, 100}
    )
    t0 = time.time()
    V2 = dp2.backward_induction(prices[:, None], np.ones((96, 1)), agc_clearing_price=6.0)
    sim2 = dp2.forward_simulate(V2, prices, actual_agc_prices=np.full(96, 6.0), init_soc=0.5)
    elapsed = time.time() - t0
    logger.info(
        f"Joint DP (AGC on): ¥{sim2['revenue_total']:,.2f} "
        f"(arb=¥{sim2['revenue_arbitrage']:,.0f}, agc=¥{sim2['revenue_agc']:,.0f}), "
        f"time {elapsed:.2f}s"
    )
    logger.info(
        f"  AGC up 平均:   {sim2['agc_up'].mean():.1f} MW, "
        f"AGC down 平均: {sim2['agc_down'].mean():.1f} MW"
    )
