"""
Tensor-based stochastic DP for energy storage arbitrage
========================================================

实现 Lee & Sun 2025 (arXiv:2511.15629) Algorithm 1 的 NumPy CPU 版本。

核心思想（和 Regime V3 对比）：
    Regime V3:    先把场景 expectation 合并到 price 上，再做 DP（deterministic DP on E[price]）
    Lee-Sun 2025: 每时段保留 R 个价格支撑点，DP 里做真正的 E_ξ[max_a Q]（stagewise stochastic DP）

Jensen: E[max_a f(a, ξ)] ≥ max_a f(a, E[ξ])，所以 Lee-Sun ≥ Regime V3 在数学上保证。
实际差多少取决于价格波动和 payoff 凸性——这是要实测的。

数据结构（论文 Eq 12-14）：
    state grid     ŝ ∈ R^S:  SoC 离散化成 S 个点
    action grid    p̂ ∈ R^P:  功率离散化，设计上让 state transition 尽量落在整数格点
    scenarios     λ̂ ∈ R^(T,R):  每时段 R 个价格支撑点
    probabilities π̂ ∈ R^(T,R):  每支撑点的概率，Σ_r π̂[t,r]=1

Algorithm 1 (论文 §III-B)：
    Input:  ŝ, p̂, {λ̂_t, π̂_t}_{t=1}^T
    Output: V̂_t ∈ R^S for t = 0, ..., T

    预计算（constant across t）：
      σ[i, j]  = ŝ[i] + F(p̂[j])      # (S, P) 原始 next-state levels（可能越界）
      z[i, j]  = clip(σ, 0, s̄) / δ + 1  # next-state grid 上的连续坐标
      z-, z+   = ⌊z⌋, ⌈z⌉             # 整数邻居
      b        = (z - z-)/(z+ - z-)   # 插值系数

    Backward:
      V̂_T = 0
      for t = T, ..., 1:
          V̂_t^next[i, j] = (1-b)·V̂_t[z-] + b·V̂_t[z+]    # linear interp on next V
          V̂_t^next[infeasible σ] = -∞
          Q̂_all[i, j, r] = p̂[j]·λ̂[t, r] + V̂_t^next[i, j]  # (S, P, R)
          Q̂[i, r] = max_j Q̂_all[i, j, r]                  # (S, R)
          V̂_{t-1}[i] = Σ_r Q̂[i, r]·π̂[t, r]                # (S,)

关键贡献点复现：
  1. 离散 DP 的 tensor 化（全 vectorized，无 Python 双重循环）
  2. Action grid 设计（Eq 13）：动作步长精心选择，让 s + F(p) 落在格点上
  3. 可选 convexification → bid curve（论文 §II-C，后续 bid_curve.py 实现）

与我们场景的适配：
  - 电池：200 MW × 400 MWh = 2h duration
  - dt = 0.25 h
  - η_c = η_d = 0.9487 (round-trip 0.9)
  - degradation cost = ¥2/MWh throughput
  - SoC ∈ [0.05, 0.95]

我们用 SoC 作 state（而不是论文的 "depth of discharge"），但算法数学等价。

Usage:
    from optimization.vfa_dp.tensor_dp import TensorDP
    from config import BatteryConfig

    battery = BatteryConfig()
    dp = TensorDP(battery, delta_soc=0.02)
    V = dp.backward_induction(price_scenarios, price_probs)
    revenue, actions = dp.forward_simulate(V, actual_prices, init_soc=0.5)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class DPConfig:
    """TensorDP 配置参数"""
    delta_soc: float = 0.02           # SoC 网格粒度（0.02 = 2%）
    power_step_divisor: int = 1       # Action grid 是否加密（1 = 按 Eq 13 自动）
    deg_cost: float = 2.0             # 降解成本 ¥/MWh throughput
    final_soc_penalty: float = 0.0    # 终态 SoC 惩罚权重（0 = 不约束）
    final_soc_target: float = 0.3     # 终态目标 SoC（若惩罚 > 0）


class TensorDP:
    """
    Tensor-based stochastic DP solver。
    """

    def __init__(self, battery, config: Optional[DPConfig] = None, dt: float = 0.25):
        """
        Args:
            battery: BatteryConfig 对象（我们项目的）
            config: DPConfig，默认用 delta_soc=0.02
            dt: 时间步长（小时）
        """
        self.battery = battery
        self.config = config or DPConfig()
        self.dt = dt

        # 电池参数
        self.p_max = battery.capacity_mw
        self.cap_mwh = battery.capacity_mwh
        self.eta_c = battery.charge_efficiency
        self.eta_d = battery.discharge_efficiency
        self.soc_min = battery.min_soc
        self.soc_max = battery.max_soc

        # 构建 state grid 和 action grid
        self._build_state_grid()
        self._build_action_grid()

        # 预计算 transition 索引（constant across t）
        self._precompute_transitions()

        logger.info(f"TensorDP init: S={self.S} states (δ_soc={self.config.delta_soc}), "
                    f"P={self.P} actions, feasibility matrix: {self.S * self.P} pairs")

    # ============================================================
    # 网格构建
    # ============================================================

    def _build_state_grid(self):
        """State grid: SoC ∈ [soc_min, soc_max]，步长 δ_soc"""
        delta = self.config.delta_soc
        # 包含端点
        n = int(round((self.soc_max - self.soc_min) / delta)) + 1
        self.state_grid = np.linspace(self.soc_min, self.soc_max, n)  # shape (S,)
        self.S = n

    def _build_action_grid(self):
        """
        Action grid: power ∈ [-p_max, p_max]
        按论文 Eq 13：action 步长设计让 transition 落在 state grid 上
          n^c = ⌈p̄ η_c δ_SoC^-1 (dt/cap)⁻¹⌉  (充电方向)
          n^d = ⌈p̄ / η_d × δ_SoC^-1 (dt/cap)⁻¹⌉ (放电方向)

        换算到 SoC 空间，每个 grid 点的 SoC 变化量：
            charge:   ΔSoC = p·dt·η_c / cap_mwh
            discharge: ΔSoC = -p·dt / (η_d × cap_mwh)
        要让 ΔSoC 恰好是 δ_soc 的整数倍：
            充电 power step: δ_soc × cap_mwh / (dt × η_c)
            放电 power step: δ_soc × cap_mwh × η_d / dt
        """
        delta = self.config.delta_soc

        # SoC 粒度换算成充/放电功率步长
        power_step_charge = delta * self.cap_mwh / (self.dt * self.eta_c)      # 每步能抬多少 SoC 所需功率
        power_step_discharge = delta * self.cap_mwh * self.eta_d / self.dt     # 每步放多少 SoC 所需功率

        # 充电方向步数（正功率 = 放电，负功率 = 充电；power 约束 p ∈ [-p_max, p_max]）
        # 我们用 power 正=放电 的惯例（和我们代码一致）
        n_dis = int(np.ceil(self.p_max / power_step_discharge))  # 放电步数
        n_chg = int(np.ceil(self.p_max / power_step_charge))     # 充电步数

        # 充电侧：-power_step_charge × i for i=1..n_chg，加入 -p_max endpoint
        # 放电侧：+power_step_discharge × j for j=1..n_dis，加入 +p_max endpoint
        # 中间：0 (待机)

        neg_powers = [-power_step_charge * i for i in range(1, n_chg)]  # 不到 -p_max
        neg_powers.append(-self.p_max)
        neg_powers.reverse()  # 从最小到 0

        pos_powers = [power_step_discharge * j for j in range(1, n_dis)]
        pos_powers.append(self.p_max)

        self.action_grid = np.array(neg_powers + [0.0] + pos_powers)
        self.P = len(self.action_grid)

    def _precompute_transitions(self):
        """
        预计算 transition matrices（Algorithm 1 Lines 2-5）

        σ[i, j] = ŝ[i] + ΔSoC(p̂[j])
        z[i, j] 是 σ 在 state grid 上的连续坐标
        z-, z+: 整数邻居
        b: 插值系数
        mask: 越界标记
        """
        # ΔSoC 向量（每个 action）
        delta_soc = np.where(
            self.action_grid >= 0,
            -self.action_grid * self.dt / (self.eta_d * self.cap_mwh),  # 放电
            -self.action_grid * self.dt * self.eta_c / self.cap_mwh     # 充电（p<0，结果正）
        )  # shape (P,)

        # σ = state + ΔSoC (broadcast)
        sigma = self.state_grid[:, None] + delta_soc[None, :]  # (S, P)

        # 越界 mask
        infeasible = (sigma < self.soc_min - 1e-9) | (sigma > self.soc_max + 1e-9)

        # 夹到合法范围后算网格坐标
        sigma_clipped = np.clip(sigma, self.soc_min, self.soc_max)
        delta = self.config.delta_soc
        z = (sigma_clipped - self.soc_min) / delta  # (S, P), 连续坐标 ∈ [0, S-1]

        z_low = np.floor(z).astype(np.int64)
        z_high = np.minimum(z_low + 1, self.S - 1)  # 防越界
        b = z - z_low                                # 插值系数 ∈ [0, 1]
        b = np.where(z_low == z_high, 0.0, b)        # 若上下邻居重合，权重置 0 避免 NaN

        self._sigma = sigma
        self._z_low = z_low
        self._z_high = z_high
        self._b = b
        self._infeasible = infeasible

    # ============================================================
    # Backward induction
    # ============================================================

    def backward_induction(
        self,
        price_scenarios: np.ndarray,   # shape (T, R) 每时段 R 个价格支撑点
        price_probs: np.ndarray,        # shape (T, R) 每支撑点的概率
    ) -> np.ndarray:
        """
        Algorithm 1 的核心循环。

        Returns:
            V: shape (T+1, S) —— V[t, i] = 从 t 时刻状态 ŝ[i] 出发的 optimal expected value
               V[T, :] = terminal value（默认 0，除非有 final_soc_penalty）
        """
        T, R = price_scenarios.shape
        assert price_probs.shape == (T, R)

        # 概率归一化（兜底）
        probs = price_probs / price_probs.sum(axis=1, keepdims=True)

        # 初始化 V
        V = np.zeros((T + 1, self.S), dtype=np.float64)

        # Terminal value: V[T, s] = 0 默认。若有 final_soc_penalty，对低 SoC 罚款
        if self.config.final_soc_penalty > 0:
            # V_T(s) = -penalty × max(0, target_soc - s)
            below = np.maximum(0.0, self.config.final_soc_target - self.state_grid)
            V[T, :] = -self.config.final_soc_penalty * below

        # degradation cost per action (throughput × deg)
        # throughput(p) = |p| × dt
        deg_cost_per_action = self.config.deg_cost * np.abs(self.action_grid) * self.dt  # (P,)

        # 预计算 "-np.inf" 兜底
        NEG_INF = -1e18

        for t in range(T - 1, -1, -1):
            # Line 7: V̂_t^next[i, j] = (1-b) V̂_{t+1}[z-] + b V̂_{t+1}[z+]
            V_next_interp = (1 - self._b) * V[t + 1, self._z_low] + \
                            self._b * V[t + 1, self._z_high]  # (S, P)

            # Line 8: 越界设 -∞
            V_next_interp = np.where(self._infeasible, NEG_INF, V_next_interp)

            # Line 9: Q̂_all[i, j, r] = p̂[j] × λ̂[t, r] - deg_cost[j] + V̂^next[i, j]
            # payoff (j, r) matrix
            payoff_jr = self.action_grid[:, None] * price_scenarios[t][None, :] * self.dt  # (P, R)
            # 减去 degradation
            payoff_jr = payoff_jr - deg_cost_per_action[:, None]

            # Q̂_all = V̂^next[:, :, None] + payoff[None, :, :]
            Q_all = V_next_interp[:, :, None] + payoff_jr[None, :, :]  # (S, P, R)

            # Line 10: max over actions (axis=1)
            Q_ir = Q_all.max(axis=1)  # (S, R)

            # Line 11: V̂_{t-1} = Q̂ @ π̂_t
            V[t, :] = Q_ir @ probs[t]

        return V

    # ============================================================
    # Forward simulation (closed-loop greedy)
    # ============================================================

    def forward_simulate(
        self,
        V: np.ndarray,                  # (T+1, S)
        actual_prices: np.ndarray,      # (T,) 真实 RT 价格
        init_soc: float = 0.5,
    ) -> dict:
        """
        按 V 做 closed-loop greedy simulation（和 Regime V3 的 forward 等价，
        但 V 是 stagewise stochastic 意义下的最优价值函数）。

        At each t：
            current soc 在 state grid 上定位（线性插值）
            对每个 action j 评估 "即时 payoff + V[t+1, next_state(j)]"
            贪心选 argmax
            更新 SoC（用实际物理，不依赖 grid）
            累加 revenue

        Returns:
            dict with revenue_total, actions, powers, soc_trajectory, prices, rewards
        """
        T = len(actual_prices)
        soc = init_soc

        powers = np.zeros(T)
        soc_traj = [soc]
        rewards = np.zeros(T)

        deg_cost_per_action = self.config.deg_cost * np.abs(self.action_grid) * self.dt

        for t in range(T):
            # 评估每个 action 的 "即时 reward + V[t+1, next_state]"
            # 即时 reward: p × λ × dt - deg × |p| × dt
            immediate = self.action_grid * actual_prices[t] * self.dt - deg_cost_per_action

            # next state: 对每个 action 算 σ（连续 SoC）
            delta_soc = np.where(
                self.action_grid >= 0,
                -self.action_grid * self.dt / (self.eta_d * self.cap_mwh),
                -self.action_grid * self.dt * self.eta_c / self.cap_mwh
            )
            next_soc = soc + delta_soc

            # 越界的动作 mask 掉（物理不可行）
            infeasible_now = (next_soc < self.soc_min - 1e-9) | (next_soc > self.soc_max + 1e-9)

            # 在 V[t+1] 上对 next_soc 做线性插值
            next_soc_clipped = np.clip(next_soc, self.soc_min, self.soc_max)
            z = (next_soc_clipped - self.soc_min) / self.config.delta_soc
            z_low = np.floor(z).astype(np.int64)
            z_high = np.minimum(z_low + 1, self.S - 1)
            b = z - z_low
            b = np.where(z_low == z_high, 0.0, b)

            V_next = (1 - b) * V[t + 1, z_low] + b * V[t + 1, z_high]

            # Q(action) = immediate + V_next
            Q = immediate + V_next
            Q = np.where(infeasible_now, -1e18, Q)

            j_star = int(np.argmax(Q))
            p_star = self.action_grid[j_star]

            # 实际物理更新 SoC（可能会微量截断）
            if p_star >= 0:
                delta = -p_star * self.dt / (self.eta_d * self.cap_mwh)
            else:
                delta = -p_star * self.dt * self.eta_c / self.cap_mwh
            new_soc = soc + delta
            # 越界截断
            if new_soc > self.soc_max:
                # 充过头：减少充电
                actual_delta = self.soc_max - soc
                p_star = -actual_delta * self.cap_mwh / (self.dt * self.eta_c)
                new_soc = self.soc_max
            elif new_soc < self.soc_min:
                # 放过头：减少放电
                actual_delta = self.soc_min - soc
                p_star = -actual_delta * self.cap_mwh * self.eta_d / self.dt
                new_soc = self.soc_min

            # 结算
            reward = p_star * actual_prices[t] * self.dt - \
                     self.config.deg_cost * abs(p_star) * self.dt

            powers[t] = p_star
            rewards[t] = reward
            soc = new_soc
            soc_traj.append(soc)

        return {
            "revenue_total": float(rewards.sum()),
            "powers": powers,
            "soc_trajectory": np.array(soc_traj),
            "rewards": rewards,
            "final_soc": soc,
        }


# ============================================================
# 快速 sanity check: 确定性 DP vs LP Oracle
# ============================================================


def _sanity_check_deterministic(deltas: list = None):
    """
    sanity check: R=1（单场景 = 确定性）下的 DP 应该逼近 LP Oracle（论文 Table I：gap ≤ 0.3%）。
    用不同 δ_soc 观察收敛。
    """
    from config import BatteryConfig
    from oracle.lp_oracle import solve_day
    import time

    battery = BatteryConfig()

    if deltas is None:
        deltas = [0.02, 0.01, 0.005, 0.002, 0.001]

    # 合成一天 96 点 RT 价（山东 typical）
    np.random.seed(42)
    hours = np.arange(96) / 4
    base = 320 - 80 * np.cos(2 * np.pi * hours / 24) + \
           40 * np.sin(4 * np.pi * hours / 24)
    prices = base + np.random.normal(0, 20, 96)

    # LP Oracle
    lp_result = solve_day(prices, battery, init_soc=0.5)
    lp_rev = lp_result["revenue"]
    logger.info(f"\n=== Deterministic DP vs LP Oracle ===")
    logger.info(f"LP Oracle: ¥{lp_rev:,.2f}")
    logger.info(f"\n{'δ_soc':>8}  {'S':>6}  {'P':>5}  {'DP revenue':>14}  {'gap':>9}  {'time':>7}")
    logger.info("-" * 60)

    for delta in deltas:
        dp = TensorDP(battery, config=DPConfig(delta_soc=delta))

        price_scen = prices[:, None]
        price_prob = np.ones((96, 1))

        t0 = time.time()
        V = dp.backward_induction(price_scen, price_prob)
        sim = dp.forward_simulate(V, prices, init_soc=0.5)
        elapsed = time.time() - t0

        dp_rev = sim["revenue_total"]
        gap = (dp_rev - lp_rev) / abs(lp_rev) * 100
        flag = "✅" if abs(gap) < 0.5 else ("🟡" if abs(gap) < 2 else "❌")
        logger.info(f"{delta:>8.4f}  {dp.S:>6d}  {dp.P:>5d}  ¥{dp_rev:>12,.2f}  {gap:>+7.3f}%  {elapsed:>5.2f}s  {flag}")


if __name__ == "__main__":
    _sanity_check_deterministic()
