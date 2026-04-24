"""
储能电站 Gymnasium 环境

支持ERCOT和中国市场（可配置特征列和价格列）
状态：市场特征(z-normalized) + SOC + 累计收益
动作：5个离散动作（等待/慢充/快充/慢放/快放）
奖励：售电收入 - 购电成本 - 降解成本
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from env.battery_params import BatteryParams


# 动作定义：power_ratio正=放电，负=充电
ACTION_POWER_RATIOS = [0.0, -0.3, -1.0, 0.3, 1.0]
ACTION_NAMES = ["wait", "slow_charge", "fast_charge", "slow_discharge", "fast_discharge"]
N_ACTIONS = len(ACTION_POWER_RATIOS)


class BatteryEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        price_col: str = "rt_price",
        battery: BatteryParams | None = None,
        episode_length: int = 96,
        randomize_start: bool = True,
        degradation_per_mwh: float = 2.0,
        price_noise_std: float = 0.0,  # 价格噪声标准差（反过拟合）
    ):
        super().__init__()
        self.battery = battery or BatteryParams()
        self.episode_length = episode_length
        self.randomize_start = randomize_start
        self.degradation_per_mwh = degradation_per_mwh
        self.feature_cols = feature_cols
        self.price_noise_std = price_noise_std

        self.N_MARKET_FEATURES = len(feature_cols)
        self.N_ACCOUNT_FEATURES = 2  # soc, cumulative_revenue_norm
        self.N_OBS = self.N_MARKET_FEATURES + self.N_ACCOUNT_FEATURES

        # 预计算特征和价格
        self._features = df[feature_cols].fillna(0).values.astype(np.float32)
        self._prices = df[price_col].fillna(0).values.astype(np.float32)
        self._n = len(df)

        # 标准化参数
        self._feat_mean = self._features.mean(axis=0)
        self._feat_std = self._features.std(axis=0) + 1e-8
        self._price_mean = float(np.abs(self._prices).mean()) + 1e-8

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.N_OBS,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._reward_scale = self.battery.capacity_mw * self._price_mean * self.battery.interval_hours

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = self._n - self.episode_length - 1
        if self.randomize_start and max_start > 0:
            self._start = self.np_random.integers(0, max_start)
        else:
            self._start = 0

        self._step = 0
        self._soc = 0.5
        self._revenue = 0.0
        self._cycles = 0.0

        return self._obs(), self._info()

    def step(self, action: int):
        idx = self._start + self._step
        price = self._prices[idx]
        # 训练时加价格噪声（反过拟合：防止模型记住精确价格模式）
        if self.price_noise_std > 0:
            noise = np.random.normal(0, self.price_noise_std * abs(price))
            price = price + noise
        bp = self.battery

        power_ratio = ACTION_POWER_RATIOS[action]
        power_mw = power_ratio * bp.capacity_mw
        energy_mwh = power_mw * bp.interval_hours

        if energy_mwh > 0:  # 放电
            soc_change = -energy_mwh / bp.capacity_mwh / bp.discharge_efficiency
        elif energy_mwh < 0:  # 充电
            soc_change = -energy_mwh * bp.charge_efficiency / bp.capacity_mwh
        else:
            soc_change = 0.0

        new_soc = self._soc + soc_change

        if new_soc > bp.max_soc or new_soc < bp.min_soc:
            power_mw = 0.0
            energy_mwh = 0.0
            new_soc = self._soc

        self._soc = new_soc
        revenue = energy_mwh * price
        degradation_cost = abs(energy_mwh) * self.degradation_per_mwh
        cycle_fraction = abs(energy_mwh) / (2 * bp.capacity_mwh)
        self._cycles += cycle_fraction

        net = revenue - degradation_cost
        self._revenue += net
        reward = float(net / self._reward_scale)

        self._step += 1
        terminated = False
        truncated = self._step >= self.episode_length or idx >= self._n - 2

        return self._obs(), reward, terminated, truncated, self._info()

    def _obs(self) -> np.ndarray:
        idx = min(self._start + self._step, self._n - 1)
        market = (self._features[idx] - self._feat_mean) / self._feat_std
        account = np.array([
            self._soc,
            self._revenue / max(abs(self._revenue), self._reward_scale),
        ], dtype=np.float32)
        return np.concatenate([market, account])

    def _info(self) -> dict:
        return {
            "soc": self._soc,
            "revenue": self._revenue,
            "cycles": self._cycles,
            "step": self._step,
        }

    def get_state(self) -> dict:
        return {
            "start": self._start,
            "step": self._step,
            "soc": self._soc,
            "revenue": self._revenue,
            "cycles": self._cycles,
        }

    def set_state(self, state: dict):
        self._start = state["start"]
        self._step = state["step"]
        self._soc = state["soc"]
        self._revenue = state["revenue"]
        self._cycles = state["cycles"]
