"""
储能RL交易策略 — 全局配置

支持多省中国电力现货市场 + ERCOT（保留兼容）
"""

from dataclasses import dataclass, field


# ============================================================
# 电池物理参数
# ============================================================

@dataclass
class BatteryConfig:
    capacity_mw: float = 200          # 额定功率 MW
    capacity_mwh: float = 400         # 额定容量 MWh
    max_soc: float = 0.95
    min_soc: float = 0.05
    round_trip_efficiency: float = 0.90
    charge_efficiency: float = 0.9487  # sqrt(0.90)
    discharge_efficiency: float = 0.9487
    degradation_cost_per_mwh: float = 2.0  # 元/MWh 吞吐量降解成本
    interval_hours: float = 0.25       # 15分钟


# ============================================================
# 动作空间
# ============================================================

ACTIONS = {
    0: {"name": "wait",            "power_ratio": 0.0},
    1: {"name": "slow_charge",     "power_ratio": -0.3},
    2: {"name": "fast_charge",     "power_ratio": -1.0},
    3: {"name": "slow_discharge",  "power_ratio": 0.3},
    4: {"name": "fast_discharge",  "power_ratio": 1.0},
}

N_ACTIONS = len(ACTIONS)


# ============================================================
# Oracle 量化阈值
# ============================================================

@dataclass
class OracleConfig:
    wait_threshold: float = 0.15      # |ratio| < 0.15 → wait
    slow_threshold: float = 0.65      # 0.15 ≤ |ratio| < 0.65 → slow


# ============================================================
# 训练配置
# ============================================================

@dataclass
class BCConfig:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 512
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    use_layer_norm: bool = True
    init_socs: list = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])


@dataclass
class PPOConfig:
    iterations: int = 500
    episodes_per_iter: int = 32
    episode_length: int = 96
    lr_policy: float = 3e-5
    lr_value: float = 1e-4
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    mini_epochs: int = 4
    max_grad_norm: float = 0.5
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64])


# ============================================================
# 回测配置
# ============================================================

@dataclass
class BacktestConfig:
    min_train_days: int = 180
    test_days: int = 90
    stride_days: int = 90
    expanding_window: bool = True
    seeds: list = field(default_factory=lambda: [42, 123, 456])


# ============================================================
# Threshold搜索范围
# ============================================================

@dataclass
class ThresholdSearchConfig:
    charge_ratios: list = field(
        default_factory=lambda: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    )
    discharge_ratios: list = field(
        default_factory=lambda: [1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55]
    )


# ============================================================
# ERCOT兼容（保留旧配置）
# ============================================================

@dataclass
class ERCOTConfig:
    hubs: list = field(default_factory=lambda: ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"])
    primary_hub: str = "HB_WEST"
    price_col: str = "spp"
