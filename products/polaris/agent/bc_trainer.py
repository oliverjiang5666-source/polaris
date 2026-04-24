"""
行为克隆(Behavior Cloning)训练器

支持两种老师：
1. LP Oracle（理论最优）— 新方法
2. Threshold规则 — 旧方法（保留用于对比）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from agent.policy_net import PolicyNet
from env.battery_env import BatteryEnv, ACTION_POWER_RATIOS


def train_bc_from_oracle(
    env: BatteryEnv,
    oracle_actions: np.ndarray,
    obs_dim: int,
    init_socs: list[float] | None = None,
    n_epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden: list[int] | None = None,
    use_layer_norm: bool = True,
) -> PolicyNet:
    """
    从LP Oracle动作序列训练BC策略

    Args:
        env: 电池环境
        oracle_actions: LP Oracle的离散动作序列 (N,)
        obs_dim: 观测维度
        init_socs: 多个初始SOC增广训练
        n_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        hidden: 网络隐层
        use_layer_norm: 是否用LayerNorm

    Returns:
        训练好的PolicyNet
    """
    if init_socs is None:
        init_socs = [0.5]  # Fix #2: 只用0.5，匹配Oracle求解的init_soc
    if hidden is None:
        hidden = [256, 128, 64]

    logger.info(f"BC from Oracle: collecting demonstrations with {len(init_socs)} init SOCs...")

    all_obs = []
    all_actions = []
    n_total = len(oracle_actions)
    steps_per_day = 96

    for init_soc in init_socs:
        n_days = n_total // steps_per_day

        for d in range(n_days):
            start = d * steps_per_day
            end = start + steps_per_day

            # 设置环境到这一天的开头
            env._start = start
            env._step = 0
            env._soc = init_soc
            env._revenue = 0.0
            env._cycles = 0.0

            for t in range(steps_per_day):
                if start + t >= n_total:
                    break

                obs = env._obs()
                action = int(oracle_actions[start + t])

                all_obs.append(obs)
                all_actions.append(action)

                # 用oracle动作推进环境（更新SOC）
                env.step(action)

    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.int64)

    logger.info(f"  Collected {len(all_obs):,} demonstration pairs")

    # Fix #1: 温和的类别权重（sqrt而非inverse）
    # Oracle 74%是wait，不能用inverse weight (0.27)压wait
    # 用sqrt让少数类权重提升，但不过度
    action_counts = np.bincount(all_actions, minlength=5)
    total = len(all_actions)
    weights = np.zeros(5, dtype=np.float32)
    for i in range(5):
        if action_counts[i] > 0:
            raw_weight = total / (5 * action_counts[i])
            weights[i] = np.sqrt(raw_weight)  # sqrt而非raw → 温和平衡
        else:
            weights[i] = 1.0

    logger.info(f"  Action distribution: {action_counts}")
    logger.info(f"  Class weights (sqrt): {weights.round(2)}")

    # 训练
    policy = PolicyNet(obs_dim, n_actions=5, hidden=hidden, use_layer_norm=use_layer_norm)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    dataset = TensorDataset(
        torch.tensor(all_obs),
        torch.tensor(all_actions),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for obs_batch, act_batch in loader:
            logits = policy(obs_batch)
            loss = F.cross_entropy(logits, act_batch, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(obs_batch)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == act_batch).sum().item()
            total_samples += len(obs_batch)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.3f}")

    logger.info(f"  BC training complete. Final accuracy: {accuracy:.3f}")
    return policy


def train_bc_from_threshold(
    env: BatteryEnv,
    df,
    obs_dim: int,
    charge_ratio: float = 0.7,
    discharge_ratio: float = 1.3,
    n_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden: list[int] | None = None,
) -> PolicyNet:
    """
    从Threshold规则策略训练BC（保留旧方法用于对比）
    """
    if hidden is None:
        hidden = [256, 128, 64]

    logger.info(f"BC from Threshold: cr={charge_ratio}, dr={discharge_ratio}")

    all_obs = []
    all_actions = []

    rt_price = df["rt_price"].values
    rt_ma_96 = df["rt_price_ma_96"].values if "rt_price_ma_96" in df.columns else df["rt_price"].rolling(96).mean().values

    env._start = 0
    env._step = 0
    env._soc = 0.5
    env._revenue = 0.0
    env._cycles = 0.0

    for i in range(len(df)):
        obs = env._obs()

        price = rt_price[i]
        ma96 = rt_ma_96[i] if not np.isnan(rt_ma_96[i]) else price
        ratio = price / max(ma96, 1.0)

        if ratio < charge_ratio:
            action = 2  # fast_charge
        elif ratio < charge_ratio + 0.15:
            action = 1  # slow_charge
        elif ratio > discharge_ratio:
            action = 4  # fast_discharge
        elif ratio > discharge_ratio - 0.15:
            action = 3  # slow_discharge
        else:
            action = 0  # wait

        all_obs.append(obs)
        all_actions.append(action)
        env.step(action)

    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.int64)

    logger.info(f"  Collected {len(all_obs):,} demonstration pairs")

    policy = PolicyNet(obs_dim, n_actions=5, hidden=hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    dataset = TensorDataset(torch.tensor(all_obs), torch.tensor(all_actions))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        n = 0
        for obs_batch, act_batch in loader:
            logits = policy(obs_batch)
            loss = F.cross_entropy(logits, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(obs_batch)
            n += len(obs_batch)

        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: loss={total_loss/n:.4f}")

    return policy


def evaluate_policy(
    policy: PolicyNet,
    env: BatteryEnv,
    n_steps: int | None = None,
    deterministic: bool = True,
) -> dict:
    """
    评估策略的总收入

    Args:
        policy: 要评估的策略
        env: 电池环境
        n_steps: 评估步数（None=整个数据集）
        deterministic: 是否确定性选择（argmax）

    Returns:
        dict with revenue, cycles, actions
    """
    if n_steps is None:
        n_steps = env._n

    env._start = 0
    env._step = 0
    env._soc = 0.5
    env._revenue = 0.0
    env._cycles = 0.0

    actions_taken = []

    for _ in range(min(n_steps, env._n - 1)):
        obs = env._obs()
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action = policy.act(obs_tensor, deterministic=deterministic)
        actions_taken.append(action)
        env.step(action)

    action_counts = np.bincount(np.array(actions_taken), minlength=5)

    return {
        "revenue": env._revenue,
        "cycles": env._cycles,
        "steps": len(actions_taken),
        "action_counts": action_counts,
    }
