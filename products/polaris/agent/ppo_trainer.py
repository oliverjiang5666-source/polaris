"""
PPO在线优化训练器（CleanRL风格）

从BC初始化的策略出发，在历史价格模拟器中通过试错学习超越Threshold。

关键设计：
- 价格回放模式：随机抽取历史天数，价格固定不受Agent影响
- 从BC checkpoint初始化：不从零开始，站在Threshold的肩膀上
- GAE优势估计：减少方差
- 独立Value网络：不共享backbone
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from agent.policy_net import PolicyNet, ValueNet
from env.battery_env import BatteryEnv


def train_ppo(
    env: BatteryEnv,
    pretrained_policy: PolicyNet | None = None,
    obs_dim: int = 33,
    n_iterations: int = 300,
    episodes_per_iter: int = 32,
    episode_length: int = 96,
    lr_policy: float = 3e-5,
    lr_value: float = 1e-4,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    gae_lambda: float = 0.95,
    gamma: float = 0.99,
    mini_epochs: int = 4,
    max_grad_norm: float = 0.5,
    hidden: list[int] | None = None,
    log_interval: int = 20,
) -> PolicyNet:
    """
    PPO训练

    Args:
        env: 电池环境（使用历史价格数据）
        pretrained_policy: BC预训练策略（None则随机初始化）
        obs_dim: 观测维度
        n_iterations: 训练迭代数
        episodes_per_iter: 每轮收集的episode数
        episode_length: 每个episode步数（96=1天）
        lr_policy: 策略网络学习率
        lr_value: 价值网络学习率
        clip_epsilon: PPO clip参数
        entropy_coeff: 熵奖励系数
        gae_lambda: GAE参数
        gamma: 折扣因子
        mini_epochs: 每轮数据上更新几次
        max_grad_norm: 梯度裁剪
        hidden: 隐层维度
        log_interval: 日志间隔

    Returns:
        训练好的PolicyNet
    """
    if hidden is None:
        hidden = [256, 128, 64]

    # 初始化策略
    if pretrained_policy is not None:
        policy = pretrained_policy
        logger.info("PPO: Starting from pretrained BC policy")
    else:
        policy = PolicyNet(obs_dim, n_actions=5, hidden=hidden, use_layer_norm=True)
        logger.info("PPO: Starting from random policy")

    # 独立的Value网络
    value_net = ValueNet(obs_dim, hidden=hidden, use_layer_norm=True)

    opt_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=lr_value)

    env.episode_length = episode_length
    env.randomize_start = True

    best_avg_reward = -float("inf")
    best_policy_state = None

    for iteration in range(n_iterations):
        # ============ Collect episodes ============
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        episode_revenues = []

        for ep in range(episodes_per_iter):
            obs, info = env.reset()
            ep_reward = 0.0

            for step in range(episode_length):
                obs_tensor = torch.tensor(obs, dtype=torch.float32)

                with torch.no_grad():
                    logits = policy(obs_tensor.unsqueeze(0))
                    probs = F.softmax(logits, dim=-1).squeeze(0)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = value_net(obs_tensor.unsqueeze(0)).squeeze()

                all_obs.append(obs)
                all_actions.append(action.item())
                all_log_probs.append(log_prob.item())
                all_values.append(value.item())

                obs, reward, terminated, truncated, info = env.step(action.item())
                all_rewards.append(reward)
                all_dones.append(terminated or truncated)
                ep_reward += reward

            episode_revenues.append(info["revenue"])

        # Convert to tensors
        obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32)
        actions_t = torch.tensor(all_actions, dtype=torch.int64)
        old_log_probs_t = torch.tensor(all_log_probs, dtype=torch.float32)
        rewards_t = torch.tensor(all_rewards, dtype=torch.float32)
        dones_t = torch.tensor(all_dones, dtype=torch.float32)
        values_t = torch.tensor(all_values, dtype=torch.float32)

        # ============ Compute GAE advantages ============
        advantages = torch.zeros_like(rewards_t)
        returns = torch.zeros_like(rewards_t)
        gae = 0.0
        n_steps = len(rewards_t)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1 or dones_t[t]:
                next_value = 0.0
            else:
                next_value = values_t[t + 1]

            delta = rewards_t[t] + gamma * next_value * (1 - dones_t[t]) - values_t[t]
            gae = delta + gamma * gae_lambda * (1 - dones_t[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values_t[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ============ PPO update ============
        batch_size = n_steps
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(mini_epochs):
            # 随机打乱
            indices = torch.randperm(batch_size)
            mini_batch_size = min(512, batch_size)

            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                idx = indices[start:end]

                mb_obs = obs_t[idx]
                mb_actions = actions_t[idx]
                mb_old_log_probs = old_log_probs_t[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # Policy loss
                logits = policy(mb_obs)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * entropy

                opt_policy.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt_policy.step()

                # Value loss
                values = value_net(mb_obs)
                value_loss = F.mse_loss(values, mb_returns)

                opt_value.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                opt_value.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # ============ Logging ============
        avg_revenue = np.mean(episode_revenues)
        n_updates = mini_epochs * (batch_size // mini_batch_size + 1)

        if avg_revenue > best_avg_reward:
            best_avg_reward = avg_revenue
            best_policy_state = {k: v.clone() for k, v in policy.state_dict().items()}

        if (iteration + 1) % log_interval == 0 or iteration == 0:
            logger.info(
                f"  PPO iter {iteration+1}/{n_iterations}: "
                f"avg_rev={avg_revenue:>10,.0f} "
                f"best={best_avg_reward:>10,.0f} "
                f"p_loss={total_policy_loss/n_updates:.4f} "
                f"v_loss={total_value_loss/n_updates:.4f} "
                f"entropy={total_entropy/n_updates:.3f}"
            )

    # 加载最佳策略
    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        logger.info(f"  PPO complete. Best avg revenue: {best_avg_reward:,.0f}")

    return policy
