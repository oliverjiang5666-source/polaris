"""
GRPO训练器 — Group Relative Policy Optimization

核心逻辑：
  1. 采样一批状态
  2. 每个状态穷举5个动作，各forward K步
  3. 组内归一化advantage
  4. 更新policy（clipped objective + entropy bonus）

无critic，无采样方差。
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import deepcopy
from loguru import logger
from agent.policy_net import PolicyNet
from env.battery_env import BatteryEnv, N_ACTIONS


def collect_group_data(
    env: BatteryEnv,
    policy: PolicyNet,
    n_states: int = 256,
    lookahead: int = 48,
    gamma: float = 0.99,
    stratified: bool = True,
    threshold_rollout: bool = False,
) -> dict:
    """
    采样n_states个状态，每个状态穷举5个动作。

    threshold_rollout=True时：
      - 后续步用Threshold策略（不用当前policy）→ rollout质量有保证
      - 同时记录Threshold在step 0的推荐动作 → 用作advantage baseline

    返回：
      states: (n_states, obs_dim)
      returns: (n_states, 5) — 每个动作的累计折扣回报
      baseline_actions: (n_states,) — Threshold推荐的动作（仅threshold_rollout时）
    """
    from agent.baselines import threshold_strategy

    states = []
    returns = []
    baseline_actions = []

    # 构建小时分桶（分层采样，均匀覆盖24小时）
    hour_buckets = None
    if stratified and env._hours is not None:
        max_start = env._n - env.episode_length - 1
        hour_buckets = {}
        for h in range(24):
            mask = (env._hours >= h) & (env._hours < h + 1)
            valid = np.where(mask)[0]
            if max_start > 0:
                valid = valid[valid < max_start]
            if len(valid) > 0:
                hour_buckets[h] = valid
        all_hours = sorted(hour_buckets.keys())

    for i in range(n_states):
        if hour_buckets:
            target_hour = all_hours[i % len(all_hours)]
            start_idx = np.random.choice(hour_buckets[target_hour])
            env._start = start_idx
            env._step = 0
            env._soc = np.random.uniform(0.1, 0.9)
            env._revenue = 0.0
            env._cycles = 0.0
            obs = env._obs()
        else:
            obs, _ = env.reset()
        saved = env.get_state()
        states.append(obs.copy())

        # 记录Threshold的推荐动作
        if threshold_rollout:
            idx0 = env._start + env._step
            p0 = float(env._prices[idx0])
            ma0 = float(env._features[idx0, env._ma96_col_idx])
            baseline_actions.append(threshold_strategy(obs, p0, ma0))

        action_returns = []
        for action in range(N_ACTIONS):
            env.set_state(deepcopy(saved))
            total_return = 0.0

            obs_next, reward, term, trunc, _ = env.step(action)
            total_return += reward

            for k in range(1, lookahead):
                if term or trunc:
                    break
                if threshold_rollout:
                    # 用Threshold策略做rollout（质量有保证）
                    ridx = env._start + env._step
                    rp = float(env._prices[ridx])
                    rma = float(env._features[ridx, env._ma96_col_idx])
                    a = threshold_strategy(obs_next, rp, rma)
                else:
                    with torch.no_grad():
                        a = policy.act(torch.FloatTensor(obs_next), deterministic=False)
                obs_next, r, term, trunc, _ = env.step(a)
                total_return += (gamma ** k) * r

            action_returns.append(total_return)

        returns.append(action_returns)

    result = {
        "states": np.array(states, dtype=np.float32),
        "returns": np.array(returns, dtype=np.float32),
    }
    if threshold_rollout:
        result["baseline_actions"] = np.array(baseline_actions, dtype=np.int64)
    return result


def compute_advantages(returns: np.ndarray, baseline_actions: np.ndarray = None) -> np.ndarray:
    """
    baseline_actions=None: 原始组内归一化
    baseline_actions给定时: advantage = R_i - R_threshold, 全局归一化
    """
    if baseline_actions is not None:
        batch_size = returns.shape[0]
        bl_ret = returns[np.arange(batch_size), baseline_actions].reshape(-1, 1)
        raw = returns - bl_ret
        return raw / (raw.std() + 1e-8)
    else:
        mean = returns.mean(axis=1, keepdims=True)
        std = returns.std(axis=1, keepdims=True) + 1e-8
        return (returns - mean) / std


def train_grpo(
    env: BatteryEnv,
    obs_dim: int,
    n_iterations: int = 200,
    batch_size: int = 256,
    lookahead: int = 4,
    gamma: float = 0.99,
    lr: float = 3e-4,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    mini_epochs: int = 4,
    pretrained_policy: PolicyNet | None = None,
) -> PolicyNet:
    """GRPO主训练循环"""
    policy = pretrained_policy if pretrained_policy is not None else PolicyNet(obs_dim, N_ACTIONS)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for iteration in range(n_iterations):
        # 1. 收集数据
        data = collect_group_data(env, policy, batch_size, lookahead, gamma,
                                  stratified=True, threshold_rollout=True)
        advantages = compute_advantages(data["returns"], data.get("baseline_actions"))

        states_t = torch.FloatTensor(data["states"])
        advantages_t = torch.FloatTensor(advantages)

        # 2. 计算旧的log概率
        with torch.no_grad():
            old_log_probs = policy.get_log_probs(states_t)  # (batch, 5)

        # 3. 多轮mini-epoch更新
        total_loss = 0.0
        for _ in range(mini_epochs):
            new_log_probs = policy.get_log_probs(states_t)  # (batch, 5)

            # 对每个动作计算ratio
            ratio = torch.exp(new_log_probs - old_log_probs)  # (batch, 5)

            # Clipped surrogate objective（对5个动作求和）
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            probs = torch.exp(new_log_probs)
            entropy = -(probs * new_log_probs).sum(dim=-1).mean()

            loss = policy_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / mini_epochs
        mean_return = data["returns"].mean()
        best_return = data["returns"].max(axis=1).mean()

        if (iteration + 1) % 20 == 0 or iteration == 0:
            # 动作分布
            with torch.no_grad():
                probs = policy.get_probs(states_t).mean(dim=0).numpy()
            action_dist = " ".join([f"{p:.0%}" for p in probs])
            logger.info(
                f"Iter {iteration+1:3d}/{n_iterations} | "
                f"loss={avg_loss:.4f} | "
                f"mean_ret={mean_return:.4f} | "
                f"best_ret={best_return:.4f} | "
                f"actions=[{action_dist}]"
            )

    return policy


def pretrain_from_baseline(
    env: BatteryEnv,
    df: pd.DataFrame,
    obs_dim: int,
    n_epochs: int = 10,
    batch_size: int = 512,
    charge_ratio: float = 0.7,
    discharge_ratio: float = 1.3,
) -> PolicyNet:
    """Behavior cloning预训练：用Threshold策略的演示数据预热policy网络"""
    from agent.baselines import threshold_strategy

    policy = PolicyNet(obs_dim, N_ACTIONS)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 收集Threshold策略的(obs, action)对
    all_obs = []
    all_actions = []

    prices = df["price"].values
    ma_96s = df["price_ma_96"].values if "price_ma_96" in df.columns else prices
    has_dam = "dam_position" in df.columns
    if has_dam:
        dam_positions = df["dam_position"].values
        from agent.baselines import dam_threshold

    bc_env = BatteryEnv(df, env.battery, episode_length=len(df) - 1, randomize_start=False)
    obs, _ = bc_env.reset()

    for step in range(len(df) - 1):
        if has_dam:
            action = dam_threshold(obs, float(prices[step]), float(ma_96s[step]),
                                   float(dam_positions[step]), bc_env._soc,
                                   charge_ratio=charge_ratio, discharge_ratio=discharge_ratio)
        else:
            action = threshold_strategy(obs, float(prices[step]), float(ma_96s[step]),
                                        charge_ratio=charge_ratio, discharge_ratio=discharge_ratio)
        all_obs.append(obs.copy())
        all_actions.append(action)

        obs, _, term, trunc, _ = bc_env.step(action)
        if term or trunc:
            break

    obs_t = torch.FloatTensor(np.array(all_obs))
    act_t = torch.LongTensor(all_actions)
    n = len(obs_t)
    logger.info(f"BC: {n:,} demonstrations collected")

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batch = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]

            logits = policy(obs_t[idx])
            loss = F.cross_entropy(logits, act_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        if (epoch + 1) % 3 == 0 or epoch == 0:
            with torch.no_grad():
                sample = obs_t[:min(1000, n)]
                probs = policy.get_probs(sample).mean(dim=0).numpy()
            dist = " ".join(f"{p:.0%}" for p in probs)
            logger.info(f"BC epoch {epoch+1}/{n_epochs}: loss={total_loss/n_batch:.4f} actions=[{dist}]")

    return policy


def pretrain_from_oracle(
    env: BatteryEnv,
    df: pd.DataFrame,
    obs_dim: int,
    n_epochs: int = 30,
    batch_size: int = 512,
) -> PolicyNet:
    """Hindsight Oracle BC：用未来价格生成最优演示，训练policy。
    多个初始SOC生成多样化轨迹，让policy在各种SOC下都知道该怎么做。
    """
    from agent.baselines import hindsight_oracle

    policy = PolicyNet(obs_dim, N_ACTIONS)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    prices = df["price"].values
    all_obs = []
    all_actions = []

    for init_soc in [0.2, 0.4, 0.5, 0.6, 0.8]:
        bc_env = BatteryEnv(df, env.battery, episode_length=len(df) - 1, randomize_start=False)
        obs, _ = bc_env.reset()
        bc_env._soc = init_soc
        obs = bc_env._obs()

        for step in range(len(df) - 1):
            action = hindsight_oracle(step, prices, bc_env._soc)
            all_obs.append(obs.copy())
            all_actions.append(action)
            obs, _, term, trunc, _ = bc_env.step(action)
            if term or trunc:
                break

    obs_t = torch.FloatTensor(np.array(all_obs))
    act_t = torch.LongTensor(all_actions)
    n = len(obs_t)

    # 类别平衡权重：上调充放电的少数类权重
    action_counts = torch.bincount(act_t, minlength=N_ACTIONS).float()
    class_weights = (action_counts.sum() / (N_ACTIONS * action_counts)).clamp(max=10)
    logger.info(f"Oracle BC: {n:,} demos, class_weights={class_weights.numpy().round(2)}")

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batch = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            logits = policy(obs_t[idx])
            loss = F.cross_entropy(logits, act_t[idx], weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                sample = obs_t[:min(2000, n)]
                probs = policy.get_probs(sample).mean(dim=0).numpy()
            dist = " ".join(f"{p:.0%}" for p in probs)
            logger.info(f"Oracle BC {epoch+1}/{n_epochs}: loss={total_loss/n_batch:.4f} actions=[{dist}]")

    return policy


def train_grpo_trajectory(
    env: BatteryEnv,
    obs_dim: int,
    n_iterations: int = 500,
    n_episodes: int = 32,
    n_samples: int = 8,
    episode_length: int = 96,
    lr: float = 1e-5,
    clip_epsilon: float = 10.0,
    kl_coeff: float = 0.001,
    ref_update_freq: int = 50,
    temperature: float = 1.0,
    pretrained_policy: PolicyNet | None = None,
) -> PolicyNet:
    """正确的GRPO（基于DeepSeek R1论文）。

    与之前的错误实现的关键区别：
    1. 采样**完整轨迹**（不是单步动作）
    2. Reward是**轨迹总收入**（不是短距lookahead）
    3. Advantage在**G条轨迹间**归一化（不是5个动作间）
    4. clip ε=10（不是0.2），lr~1e-5（不是1e-4），单次epoch
    5. KL penalty到frozen reference policy
    """
    policy = pretrained_policy if pretrained_policy is not None else PolicyNet(obs_dim, N_ACTIONS)
    ref_policy = deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # 分层采样的小时分桶
    hour_buckets = None
    if env._hours is not None:
        max_start = env._n - episode_length - 1
        hour_buckets = {}
        for h in range(24):
            mask = (env._hours >= h) & (env._hours < h + 1)
            valid = np.where(mask)[0]
            if max_start > 0:
                valid = valid[valid < max_start]
            if len(valid) > 0:
                hour_buckets[h] = valid
        all_hours = sorted(hour_buckets.keys())

    for iteration in range(n_iterations):
        # ===== 采样阶段：收集完整轨迹 =====
        all_states = []
        all_actions = []
        all_old_lp = []
        all_advantages = []
        iter_revenues = []

        for ep in range(n_episodes):
            # 分层采样起始状态
            if hour_buckets:
                target_hour = all_hours[ep % len(all_hours)]
                start_idx = np.random.choice(hour_buckets[target_hour])
                env._start = start_idx
                env._step = 0
                env._soc = np.random.uniform(0.1, 0.9)
                env._revenue = 0.0
                env._cycles = 0.0
            else:
                env.reset()
            saved = env.get_state()

            # 对同一起始状态采样G条完整轨迹
            group_data = []
            group_revenues = []

            for g in range(n_samples):
                env.set_state(deepcopy(saved))
                traj_states = []
                traj_actions = []
                traj_log_probs = []

                obs = env._obs()
                for step in range(episode_length):
                    state_t = torch.FloatTensor(obs)
                    with torch.no_grad():
                        logits = policy(state_t.unsqueeze(0)).squeeze(0)
                        probs = F.softmax(logits / temperature, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()

                    traj_states.append(obs.copy())
                    traj_actions.append(action.item())
                    traj_log_probs.append(dist.log_prob(action).item())

                    obs, _, term, trunc, info = env.step(action.item())
                    if term or trunc:
                        break

                group_data.append((traj_states, traj_actions, traj_log_probs))
                group_revenues.append(info["revenue"])

            # 组内advantage归一化
            revs = np.array(group_revenues)
            mean_r = revs.mean()
            std_r = revs.std() + 1e-8
            iter_revenues.extend(group_revenues)

            for g in range(n_samples):
                adv = (group_revenues[g] - mean_r) / std_r
                states_g, actions_g, lp_g = group_data[g]
                all_states.extend(states_g)
                all_actions.extend(actions_g)
                all_old_lp.extend(lp_g)
                all_advantages.extend([adv] * len(states_g))

        # ===== 更新阶段：单次epoch =====
        states_t = torch.FloatTensor(np.array(all_states))
        actions_t = torch.LongTensor(all_actions)
        old_lp_t = torch.FloatTensor(all_old_lp)
        adv_t = torch.FloatTensor(all_advantages)

        # 新的log prob
        new_all_lp = policy.get_log_probs(states_t)
        new_lp = new_all_lp[torch.arange(len(actions_t)), actions_t]

        # Clipped surrogate（ε=10，远比PPO的0.2宽）
        ratio = torch.exp(new_lp - old_lp_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty到reference policy
        with torch.no_grad():
            ref_lp = ref_policy.get_log_probs(states_t)
        kl = (torch.exp(ref_lp) * (ref_lp - new_all_lp)).sum(dim=-1).mean()

        loss = policy_loss + kl_coeff * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # 定期更新reference model
        if (iteration + 1) % ref_update_freq == 0:
            ref_policy = deepcopy(policy)
            for p in ref_policy.parameters():
                p.requires_grad = False

        # 日志
        if (iteration + 1) % 20 == 0 or iteration == 0:
            mean_rev = np.mean(iter_revenues)
            std_rev = np.std(iter_revenues)
            with torch.no_grad():
                sample_probs = policy.get_probs(states_t[:min(2000, len(states_t))]).mean(dim=0).numpy()
            dist_str = " ".join(f"{p:.0%}" for p in sample_probs)
            logger.info(
                f"GRPO {iteration+1:3d}/{n_iterations} | "
                f"loss={loss.item():.4f} | "
                f"rev=${mean_rev:+,.0f}(±{std_rev:,.0f}) | "
                f"kl={kl.item():.4f} | "
                f"actions=[{dist_str}]"
            )

    return policy
