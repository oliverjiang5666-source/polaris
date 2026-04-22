"""
DAgger: Dataset Aggregation 训练器 (Modification #6)

问题：BC 在 Oracle 轨迹上训练，但测试时用自己的策略——分布偏移。
解决：迭代地收集策略访问到的状态，在这些状态重新用 Oracle 标注正确动作。

算法：
    Input: 初始 BC 策略 π_0（在 Oracle 轨迹上训练）
    for i = 1..N:
        1. 用 π_{i-1} 在训练集上跑，收集访问到的状态 S_i
        2. 对 S_i 每个状态，用 Oracle 算"正确动作" a*
        3. D_i = {(s, a*) for s in S_i}
        4. D_total = D_0 ∪ D_1 ∪ ... ∪ D_i
        5. 在 D_total 上重训 π_i
    return π_N

Reference:
    Ross, Gordon, Bagnell (2011). "A Reduction of Imitation Learning and
    Structured Prediction to No-Regret Online Learning." AISTATS.

用法：
    from agent.dagger_trainer import DAggerTrainer
    trainer = DAggerTrainer(env, oracle_fn, n_iterations=5)
    policy = trainer.train()
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from config import BatteryConfig, BCConfig
from agent.policy_net import PolicyNet
from oracle.lp_oracle import solve_day


def get_oracle_action_at_state(
    soc: float,
    remaining_prices: np.ndarray,
    battery: BatteryConfig,
) -> int:
    """
    给定当前 SoC 和剩余时段的真实价格，用 LP Oracle 求最优动作。

    这是 DAgger 的"专家标注器"。
    """
    if len(remaining_prices) < 2:
        return 0  # wait
    result = solve_day(remaining_prices, battery, init_soc=soc)
    # 返回第一个动作
    return int(result["actions"][0])


class DAggerTrainer:
    """
    DAgger 训练器。

    流程：
        Step 0: 从 Oracle 轨迹训练初始 BC
        Step 1..N: 滚动策略 → 收集状态 → Oracle 标注 → 合并数据 → 重训
    """

    def __init__(
        self,
        train_df,
        feature_cols: list[str],
        oracle_features_fn,  # 从 df 提取 oracle 期望 features 的函数
        battery: BatteryConfig | None = None,
        n_iterations: int = 5,
        samples_per_iter: int = 5000,
        bc_config: BCConfig | None = None,
    ):
        self.df = train_df
        self.feature_cols = feature_cols
        self.oracle_features_fn = oracle_features_fn
        self.battery = battery or BatteryConfig()
        self.n_iterations = n_iterations
        self.samples_per_iter = samples_per_iter
        self.bc_config = bc_config or BCConfig()

        self._features = train_df[feature_cols].fillna(0).values.astype(np.float32)
        self._prices = train_df["rt_price"].fillna(0).values.astype(np.float32)

        # 标准化参数
        self._feat_mean = self._features.mean(axis=0)
        self._feat_std = self._features.std(axis=0) + 1e-8

        # 累积数据集
        self.aggregated_X = []
        self.aggregated_y = []

    def train_bc(self, X: np.ndarray, y: np.ndarray) -> PolicyNet:
        """在数据 (X, y) 上训练一个 BC 策略"""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_dim = X.shape[1]
        policy = PolicyNet(
            input_dim=input_dim,
            hidden_dims=self.bc_config.hidden_dims,
            output_dim=5,  # 5 actions
            use_layer_norm=self.bc_config.use_layer_norm,
        ).to(device)

        optimizer = torch.optim.Adam(policy.parameters(), lr=self.bc_config.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).long().to(device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.bc_config.batch_size, shuffle=True)

        policy.train()
        for epoch in range(self.bc_config.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = policy(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            if (epoch + 1) % 10 == 0:
                logger.info(f"    Epoch {epoch+1}/{self.bc_config.epochs}: loss={total_loss/len(dataset):.4f}")

        policy.eval()
        return policy

    def rollout_policy(self, policy: PolicyNet, n_samples: int) -> tuple[np.ndarray, np.ndarray, list]:
        """
        用当前 policy 在训练集上 rollout，采样访问到的状态。

        Returns:
            X_visited: 策略访问的状态特征（normalized）
            socs_visited: 对应的 SoC
            time_idx_visited: 对应的时间索引（用于 Oracle 标注）
        """
        device = next(policy.parameters()).device
        n = len(self._features)

        # 随机选起始点
        start_candidates = np.random.choice(
            range(n - 96), size=min(n_samples // 96, (n - 96)), replace=False)

        X_visited = []
        socs_visited = []
        time_idx_visited = []

        for start in start_candidates:
            soc = 0.5
            for step in range(96):
                idx = start + step
                feat = (self._features[idx] - self._feat_mean) / self._feat_std
                account = np.array([soc, 0.0], dtype=np.float32)
                obs = np.concatenate([feat, account])

                # 记录状态
                X_visited.append(obs)
                socs_visited.append(soc)
                time_idx_visited.append(idx)

                # 策略决策
                with torch.no_grad():
                    x_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    logits = policy(x_t)
                    action = int(logits.argmax(dim=1).item())

                # 模拟物理 step（简化：只更新 SoC）
                power_ratios = [0.0, -0.3, -1.0, 0.3, 1.0]
                pw = power_ratios[action] * self.battery.capacity_mw
                dt = self.battery.interval_hours
                e = pw * dt
                if e > 0:
                    sc = -e / self.battery.capacity_mwh / self.battery.discharge_efficiency
                elif e < 0:
                    sc = -e * self.battery.charge_efficiency / self.battery.capacity_mwh
                else:
                    sc = 0.0
                soc = np.clip(soc + sc, self.battery.min_soc, self.battery.max_soc)

        return (np.array(X_visited),
                np.array(socs_visited),
                time_idx_visited)

    def label_with_oracle(
        self,
        X_visited: np.ndarray,
        socs_visited: np.ndarray,
        time_idx_visited: list,
    ) -> np.ndarray:
        """用 Oracle 标注访问到的状态的正确动作"""
        labels = []
        for soc, idx in zip(socs_visited, time_idx_visited):
            # 当天剩余价格
            day_start = (idx // 96) * 96
            day_end = day_start + 96
            remaining = self._prices[idx:day_end]
            if len(remaining) < 2:
                labels.append(0)
                continue
            try:
                action = get_oracle_action_at_state(soc, remaining, self.battery)
            except Exception:
                action = 0
            labels.append(action)
        return np.array(labels, dtype=np.int64)

    def train(self) -> PolicyNet:
        """完整 DAgger 训练流程"""
        # Step 0: 初始 BC on Oracle 轨迹
        logger.info("DAgger Iteration 0: Initial BC on Oracle")
        X_oracle, y_oracle = self.oracle_features_fn(self.df)
        self.aggregated_X.append(X_oracle)
        self.aggregated_y.append(y_oracle)

        X_all = np.concatenate(self.aggregated_X, axis=0)
        y_all = np.concatenate(self.aggregated_y, axis=0)
        policy = self.train_bc(X_all, y_all)

        for i in range(1, self.n_iterations + 1):
            logger.info(f"\nDAgger Iteration {i}/{self.n_iterations}")

            # 1. Rollout 策略
            logger.info(f"  Rollout 策略收集 {self.samples_per_iter} 个状态...")
            X_vis, socs_vis, idx_vis = self.rollout_policy(policy, self.samples_per_iter)
            logger.info(f"  收集到 {len(X_vis)} 个状态")

            # 2. Oracle 标注
            logger.info(f"  Oracle 标注中...")
            y_vis = self.label_with_oracle(X_vis, socs_vis, idx_vis)

            # 3. 合并数据
            self.aggregated_X.append(X_vis)
            self.aggregated_y.append(y_vis)

            X_all = np.concatenate(self.aggregated_X, axis=0)
            y_all = np.concatenate(self.aggregated_y, axis=0)
            logger.info(f"  累积数据集大小：{len(X_all)}")

            # 4. 重训
            logger.info(f"  重训 BC...")
            policy = self.train_bc(X_all, y_all)

        return policy


if __name__ == "__main__":
    logger.info("DAgger 训练器模块加载成功。")
    logger.info("用法示例：")
    logger.info("  trainer = DAggerTrainer(train_df, feature_cols, oracle_fn, n_iterations=3)")
    logger.info("  policy = trainer.train()")
    logger.info("  eval_on_test(policy)")
