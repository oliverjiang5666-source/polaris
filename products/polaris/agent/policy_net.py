"""
Policy网络 — MLP with optional LayerNorm
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 5,
        hidden: list[int] | None = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]

        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def get_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward(x), dim=-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        with torch.no_grad():
            probs = self.get_probs(obs.unsqueeze(0)).squeeze(0)
            if deterministic:
                return probs.argmax().item()
            return torch.multinomial(probs, 1).item()


class ValueNet(nn.Module):
    """Value network for PPO (separate from policy)"""
    def __init__(
        self,
        obs_dim: int,
        hidden: list[int] | None = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]

        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
