"""
Stream AC(λ) — CNN actor-critic that maps an input frame to a VQ-VAE action grid.

Input:  (H, W, 1) uint8 grayscale frame
Output: (grid_size, grid_size) action grid with values in [0, num_codes)

Algorithm:
    "Streaming Deep Reinforcement Learning Finally Works"
    Elsayed, Vasan & Mahmood (2024) — https://arxiv.org/abs/2410.14606
    Reference implementation: https://github.com/mohmdelsayed/streaming-drl
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ── ObGD Optimizer ────────────────────────────────────────────────────────────

class ObGD(torch.optim.Optimizer):
    """
    Observed Gradient Descent with eligibility traces.
    Updates parameters following the streaming-DRL criteria:
        e  ←  γλ · e  +  ∇loss(θ)
        α' ←  α / max(1,  α · κ · |δ| · Σ|e|)
        θ  ←  θ  −  α' · δ · e
    """

    def __init__(
        self,
        params,
        lr:    float = 1.0,
        gamma: float = 0.99,
        lamda: float = 0.8,
        kappa: float = 2.0,
    ):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super().__init__(params, defaults)

    def step(self, delta: float, reset: bool = False):  # type: ignore[override]
        z_sum = 0.0
        for group in self.param_groups:
            gam_lam = group["gamma"] * group["lamda"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "e" not in state:
                    state["e"] = torch.zeros_like(p.data)
                e = state["e"]
                e.mul_(gam_lam).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        lr          = self.param_groups[0]["lr"]
        kappa       = self.param_groups[0]["kappa"]
        delta_bar   = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * lr * kappa
        step_size   = lr / dot_product if dot_product > 1.0 else lr

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = self.state[p]["e"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()


# ── Adaptive CNN Building Blocks ──────────────────────────────────────────────

class ActorCNN(nn.Module):
    def __init__(self, num_codes: int = 64, hidden: int = 64, grid_size: int = 7):
        super().__init__()
        self.grid_size = grid_size
        
        # Base shared processing layers
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=4, stride=2, padding=1)       # 128x128 -> 64x64
        self.conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1) # 64x64 -> 32x32
        self.conv3 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1) # 32x32 -> 16x16
        
        # Adapt final conv stride strategy depending on your target VQ-VAE grid setup
        if grid_size == 7:
            # 16x16 -> 7x7 (Uses stride 2, no padding on a 4x4 kernel window collapse)
            self.conv4 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=0)
        elif grid_size == 8:
            # 16x16 -> 8x8 (Standard symmetric downsampling layout)
            self.conv4 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1)
        else:
            # Universal fallback option using Adaptive pooling routines
            self.conv4 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1)
            self.pool_fallback = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        self.policy_head = nn.Conv2d(hidden * 2, num_codes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.conv1(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv2(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv3(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv4(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))

        if hasattr(self, 'pool_fallback'):
            x = self.pool_fallback(x)

        logits = self.policy_head(x)                # (1, num_codes, grid_size, grid_size)
        return logits.squeeze(0).permute(1, 2, 0)  # (grid_size, grid_size, num_codes)
    

class CriticCNN(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1,          hidden,     kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden,     hidden * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,           padding=1)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.conv1(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv2(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv3(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))

        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze()


# ── Agent ─────────────────────────────────────────────────────────────────────

class PolicyAgent:
    def __init__(
        self,
        num_codes:    int   = 64,
        grid_size:    int   = 7,
        hidden:       int   = 64,
        lr:           float = 1.0,
        gamma:        float = 0.99,
        lamda:        float = 0.8,
        entropy_coef: float = 0.01,
        kappa_policy: float = 3.0,
        kappa_value:  float = 2.0,
        device:       str   = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device       = torch.device(device)
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.grid_size    = grid_size

        # Build networks passing down the target grid map properties
        self.actor  = ActorCNN(num_codes=num_codes, hidden=hidden, grid_size=grid_size).to(self.device)
        self.critic = CriticCNN(hidden=hidden).to(self.device)

        self.opt_policy = ObGD(
            self.actor.parameters(),
            lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy,
        )
        self.opt_value = ObGD(
            self.critic.parameters(),
            lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value,
        )

        self._frame:          torch.Tensor = None
        self._log_prob:       torch.Tensor = None
        self._entropy:        torch.Tensor = None
        self._value:          torch.Tensor = None
        self._current_reward: float        = 0.0

        self.stats = {
            "steps":        0,
            "last_reward":  0.0,
            "last_delta":   0.0,
            "last_entropy": 0.0,
        }

    def select_action(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes your (128, 128, 1) frame image matrix and samples 
        a perfectly aligned discrete (grid_size, grid_size) spatial map.
        """
        t = torch.from_numpy(frame).permute(2, 0, 1).to(self.device)

        self.actor.train()
        self.critic.train()

        # ── Actor Forward Run ─────────────────────────────────────────
        logits      = self.actor(t)                         # (grid_size, grid_size, num_codes)
        dist        = Categorical(logits=logits)
        action_grid = dist.sample()                         # (grid_size, grid_size)

        self._log_prob = -dist.log_prob(action_grid).sum()
        self._entropy  =  dist.entropy().mean()

        # ── Critic Forward Run ────────────────────────────────────────
        self._value = -self.critic(t)

        self._frame = t
        self.stats["steps"] += 1

        return action_grid.detach().cpu().numpy().astype(np.uint8)

    def record_reward(self, reward: float):
        self._current_reward      = reward
        self.stats["last_reward"] = reward

    def update(self, next_frame: np.ndarray = None, done: bool = False):
        if self._frame is None:
            return

        with torch.no_grad():
            if done or next_frame is None:
                v_next = 0.0
            else:
                t_next = (torch.from_numpy(next_frame)
                          .permute(2, 0, 1).to(self.device))
                v_next = self.critic(t_next).item()

        v_s   = -self._value.item()
        delta =  self._current_reward + self.gamma * v_next - v_s

        entropy_bonus = (
            self.entropy_coef
            * self._entropy
            * torch.sign(torch.tensor(delta, device=self.device))
        )
        policy_loss = self._log_prob - entropy_bonus

        self.opt_policy.zero_grad()
        policy_loss.backward()

        self.opt_value.zero_grad()
        self._value.backward()

        self.opt_policy.step(delta, reset=done)
        self.opt_value.step(delta, reset=done)

        self.stats["last_delta"]   = delta
        self.stats["last_entropy"] = self._entropy.item()

        self._frame    = None
        self._log_prob = None
        self._entropy  = None
        self._value    = None

    def save(self, path: str = "policy.pt"):
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            path,
        )
        print(f"Saved policy → {path}")

    def load(self, path: str = "policy.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        print(f"Loaded policy ← {path}")