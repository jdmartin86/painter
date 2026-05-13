"""
PolicyNetwork — CNN that maps an input frame to a VQ-VAE action grid.

Input:  (1, 128, 128) grayscale frame
Output: (32, 32) action grid with values in [0, num_codes)

Training algorithm: Streaming PPO
  - A value head provides per-step TD(0) advantage estimates.
  - The PPO clipped surrogate objective prevents destructively large updates.
  - One gradient step is taken per environment step (streaming / online mode).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional


class PolicyNetwork(nn.Module):
    def __init__(self, num_codes: int = 64, hidden: int = 64):
        super().__init__()
        self.num_codes = num_codes

        # 128×128 → 64×64 → 32×32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Policy head: per-pixel logits over codebook entries
        self.policy_head = nn.Conv2d(hidden * 2, num_codes, kernel_size=1)

        # Value head: global average-pool → scalar baseline
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, hidden*2, 1, 1)
            nn.Flatten(),              # (B, hidden*2)
            nn.Linear(hidden * 2, 1), # (B, 1)
        )

    # ------------------------------------------------------------------
    # Full forward pass (used during action selection)
    # ------------------------------------------------------------------
    def forward(self, frame: torch.Tensor):
        """
        Args:
            frame: (1, H, W) or (H, W) uint8 tensor (normalised internally)

        Returns:
            action_grid : (32, 32) LongTensor
            log_prob    : scalar — sum of per-pixel log-probs
            entropy     : scalar — mean per-pixel entropy
            value       : scalar — state value estimate V(s)
        """
        x = frame.float() / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)                              # (1, C, H, W)

        features = self.encoder(x)                          # (1, hidden*2, 32, 32)

        logits = self.policy_head(features)                 # (1, num_codes, 32, 32)
        logits = logits.squeeze(0).permute(1, 2, 0)        # (32, 32, num_codes)

        value = self.value_head(features).squeeze()         # scalar

        dist        = Categorical(logits=logits)
        action_grid = dist.sample()                         # (32, 32)
        log_prob    = dist.log_prob(action_grid).sum()      # scalar
        entropy     = dist.entropy().mean()                 # scalar

        return action_grid, log_prob, entropy, value

    # ------------------------------------------------------------------
    # Re-evaluation pass (used during the PPO update)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        frame: torch.Tensor,
        action_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-score a previously sampled (frame, action_grid) pair under the
        *current* parameters — required for computing the PPO importance ratio.

        Args:
            frame       : (1, H, W) uint8 tensor
            action_grid : (32, 32) LongTensor

        Returns:
            log_prob : scalar
            entropy  : scalar
            value    : scalar V(s)
        """
        x = frame.float() / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)

        features = self.encoder(x)

        logits = self.policy_head(features).squeeze(0).permute(1, 2, 0)
        value  = self.value_head(features).squeeze()

        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(action_grid).sum()
        entropy  = dist.entropy().mean()

        return log_prob, entropy, value


class PolicyAgent:
    """
    Streaming PPO agent.

    Usage per environment step
    --------------------------
        action = agent.select_action(frame)
        # … apply action, receive reward and next_frame …
        agent.record_reward(reward)
        agent.update(next_frame, done=done)
    """

    def __init__(
        self,
        num_codes:     int   = 64,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        clip_eps:      float = 0.2,
        vf_coef:       float = 0.5,
        entropy_coef:  float = 0.01,
        max_grad_norm: float = 0.5,
        device:        str   = None,
    ):
        """
        Args:
            num_codes     : VQ-VAE codebook size (== number of policy outputs).
            lr            : Adam learning rate.
            gamma         : Discount factor for TD bootstrapping.
            clip_eps      : PPO probability-ratio clip range (±clip_eps around 1).
            vf_coef       : Coefficient for the value-function MSE loss.
            entropy_coef  : Coefficient for the entropy bonus.
            max_grad_norm : L2 gradient-clipping norm.
            device        : 'cpu' | 'cuda' | 'mps' (auto-detected if None).
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device        = torch.device(device)
        self.gamma         = gamma
        self.clip_eps      = clip_eps
        self.vf_coef       = vf_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.net       = PolicyNetwork(num_codes=num_codes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # Storage for the current transition
        self._frame:         torch.Tensor | None = None
        self._action_grid:   torch.Tensor | None = None
        self._log_prob_old:  torch.Tensor | None = None  # detached — behaviour policy
        self._value:         torch.Tensor | None = None  # V(s_t)
        self._current_reward: float               = 0.0

        self.stats = {
            "steps":           0,
            "last_reward":     0.0,
            "last_loss":       0.0,
            "last_policy_loss": 0.0,
            "last_value_loss":  0.0,
            "last_entropy":     0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, frame: np.ndarray) -> np.ndarray:
        """
        Sample an action from the current policy and cache the transition data
        needed by update().

        Args:
            frame: (H, W, 1) uint8 numpy array

        Returns:
            action_grid: (32, 32) uint8 numpy array, values in [0, num_codes)
        """
        t = torch.from_numpy(frame).permute(2, 0, 1).to(self.device)  # (1, H, W)

        self.net.train()
        action_grid, log_prob, _, value = self.net(t)

        # Store tensors needed for the PPO update.
        # log_prob_old is detached so that it acts as a fixed reference point
        # for the behaviour policy (π_old) regardless of subsequent gradient steps.
        self._frame        = t
        self._action_grid  = action_grid.detach()
        self._log_prob_old = log_prob.detach()
        self._value        = value                # kept for TD target — no detach yet

        self.stats["steps"] += 1
        return action_grid.cpu().numpy().astype(np.uint8)

    def record_reward(self, reward: float):
        """Cache the reward received after the most recent select_action() call."""
        self._current_reward      = reward
        self.stats["last_reward"] = reward

    def update(self, next_frame: np.ndarray | None = None, done: bool = False):
        """
        Streaming PPO update — one gradient step per environment step.

        Args:
            next_frame : (H, W, 1) uint8 numpy array — the observation after
                         the action.  Used to bootstrap V(s_{t+1}).
                         If None, V(s_{t+1}) is treated as 0 (e.g. episode end).
            done       : If True, the bootstrap value is zeroed out regardless
                         of next_frame (terminal state).
        """
        if self._frame is None:
            return

        # ── 1. Bootstrap V(s_{t+1}) ──────────────────────────────────
        with torch.no_grad():
            if done or next_frame is None:
                next_value = torch.tensor(0.0, device=self.device)
            else:
                t_next    = (torch.from_numpy(next_frame)
                             .permute(2, 0, 1)
                             .to(self.device))
                _, _, _, next_value = self.net(t_next)

        # ── 2. TD(0) advantage ───────────────────────────────────────
        reward     = torch.tensor(self._current_reward, device=self.device)
        td_target  = reward + self.gamma * next_value          # y_t
        advantage  = (td_target - self._value).detach()        # A_t  (stop-grad)

        # ── 3. Re-evaluate (s_t, a_t) under current parameters ──────
        log_prob_new, entropy, value_new = self.net.evaluate(
            self._frame, self._action_grid
        )

        # ── 4. PPO clipped surrogate loss ────────────────────────────
        ratio       = torch.exp(log_prob_new - self._log_prob_old)
        surr1       = ratio * advantage
        surr2       = torch.clamp(ratio, 1.0 - self.clip_eps,
                                         1.0 + self.clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2)

        # ── 5. Value-function MSE loss ───────────────────────────────
        value_loss  = nn.functional.mse_loss(value_new, td_target.detach())

        # ── 6. Total loss ────────────────────────────────────────────
        loss = (policy_loss
                + self.vf_coef      * value_loss
                - self.entropy_coef * entropy)

        # ── 7. Gradient step ─────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ── 8. Logging ───────────────────────────────────────────────
        self.stats["last_loss"]        = loss.item()
        self.stats["last_policy_loss"] = policy_loss.item()
        self.stats["last_value_loss"]  = value_loss.item()
        self.stats["last_entropy"]     = entropy.item()

        # Clear transition buffer
        self._frame       = None
        self._action_grid = None
        self._log_prob_old = None
        self._value        = None

    def save(self, path: str = "policy.pt"):
        torch.save(self.net.state_dict(), path)
        print(f"Saved policy → {path}")

    def load(self, path: str = "policy.pt"):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded policy ← {path}")