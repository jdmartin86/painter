"""
Stream AC(λ) — CNN actor-critic that maps an input frame to a VQ-VAE action grid.

Input:  (H, W, 1) uint8 grayscale frame
Output: (32, 32) action grid with values in [0, num_codes)

Algorithm:
    "Streaming Deep Reinforcement Learning Finally Works"
    Elsayed, Vasan & Mahmood (2024) — https://arxiv.org/abs/2410.14606
    Reference implementation: https://github.com/mohmdelsayed/streaming-drl

Key differences from standard actor-critic / PPO:
  - ObGD optimizer: eligibility traces (λ) replace momentum/Adam state.
    Per-step update: θ ← θ − α·δ·e,  e ← γλe + ∇loss
  - Overshooting prevention: step size α is normalised so that
    |δ| · α · κ · Σ|e| ≤ 1, preventing catastrophically large updates.
  - Entropy bonus is sign-flipped with δ so it always encourages
    exploration regardless of the TD error's sign (|δ| weighting).
  - Layer normalisation after every conv layer for activation stability.
  - No experience replay, no target network, no batch updates.
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

    The update for each parameter θ at every step:

        e  ←  γλ · e  +  ∇loss(θ)                       (accumulate trace)
        α' ←  α / max(1,  α · κ · |δ| · Σ|e|)           (shrink if overshooting)
        θ  ←  θ  −  α' · δ · e                           (TD-weighted descent)

    On episode end (reset=True) all traces are zeroed.

    Args:
        params : iterable of parameters (same as any torch optimizer)
        lr     : base learning rate α  (default 1.0 — ObGD is self-normalising)
        gamma  : discount factor γ for trace decay
        lamda  : eligibility trace decay λ
        kappa  : overshooting-prevention scale  (policy: ~3.0, value: ~2.0)
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
        """
        Args:
            delta : TD error scalar — the per-step learning signal.
            reset : Zero eligibility traces after the update (episode end).
        """
        # ── Pass 1: update traces, accumulate |e| sum ─────────────────
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

        # ── Overshooting-prevention normalisation ─────────────────────
        # If the natural update δ·α·Σ|e| would overshoot by factor κ,
        # shrink α proportionally.
        lr          = self.param_groups[0]["lr"]
        kappa       = self.param_groups[0]["kappa"]
        delta_bar   = max(abs(delta), 1.0)           # avoid div-by-zero
        dot_product = delta_bar * z_sum * lr * kappa
        step_size   = lr / dot_product if dot_product > 1.0 else lr

        # ── Pass 2: parameter update ──────────────────────────────────
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = self.state[p]["e"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()


# ── CNN building blocks ───────────────────────────────────────────────────────

class ActorCNN(nn.Module):
    """
    CNN policy network.

    128×128 → 64×64 → 32×32 feature maps, then a 1×1 conv that produces
    per-pixel logits over the VQ-VAE codebook.

    Layer normalisation (over C×H×W per sample) is applied after each conv,
    matching the stream_ac reference which uses per-sample normalisation for
    training stability without batch statistics.
    """

    def __init__(self, num_codes: int = 64, hidden: int = 64):
        super().__init__()
        self.conv1       = nn.Conv2d(1,          hidden,     kernel_size=4, stride=2, padding=1)
        self.conv2       = nn.Conv2d(hidden,     hidden * 2, kernel_size=4, stride=2, padding=1)
        self.conv3       = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,           padding=1)
        self.policy_head = nn.Conv2d(hidden * 2, num_codes,  kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (1, H, W) uint8 tensor

        Returns:
            logits: (32, 32, num_codes)
        """
        x = x.float() / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)                                  # (1, 1, H, W)

        x = self.conv1(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv2(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv3(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))

        logits = self.policy_head(x)                            # (1, num_codes, 32, 32)
        return logits.squeeze(0).permute(1, 2, 0)              # (32, 32, num_codes)


class CriticCNN(nn.Module):
    """
    CNN value network.

    Same convolutional stack as the actor (separate weights), then global
    average pooling and a linear layer to produce a scalar V(s).
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1,          hidden,     kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden,     hidden * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,           padding=1)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (1, H, W) uint8 tensor

        Returns:
            value: scalar tensor V(s)
        """
        x = x.float() / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.conv1(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv2(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv3(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))

        x = self.pool(x).flatten(1)                             # (1, hidden*2)
        return self.fc(x).squeeze()                             # scalar


# ── Agent ─────────────────────────────────────────────────────────────────────

class PolicyAgent:
    """
    Stream AC(λ) agent.

    Per-step usage
    --------------
        action = agent.select_action(frame)      # (H,W,1) uint8 → (32,32) uint8
        # … apply action to environment …
        agent.record_reward(reward)
        agent.update(next_frame, done=terminated_or_truncated)
    """

    def __init__(
        self,
        num_codes:    int   = 64,
        hidden:       int   = 64,
        lr:           float = 1.0,    # ObGD is self-normalising; 1.0 is a safe default
        gamma:        float = 0.99,
        lamda:        float = 0.8,    # λ — trace decay
        entropy_coef: float = 0.01,
        kappa_policy: float = 3.0,    # overshooting scale for actor
        kappa_value:  float = 2.0,    # overshooting scale for critic
        device:       str   = None,
    ):
        """
        Args:
            num_codes    : VQ-VAE codebook size.
            hidden       : Base channel count for both CNN stacks.
            lr           : ObGD learning rate (shared for actor and critic).
            gamma        : Discount / trace-decay base γ.
            lamda        : Eligibility trace decay λ (the λ in AC(λ)).
            entropy_coef : Entropy bonus weight.
            kappa_policy : Overshooting-prevention scale for the actor ObGD.
            kappa_value  : Overshooting-prevention scale for the critic ObGD.
            device       : 'cpu' | 'cuda' | 'mps' (auto-detected if None).
        """
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

        # Separate networks — no shared encoder — matching the reference.
        # Each has its own ObGD state (independent eligibility traces).
        self.actor  = ActorCNN(num_codes=num_codes, hidden=hidden).to(self.device)
        self.critic = CriticCNN(hidden=hidden).to(self.device)

        self.opt_policy = ObGD(
            self.actor.parameters(),
            lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy,
        )
        self.opt_value = ObGD(
            self.critic.parameters(),
            lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value,
        )

        # One-step transition buffer.
        # _log_prob and _value are held graph-attached until update() so that
        # backward() can fill .grad for ObGD to consume.
        self._frame:          torch.Tensor = None
        self._log_prob:       torch.Tensor = None  # −log π(a|s)  (pre-negated)
        self._entropy:        torch.Tensor = None
        self._value:          torch.Tensor = None  # −V(s_t)      (pre-negated)
        self._current_reward: float        = 0.0

        self.stats = {
            "steps":        0,
            "last_reward":  0.0,
            "last_delta":   0.0,
            "last_entropy": 0.0,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def select_action(self, frame: np.ndarray) -> np.ndarray:
        """
        Forward pass through actor and critic.  Caches graph-attached tensors
        needed by update().  Do not call select_action again before update().

        Args:
            frame: (H, W, 1) uint8 numpy array

        Returns:
            action_grid: (32, 32) uint8 numpy array, values in [0, num_codes)
        """
        t = torch.from_numpy(frame).permute(2, 0, 1).to(self.device)  # (1, H, W)

        self.actor.train()
        self.critic.train()

        # ── Actor ─────────────────────────────────────────────────────
        logits      = self.actor(t)                         # (32, 32, num_codes)
        dist        = Categorical(logits=logits)
        action_grid = dist.sample()                         # (32, 32)

        # Pre-negate: ObGD does θ -= step*δ*e and e accumulates ∇loss.
        # For the policy we want θ += step*δ*∇log_π, so loss = -log π(a|s).
        self._log_prob = -dist.log_prob(action_grid).sum()  # scalar, graph-attached
        self._entropy  =  dist.entropy().mean()             # scalar, graph-attached

        # ── Critic ────────────────────────────────────────────────────
        # Pre-negate: grad of (-V) is -∇V; ObGD update θ -= step*δ*(-∇V)
        # = θ += step*δ*∇V, which moves V toward V+δ (correct TD direction).
        self._value = -self.critic(t)                       # scalar, graph-attached

        self._frame = t
        self.stats["steps"] += 1

        return action_grid.detach().cpu().numpy().astype(np.uint8)

    def record_reward(self, reward: float):
        """Store the reward received after the most recent select_action() call."""
        self._current_reward      = reward
        self.stats["last_reward"] = reward

    def update(self, next_frame: np.ndarray = None, done: bool = False):
        """
        Stream AC(λ) update — one ObGD step per environment step.

        The TD error δ = r + γV(s') − V(s) is the sole learning signal.
        It is passed directly to ObGD which uses it to scale the eligibility
        trace update; no separate loss value is computed.

        Args:
            next_frame : (H, W, 1) uint8 numpy array — observation after action.
                         Treated as terminal (V=0) when None or done=True.
            done       : True on episode end — resets eligibility traces.
        """
        if self._frame is None:
            return

        # ── 1. Bootstrap V(s') ────────────────────────────────────────
        with torch.no_grad():
            if done or next_frame is None:
                v_next = 0.0
            else:
                t_next = (torch.from_numpy(next_frame)
                          .permute(2, 0, 1).to(self.device))
                v_next = self.critic(t_next).item()

        # ── 2. TD error δ ─────────────────────────────────────────────
        # self._value is −V(s_t); negate to recover V(s_t).
        v_s   = -self._value.item()
        delta =  self._current_reward + self.gamma * v_next - v_s

        # ── 3. Actor loss → .grad for ObGD ────────────────────────────
        # policy_loss  = -log π(a|s)                  (already negated)
        #
        # Entropy bonus: multiply by sign(δ) so that:
        #   δ · sign(δ) · ∇H  =  |δ| · ∇H
        # This ensures the entropy bonus always pushes toward higher entropy,
        # regardless of whether the TD error is positive or negative.
        entropy_bonus = (
            self.entropy_coef
            * self._entropy
            * torch.sign(torch.tensor(delta, device=self.device))
        )
        policy_loss = self._log_prob - entropy_bonus

        self.opt_policy.zero_grad()
        policy_loss.backward()

        # ── 4. Critic loss → .grad for ObGD ──────────────────────────
        # value_loss = −V(s)  (already negated in select_action)
        self.opt_value.zero_grad()
        self._value.backward()

        # ── 5. ObGD step (δ scales + clips via eligibility traces) ────
        self.opt_policy.step(delta, reset=done)
        self.opt_value.step(delta, reset=done)

        # ── 6. Logging ────────────────────────────────────────────────
        self.stats["last_delta"]   = delta
        self.stats["last_entropy"] = self._entropy.item()

        # Clear transition buffer
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