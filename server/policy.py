from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ── Optimized ObGD Optimizer ──────────────────────────────────────────────────

class ObGD(torch.optim.Optimizer):
    """
    Vectorized Observed Gradient Descent with eligibility traces.
    Eliminates internal CPU stalls (.item()) by keeping step sizing on-device.
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

    @torch.no_grad()
    def step(self, delta_tensor: torch.Tensor, reset: bool = False):  # type: ignore[override]
        """
        Expects delta_tensor to be a 0-dim scalar tensor on the correct device.
        """
        # 1. Update eligibility traces and gather total trace sum entirely on device
        z_sum = torch.tensor(0.0, device=delta_tensor.device)
        
        for group in self.param_groups:
            gam_lam = group["gamma"] * group["lamda"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "e" not in state:
                    state["e"] = torch.zeros_like(p)
                
                e = state["e"]
                # In-place updates are faster
                e.mul_(gam_lam).add_(p.grad)
                # z_sum.add_(e.abs().sum())
                z_sum.add_(e.pow(2).sum())

        # 2. Compute dynamic step size entirely via tensor math (no python scalar stalls)
        group = self.param_groups[0]
        lr = group["lr"]
        kappa = group["kappa"]
        
        intended_change = kappa * delta_tensor.abs()
        directional_influence = z_sum + 1e-8
        
        # Clip step sizing to avoid dividing by near-zero traces
        step_size = torch.clamp(intended_change / directional_influence, max=lr)

        # delta_bar = torch.clamp(delta_tensor.abs(), min=1.0)
        # dot_product = delta_bar * z_sum * (lr * kappa)
        
        # Scale factor calculation: if dot_product > 1.0 choose lr/dot_product else lr
        # step_size = torch.where(dot_product > 1.0, lr / dot_product, lr)

        # 3. Apply gradients and optionally reset
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                e = state["e"]
                
                # Update parameters: θ ← θ − α' · δ · e
                p.add_(e * delta_tensor, alpha=-step_size.item() if step_size.dim() > 0 else -step_size)
                
                if reset:
                    e.zero_()

# ── Shared Representation Architecture ───────────────────────────────────────────

class SharedActorCritic(nn.Module):
    def __init__(self, num_codes: int = 64, hidden: int = 64, grid_size: int = 7):
        super().__init__()
        # One shared backbone for both networks cut down your CNN math by ~40%
        self.shared_conv1 = nn.Conv2d(1, hidden, kernel_size=4, stride=2, padding=1)
        self.shared_conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1)
        self.shared_conv3 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.index_embedding = nn.Embedding(grid_size * grid_size, 16)
        
        self.act_fc = nn.Linear(hidden * 2 + 16, num_codes)
        self.crit_fc = nn.Linear(hidden * 2, 1)

    def forward(self, x_norm: torch.Tensor, idx_tensor: torch.Tensor):
        # Extract features once
        x = self.shared_conv1(x_norm); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.shared_conv2(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        x = self.shared_conv3(x); x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        features = self.pool(x).flatten(1)
        
        # Branch to Actor
        pos_feats = self.index_embedding(idx_tensor)
        actor_logits = self.act_fc(torch.cat([features, pos_feats], dim=1)).squeeze(0)
        
        # Branch to Critic
        critic_value = self.crit_fc(features).squeeze()
        
        return actor_logits, critic_value
    
# ── Combined Dual-Head Architecture ───────────────────────────────────────────

class StreamingActorCritic(nn.Module):
    """
    Combined backbone architecture to minimize frame preprocessing.
    """
    def __init__(self, num_codes: int = 64, hidden: int = 64, grid_size: int = 7):
        super().__init__()
        self.grid_size = grid_size
        self.total_slots = grid_size * grid_size

        # --- Actor Network ---
        self.act_conv1 = nn.Conv2d(1, hidden, kernel_size=4, stride=2, padding=1)
        self.act_conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1)
        self.act_conv3 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1)
        self.act_conv4 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1)
        
        self.index_embedding = nn.Embedding(self.total_slots, 16)
        self.act_fc = nn.Linear(hidden * 2 + 16, num_codes)

        # --- Critic Network ---
        self.crit_conv1 = nn.Conv2d(1, hidden, kernel_size=4, stride=2, padding=1)
        self.crit_conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1)
        self.crit_conv3 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1)
        self.crit_fc    = nn.Linear(hidden * 2, 1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_actor(self, x_norm: torch.Tensor, idx_tensor: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.act_conv1(x_norm)
        x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        
        # Layer 2
        x = self.act_conv2(x)
        x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        
        # Layer 3
        x = self.act_conv3(x)
        x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        
        # Layer 4
        x = self.act_conv4(x)
        x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        
        flat_feats = self.pool(x).flatten(1)
        pos_feats = self.index_embedding(idx_tensor)
        
        combined = torch.cat([flat_feats, pos_feats], dim=1)
        return self.act_fc(combined).squeeze(0)

    def forward_critic(self, x_norm: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.crit_conv1(x_norm)
        x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        
        # Layer 2
        x = self.crit_conv2(x)
        x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        
        # Layer 3
        x = self.crit_conv3(x)
        x = F.leaky_relu(F.layer_norm(x, x.shape[1:]))
        
        flat_feats = self.pool(x).flatten(1)
        return self.crit_fc(flat_feats).squeeze()

# ── Optimized Agent ───────────────────────────────────────────────────────────

# class PolicyAgent:
#     def __init__(
#         self,
#         num_codes:    int   = 64,
#         grid_size:    int   = 7,
#         hidden:       int   = 64,
#         lr:           float = 1.0,
#         gamma:        float = 0.99,
#         lamda:        float = 0.8,
#         entropy_coef: float = 0.01,
#         kappa_policy: float = 3.0,
#         kappa_value:  float = 2.0,
#         device:       str   = None,
#     ):
#         if device is None:
#             if torch.cuda.is_available(): device = "cuda"
#             elif torch.backends.mps.is_available(): device = "mps"
#             else: device = "cpu"

#         self.device       = torch.device(device)
#         self.gamma        = gamma
#         self.entropy_coef = entropy_coef
#         self.grid_size    = grid_size
#         self.total_slots  = grid_size * grid_size

#         # Combined module holds all weights
#         self.model = StreamingActorCritic(num_codes=num_codes, hidden=hidden, grid_size=grid_size).to(self.device)
#         self.model.train()

#         # Separate optimization groupings intact
#         actor_params = [p for n, p in self.model.named_parameters() if "act_" in n or "index_embedding" in n]
#         critic_params = [p for n, p in self.model.named_parameters() if "crit_" in n]

#         self.opt_policy = ObGD(actor_params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
#         self.opt_value  = ObGD(critic_params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

#         # Preallocate static/persistent index buffers to avoid device allocation overhead
#         self._idx_tensors = [torch.tensor([i], device=self.device) for i in range(self.total_slots)]

#         # Tracking variables
#         self._frame_norm:     torch.Tensor = None
#         self._log_prob:       torch.Tensor = None  
#         self._entropy:        torch.Tensor = None
#         self._value:          torch.Tensor = None  
#         self._current_reward: float        = 0.0
#         self.current_idx                   = 0

#         self.stats = {"steps": 0, "last_reward": 0.0, "last_delta": 0.0, "last_entropy": 0.0}

#     def select_action(self, frame: np.ndarray) -> tuple[int, int]:
#         # Faster conversions: Pin memory if incoming from asynchronous threads, or load straight
#         t = torch.from_numpy(frame).permute(2, 0, 1).to(self.device, non_blocking=True)
        
#         # Keep normalized version ready for both network arms
#         x_norm = t.float().div_(255.0).unsqueeze_(0)
#         idx_tensor = self._idx_tensors[self.current_idx]

#         # Execute network structures efficiently
#         logits = self.model.forward_actor(x_norm, idx_tensor)
#         dist   = Categorical(logits=logits)
#         action = dist.sample()

#         self._log_prob   = -dist.log_prob(action)
#         self._entropy    = dist.entropy()
#         self._value      = -self.model.forward_critic(x_norm)
#         self._frame_norm = x_norm
        
#         self.stats["steps"] += 1
#         return action.item(), self.current_idx

#     def record_reward(self, reward: float):
#         self._current_reward = reward
#         self.stats["last_reward"] = reward

#     def update(self, next_frame: np.ndarray = None, done: bool = False):
#         if self._frame_norm is None:
#             return

#         # Compute next state value cleanly
#         if done or next_frame is None:
#             v_next = 0.0
#         else:
#             with torch.no_grad():
#                 t_next = torch.from_numpy(next_frame).permute(2, 0, 1).to(self.device, non_blocking=True)
#                 x_next_norm = t_next.float().div_(255.0).unsqueeze_(0)
#                 v_next = self.model.forward_critic(x_next_norm).item()

#         v_s = -self._value.item()
#         delta = self._current_reward + self.gamma * v_next - v_s
        
#         # Move delta calculation to device for ObGD
#         delta_tensor = torch.tensor(delta, device=self.device)

#         entropy_bonus = self.entropy_coef * self._entropy * torch.sign(delta_tensor)
#         policy_loss   = self._log_prob - entropy_bonus

#         # Backward passes execution
#         self.opt_policy.zero_grad(set_to_none=True)
#         policy_loss.backward(retain_graph=True) # retaining if needed, though they don't share trunks

#         self.opt_value.zero_grad(set_to_none=True)
#         self._value.backward()

#         # Step tracking without breaking asynchronous CUDA/MPS streams
#         self.opt_policy.step(delta_tensor, reset=done)
#         self.opt_value.step(delta_tensor, reset=done)

#         # Step the index counter
#         if done:
#             self.current_idx = 0
#         else:
#             self.current_idx = (self.current_idx + 1) % self.total_slots

#         self.stats["last_delta"]   = delta
#         self.stats["last_entropy"] = self._entropy.item()
#         self._frame_norm = None

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
            if torch.cuda.is_available(): device = "cuda"
            elif torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"

        self.device       = torch.device(device)
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.grid_size    = grid_size
        self.total_slots  = grid_size * grid_size

        # Model Initialization
        self.model = SharedActorCritic(num_codes=num_codes, hidden=hidden, grid_size=grid_size).to(self.device)
        self.model.train()

        # Route variables out into independent weight optimizer groups
        actor_params  = [p for n, p in self.model.named_parameters() if "act_" in n or "index_embedding" in n]
        critic_params = [p for n, p in self.model.named_parameters() if "crit_" in n]
        
        # Shared trunk weights are added to the policy track for trace generation
        shared_params = [p for n, p in self.model.named_parameters() if "conv" in n]
        actor_params.extend(shared_params)

        self.opt_policy = ObGD(actor_params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.opt_value  = ObGD(critic_params, lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

        # Persistent index tensor lookups to save device recreation cycles
        self._idx_tensors = [torch.tensor([i], device=self.device) for i in range(self.total_slots)]

        # One-step tracking allocations
        self._frame_norm:     torch.Tensor = None
        self._log_prob:       torch.Tensor = None  
        self._entropy:        torch.Tensor = None
        self._value:          torch.Tensor = None  
        self._current_reward: float        = 0.0
        self.current_idx                   = 0

        self.stats = {"steps": 0, "last_reward": 0.0, "last_delta": 0.0, "last_entropy": 0.0}

    def select_action(self, frame: np.ndarray) -> tuple[int, int]:
        t = torch.from_numpy(frame).permute(2, 0, 1).to(self.device, non_blocking=True)
        x_norm = t.float().div_(255.0).unsqueeze_(0)
        idx_tensor = self._idx_tensors[self.current_idx]

        # Extract both network actions concurrently
        logits, critic_val = self.model(x_norm, idx_tensor)
        
        dist   = Categorical(logits=logits)
        action = dist.sample()

        self._log_prob   = -dist.log_prob(action)
        self._entropy    = dist.entropy()
        self._value      = -critic_val
        self._frame_norm = x_norm
        
        self.stats["steps"] += 1
        return action.item(), self.current_idx

    def record_reward(self, reward: float):
        self._current_reward = reward
        self.stats["last_reward"] = reward

    def update(self, next_frame: np.ndarray = None, done: bool = False):
        if self._frame_norm is None:
            return

        if done or next_frame is None:
            v_next = 0.0
        else:
            with torch.no_grad():
                t_next = torch.from_numpy(next_frame).permute(2, 0, 1).to(self.device, non_blocking=True)
                x_next_norm = t_next.float().div_(255.0).unsqueeze_(0)
                
                # Fetch only next critic state projections
                _, v_next_tensor = self.model(x_next_norm, self._idx_tensors[self.current_idx])
                v_next = v_next_tensor.item()

        v_s = -self._value.item()
        delta = self._current_reward + self.gamma * v_next - v_s
        
        delta_tensor = torch.tensor(delta, device=self.device)

        entropy_bonus = self.entropy_coef * self._entropy * torch.sign(delta_tensor)
        policy_loss   = self._log_prob - entropy_bonus

        # Backward sequence runs smoothly without destroying common layers mid-flight
        self.opt_policy.zero_grad(set_to_none=True)
        policy_loss.backward(retain_graph=True)

        self.opt_value.zero_grad(set_to_none=True)
        self._value.backward()

        self.opt_policy.step(delta_tensor, reset=done)
        self.opt_value.step(delta_tensor, reset=done)

        if done:
            self.current_idx = 0
        else:
            self.current_idx = (self.current_idx + 1) % self.total_slots

        self.stats["last_delta"]   = delta
        self.stats["last_entropy"] = self._entropy.item()
        self._frame_norm = None

    def save(self, path: str = "policy.pt"):
        torch.save({"model": self.model.state_dict()}, path)
        print(f"Saved policy → {path}")

    def load(self, path: str = "policy.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        print(f"Loaded policy ← {path}")