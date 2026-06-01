import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        gamma: float = 0.999,
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
                e.mul_(gam_lam).add_(p.grad)
                z_sum.add_(e.pow(2).sum())

        # 2. Compute dynamic step size entirely via tensor math (no python scalar stalls)
        group = self.param_groups[0]
        lr    = group["lr"]
        kappa = group["kappa"]

        intended_change       = kappa * delta_tensor.abs()
        directional_influence = z_sum + 1e-8
        step_size = torch.clamp(intended_change / directional_influence, max=lr)

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

class AdaptiveObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, beta2=beta2, eps=eps)
        self.counter = 0
        super(AdaptiveObGD, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        z_sum = 0.0
        self.counter += 1
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                e, v = state["eligibility_trace"], state["v"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)

                v.mul_(group["beta2"]).addcmul_(delta*e, delta*e, value=1.0 - group["beta2"])
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                z_sum += (e / (v_hat + group["eps"]).sqrt()).abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                v, e = state["v"], state["eligibility_trace"]
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                p.data.addcdiv_(delta * e, (v_hat + group["eps"]).sqrt(), value=-step_size)
                if reset:
                    e.zero_()

# ── Stream Actor Critic ──────────────────────────────────────────────────

class StreamAC(nn.Module):
    def __init__(self, num_codes: int = 64,
                 hidden: int = 64,
                 grid_size: int = 7,
                 lr: float = 1.0,
                 gamma: float = 0.99,
                 lamda: float = 0.8,
                 kappa_policy: float = 3.0,
                 kappa_value: float = 2.0,
                 entropy_coeff: float = 0.01,
                 device: str = None):
        super().__init__()

        if device is None:
            if torch.cuda.is_available(): device = "cuda"
            elif torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"
            print("Device: ", device)
       
        self.device = torch.device(device)
        print("Device: ", self.device)

        self.gamma         = gamma
        self.grid_size     = grid_size
        self.entropy_coeff = entropy_coeff
        self.n_cells = self.grid_size * self.grid_size
 
        feat_dim = hidden * 2  # 128
 
        # Shared CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([hidden, 64, 64]),
            nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([hidden * 2, 32, 32]),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([hidden * 2, 16, 16]),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
 
        # Shared GRU — stepped grid_size * grid_size times per frame.
        self.gru = nn.GRUCell(feat_dim, feat_dim)
 
        # Actor and critic heads both read from the same recurrent features.
        self.network_policy_head = nn.Linear(feat_dim, num_codes)
        self.network_value_head  = nn.Linear(feat_dim, 1)
 
        # self.optimizer_policy = ObGD(
        #     list(self.backbone.parameters()) +
        #     list(self.gru.parameters()) +
        #     list(self.network_policy_head.parameters()),
        #     lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy,
        # )
        # self.optimizer_value = ObGD(
        #     list(self.backbone.parameters()) +
        #     list(self.gru.parameters()) +
        #     list(self.network_value_head.parameters()),
        #     lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value,
        # )

        self.optimizer_policy = AdaptiveObGD(
            list(self.backbone.parameters()) +
            list(self.gru.parameters()) +
            list(self.network_policy_head.parameters()),
            lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy,
        )
        self.optimizer_value = AdaptiveObGD(
            list(self.backbone.parameters()) +
            list(self.gru.parameters()) +
            list(self.network_value_head.parameters()),
            lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value,
        )

        # Shared recurrent hidden state — persists across frames, never reset
        self.h: torch.Tensor = torch.zeros(1, feat_dim)
 
        # One-step tracking allocations
        self._frame_norm:     torch.Tensor = None
        self._log_prob:       torch.Tensor = None
        self._entropy:        torch.Tensor = None
        self._value:          torch.Tensor = None
        self._current_reward: float        = 0.0
        self.current_idx                   = 0
 
        self.stats = {"steps": 0,  "updates": 0, "last_reward": 0.0, "last_delta": 0.0, "last_entropy": 0.0}
 
    def _features(self, x_norm: torch.Tensor):
        x = self.backbone(x_norm)
        return self.pool(x).flatten(1)                        # (B, feat_dim)
 
    def _rollout(self, x_norm: torch.Tensor):
        """
        Steps the shared GRU across all grid_size² cells, updating self.h.
 
        Returns:
            all_h: shape (B, grid_size², feat_dim) — hidden state at each cell
        """
        features = self._features(x_norm)                     # (B, feat_dim)
 
        all_h = []
        for _ in range(self.n_cells):
            self.h = self.gru(features, self.h)
            all_h.append(self.h)
 
        return torch.stack(all_h, dim=1)                      # (B, N, feat_dim)
 
    def pi(self, x_norm: torch.Tensor):
        """
        Returns:
            all_probs: shape (B, grid_size², num_codes) — softmax over codes
        """
        all_h = self._rollout(x_norm)                         # (B, N, feat_dim)
        logits = self.network_policy_head(all_h)              # (B, N, num_codes) — linear
        return F.softmax(logits, dim=-1)
 
    def v(self, x_norm: torch.Tensor):
        """
        Returns:
            all_values: shape (B, grid_size²) — linear function of recurrent features
        """
        all_h = self._rollout(x_norm)                         # (B, N, feat_dim)
        return self.network_value_head(all_h).squeeze(-1)     # (B, N)
 
    def sample_action(self, x_norm: torch.Tensor):
        """
        Runs the full grid rollout and samples one code per cell.
 
        Returns:
            grid (Tensor): sampled code indices, shape (grid_size, grid_size)
        """
        all_probs = self.pi(x_norm)                           # (B, N, num_codes)
        dist = torch.distributions.Categorical(all_probs[0])  # batch over N cells
        action = dist.sample().view(self.grid_size, self.grid_size)
        # Detach so the inference graph built during sample_action does not
        # persist into the next update_params call via the shared self.h state.
        self.h = self.h.detach()
        return action
    
    def greedy_action(self, x_norm: torch.Tensor):
        """
        Runs the full grid rollout and selects the highest-probability code per cell.

        Returns:
            grid (Tensor): greedy code indices, shape (grid_size, grid_size)
        """
        all_probs = self.pi(x_norm)                                    # (B, N, num_codes)
        action = all_probs[0].argmax(dim=-1).view(self.grid_size, self.grid_size)
        self.h = self.h.detach()
        return action

    def epsilon_greedy_action(self, x_norm: torch.Tensor, epsilon: float = 0.1):
        """
        Runs the full grid rollout and selects greedily with probability (1 - epsilon),
        otherwise samples uniformly at random per cell.

        Returns:
            grid (Tensor): code indices, shape (grid_size, grid_size)
        """
        all_probs = self.pi(x_norm)                                    # (B, N, num_codes)
        if torch.rand(1).item() < epsilon:
            num_codes = all_probs.shape[-1]
            action = torch.randint(num_codes, (self.grid_size, self.grid_size))
        else:
            action = all_probs[0].argmax(dim=-1).view(self.grid_size, self.grid_size)
        self.h = self.h.detach()
        return action
 
    def update_params(self, s, a, r, s_prime):
        s       = torch.tensor(np.array(s), dtype=torch.float32)
        a       = torch.tensor(np.array(a))
        r       = torch.tensor(np.array(r))
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float32)
 
        # Compute TD error
        v_s     = self.v(s)[0, -1]        
        v_prime = self.v(s_prime)[0, -1].detach()
        delta   = r + self.gamma * v_prime - v_s
 
        # Compute policy loss
        probs = self.pi(s)
        dist  = torch.distributions.Categorical(probs)

        log_prob_pi = -(dist.log_prob(a)[0, -1]).sum()
        entropy_pi  = -self.entropy_coeff * dist.entropy()[0, -1].sum() * torch.sign(delta).item()
  
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()

        (-v_s).backward(retain_graph=True)
        (log_prob_pi + entropy_pi).backward()

        self.optimizer_policy.step(delta.detach(), reset=False)
        self.optimizer_value.step(delta.detach(), reset=False)

        # Detach self.h so the training graph built this step doesn't chain into
        # the next step's rollout. Without this, self.h retains a reference into
        # the freed graph and the second call to .backward() crashes.
        self.h = self.h.detach()