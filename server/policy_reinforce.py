"""
PolicyNetwork — CNN that maps an input frame to a VQ-VAE action grid.

Input:  (1, 128, 128) grayscale frame
Output: (32, 32) action grid with values in [0, num_codes)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, num_codes: int = 64, hidden: int = 64):
        super().__init__()
        self.num_codes = num_codes

        # 128x128 → 64x64 → 32x32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(hidden * 2, num_codes, kernel_size=1)

    def forward(self, frame: torch.Tensor):
        x = frame.float() / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)

        logits = self.head(self.encoder(x))           # (1, num_codes, 32, 32)
        logits = logits.squeeze(0).permute(1, 2, 0)   # (32, 32, num_codes)

        dist        = Categorical(logits=logits)
        action_grid = dist.sample()
        log_prob    = dist.log_prob(action_grid).sum()
        entropy     = dist.entropy().mean()

        return action_grid, log_prob, entropy


class PolicyAgent:
    def __init__(self, num_codes: int = 64, lr: float = 1e-3,
                 entropy_coef: float = 0.01, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device       = torch.device(device)
        self.entropy_coef = entropy_coef
        self.net          = PolicyNetwork(num_codes=num_codes).to(self.device)
        self.optimizer    = torch.optim.Adam(self.net.parameters(), lr=lr)

        self._log_prob       = None
        self._entropy        = None
        self._current_reward = 0.0
        self.stats = {"steps": 0, "last_reward": 0.0, "last_loss": 0.0}

    def select_action(self, frame: np.ndarray) -> np.ndarray:
        """
        Forward pass. Call update() after each call to this method.

        Args:
            frame: (H, W, 1) uint8 numpy array

        Returns:
            action_grid: (32, 32) numpy int array, values in [0, num_codes)
        """
        t = torch.from_numpy(frame).permute(2, 0, 1).to(self.device)

        self.net.train()
        action_grid, log_prob, entropy = self.net(t)

        self._log_prob = log_prob
        self._entropy  = entropy
        self.stats["steps"] += 1

        return action_grid.cpu().numpy().astype(np.uint8)

    def record_reward(self, reward: float):
        """Store the latest reward. Takes effect on the next update() call."""
        self._current_reward      = reward
        self.stats["last_reward"] = reward

    def update(self):
        """REINFORCE update. Call once per timestep after select_action()."""
        if self._log_prob is None:
            return

        loss = -(self._log_prob * self._current_reward
                 + self.entropy_coef * self._entropy)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.stats["last_loss"] = loss.item()
        self._log_prob = None
        self._entropy  = None

    def save(self, path: str = "policy.pt"):
        torch.save(self.net.state_dict(), path)
        print(f"Saved policy → {path}")

    def load(self, path: str = "policy.pt"):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded policy ← {path}")