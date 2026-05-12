"""
agent.py — JAX/Flax RL agent with human-in-the-loop reward.

Architecture:
  - Small CNN encoder  →  flattened features
  - MLP policy head    →  discrete action logits
  - REINFORCE update triggered after each complete episode
    (episode = buffer fills up OR explicit done signal)

Human reward:
  The reward sent from the iOS app is accumulated into the current
  trajectory.  If no human reward arrives for a step, the environment
  reward defaults to 0.0.  The update uses the sum across the buffer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.training import train_state

# ── Hyperparameters ──────────────────────────────────────────────────────────

IMAGE_H = 128
IMAGE_W = 128
IMAGE_C = 3

# Discrete action space — customise these labels for your task
ACTION_LABELS = ["forward", "backward", "turn_left", "turn_right", "stop"]
NUM_ACTIONS = len(ACTION_LABELS)

BUFFER_SIZE = 32          # steps before a gradient update
GAMMA = 0.99              # discount factor
LEARNING_RATE = 3e-4

# ── Network ──────────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    """Tiny CNN → MLP policy that fits comfortably in CPU RAM."""

    num_actions: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: float32 image tensor, shape (H, W, C), values in [0, 1].
        Returns:
            logits: shape (num_actions,)
        """
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape(-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x


# ── Rollout buffer ────────────────────────────────────────────────────────────

@dataclass
class Transition:
    obs: np.ndarray        # (H, W, C)  float32
    action: int
    log_prob: float
    reward: float = 0.0    # filled in later by human signal


@dataclass
class RolloutBuffer:
    transitions: List[Transition] = field(default_factory=list)

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def set_last_reward(self, reward: float) -> None:
        """Assign reward to the most recently added transition."""
        if self.transitions:
            self.transitions[-1].reward += reward

    def is_full(self, capacity: int) -> bool:
        return len(self.transitions) >= capacity

    def clear(self) -> None:
        self.transitions.clear()

    def discounted_returns(self, gamma: float) -> np.ndarray:
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        returns = np.zeros_like(rewards)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + gamma * running
            returns[i] = running
        # Normalise for training stability
        std = returns.std()
        if std > 1e-8:
            returns = (returns - returns.mean()) / std
        return returns


# ── Training state ────────────────────────────────────────────────────────────

def create_train_state(rng: jax.Array, learning_rate: float) -> train_state.TrainState:
    model = PolicyNet(num_actions=NUM_ACTIONS)
    dummy = jnp.zeros((IMAGE_H, IMAGE_W, IMAGE_C))
    params = model.init(rng, dummy)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def policy_loss_and_grad(params, apply_fn, obs_batch, actions, returns):
    """
    REINFORCE loss:  -E[ log π(a|s) · G_t ]
    """
    def loss_fn(params):
        logits = jax.vmap(lambda o: apply_fn(params, o))(obs_batch)
        log_probs = jax.nn.log_softmax(logits)
        chosen_log_probs = log_probs[jnp.arange(len(actions)), actions]
        return -jnp.mean(chosen_log_probs * returns)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads


# ── Agent ─────────────────────────────────────────────────────────────────────

class RLAgent:
    """Thread-safe RL agent.  Inference happens on the WebSocket thread;
    training happens in a background thread to avoid stalling the server."""

    def __init__(self):
        rng = jax.random.PRNGKey(0)
        self.state = create_train_state(rng, LEARNING_RATE)
        self.buffer = RolloutBuffer()
        self._lock = threading.Lock()
        self._step = 0
        self._total_updates = 0

    # ── Inference ────────────────────────────────────────────────────────────

    def select_action(self, frame: np.ndarray) -> tuple[int, float, str]:
        """
        Args:
            frame: uint8 numpy array, shape (H, W, C)
        Returns:
            action_idx, log_prob, action_label
        """
        obs = frame.astype(np.float32) / 255.0             # normalise to [0,1]
        logits = self.state.apply_fn(self.state.params, jnp.array(obs))
        logits_np = np.array(logits)

        # Sample from the policy distribution
        probs = np.exp(logits_np - logits_np.max())
        probs /= probs.sum()
        action = int(np.random.choice(NUM_ACTIONS, p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))

        # Record transition (reward will be patched in later)
        with self._lock:
            self.buffer.add(Transition(obs=obs, action=action, log_prob=log_prob))
            self._step += 1

        return action, log_prob, ACTION_LABELS[action]

    # ── Human reward ─────────────────────────────────────────────────────────

    def record_reward(self, value: float) -> None:
        """Add a human reward signal to the latest transition."""
        with self._lock:
            self.buffer.set_last_reward(value)

        # Trigger an update if the buffer is full
        if self.buffer.is_full(BUFFER_SIZE):
            self._run_update()

    # ── Training update ───────────────────────────────────────────────────────

    def _run_update(self) -> None:
        """Perform a REINFORCE gradient update in a background thread."""
        with self._lock:
            transitions = list(self.buffer.transitions)
            returns = self.buffer.discounted_returns(GAMMA)
            self.buffer.clear()

        threading.Thread(target=self._update_step,
                         args=(transitions, returns),
                         daemon=True).start()

    def _update_step(self, transitions: List[Transition], returns: np.ndarray) -> None:
        obs_batch = jnp.array(np.stack([t.obs for t in transitions]))
        actions   = jnp.array([t.action for t in transitions])
        returns_j = jnp.array(returns)

        loss, grads = policy_loss_and_grad(
            self.state.params, self.state.apply_fn,
            obs_batch, actions, returns_j
        )

        with self._lock:
            self.state = self.state.apply_gradients(grads=grads)
            self._total_updates += 1

        print(f"[agent] update {self._total_updates:4d} | "
              f"steps {self._step:6d} | loss {float(loss):.4f}")

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "steps": self._step,
            "updates": self._total_updates,
            "buffer_len": len(self.buffer.transitions),
        }
