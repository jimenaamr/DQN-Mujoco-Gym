# src/DQN_walker2d/dqn.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DQNConfig:
    """Hyperparameters and runtime options for DQN."""

    gamma: float
    lr: float
    batch_size: int
    buffer_size: int
    learning_starts: int
    train_freq: int
    target_update_freq: int
    grad_clip_norm: float

    eps_start: float
    eps_end: float
    eps_decay_steps: int

    device: str


class _QNet(nn.Module):
    """Simple CNN Q-network for pixel observations (C,H,W)."""

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int) -> None:
        super().__init__()
        c: int
        h: int
        w: int
        c, h, w = obs_shape

        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy: torch.Tensor = torch.zeros(size=(1, c, h, w), dtype=torch.float32)
            out: torch.Tensor = self.conv(dummy)
            flat_dim: int = int(out.view(1, -1).shape[1])

        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=int(n_actions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values.

        Args:
            x: Input tensor with shape (B, C, H, W) in [0,255] or [0,1].

        Returns:
            Q-values with shape (B, n_actions).
        """
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        y: torch.Tensor = self.conv(x)
        y = y.view(y.shape[0], -1)
        return self.head(y)


class _ReplayBuffer:
    """Simple numpy replay buffer for pixel observations."""

    def __init__(self, capacity: int, obs_shape: tuple[int, int, int]) -> None:
        self.capacity: int = int(capacity)
        self.obs_shape: tuple[int, int, int] = obs_shape

        self.obs: np.ndarray = np.zeros(
            shape=(self.capacity, *obs_shape), dtype=np.uint8
        )
        self.next_obs: np.ndarray = np.zeros(
            shape=(self.capacity, *obs_shape), dtype=np.uint8
        )
        self.actions: np.ndarray = np.zeros(shape=(self.capacity,), dtype=np.int64)
        self.rewards: np.ndarray = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.dones: np.ndarray = np.zeros(shape=(self.capacity,), dtype=np.bool_)

        self.size: int = 0
        self.pos: int = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Insert a transition into the buffer."""
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = bool(done)

        self.pos = int((self.pos + 1) % self.capacity)
        self.size = int(min(self.size + 1, self.capacity))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of samples.

        Returns:
            Batch dict of numpy arrays.
        """
        idx: np.ndarray = np.random.randint(low=0, high=self.size, size=(batch_size,))
        return {
            "obs": self.obs[idx],
            "next_obs": self.next_obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
        }


class DQNAgent:
    """DQN agent with epsilon-greedy policy and target network."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        cfg: DQNConfig,
    ) -> None:
        self.cfg: DQNConfig = cfg
        self.device: torch.device = torch.device(str(cfg.device))

        self.n_actions: int = int(n_actions)
        self.q: _QNet = _QNet(obs_shape=obs_shape, n_actions=self.n_actions).to(
            device=self.device
        )
        self.q_target: _QNet = _QNet(obs_shape=obs_shape, n_actions=self.n_actions).to(
            device=self.device
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt: torch.optim.Optimizer = torch.optim.Adam(
            params=self.q.parameters(),
            lr=float(cfg.lr),
        )

        self.buffer: _ReplayBuffer = _ReplayBuffer(
            capacity=int(cfg.buffer_size),
            obs_shape=obs_shape,
        )

        self.global_step: int = 0

    def act(self, obs: np.ndarray, eval_mode: bool) -> int:
        """Select an action with epsilon-greedy exploration.

        Args:
            obs: Observation (C,H,W) uint8.
            eval_mode: If True, disable exploration.

        Returns:
            Discrete action index.
        """
        if eval_mode:
            eps: float = 0.0
        else:
            eps = float(
                _linear_epsilon(
                    step=int(self.global_step),
                    eps_start=float(self.cfg.eps_start),
                    eps_end=float(self.cfg.eps_end),
                    decay_steps=int(self.cfg.eps_decay_steps),
                )
            )

        if (not eval_mode) and (float(np.random.rand()) < eps):
            return int(np.random.randint(low=0, high=self.n_actions))

        with torch.no_grad():
            x: torch.Tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            qvals: torch.Tensor = self.q(x)
            a: int = int(torch.argmax(qvals, dim=1).item())
        return a

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in replay buffer."""
        self.buffer.add(
            obs=obs,
            action=int(action),
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
        )

    def can_update(self) -> bool:
        """Check whether the agent should run a gradient update."""
        if int(self.buffer.size) < int(self.cfg.learning_starts):
            return False
        return (int(self.global_step) % int(self.cfg.train_freq)) == 0

    def update(self) -> dict[str, float]:
        """Run a single DQN update step.

        Returns:
            Dict with scalar metrics for logging.
        """
        batch: dict[str, np.ndarray] = self.buffer.sample(
            batch_size=int(self.cfg.batch_size)
        )

        obs: torch.Tensor = torch.as_tensor(batch["obs"], device=self.device)
        next_obs: torch.Tensor = torch.as_tensor(batch["next_obs"], device=self.device)
        actions: torch.Tensor = torch.as_tensor(batch["actions"], device=self.device)
        rewards: torch.Tensor = torch.as_tensor(batch["rewards"], device=self.device)
        dones: torch.Tensor = torch.as_tensor(batch["dones"], device=self.device)

        qvals: torch.Tensor = self.q(obs)
        q_a: torch.Tensor = qvals.gather(dim=1, index=actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_qvals: torch.Tensor = self.q_target(next_obs)
            next_max: torch.Tensor = next_qvals.max(dim=1).values
            target: torch.Tensor = (
                rewards + (1.0 - dones.float()) * float(self.cfg.gamma) * next_max
            )

        loss: torch.Tensor = F.smooth_l1_loss(q_a, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.q.parameters(), max_norm=float(self.cfg.grad_clip_norm)
        )
        self.opt.step()

        if (int(self.global_step) % int(self.cfg.target_update_freq)) == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return {"loss": float(loss.detach().cpu().item())}

    def save(self, path: str) -> None:
        """Save agent parameters to disk.

        Args:
            path: Output checkpoint path.
        """
        payload: dict[str, Any] = {
            "q_state": self.q.state_dict(),
            "q_target_state": self.q_target.state_dict(),
            "opt_state": self.opt.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        torch.save(obj=payload, f=str(path))

    def load(self, path: str) -> None:
        """Load agent weights and optimizer state from a checkpoint."""
        ckpt: dict[str, Any] = torch.load(path, map_location=self.device)

        q_state: dict[str, Any] | None = ckpt.get("q")
        if q_state is None:
            q_state = ckpt.get("q_state")

        if q_state is None:
            raise KeyError(f"Checkpoint missing Q state. Keys: {list(ckpt.keys())}")

        self.q.load_state_dict(q_state)

        q_target_state: dict[str, Any] | None = ckpt.get("q_target")
        if q_target_state is not None and hasattr(self, "q_target"):
            self.q_target.load_state_dict(q_target_state)

        opt_state: dict[str, Any] | None = ckpt.get("opt")
        if opt_state is not None and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(opt_state)


def _linear_epsilon(
    step: int, eps_start: float, eps_end: float, decay_steps: int
) -> float:
    """Linear epsilon schedule.

    Args:
        step: Current global step.
        eps_start: Initial epsilon.
        eps_end: Final epsilon.
        decay_steps: Steps over which epsilon decays linearly.

    Returns:
        Epsilon value at `step`.
    """
    if decay_steps <= 0:
        return float(eps_end)
    t: float = float(np.clip(step / decay_steps, 0.0, 1.0))
    return float((1.0 - t) * eps_start + t * eps_end)
