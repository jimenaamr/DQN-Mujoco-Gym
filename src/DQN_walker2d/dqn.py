# src/DQN_walker2d/dqn.py

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_STEP_ANYWHERE_RE: re.Pattern[str] = re.compile(pattern=r"step_(\d+)_.*\.pt$")


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
            nn.Conv2d(in_channels=int(c), out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy: torch.Tensor = torch.zeros(
                size=(1, int(c), int(h), int(w)), dtype=torch.float32
            )
            out: torch.Tensor = self.conv(dummy)
            flat_dim: int = int(out.reshape(1, -1).shape[1])

        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=int(n_actions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values.

        Args:
            x: (B, C, H, W) uint8 in [0,255] or float in [0,1].

        Returns:
            (B, n_actions) Q-values.
        """
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        elif x.dtype != torch.float32:
            x = x.float()

        y: torch.Tensor = self.conv(x)
        y = y.reshape(y.shape[0], -1)
        return self.head(y)


class _ReplayBuffer:
    """Simple numpy replay buffer for pixel observations."""

    def __init__(self, capacity: int, obs_shape: tuple[int, int, int]) -> None:
        self.capacity: int = int(capacity)
        self.obs_shape: tuple[int, int, int] = obs_shape

        self.obs: np.ndarray = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self.next_obs: np.ndarray = np.zeros(
            (self.capacity, *obs_shape), dtype=np.uint8
        )
        self.actions: np.ndarray = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards: np.ndarray = np.zeros((self.capacity,), dtype=np.float32)

        # uint8 is cheaper than bool_ and converts cleanly to float mask
        self.dones: np.ndarray = np.zeros((self.capacity,), dtype=np.uint8)

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
        self.dones[self.pos] = np.uint8(1 if done else 0)

        self.pos = int((self.pos + 1) % self.capacity)
        self.size = int(min(self.size + 1, self.capacity))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions."""
        bs: int = int(batch_size)
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx: np.ndarray = np.random.randint(low=0, high=self.size, size=(bs,))
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
            self.device
        )
        self.q_target: _QNet = _QNet(obs_shape=obs_shape, n_actions=self.n_actions).to(
            self.device
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
        self.updates: int = 0

    def act(self, obs: np.ndarray, eval_mode: bool) -> int:
        """Select an action with epsilon-greedy exploration."""
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
            return int(torch.argmax(qvals, dim=1).item())

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
        if int(self.global_step) < int(self.cfg.learning_starts):
            return False
        if int(self.buffer.size) < int(self.cfg.batch_size):
            return False
        return (int(self.global_step) % int(self.cfg.train_freq)) == 0

    def update(self) -> dict[str, float]:
        """Run a single DQN update step (Double DQN + target sync by updates)."""
        batch: dict[str, np.ndarray] = self.buffer.sample(
            batch_size=int(self.cfg.batch_size)
        )

        obs: torch.Tensor = torch.as_tensor(batch["obs"], device=self.device)
        next_obs: torch.Tensor = torch.as_tensor(batch["next_obs"], device=self.device)

        actions: torch.Tensor = torch.as_tensor(
            batch["actions"], device=self.device
        ).long()
        rewards: torch.Tensor = torch.as_tensor(
            batch["rewards"], device=self.device
        ).float()
        dones: torch.Tensor = torch.as_tensor(
            batch["dones"], device=self.device
        ).float()

        q_a: torch.Tensor = (
            self.q(obs).gather(dim=1, index=actions.view(-1, 1)).squeeze(1)
        )

        with torch.no_grad():
            a_star: torch.Tensor = self.q(next_obs).argmax(dim=1, keepdim=True)
            q_next: torch.Tensor = (
                self.q_target(next_obs).gather(dim=1, index=a_star).squeeze(1)
            )
            target: torch.Tensor = (
                rewards + (1.0 - dones) * float(self.cfg.gamma) * q_next
            )

        loss: torch.Tensor = F.smooth_l1_loss(q_a, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        if float(self.cfg.grad_clip_norm) > 0.0:
            nn.utils.clip_grad_norm_(
                self.q.parameters(), max_norm=float(self.cfg.grad_clip_norm)
            )

        self.opt.step()

        self.updates += 1
        if (self.updates % int(self.cfg.target_update_freq)) == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        eps: float = float(
            _linear_epsilon(
                step=int(self.global_step),
                eps_start=float(self.cfg.eps_start),
                eps_end=float(self.cfg.eps_end),
                decay_steps=int(self.cfg.eps_decay_steps),
            )
        )

        return {
            "loss": float(loss.detach().cpu().item()),
            "q_mean": float(q_a.detach().mean().cpu().item()),
            "epsilon": float(eps),
        }

    def save(self, path: str) -> None:
        """Save agent parameters and training state to disk."""
        payload: dict[str, Any] = {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "opt": self.opt.state_dict(),
            "global_step": int(self.global_step),
            "updates": int(self.updates),
            "cfg": dict(self.cfg.__dict__),
            "q_state": self.q.state_dict(),
            "q_target_state": self.q_target.state_dict(),
            "opt_state": self.opt.state_dict(),
        }
        torch.save(obj=payload, f=str(path))

    def load(self, path: str) -> int:
        """Load agent weights and optimizer state from a checkpoint.

        Args:
            path: Checkpoint path.

        Returns:
            The loaded global step (0 if missing and filename doesn't encode it).
        """
        ckpt: dict[str, Any] = torch.load(f=str(path), map_location=self.device)

        q_state: dict[str, Any] | None = ckpt.get("q")
        if q_state is None:
            q_state = ckpt.get("q_state")
        if q_state is None:
            raise KeyError(
                f"Checkpoint missing Q state. Keys: {sorted(list(ckpt.keys()))}"
            )
        self.q.load_state_dict(state_dict=q_state)

        q_target_state: dict[str, Any] | None = ckpt.get("q_target")
        if q_target_state is None:
            q_target_state = ckpt.get("q_target_state")
        if q_target_state is not None:
            self.q_target.load_state_dict(state_dict=q_target_state)

        opt_state: dict[str, Any] | None = ckpt.get("opt")
        if opt_state is None:
            opt_state = ckpt.get("opt_state")
        if opt_state is not None:
            self.opt.load_state_dict(state_dict=opt_state)

        loaded_step: int = 0
        if "global_step" in ckpt and ckpt.get("global_step") is not None:
            loaded_step = int(ckpt["global_step"])
        else:
            m: re.Match[str] | None = _STEP_ANYWHERE_RE.search(Path(path).name)
            if m is not None:
                loaded_step = int(m.group(1))

        self.global_step = int(loaded_step)
        self.updates = int(ckpt.get("updates", 0))
        return int(loaded_step)


def _linear_epsilon(
    step: int, eps_start: float, eps_end: float, decay_steps: int
) -> float:
    """Linear epsilon schedule."""
    if decay_steps <= 0:
        return float(eps_end)
    t: float = float(np.clip(step / decay_steps, 0.0, 1.0))
    return float((1.0 - t) * eps_start + t * eps_end)
