from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DQNConfig:
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


def linear_epsilon(
    step: int, eps_start: float, eps_end: float, decay_steps: int
) -> float:
    if step >= decay_steps:
        return eps_end
    frac: float = step / float(decay_steps)
    return eps_start + frac * (eps_end - eps_start)


class ReplayBuffer:
    def __init__(self, obs_shape: tuple[int, int, int], size: int) -> None:
        self.size = int(size)
        self.idx = 0
        self.full = False

        c: int
        h: int
        w: int
        c, h, w = obs_shape
        self.obs = np.zeros((self.size, c, h, w), dtype=np.uint8)
        self.next_obs = np.zeros((self.size, c, h, w), dtype=np.uint8)
        self.actions = np.zeros((self.size,), dtype=np.int64)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        self.dones = np.zeros((self.size,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size if self.full else self.idx

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = 1.0 if done else 0.0

        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0
            self.full = True

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        max_n: int = len(self)
        idxs = np.random.randint(low=0, high=max_n, size=batch_size)

        obs = torch.from_numpy(self.obs[idxs]).float().to(device) / 255.0
        next_obs = torch.from_numpy(self.next_obs[idxs]).float().to(device) / 255.0
        actions = torch.from_numpy(self.actions[idxs]).to(device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(device)
        dones = torch.from_numpy(self.dones[idxs]).to(device)
        
        return obs, actions, rewards, next_obs, dones


class QNetwork(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int) -> None:
        super().__init__()
        c: int
        h: int
        w: int
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy: Tensor = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            flat = out.view(1, -1).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(in_features=flat, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(dim=0), -1)
        return self.mlp(x)


class DQNAgent:
    def __init__(
        self, obs_shape: tuple[int, int, int], n_actions: int, cfg: DQNConfig
    ) -> None:
        self.cfg: DQNConfig = cfg
        self.device = torch.device(device=cfg.device)

        self.q: QNetwork = QNetwork(obs_shape=obs_shape, n_actions=n_actions).to(
            device=self.device
        )
        self.q_target: QNetwork = QNetwork(obs_shape=obs_shape, n_actions=n_actions).to(
            device=self.device
        )
        self.q_target.load_state_dict(state_dict=self.q.state_dict())
        self.q_target.eval()

        self.opt = torch.optim.Adam(params=self.q.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(obs_shape=obs_shape, size=cfg.buffer_size)

        self.n_actions: int = n_actions
        self.global_step = 0
        self.updates = 0

    def epsilon(self) -> float:
        return linear_epsilon(
            step=self.global_step,
            eps_start=self.cfg.eps_start,
            eps_end=self.cfg.eps_end,
            decay_steps=self.cfg.eps_decay_steps,
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        if (not eval_mode) and (np.random.rand() < self.epsilon()):
            return int(np.random.randint(low=0, high=self.n_actions))

        x = torch.from_numpy(obs).float().to(self.device).unsqueeze(0) / 255.0
        q = self.q(x)
        return int(torch.argmax(input=q, dim=1).item())

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def can_update(self) -> bool:
        if self.global_step < self.cfg.learning_starts:
            return False
        if len(self.rb) < self.cfg.batch_size:
            return False
        return (self.global_step % self.cfg.train_freq) == 0

    def update(self) -> dict[str, float]:
        obs: Tensor
        actions: Tensor
        rewards: Tensor
        next_obs: Tensor
        dones: Tensor
        obs, actions, rewards, next_obs, dones = self.rb.sample(
            batch_size=self.cfg.batch_size, device=self.device
        )

        with torch.no_grad():
            q_next = self.q_target(next_obs).max(dim=1).values
            target = rewards + self.cfg.gamma * (1.0 - dones) * q_next

        q_pred = self.q(obs).gather(1, actions.view(-1, 1)).squeeze(1)
        loss: Tensor = F.smooth_l1_loss(input=q_pred, target=target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                parameters=self.q.parameters(), max_norm=self.cfg.grad_clip_norm
            )
        self.opt.step()

        self.updates += 1

        if (self.global_step % self.cfg.target_update_freq) == 0:
            self.q_target.load_state_dict(state_dict=self.q.state_dict())

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_pred.mean().item()),
            "epsilon": float(self.epsilon()),
        }

    def save(self, path: str) -> None:
        payload = {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "opt": self.opt.state_dict(),
            "global_step": self.global_step,
            "updates": self.updates,
            "cfg": self.cfg.__dict__,
        }
        torch.save(obj=payload, f=path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state_dict=payload["q"])
        self.q_target.load_state_dict(state_dict=payload["q_target"])
        self.opt.load_state_dict(state_dict=payload["opt"])
        self.global_step = int(payload["global_step"])
        self.updates = int(payload["updates"])
