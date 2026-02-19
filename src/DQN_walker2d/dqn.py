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
        """Initialize the CNN + MLP Q-network.

        Args:
            obs_shape: Observation shape as (C, H, W).
            n_actions: Number of discrete actions.
        """
        super().__init__()
        c: int
        h: int
        w: int
        c, h, w = obs_shape

        # self.conv: nn.Sequential = nn.Sequential(
        #     nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        # )

        self.conv = nn.Sequential(
            # Aumentamos el stride para reducir drásticamente el mapa de características rápido
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.SiLU(inplace=True),
        )

        with torch.no_grad():
            dummy: torch.Tensor = torch.zeros(
                size=(1, c, h, w),
                dtype=torch.float32,
            )
            out: torch.Tensor = self.conv(dummy)
            flat_dim: int = int(out.flatten(start_dim=1).shape[1])

        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values.

        Args:
            x: (B, C, H, W) uint8 in [0,255] or float.

        Returns:
            (B, n_actions) Q-values.
        """
        if x.dtype is torch.uint8:
            x = x.to(dtype=torch.float32).mul_(other=(1.0 / 255.0))
        elif x.dtype is not torch.float32:
            x = x.to(dtype=torch.float32)

        # For CUDA, channels_last often speeds up Conv2d without changing numerics.
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)

        y: torch.Tensor = self.conv(x)
        y = y.flatten(start_dim=1)
        return self.head(y)


class _ReplayBuffer:
    """Simple numpy replay buffer for pixel observations."""

    def __init__(self, capacity: int, obs_shape: tuple[int, int, int]) -> None:
        """Create a replay buffer.

        Args:
            capacity: Maximum number of transitions stored.
            obs_shape: Observation shape as (C, H, W).
        """
        self.capacity: int = int(capacity)
        self.obs_shape: tuple[int, int, int] = obs_shape

        self.obs: np.ndarray = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self.next_obs: np.ndarray = np.zeros(
            (self.capacity, *obs_shape),
            dtype=np.uint8,
        )
        self.actions: np.ndarray = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards: np.ndarray = np.zeros((self.capacity,), dtype=np.float32)
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
        """Insert a transition into the buffer.

        Args:
            obs: Current observation (C,H,W) uint8.
            action: Discrete action index.
            reward: Scalar reward.
            next_obs: Next observation (C,H,W) uint8.
            done: Episode termination flag.
        """
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = np.uint8(1 if done else 0)

        self.pos = int((self.pos + 1) % self.capacity)
        self.size = int(min(self.size + 1, self.capacity))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict of numpy arrays: obs, next_obs, actions, rewards, dones.
        """
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


def _compile_module(m: nn.Module) -> nn.Module:
    """Compile a module with torch.compile when available.

    Args:
        m: Module to compile.

    Returns:
        Compiled module if torch.compile exists, otherwise the original module.
    """
    compile_fn: Any = getattr(torch, "compile", None)
    if compile_fn is None:
        return m
    print(f"compile {m.__class__.__name__}")

    return compile_fn(
        m,
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
    )


class DQNAgent:
    """DQN agent with epsilon-greedy policy and target network."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        cfg: DQNConfig,
    ) -> None:
        """Initialize networks, optimizer, and replay buffer.

        Args:
            obs_shape: Observation shape as (C, H, W).
            n_actions: Number of discrete actions.
            cfg: DQN hyperparameters and runtime config.
        """
        self.cfg: DQNConfig = cfg
        self.device: torch.device = torch.device(str(cfg.device))

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.n_actions: int = int(n_actions)

        q: _QNet = _QNet(obs_shape=obs_shape, n_actions=self.n_actions).to(self.device)
        q_target: _QNet = _QNet(obs_shape=obs_shape, n_actions=self.n_actions).to(
            self.device
        )

        # For CUDA, channels_last can speed up Conv2d without changing behavior.
        if self.device.type == "cuda":
            q = q.to(memory_format=torch.channels_last)
            q_target = q_target.to(memory_format=torch.channels_last)

        q_target.load_state_dict(q.state_dict())
        q_target.eval()
        q.train()

        # Always compile when available.
        self.q: nn.Module = _compile_module(m=q)
        self.q_target: nn.Module = _compile_module(m=q_target)

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

        self._use_cuda: bool = self.device.type == "cuda"
        self._bs: int = int(cfg.batch_size)

        self._st_obs: torch.Tensor | None = None
        self._st_next_obs: torch.Tensor | None = None
        self._st_actions: torch.Tensor | None = None
        self._st_rewards: torch.Tensor | None = None
        self._st_dones: torch.Tensor | None = None

        if self._use_cuda:
            c: int
            h: int
            w: int
            c, h, w = obs_shape

            self._st_obs = torch.empty(
                (self._bs, c, h, w),
                dtype=torch.uint8,
                pin_memory=True,
            )
            self._st_next_obs = torch.empty(
                (self._bs, c, h, w),
                dtype=torch.uint8,
                pin_memory=True,
            )
            self._st_actions = torch.empty(
                (self._bs,),
                dtype=torch.int64,
                pin_memory=True,
            )
            self._st_rewards = torch.empty(
                (self._bs,),
                dtype=torch.float32,
                pin_memory=True,
            )
            self._st_dones = torch.empty(
                (self._bs,),
                dtype=torch.uint8,
                pin_memory=True,
            )

    def act(self, obs: np.ndarray, eval_mode: bool) -> int:
        """Select an action with epsilon-greedy exploration.

        Args:
            obs: Observation (C,H,W) uint8.
            eval_mode: If True, disables exploration.

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

        with torch.inference_mode():
            x: torch.Tensor = (
                torch
                .as_tensor(obs)
                .unsqueeze(0)
                .to(
                    device=self.device,
                    non_blocking=True,
                )
            )
            qvals: torch.Tensor = self.q(x)  # type: ignore[operator]
            return int(torch.argmax(qvals, dim=1).item())

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in replay buffer.

        Args:
            obs: Current observation.
            action: Discrete action index.
            reward: Scalar reward.
            next_obs: Next observation.
            done: Episode termination flag.
        """
        self.buffer.add(
            obs=obs,
            action=int(action),
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
        )

    def can_update(self) -> bool:
        """Check whether the agent should run a gradient update.

        Returns:
            True if update conditions are met.
        """
        if int(self.global_step) < int(self.cfg.learning_starts):
            return False
        if int(self.buffer.size) < int(self.cfg.batch_size):
            return False
        return (int(self.global_step) % int(self.cfg.train_freq)) == 0

    def _batch_to_device(
        self, batch_np: dict[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        """Convert a numpy batch into tensors on the agent device.

        Args:
            batch_np: Dict with numpy arrays.

        Returns:
            Dict with tensors on self.device.
        """
        if not self._use_cuda:
            obs: torch.Tensor = torch.from_numpy(batch_np["obs"]).to(device=self.device)
            next_obs: torch.Tensor = torch.from_numpy(batch_np["next_obs"]).to(
                device=self.device
            )
            actions: torch.Tensor = torch.from_numpy(batch_np["actions"]).to(
                device=self.device,
                dtype=torch.int64,
            )
            rewards: torch.Tensor = torch.from_numpy(batch_np["rewards"]).to(
                device=self.device,
                dtype=torch.float32,
            )
            dones_u8: torch.Tensor = torch.from_numpy(batch_np["dones"]).to(
                device=self.device
            )
            dones: torch.Tensor = dones_u8.to(dtype=torch.float32)
            return {
                "obs": obs,
                "next_obs": next_obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
            }

        assert self._st_obs is not None
        assert self._st_next_obs is not None
        assert self._st_actions is not None
        assert self._st_rewards is not None
        assert self._st_dones is not None

        self._st_obs.copy_(torch.from_numpy(batch_np["obs"]), non_blocking=False)
        self._st_next_obs.copy_(
            torch.from_numpy(batch_np["next_obs"]),
            non_blocking=False,
        )
        self._st_actions.copy_(
            torch.from_numpy(batch_np["actions"]).to(dtype=torch.int64),
            non_blocking=False,
        )
        self._st_rewards.copy_(
            torch.from_numpy(batch_np["rewards"]).to(dtype=torch.float32),
            non_blocking=False,
        )
        self._st_dones.copy_(
            torch.from_numpy(batch_np["dones"]).to(dtype=torch.uint8),
            non_blocking=False,
        )

        obs: torch.Tensor = self._st_obs.to(device=self.device, non_blocking=True)
        next_obs: torch.Tensor = self._st_next_obs.to(
            device=self.device,
            non_blocking=True,
        )
        actions: torch.Tensor = self._st_actions.to(
            device=self.device,
            non_blocking=True,
        )
        rewards: torch.Tensor = self._st_rewards.to(
            device=self.device,
            non_blocking=True,
        )
        dones_u8: torch.Tensor = self._st_dones.to(
            device=self.device,
            non_blocking=True,
        )
        dones: torch.Tensor = dones_u8.to(dtype=torch.float32)

        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    def update(self) -> dict[str, float]:
        """Run a single DQN update step (Double DQN + periodic target sync).

        Returns:
            Metrics dict with loss, mean Q(s,a), and current epsilon.
        """
        batch_np: dict[str, np.ndarray] = self.buffer.sample(
            batch_size=int(self.cfg.batch_size),
        )
        batch: dict[str, torch.Tensor] = self._batch_to_device(batch_np=batch_np)

        obs: torch.Tensor = batch["obs"]
        next_obs: torch.Tensor = batch["next_obs"]
        actions: torch.Tensor = batch["actions"]
        rewards: torch.Tensor = batch["rewards"]
        dones: torch.Tensor = batch["dones"]

        # Keep input layout consistent with channels_last networks on CUDA.
        if obs.is_cuda:
            obs = obs.contiguous(memory_format=torch.channels_last)
            next_obs = next_obs.contiguous(memory_format=torch.channels_last)

        gamma: float = float(self.cfg.gamma)

        q_sa_all: torch.Tensor = self.q(obs)  # type: ignore[operator]
        q_a: torch.Tensor = q_sa_all.gather(dim=1, index=actions.unsqueeze(1)).squeeze(
            1
        )

        with torch.no_grad():
            a_star: torch.Tensor = self.q(next_obs).argmax(dim=1, keepdim=True)  # type: ignore[operator]
            q_next: torch.Tensor = (
                self
                .q_target(next_obs)  # type: ignore[operator]
                .gather(dim=1, index=a_star)
                .squeeze(1)
            )
            target: torch.Tensor = rewards + (1.0 - dones) * gamma * q_next

        loss: torch.Tensor = F.smooth_l1_loss(q_a, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        grad_clip: float = float(self.cfg.grad_clip_norm)
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(
                parameters=self.q.parameters(),
                max_norm=grad_clip,
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
            "loss": float(loss.detach().item()),
            "q_mean": float(q_a.detach().mean().item()),
            "epsilon": eps,
        }

    def save(self, path: str) -> None:
        """Save agent parameters and training state to disk.

        Args:
            path: Output checkpoint path.
        """
        payload: dict[str, Any] = {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "opt": self.opt.state_dict(),
            "global_step": int(self.global_step),
            "updates": int(self.updates),
            "cfg": dict(self.cfg.__dict__),
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
            raise KeyError(
                f"Checkpoint missing Q state. Keys: {sorted(list(ckpt.keys()))}"
            )
        self.q.load_state_dict(state_dict=q_state)

        q_target_state: dict[str, Any] | None = ckpt.get("q_target")
        if q_target_state is not None:
            self.q_target.load_state_dict(state_dict=q_target_state)

        opt_state: dict[str, Any] | None = ckpt.get("opt")
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
    step: int,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
) -> float:
    """Linear epsilon schedule.

    Args:
        step: Current environment step.
        eps_start: Starting epsilon.
        eps_end: Final epsilon.
        decay_steps: Steps over which epsilon decays linearly.

    Returns:
        Epsilon value for the given step.
    """
    if decay_steps <= 0:
        return float(eps_end)
    t: float = float(np.clip(step / decay_steps, 0.0, 1.0))
    return float((1.0 - t) * eps_start + t * eps_end)
