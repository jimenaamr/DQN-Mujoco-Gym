# src/rdqn/rdqn.py

from __future__ import annotations

import math
import random
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
class RDQNConfig:
    """Hyperparameters and runtime options for Rainbow DQN."""

    gamma: float
    lr: float
    batch_size: int
    buffer_size: int
    learning_starts: int
    train_freq: int
    target_update_freq: int
    grad_clip_norm: float

    # Exploration / Noisy Nets.
    noisy_sigma0: float

    # N-step returns.
    n_step: int

    # Prioritized replay.
    prio_alpha: float
    prio_beta_start: float
    prio_beta_end: float
    prio_beta_steps: int
    prio_eps: float

    # Distributional RL (C51).
    v_min: float
    v_max: float
    n_atoms: int

    device: str


class NoisyLinear(nn.Module):
    """Factorized Gaussian Noisy Linear layer (Fortunato et al.)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma0: float = 0.5,
        bias: bool = True,
    ) -> None:
        """Initialize the layer.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            sigma0: Initial std for noise parameters.
            bias: Whether to include a bias term.
        """
        super().__init__()
        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)
        self.sigma0: float = float(sigma0)

        self.weight_mu: nn.Parameter = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.weight_sigma: nn.Parameter = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.register_buffer(
            "weight_eps", torch.empty(self.out_features, self.in_features)
        )

        if bias:
            self.bias_mu: nn.Parameter = nn.Parameter(torch.empty(self.out_features))
            self.bias_sigma: nn.Parameter = nn.Parameter(torch.empty(self.out_features))
            self.register_buffer("bias_eps", torch.empty(self.out_features))
        else:
            self.bias_mu = None
            self.bias_sigma = None
            self.register_buffer("bias_eps", torch.empty(0))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        mu_range: float = 1.0 / math.sqrt(float(self.in_features))
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.constant_(
            self.weight_sigma,
            float(self.sigma0) * mu_range,
        )
        if self.bias_mu is not None and self.bias_sigma is not None:
            nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
            nn.init.constant_(self.bias_sigma, float(self.sigma0) * mu_range)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """Noise transform for factorized Gaussian noise.

        Args:
            x: Standard normal samples.

        Returns:
            Transformed samples.
        """
        return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)

    def reset_noise(self) -> None:
        """Resample noise buffers."""
        eps_in: torch.Tensor = torch.randn(
            self.in_features, device=self.weight_mu.device
        )
        eps_out: torch.Tensor = torch.randn(
            self.out_features, device=self.weight_mu.device
        )
        f_in: torch.Tensor = self._f(eps_in)
        f_out: torch.Tensor = self._f(eps_out)

        self.weight_eps.copy_(torch.ger(f_out, f_in))
        if self.bias_mu is not None and self.bias_sigma is not None:
            self.bias_eps.copy_(f_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the noisy linear transformation.

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        w: torch.Tensor = self.weight_mu + self.weight_sigma * self.weight_eps
        if self.bias_mu is None or self.bias_sigma is None:
            b: torch.Tensor | None = None
        else:
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        return F.linear(x, w, b)


class _RainbowQNet(nn.Module):
    """Rainbow CNN Q-network for pixel observations (C,H,W), aligned with DQN's _QNet.

    Architecture mirrors your _QNet:
      - conv stack: 16/32 channels, SiLU activations
      - flatten
      - two-layer MLP trunk: 256 -> 64

    Then it splits into dueling distributional heads (C51) using NoisyLinear:
      - value head: 64 -> n_atoms
      - advantage head: 64 -> n_actions * n_atoms
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        n_atoms: int,
        noisy_sigma0: float,
    ) -> None:
        """Initialize the CNN + dueling distributional heads.

        Args:
            obs_shape: Observation shape as (C, H, W).
            n_actions: Number of discrete actions.
            n_atoms: Number of atoms for C51 distribution.
            noisy_sigma0: Initial sigma for NoisyLinear layers.
        """
        super().__init__()
        c: int
        h: int
        w: int
        c, h, w = obs_shape

        self.n_actions: int = int(n_actions)
        self.n_atoms: int = int(n_atoms)

        # Match _QNet conv stack (channels + activations).
        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4),
            nn.SiLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.SiLU(inplace=False),
        )

        with torch.no_grad():
            dummy: torch.Tensor = torch.zeros(
                size=(1, int(c), int(h), int(w)),
                dtype=torch.float32,
            )
            out: torch.Tensor = self.conv(dummy)
            flat_dim: int = int(out.flatten(start_dim=1).shape[1])

        # Match _QNet head trunk sizes (256 -> 64), then branch.
        self.trunk: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=256),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(inplace=False),
        )

        self.value: nn.Sequential = nn.Sequential(
            NoisyLinear(in_features=64, out_features=64, sigma0=noisy_sigma0),
            nn.ReLU(inplace=False),
            NoisyLinear(in_features=64, out_features=self.n_atoms, sigma0=noisy_sigma0),
        )

        self.adv: nn.Sequential = nn.Sequential(
            NoisyLinear(in_features=64, out_features=64, sigma0=noisy_sigma0),
            nn.ReLU(inplace=False),
            NoisyLinear(
                in_features=64,
                out_features=self.n_actions * self.n_atoms,
                sigma0=noisy_sigma0,
            ),
        )

    def reset_noise(self) -> None:
        """Reset noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distributional logits for each action.

        Args:
            x: (B, C, H, W) uint8 in [0,255] or float.

        Returns:
            (B, n_actions, n_atoms) logits.
        """
        if x.dtype is torch.uint8:
            x = x.to(dtype=torch.float32).mul(1.0 / 255.0)
        elif x.dtype is not torch.float32:
            x = x.to(dtype=torch.float32)

        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)

        y: torch.Tensor = self.conv(x)
        y = y.flatten(start_dim=1)

        z: torch.Tensor = self.trunk(y)  # (B, 64)

        v: torch.Tensor = self.value(z)  # (B, n_atoms)
        a: torch.Tensor = self.adv(z)  # (B, n_actions * n_atoms)
        a = a.view(z.shape[0], self.n_actions, self.n_atoms)

        v = v.view(z.shape[0], 1, self.n_atoms)
        q_logits: torch.Tensor = v + (a - a.mean(dim=1, keepdim=True))
        return q_logits


class _PrioritizedReplay:
    """Simple proportional prioritized replay buffer (sum-tree via numpy arrays)."""

    def __init__(
        self,
        capacity: int,
        alpha: float,
        eps: float,
    ) -> None:
        """Initialize the buffer.

        Args:
            capacity: Max transitions.
            alpha: Priority exponent.
            eps: Small constant added to priorities.
        """
        self.capacity: int = int(capacity)
        self.alpha: float = float(alpha)
        self.eps: float = float(eps)

        self.pos: int = 0
        self.size: int = 0

        self.obs: list[np.ndarray | None] = [None] * self.capacity
        self.next_obs: list[np.ndarray | None] = [None] * self.capacity
        self.action: np.ndarray = np.zeros((self.capacity,), dtype=np.int64)
        self.reward: np.ndarray = np.zeros((self.capacity,), dtype=np.float32)
        self.done: np.ndarray = np.zeros((self.capacity,), dtype=np.bool_)

        self.prio: np.ndarray = np.zeros((self.capacity,), dtype=np.float32)
        self.max_prio: float = 1.0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition.

        Args:
            obs: Current observation (C,H,W).
            action: Discrete action.
            reward: Scalar reward.
            next_obs: Next observation (C,H,W).
            done: Episode done flag.
        """
        i: int = int(self.pos)

        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.action[i] = int(action)
        self.reward[i] = float(reward)
        self.done[i] = bool(done)

        self.prio[i] = float(self.max_prio)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        beta: float,
        rng: random.Random,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample a batch with importance weights.

        Args:
            batch_size: Batch size.
            beta: IS correction exponent.
            rng: Python RNG for reproducibility.

        Returns:
            batch: Dict with arrays.
            indices: Buffer indices.
            weights: IS weights (batch,).
        """
        n: int = int(self.size)
        if n <= 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        p_raw: np.ndarray = self.prio[:n].astype(np.float64, copy=False)
        p: np.ndarray = np.power(p_raw + float(self.eps), float(self.alpha))
        p_sum: float = float(np.sum(p))
        if not math.isfinite(p_sum) or p_sum <= 0.0:
            p = np.ones_like(p) / float(len(p))
        else:
            p = p / p_sum

        idx: np.ndarray = np.array(
            [rng.choices(range(n), weights=p, k=1)[0] for _ in range(int(batch_size))],
            dtype=np.int64,
        )

        w: np.ndarray = np.power(float(n) * p[idx], -float(beta))
        w = w / (float(np.max(w)) + 1e-12)
        w = w.astype(np.float32, copy=False)

        obs_b: np.ndarray = np.stack([self.obs[int(i)] for i in idx], axis=0)  # type: ignore[arg-type]
        next_obs_b: np.ndarray = np.stack(
            [self.next_obs[int(i)] for i in idx],
            axis=0,  # type: ignore[arg-type]
        )

        batch: dict[str, np.ndarray] = {
            "obs": obs_b,
            "action": self.action[idx].astype(np.int64, copy=False),
            "reward": self.reward[idx].astype(np.float32, copy=False),
            "next_obs": next_obs_b,
            "done": self.done[idx].astype(np.bool_, copy=False),
        }
        return batch, idx, w

    def update_priorities(self, indices: np.ndarray, prios: np.ndarray) -> None:
        """Update priorities for sampled indices.

        Args:
            indices: Indices to update.
            prios: New priorities (typically TD error magnitudes).
        """
        indices_i64: np.ndarray = indices.astype(np.int64, copy=False)
        prios_f32: np.ndarray = prios.astype(np.float32, copy=False)

        for j in range(int(indices_i64.shape[0])):
            i: int = int(indices_i64[j])
            p: float = float(prios_f32[j])
            p = float(abs(p)) + float(self.eps)
            self.prio[i] = p
            if p > self.max_prio:
                self.max_prio = float(p)


class RDQNAgent:
    """Rainbow DQN agent with a DQN-like interface used by your scripts."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        cfg: RDQNConfig,
        seed: int = 0,
    ) -> None:
        """Initialize the agent.

        Args:
            obs_shape: Observation shape (C,H,W).
            n_actions: Discrete action count.
            cfg: RDQNConfig.
            seed: RNG seed for sampling / action selection.
        """
        self.cfg: RDQNConfig = cfg
        self.device: torch.device = torch.device(str(cfg.device))

        self.n_actions: int = int(n_actions)
        self.n_atoms: int = int(cfg.n_atoms)

        self.v_min: float = float(cfg.v_min)
        self.v_max: float = float(cfg.v_max)
        self.support: torch.Tensor = torch.linspace(
            float(self.v_min),
            float(self.v_max),
            int(self.n_atoms),
            device=self.device,
            dtype=torch.float32,
        )
        self.delta_z: float = float((self.v_max - self.v_min) / (self.n_atoms - 1))

        self.q: _RainbowQNet = _RainbowQNet(
            obs_shape=obs_shape,
            n_actions=int(n_actions),
            n_atoms=int(cfg.n_atoms),
            noisy_sigma0=float(cfg.noisy_sigma0),
        ).to(self.device)
        self.q_targ: _RainbowQNet = _RainbowQNet(
            obs_shape=obs_shape,
            n_actions=int(n_actions),
            n_atoms=int(cfg.n_atoms),
            noisy_sigma0=float(cfg.noisy_sigma0),
        ).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.q_targ.eval()

        self.opt: torch.optim.Optimizer = torch.optim.Adam(
            params=self.q.parameters(),
            lr=float(cfg.lr),
        )

        self.replay: _PrioritizedReplay = _PrioritizedReplay(
            capacity=int(cfg.buffer_size),
            alpha=float(cfg.prio_alpha),
            eps=float(cfg.prio_eps),
        )

        self.rng: random.Random = random.Random(int(seed))
        self.global_step: int = 0
        self.updates: int = 0

        self._beta: float = float(cfg.prio_beta_start)

    def _beta_by_step(self, step: int) -> float:
        """Linearly anneal beta from start to end.

        Args:
            step: Global step.

        Returns:
            Annealed beta.
        """
        t: float = min(1.0, float(step) / float(max(1, self.cfg.prio_beta_steps)))
        return float(
            self.cfg.prio_beta_start
            + t * (self.cfg.prio_beta_end - self.cfg.prio_beta_start)
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, eval_mode: bool) -> int:
        """Select action from the current policy.

        Args:
            obs: Observation (C,H,W) uint8.
            eval_mode: If True, disables noise for stable evaluation.

        Returns:
            Discrete action index.
        """
        self.q.eval()
        if eval_mode:
            # In eval, we still compute deterministically; NoisyLinear still has
            # fixed sampled eps buffers. We reset noise less frequently by not
            # calling reset_noise here.
            pass
        else:
            self.q.reset_noise()

        obs_t: torch.Tensor = torch.from_numpy(obs[None, ...]).to(self.device)
        logits: torch.Tensor = self.q(obs_t)  # (1,A,Z)
        prob: torch.Tensor = F.softmax(logits, dim=-1)
        q_values: torch.Tensor = torch.sum(prob * self.support.view(1, 1, -1), dim=-1)
        a: int = int(torch.argmax(q_values, dim=1).item())
        self.q.train()
        return a

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in replay.

        Args:
            obs: Observation.
            action: Action.
            reward: Reward.
            next_obs: Next observation.
            done: Done flag.
        """
        self.replay.add(
            obs=obs,
            action=int(action),
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
        )

    def can_update(self) -> bool:
        """Check if the agent should start updating.

        Returns:
            True if enough samples exist and learning_starts reached.
        """
        return bool(
            self.replay.size >= self.cfg.batch_size
            and self.global_step >= self.cfg.learning_starts
        )

    def _project_distribution(
        self,
        next_prob: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """C51 projection operator.

        Args:
            next_prob: Next-state action distribution (B, Z).
            rewards: Rewards (B,).
            dones: Done flags (B,) as float {0,1}.

        Returns:
            Projected target distribution (B, Z).
        """
        bsz: int = int(rewards.shape[0])

        tz: torch.Tensor = (
            rewards[:, None]
            + (1.0 - dones[:, None]) * float(self.cfg.gamma) * self.support[None, :]
        )
        tz = torch.clamp(tz, min=float(self.v_min), max=float(self.v_max))

        bj: torch.Tensor = (tz - float(self.v_min)) / float(self.delta_z)
        l: torch.Tensor = torch.floor(bj).to(torch.int64)
        u: torch.Tensor = torch.ceil(bj).to(torch.int64)

        m: torch.Tensor = torch.zeros(
            (bsz, int(self.n_atoms)), device=self.device, dtype=torch.float32
        )

        for i in range(bsz):
            for j in range(int(self.n_atoms)):
                lj: int = int(l[i, j].item())
                uj: int = int(u[i, j].item())

                pj: float = float(next_prob[i, j].item())
                bjv: float = float(bj[i, j].item())

                if lj == uj:
                    m[i, lj] += pj
                else:
                    m[i, lj] += pj * float(uj - bjv)
                    m[i, uj] += pj * float(bjv - lj)

        m = torch.clamp(m, min=0.0)
        m = m / (m.sum(dim=1, keepdim=True) + 1e-12)
        return m

    def update(self) -> dict[str, float]:
        """Run a single optimization step.

        Returns:
            Metrics dict for logging.
        """
        if (self.global_step % int(self.cfg.train_freq)) != 0:
            return {}

        self._beta = float(self._beta_by_step(step=int(self.global_step)))

        batch_np: dict[str, np.ndarray]
        idx: np.ndarray
        w_np: np.ndarray
        batch_np, idx, w_np = self.replay.sample(
            batch_size=int(self.cfg.batch_size),
            beta=float(self._beta),
            rng=self.rng,
        )

        obs: torch.Tensor = torch.from_numpy(batch_np["obs"]).to(self.device)
        next_obs: torch.Tensor = torch.from_numpy(batch_np["next_obs"]).to(self.device)
        action: torch.Tensor = torch.from_numpy(batch_np["action"]).to(self.device)
        reward: torch.Tensor = torch.from_numpy(batch_np["reward"]).to(self.device)
        done: torch.Tensor = torch.from_numpy(batch_np["done"].astype(np.float32)).to(
            self.device
        )
        w: torch.Tensor = torch.from_numpy(w_np).to(self.device)

        self.q.train()
        self.q.reset_noise()
        self.q_targ.eval()
        self.q_targ.reset_noise()

        # Current logits for chosen actions.
        logits: torch.Tensor = self.q(obs)  # (B,A,Z)
        logits_a: torch.Tensor = logits[torch.arange(logits.shape[0]), action]  # (B,Z)
        log_prob_a: torch.Tensor = F.log_softmax(logits_a, dim=-1)

        with torch.no_grad():
            # Double DQN action selection using online net expected values.
            next_logits_online: torch.Tensor = self.q(next_obs)  # (B,A,Z)
            next_prob_online: torch.Tensor = F.softmax(next_logits_online, dim=-1)
            next_q_online: torch.Tensor = torch.sum(
                next_prob_online * self.support.view(1, 1, -1),
                dim=-1,
            )  # (B,A)
            next_a: torch.Tensor = torch.argmax(next_q_online, dim=1)  # (B,)

            # Target distribution from target net for selected actions.
            next_logits_t: torch.Tensor = self.q_targ(next_obs)  # (B,A,Z)
            next_prob_t: torch.Tensor = F.softmax(next_logits_t, dim=-1)
            next_prob_t_a: torch.Tensor = next_prob_t[
                torch.arange(next_prob_t.shape[0]), next_a
            ]  # (B,Z)

            target_dist: torch.Tensor = self._project_distribution(
                next_prob=next_prob_t_a,
                rewards=reward,
                dones=done,
            )

        # Cross-entropy loss between target_dist and current log probs.
        ce: torch.Tensor = -(target_dist * log_prob_a).sum(dim=1)  # (B,)
        loss: torch.Tensor = (w * ce).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(self.cfg.grad_clip_norm) > 0.0:
            nn.utils.clip_grad_norm_(
                self.q.parameters(), max_norm=float(self.cfg.grad_clip_norm)
            )
        self.opt.step()

        td_err: torch.Tensor = ce.detach()
        self.replay.update_priorities(
            indices=idx,
            prios=td_err.detach().cpu().numpy().astype(np.float32, copy=False),
        )

        self.updates += 1

        if (self.updates % int(self.cfg.target_update_freq)) == 0:
            self.q_targ.load_state_dict(self.q.state_dict())

        return {
            "loss": float(loss.item()),
            "prio_beta": float(self._beta),
            "updates": float(self.updates),
        }

    def save(self, path: str) -> None:
        """Save model weights and counters.

        Args:
            path: Destination .pt file path.
        """
        p: Path = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "q": self.q.state_dict(),
            "q_targ": self.q_targ.state_dict(),
            "opt": self.opt.state_dict(),
            "global_step": int(self.global_step),
            "updates": int(self.updates),
            "cfg": self.cfg.__dict__,
        }
        torch.save(payload, str(p))

    def load(self, path: str) -> int:
        """Load model weights and counters.

        Args:
            path: Checkpoint .pt file.

        Returns:
            Loaded global_step.
        """
        payload: dict[str, Any] = torch.load(str(path), map_location=self.device)
        self.q.load_state_dict(payload["q"])
        self.q_targ.load_state_dict(payload.get("q_targ", payload["q"]))
        self.opt.load_state_dict(payload["opt"])
        self.global_step = int(payload.get("global_step", 0))
        self.updates = int(payload.get("updates", 0))

        m: re.Match[str] | None = _STEP_ANYWHERE_RE.search(str(Path(path).name))
        if self.global_step <= 0 and m is not None:
            self.global_step = int(m.group(1))
        return int(self.global_step)
