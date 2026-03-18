# src/rdqn_emma/rdqn.py

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
    """Hiperparámetros sincronizados con tu YAML y entrenamiento."""
    gamma: float
    lr: float
    batch_size: int
    buffer_size: int
    learning_starts: int
    train_freq: int
    target_update_freq: int
    grad_clip_norm: float
    noisy_sigma0: float
    n_step: int
    prio_alpha: float
    prio_beta_start: float
    prio_beta_end: float
    prio_beta_steps: int
    prio_eps: float
    v_min: float
    v_max: float
    n_atoms: int
    device: str


class NoisyLinear(nn.Module):
    """Capa Noisy Linear para exploración paramétrica."""
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma0 = float(sigma0)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(float(self.in_features))
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.sigma0 * mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.bias_sigma, self.sigma0 * mu_range)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_features, device=self.weight_mu.device)
        f_in, f_out = self._f(eps_in), self._f(eps_out)
        self.weight_eps.copy_(f_out.ger(f_in))
        self.bias_eps.copy_(f_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class _RainbowQNet(nn.Module):
    """Red Dueling Distributional con nombres unificados para compatibilidad de carga."""
    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, n_atoms: int, noisy_sigma0: float):
        super().__init__()
        c, h, w = obs_shape
        self.n_actions, self.n_atoms = int(n_actions), int(n_atoms)

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, 8, stride=4), nn.SiLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.SiLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.conv(dummy).flatten(1).shape[1]

        # Nombres de streams unificados con tus checkpoints (value_stream/advantage_stream)
        self.value_stream = nn.Sequential(
            NoisyLinear(flat_dim, 256, noisy_sigma0), nn.ReLU(),
            NoisyLinear(256, self.n_atoms, noisy_sigma0)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(flat_dim, 256, noisy_sigma0), nn.ReLU(),
            NoisyLinear(256, self.n_actions * self.n_atoms, noisy_sigma0)
        )

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8: x = x.float() / 255.0
        y = self.conv(x).flatten(1)
        v = self.value_stream(y).view(-1, 1, self.n_atoms)
        a = self.advantage_stream(y).view(-1, self.n_actions, self.n_atoms)
        q_logits = v + (a - a.mean(dim=1, keepdim=True))
        return q_logits


class _PrioritizedReplay:
    """Buffer PER eficiente integrado."""
    def __init__(self, capacity: int, alpha: float, eps: float):
        self.capacity, self.alpha, self.eps = int(capacity), float(alpha), float(eps)
        self.pos = 0
        self.size = 0
        self.obs = [None] * self.capacity
        self.next_obs = [None] * self.capacity
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.bool_)
        self.prio = np.zeros(self.capacity, dtype=np.float32)
        self.max_prio = 1.0

    def add(self, obs, action, reward, next_obs, done):
        i = self.pos
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.action[i], self.reward[i], self.done[i] = action, reward, done
        self.prio[i] = self.max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float, rng: random.Random):
        n = self.size
        p = (self.prio[:n] + self.eps) ** self.alpha
        p /= p.sum()
        idx = np.array([rng.choices(range(n), weights=p, k=1)[0] for _ in range(batch_size)])
        
        w = (n * p[idx]) ** (-beta)
        w /= (w.max() + 1e-12)
        
        batch = {
            "obs": np.stack([self.obs[i] for i in idx]),
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": np.stack([self.next_obs[i] for i in idx]),
            "done": self.done[idx]
        }
        return batch, idx, w.astype(np.float32)

    def update_priorities(self, indices, prios):
        for i, p in zip(indices, prios):
            self.prio[i] = abs(p) + self.eps
            self.max_prio = max(self.max_prio, self.prio[i])


class RDQNAgent:
    """Agente Rainbow unificado y optimizado."""
    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, cfg: RDQNConfig, seed: int = 0):
        self.cfg, self.device = cfg, torch.device(cfg.device)
        self.n_actions, self.n_atoms = int(n_actions), int(cfg.n_atoms)
        
        self.support = torch.linspace(cfg.v_min, cfg.v_max, self.n_atoms).to(self.device)
        self.delta_z = (cfg.v_max - cfg.v_min) / (self.n_atoms - 1)

        self.q = _RainbowQNet(obs_shape, n_actions, self.n_atoms, cfg.noisy_sigma0).to(self.device)
        self.q_target = _RainbowQNet(obs_shape, n_actions, self.n_atoms, cfg.noisy_sigma0).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, eps=1.5e-4)
        self.replay = _PrioritizedReplay(cfg.buffer_size, cfg.prio_alpha, cfg.prio_eps)
        self.rng = random.Random(seed)
        self.global_step = 0
        self.updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, eval_mode: bool) -> int:
        self.q.eval() if eval_mode else self.q.train()
        if not eval_mode: self.q.reset_noise()
        
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.uint8).unsqueeze(0)
        prob = F.softmax(self.q(obs_t), dim=-1)
        q_values = (prob * self.support).sum(2)
        return int(q_values.argmax(1).item())

    def store(self, obs, action, reward, next_obs, done):
        self.replay.add(obs, action, reward, next_obs, done)

    def can_update(self) -> bool:
        return self.replay.size >= self.cfg.batch_size and self.global_step >= self.cfg.learning_starts

    def update(self) -> dict[str, float]:
        if (self.global_step % self.cfg.train_freq) != 0: return {}

        beta = min(1.0, self.cfg.prio_beta_start + self.global_step * (self.cfg.prio_beta_end - self.cfg.prio_beta_start) / self.cfg.prio_beta_steps)
        batch_np, idx, w_np = self.replay.sample(self.cfg.batch_size, beta, self.rng)

        # Carga optimizada a dispositivo
        obs = torch.as_tensor(batch_np["obs"], device=self.device)
        next_obs = torch.as_tensor(batch_np["next_obs"], device=self.device)
        action = torch.as_tensor(batch_np["action"], device=self.device)
        reward = torch.as_tensor(batch_np["reward"], device=self.device)
        done = torch.as_tensor(batch_np["done"], device=self.device, dtype=torch.float32)
        w = torch.as_tensor(w_np, device=self.device)

        self.q.train()
        self.q.reset_noise()

        logits = self.q(obs)[range(self.cfg.batch_size), action]
        log_prob = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            # Double DQN + Distributional
            next_action = (F.softmax(self.q(next_obs), dim=-1) * self.support).sum(2).argmax(1)
            next_prob = F.softmax(self.q_target(next_obs)[range(self.cfg.batch_size), next_action], dim=-1)

            # Proyección C51 Vectorizada (Optimización x10)
            tz = reward.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * self.cfg.gamma * self.support
            tz = tz.clamp(self.cfg.v_min, self.cfg.v_max)
            
            b = (tz - self.cfg.v_min) / self.delta_z
            l, u = b.floor().long(), b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.cfg.n_atoms - 1)) * (l == u)] += 1

            target_dist = torch.zeros(self.cfg.batch_size, self.n_atoms, device=self.device)
            offset = torch.linspace(0, (self.cfg.batch_size - 1) * self.n_atoms, self.cfg.batch_size).to(self.device).long().unsqueeze(1)
            
            target_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_prob * (u.float() - b)).view(-1).float())
            target_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_prob * (b - l.float())).view(-1).float())

        loss = (-(target_dist * log_prob).sum(1) * w).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm > 0: nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.opt.step()

        self.replay.update_priorities(idx, (-(target_dist * log_prob).sum(1)).detach().cpu().numpy())
        self.updates += 1
        if (self.updates % self.cfg.target_update_freq) == 0: self.q_target.load_state_dict(self.q.state_dict())

        return {"loss": loss.item(), "updates": self.updates}

    def save(self, path: str):
        torch.save({"q": self.q.state_dict(), "step": self.global_step, "updates": self.updates}, path)

    def load(self, path: str) -> int:
        payload = torch.load(path, map_location=self.device)
        self.q.load_state_dict(payload["q"])
        self.q_target.load_state_dict(payload.get("q_target", payload["q"]))
        self.global_step = payload.get("step", 0)
        self.updates = payload.get("updates", 0)
        return self.global_step