from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any

@dataclass
class RDQNConfig:
    gamma: float
    lr: float
    batch_size: int
    buffer_size: int
    learning_starts: int
    train_freq: int
    target_update_freq: int
    grad_clip_norm: float
    device: str
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

# --- ESTRUCTURAS PARA PRIORITIZED EXPERIENCE REPLAY (PER) ---

class SumTree:
    """Estructura de datos para muestreo eficiente O(log n)."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: Any):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    """Buffer PER integrado para evitar dependencias externas."""
    def __init__(self, capacity: int, alpha: float, eps: float):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.eps = eps
        self.capacity = capacity

    def _get_priority(self, error: float) -> float:
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, obs, action, reward, next_obs, done):
        # Nuevas transiciones entran con prioridad máxima
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0: max_p = 1.0
        self.tree.add(max_p, (obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, beta: float):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        # Desempaquetar batch
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return {
            "obs": np.array(obs), "actions": np.array(actions),
            "rewards": np.array(rewards), "next_obs": np.array(next_obs),
            "dones": np.array(dones), "indices": idxs,
            "weights": is_weights.astype(np.float32)
        }

    def update_priorities(self, idxs: list[int], errors: np.ndarray):
        for idx, error in zip(idxs, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)

# --- ARQUITECTURA RAINBOW ---

class NoisyLinear(nn.Module):
    """Capa Noisy Linear para exploración paramétrica[cite: 69, 75]."""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size: int):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)

class _RainbowNet(nn.Module):
    """Dueling Distributional Network[cite: 66, 68, 70]."""
    def __init__(self, obs_shape, n_actions, n_atoms, v_min, v_max, noisy_std):
        super().__init__()
        c, h, w = obs_shape
        self.n_actions, self.n_atoms = n_actions, n_atoms
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, 8, stride=4), nn.SiLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.SiLU(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, c, h, w)).flatten(1).shape[1]

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.value_stream = nn.Sequential(NoisyLinear(flat_dim, 256, noisy_std), nn.ReLU(), NoisyLinear(256, n_atoms, noisy_std))
        self.advantage_stream = nn.Sequential(NoisyLinear(flat_dim, 256, noisy_std), nn.ReLU(), NoisyLinear(256, n_actions * n_atoms, noisy_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8: x = x.float() / 255.0
        x = self.conv(x).flatten(1)
        dist_v = self.value_stream(x).view(-1, 1, self.n_atoms)
        dist_a = self.advantage_stream(x).view(-1, self.n_actions, self.n_atoms)
        dist_q = dist_v + (dist_a - dist_a.mean(dim=1, keepdim=True))
        return F.softmax(dist_q, dim=-1)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        return (self.forward(x) * self.support).sum(2)

class RDQNAgent:
    """Agente Rainbow optimizado y autocontenido."""
    def __init__(self, obs_shape, n_actions, cfg: RDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = _RainbowNet(obs_shape, n_actions, cfg.n_atoms, cfg.v_min, cfg.v_max, cfg.noisy_sigma0).to(self.device)
        self.q_target = _RainbowNet(obs_shape, n_actions, cfg.n_atoms, cfg.v_min, cfg.v_max, cfg.noisy_sigma0).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, eps=1.5e-4)
        self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, cfg.prio_alpha, cfg.prio_eps)
        self.global_step, self.updates = 0, 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        self.q.eval() if eval_mode else self.q.train()
        x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        return int(self.q.get_q_values(x).argmax(1).item())

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def can_update(self) -> bool:
        return self.global_step >= self.cfg.learning_starts and (self.global_step % self.cfg.train_freq) == 0

    def update(self) -> dict[str, float]:
        beta = min(1.0, self.cfg.prio_beta_start + self.global_step * (1.0 - self.cfg.prio_beta_start) / self.cfg.prio_beta_steps)
        batch = self.buffer.sample(self.cfg.batch_size, beta)
        
        obs = torch.as_tensor(batch["obs"], device=self.device, non_blocking=True)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device, non_blocking=True)
        actions = torch.as_tensor(batch["actions"], device=self.device, non_blocking=True)
        rewards = torch.as_tensor(batch["rewards"], device=self.device, non_blocking=True)
        dones = torch.as_tensor(batch["dones"], device=self.device, non_blocking=True)
        weights = torch.as_tensor(batch["weights"], device=self.device, non_blocking=True)

        with torch.no_grad():
            next_action = self.q.get_q_values(next_obs).argmax(1)
            next_dist = self.q_target(next_obs)[range(self.cfg.batch_size), next_action]
            
            # Proyección categórica vectorizada (Optimización x10) [cite: 68, 76]
            target_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.cfg.gamma**self.cfg.n_step) * self.q.support
            target_z = target_z.clamp(self.cfg.v_min, self.cfg.v_max)
            b = (target_z - self.cfg.v_min) / ((self.cfg.v_max - self.cfg.v_min) / (self.cfg.n_atoms - 1))
            l, u = b.floor().long(), b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.cfg.n_atoms - 1)) * (l == u)] += 1

            m = torch.zeros(self.cfg.batch_size, self.cfg.n_atoms, device=self.device)
            offset = torch.linspace(0, (self.cfg.batch_size - 1) * self.cfg.n_atoms, self.cfg.batch_size, device=self.device).long().unsqueeze(1)
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        log_p = torch.log(self.q(obs)[range(self.cfg.batch_size), actions] + 1e-9)
        loss = (-(m * log_p).sum(dim=1) * weights).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm > 0: nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.opt.step()

        self.updates += 1
        if self.updates % self.cfg.target_update_freq == 0: self.q_target.load_state_dict(self.q.state_dict())
        self.buffer.update_priorities(batch["indices"], (-(m * log_p).sum(dim=1)).detach().cpu().numpy())
        self._reset_noise()
        return {"loss": loss.item()}

    def _reset_noise(self):
        for m in self.q.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()

    def save(self, path: str): torch.save({"q": self.q.state_dict(), "step": self.global_step}, path)
    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q"])
        return ckpt.get("step", 0)