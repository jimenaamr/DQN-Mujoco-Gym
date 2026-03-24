from __future__ import annotations
from random import random
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
        max_p = np.max(self.tree.tree[-self.tree.capacity :])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, (obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, beta: float, rng=None):
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
            "obs": np.array(obs),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_obs": np.array(next_obs),
            "dones": np.array(dones),
            "indices": idxs,
            "weights": is_weights.astype(np.float32),
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
        self.in_features, self.out_features, self.std_init = (
            in_features,
            out_features,
            std_init,
        )
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
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        return F.linear(x, self.weight_mu, self.bias_mu)


class _RainbowNet(nn.Module):
    """Dueling Distributional Network."""

    def __init__(self, obs_shape, n_actions, n_atoms, v_min, v_max, noisy_std):
        super().__init__()
        c, h, w = obs_shape
        self.n_actions, self.n_atoms = n_actions, n_atoms
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),   # FIX: 16→32 filtros
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # FIX: 32→64 filtros
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=1),  # FIX: tercera capa añadida (estándar DQN)
            nn.SiLU(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, c, h, w)).flatten(1).shape[1]

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.value_stream = nn.Sequential(
            NoisyLinear(flat_dim, 256, noisy_std),
            nn.ReLU(),
            NoisyLinear(256, n_atoms, noisy_std),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(flat_dim, 256, noisy_std),
            nn.ReLU(),
            NoisyLinear(256, n_actions * n_atoms, noisy_std),
        )

    def reset_noise(self):
        """Resetea el ruido en todas las capas NoisyLinear de la red."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
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
            self.n_actions = n_actions  # FIX: guardado para usar en act()
            self.device = torch.device(cfg.device)
            
            # 1. Soporte para C51
            self.support = torch.linspace(cfg.v_min, cfg.v_max, cfg.n_atoms).to(self.device)
            self.delta_z = (cfg.v_max - cfg.v_min) / (cfg.n_atoms - 1)

            # 2. Redes
            self.q = _RainbowNet(obs_shape, n_actions, cfg.n_atoms, cfg.v_min, cfg.v_max, cfg.noisy_sigma0).to(self.device)
            self.q_target = _RainbowNet(obs_shape, n_actions, cfg.n_atoms, cfg.v_min, cfg.v_max, cfg.noisy_sigma0).to(self.device)
            self.q_target.load_state_dict(self.q.state_dict())
            
            # 3. Optimizador y Buffer
            self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, eps=1.5e-4)
            self.replay = PrioritizedReplayBuffer(cfg.buffer_size, cfg.prio_alpha, cfg.prio_eps)
            
            # 4. Buffer n-step para acumular retornos antes de guardar en replay
            self._nstep_buf: list = []  # FIX: buffer temporal para n-step returns

            # 5. Generador aleatorio para el muestreo
            import random
            self.rng = random.Random(0)
            
            self.global_step, self.updates = 0, 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, eval_mode: bool) -> int:
        self.q.eval() if eval_mode else self.q.train()
        
        if not eval_mode and self.cfg.noisy_sigma0 > 0:
            self.q.reset_noise()
        
        # Si NO hay Noisy Nets, usamos epsilon-greedy manual
        if not eval_mode and self.cfg.noisy_sigma0 == 0 and random() < 0.05:
            return np.random.randint(self.n_actions)  # FIX: self.n_actions ahora existe

        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.uint8).unsqueeze(0)
        # FIX: self.q() ya devuelve softmax internamente, NO aplicar F.softmax de nuevo
        prob = self.q(obs_t)
        q_values = (prob * self.support).sum(2)
        return int(q_values.argmax(1).item())

    def store(self, obs, action, reward, next_obs, done):
        # FIX: Implementación de n-step returns
        # Acumulamos transiciones en el buffer temporal
        self._nstep_buf.append((obs, action, reward, next_obs, done))

        # Solo guardamos en replay cuando tenemos n pasos O el episodio termina
        if len(self._nstep_buf) < self.cfg.n_step and not done:
            return

        # Calcular retorno acumulado n-step: R = r0 + gamma*r1 + gamma^2*r2 + ...
        R = 0.0
        for i, (_, _, r, _, _) in enumerate(self._nstep_buf):
            R += (self.cfg.gamma ** i) * r

        # La transición que guardamos usa obs del primer paso y next_obs del último
        obs_0, action_0, _, _, _ = self._nstep_buf[0]
        _, _, _, next_obs_n, done_n = self._nstep_buf[-1]

        self.replay.add(obs_0, action_0, R, next_obs_n, done_n)

        # --- LOGGING TEMPORAL: retornos n-step para calibrar v_min/v_max ---
        if not hasattr(self, '_r_log'):
            self._r_log = []
        self._r_log.append(R)
        if len(self._r_log) % 1000 == 0:
            recent = self._r_log[-1000:]
            print(
                f"[n-step R | last 1000] "
                f"min={np.min(recent):.3f}  "
                f"max={np.max(recent):.3f}  "
                f"mean={np.mean(recent):.3f}  "
                f"p5={np.percentile(recent, 5):.3f}  "
                f"p95={np.percentile(recent, 95):.3f}  "
                f"(step={self.global_step})"
            )
        # --- FIN LOGGING TEMPORAL ---

        # Si el episodio termina, vaciamos el buffer completamente
        # (guardamos también las transiciones finales con menos de n pasos)
        if done:
            self._nstep_buf.clear()
        else:
            # Avanzamos la ventana deslizante eliminando el primer elemento
            self._nstep_buf.pop(0)

    def can_update(self) -> bool:
        # Usamos .tree.n_entries para saber cuántos datos hay realmente
        return (self.global_step >= self.cfg.learning_starts and 
                self.replay.tree.n_entries >= self.cfg.batch_size)

    def update(self) -> dict[str, float]:
        # Si el paso actual no toca entrenar, salimos
        if (self.global_step % self.cfg.train_freq) != 0: 
            return {}

        # --- LÓGICA DE ABLACIÓN PARA PER ---
        # Si prio_alpha es 0, no hay priorización, beta no importa (muestreo uniforme)
        if self.cfg.prio_alpha > 0:
            beta = min(1.0, self.cfg.prio_beta_start + self.global_step * (self.cfg.prio_beta_end - self.cfg.prio_beta_start) / self.cfg.prio_beta_steps)
        else:
            beta = 0.0 # Valor neutral para muestreo uniforme
        # -----------------------------------

        # FIX: sample() devuelve un único dict, no una tupla de 3 elementos
        batch_np = self.replay.sample(self.cfg.batch_size, beta, self.rng)
        idx = batch_np["indices"]

        # Carga de tensores (con las correcciones de tipos anteriores)
        obs = torch.as_tensor(batch_np["obs"], device=self.device)
        next_obs = torch.as_tensor(batch_np["next_obs"], device=self.device)
        action = torch.as_tensor(batch_np["actions"], device=self.device)
        reward = torch.as_tensor(batch_np["rewards"], device=self.device)
        done = torch.as_tensor(batch_np["dones"], device=self.device, dtype=torch.float32)
        w = torch.as_tensor(batch_np["weights"], device=self.device)

        self.q.train()
        # Solo reseteamos ruido si Noisy Nets están activas (noisy_sigma0 > 0)
        if self.cfg.noisy_sigma0 > 0:
            self.q.reset_noise()

        # Probabilidades actuales para la acción tomada
        # FIX: self.q() ya devuelve softmax, log_softmax aplicaría softmax dos veces.
        # Usamos log directo con clamp para estabilidad numérica.
        prob_current = self.q(obs)[range(self.cfg.batch_size), action]
        log_prob = torch.log(prob_current.clamp(min=1e-8))

        with torch.no_grad():
            # Double DQN: Selección con Q, evaluación con Q_target
            # FIX: self.q() ya devuelve softmax, no aplicar F.softmax de nuevo
            next_action = (self.q(next_obs) * self.support).sum(2).argmax(1)
            next_prob = self.q_target(next_obs)[range(self.cfg.batch_size), next_action]

            # Proyección C51
            # FIX: usar gamma**n_step porque reward ya acumula n pasos de descuento
            gamma_n = self.cfg.gamma ** self.cfg.n_step
            tz = reward.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * gamma_n * self.support
            tz = tz.clamp(self.cfg.v_min, self.cfg.v_max)
            
            b = (tz - self.cfg.v_min) / self.delta_z
            l, u = b.floor().long(), b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.cfg.n_atoms - 1)) * (l == u)] += 1

            l = l.clamp(0, self.cfg.n_atoms - 1)
            u = u.clamp(0, self.cfg.n_atoms - 1)

            target_dist = torch.zeros(self.cfg.batch_size, self.cfg.n_atoms, device=self.device)
            offset = torch.linspace(0, (self.cfg.batch_size - 1) * self.cfg.n_atoms, self.cfg.batch_size).to(self.device).long().unsqueeze(1)
            
            # Suma de probabilidades proyectadas (float() evita error de Double)
            target_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_prob * (u.float() - b)).view(-1).float())
            target_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_prob * (b - l.float())).view(-1).float())

        # Pérdida de Cross-Entropy (pesada por w del PER)
        loss = (-(target_dist * log_prob).sum(1) * w).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm > 0: 
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.opt.step()

        # Actualizar prioridades solo si PER está activo
        if self.cfg.prio_alpha > 0:
            td_errors = (-(target_dist * log_prob).sum(1)).detach().cpu().numpy()
            self.replay.update_priorities(idx, td_errors)

        self.updates += 1
        if (self.updates % self.cfg.target_update_freq) == 0: 
            self.q_target.load_state_dict(self.q.state_dict())

        return {"loss": loss.item(), "updates": self.updates}

    def _reset_noise(self):
        for m in self.q.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def save(self, path: str):
        torch.save({"q": self.q.state_dict(), "step": self.global_step}, path)

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q"])
        return ckpt.get("step", 0)