# src/ablation/helper.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


def resolve_device(name: str) -> str:
    """Resolve a user-facing device option into a torch device string.

    Args:
        name: User-provided device name. Allowed: "cpu", "gpu".

    Returns:
        "cpu" or "cuda" (if available) depending on `name`.

    Raises:
        ValueError: If `name` is not one of {"cpu", "gpu"}.
    """
    import torch

    normalized: str = str(name).strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "gpu":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Invalid device '{name}'. Use 'cpu' or 'gpu'.")


@dataclass(frozen=True)
class StabilizerConfig:
    """Configuration for the head stabilizer forces."""

    initial_intensity: float
    decay: float
    ref_height: float
    ref_fw: float


class HeadStabilizerWrapper(gym.Wrapper):
    """Inject stabilizing external forces on the head body in a MuJoCo env.

    Updated spec implemented here:

    - Build a target point `p` in the (x,z) plane using:
        * x-target = mean joint x + (ref_fw * initial_head_z)
        * z-target = ref_height * initial_head_z if head_z below it, else head_z
    - Let `h` be current head position in the (x,z) plane.
    - Compute beta = |speed(h_t) - speed(h_{t-1})| using MuJoCo body linear speed.
    - Compute alpha = k(x) * base_force * ||p - h||^2
        * k(x) = 1 / (1 + (x/a)^2), a = decay, x = accumulated forward reward
        * base_force = initial_intensity * (total_mass * |g|)
    - Apply an external force at the head with:
        * direction = (p - h)
        * magnitude = alpha
      i.e., force = alpha * unit(p - h)
    - Scale reward by (beta / (alpha + beta)) each step.
    """

    def __init__(
        self,
        env: gym.Env,
        cfg: StabilizerConfig,
        head_body_name: str = "head",
        fallback_head_body_name: str = "torso",
        fallback_body_id: int = 1,
    ) -> None:
        super().__init__(env=env)
        self.cfg: StabilizerConfig = cfg

        unwrapped: gym.Env = self.env.unwrapped
        if (not hasattr(unwrapped, "model")) or (not hasattr(unwrapped, "data")):
            raise TypeError("HeadStabilizerWrapper requires a MuJoCo-based env.")

        self._model: Any = unwrapped.model
        self._data: Any = unwrapped.data

        self._head_id: int = _resolve_body_id(
            model=self._model,
            primary_name=head_body_name,
            fallback_name=fallback_head_body_name,
            fallback_body_id=fallback_body_id,
        )

        joint_body_ids: np.ndarray = np.asarray(self._model.jnt_bodyid, dtype=np.int32)
        joint_body_ids = joint_body_ids[joint_body_ids != 0]
        if joint_body_ids.size == 0:
            joint_body_ids = np.arange(1, int(self._model.nbody), dtype=np.int32)
        self._joint_body_ids: np.ndarray = joint_body_ids

        gravity: np.ndarray = np.asarray(self._model.opt.gravity, dtype=np.float64)
        self._g_mag: float = float(np.linalg.norm(gravity))

        body_mass: np.ndarray = np.asarray(self._model.body_mass, dtype=np.float64)
        self._total_mass: float = float(np.sum(body_mass[1:]))

        self._initial_head_z: float | None = None
        self._forward_accum: float = 0.0
        self._prev_head_speed: float | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset episode state and cache initial head height.

        Args:
            **kwargs: Forwarded to the wrapped env reset().

        Returns:
            Observation and info from the wrapped environment.
        """
        obs: np.ndarray
        info: dict[str, Any]
        obs, info = self.env.reset(**kwargs)

        head_pos: np.ndarray = np.asarray(
            self._data.xpos[self._head_id], dtype=np.float64
        )
        self._initial_head_z = float(head_pos[2])
        self._forward_accum = 0.0

        head_speed: float = _body_speed(data=self._data, body_id=self._head_id)
        self._prev_head_speed = float(head_speed)

        self._data.xfrc_applied[:] = 0.0
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply stabilizer force, step the environment, and rescale reward.

        Args:
            action: Discrete action index (passed through to the wrapped env).

        Returns:
            obs, reward, terminated, truncated, info
        """
        if self._initial_head_z is None or self._prev_head_speed is None:
            raise RuntimeError("reset() must be called before step().")

        self._data.xfrc_applied[:] = 0.0

        head_pos: np.ndarray = np.asarray(
            self._data.xpos[self._head_id], dtype=np.float64
        )
        head_x: float = float(head_pos[0])
        head_z: float = float(head_pos[2])

        joints_x: np.ndarray = np.asarray(
            self._data.xpos[self._joint_body_ids, 0], dtype=np.float64
        )
        mean_x: float = float(np.mean(joints_x)) if joints_x.size > 0 else head_x

        fw_offset: float = float(self.cfg.ref_fw) * float(self._initial_head_z)
        target_x: float = float(mean_x + fw_offset)

        ref_z: float = float(self.cfg.ref_height) * float(self._initial_head_z)
        target_z: float = float(ref_z) if head_z < ref_z else float(head_z)

        dx: float = float(target_x - head_x)
        dz: float = float(target_z - head_z)
        d_xz: np.ndarray = np.array([dx, 0.0, dz], dtype=np.float64)

        base_gravity_force: float = float(self._total_mass * self._g_mag)
        base_force: float = float(self.cfg.initial_intensity) * base_gravity_force

        a: float = float(self.cfg.decay)
        x: float = float(self._forward_accum)
        k_decay: float = 1.0 if a <= 0.0 else float(1.0 / (1.0 + (x / a) ** 2))

        dist2: float = float(dx * dx + dz * dz)
        alpha: float = float(base_force * k_decay * dist2)

        head_speed_now: float = _body_speed(data=self._data, body_id=self._head_id)
        beta: float = float(abs(float(head_speed_now) - float(self._prev_head_speed)))

        if False:
            print(f"alpha: {alpha:.2f}, beta: {beta:.2f}")
            print(
                f"base_force: {base_force:.3f} k_decay: {k_decay:.3f}, dist2: {dist2:.3f}"
            )
            print(
                f"head speed: {head_speed_now:.3f}, prev speed: {self._prev_head_speed:.3f}, head acceleration: {beta:.3f}"
            )

            time.sleep(5.0)

        force_vec: np.ndarray = np.zeros((3,), dtype=np.float64)
        d_norm: float = float(np.linalg.norm(d_xz))
        if alpha > 0.0 and d_norm > 1e-12:
            force_vec = (alpha / d_norm) * d_xz

        self._data.xfrc_applied[self._head_id, :3] = force_vec.astype(np.float64)

        obs: np.ndarray
        reward: float
        terminated: bool
        truncated: bool
        info: dict[str, Any]
        obs, reward, terminated, truncated, info = self.env.step(action)

        forward_inc: float = float(info.get("reward_forward", 0.0))
        self._forward_accum += forward_inc

        head_speed_after: float = _body_speed(data=self._data, body_id=self._head_id)
        self._prev_head_speed = float(head_speed_after)

        _denom: float = float(alpha + beta)
        _agent_contrib: float = float(beta / _denom) if _denom > 0.0 else 0.0
        _real_reward: float = float(reward) * _agent_contrib

        return obs, float(reward), bool(terminated), bool(truncated), info


def _body_speed(data: Any, body_id: int) -> float:
    """Compute body linear speed magnitude.

    Args:
        data: MuJoCo MjData-like object.
        body_id: Body index.

    Returns:
        Linear speed magnitude.
    """
    if hasattr(data, "xvelp"):
        v: np.ndarray = np.asarray(data.xvelp[body_id], dtype=np.float64)
        return float(np.linalg.norm(v))

    if hasattr(data, "cvel"):
        cvel: np.ndarray = np.asarray(data.cvel[body_id], dtype=np.float64)
        v_local: np.ndarray = cvel[3:6]
        return float(np.linalg.norm(v_local))

    return 0.0


def _resolve_body_id(
    model: Any,
    primary_name: str,
    fallback_name: str,
    fallback_body_id: int,
) -> int:
    """Resolve a MuJoCo body id robustly.

    Args:
        model: MuJoCo model object.
        primary_name: First name to try.
        fallback_name: Second name to try.
        fallback_body_id: Final fallback numeric id.

    Returns:
        Selected body id.
    """
    try:
        return int(model.body(primary_name).id)
    except Exception:
        pass

    try:
        return int(model.body(fallback_name).id)
    except Exception:
        pass

    nbody: int = int(getattr(model, "nbody", fallback_body_id + 1))
    if nbody <= 0:
        return int(fallback_body_id)

    safe_id: int = int(np.clip(fallback_body_id, 0, nbody - 1))
    return safe_id


def stabilizer_config_from_yaml(cfg: dict[str, Any]) -> StabilizerConfig | None:
    """Parse `stabilizer:` section from a loaded YAML config.

    Args:
        cfg: Full experiment configuration.

    Returns:
        Parsed StabilizerConfig or None if not present.
    """
    raw: Any = cfg.get("stabilizer")
    if raw is None:
        return None
    s: dict[str, Any] = dict(raw)
    return StabilizerConfig(
        initial_intensity=float(s["initial_intensity"]),
        decay=float(s["decay"]),
        ref_height=float(s["ref_height"]),
        ref_fw=float(s.get("ref_fw", 0.0)),
    )
