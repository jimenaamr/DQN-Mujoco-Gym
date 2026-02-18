# src/DQN_walker2d/helper.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from src.DQN_walker2d.monitoring import MONITOR


@dataclass(frozen=True)
class StabilizerConfig:
    """Configuration for the head stabilizer forces."""

    initial_intensity: float
    decay: float
    ref_height: float


class HeadStabilizerWrapper(gym.Wrapper):
    """Inject stabilizing external forces on the head body in a MuJoCo env.

    See your spec in the previous message; this implementation follows it and
    avoids relying on `model.body_names` (not present in mujoco Python structs).
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

        self._model = unwrapped.model
        self._data = unwrapped.data

        self._head_id = _resolve_body_id(
            model=self._model,
            primary_name=head_body_name,
            fallback_name=fallback_head_body_name,
            fallback_body_id=fallback_body_id,
        )

        joint_body_ids: np.ndarray = np.asarray(self._model.jnt_bodyid, dtype=np.int32)
        joint_body_ids = joint_body_ids[joint_body_ids != 0]  # drop world
        if joint_body_ids.size == 0:
            joint_body_ids = np.arange(1, int(self._model.nbody), dtype=np.int32)
        self._joint_body_ids: np.ndarray = joint_body_ids

        gravity: np.ndarray = np.asarray(self._model.opt.gravity, dtype=np.float64)
        self._g_mag: float = float(np.linalg.norm(gravity))

        body_mass: np.ndarray = np.asarray(self._model.body_mass, dtype=np.float64)
        self._total_mass: float = float(np.sum(body_mass[1:]))

        self._initial_head_z: float | None = None
        self._forward_accum: float = 0.0

        self._beta: float = 1.0

    def set_beta(self, beta: float) -> None:
        """Set the fixed beta multiplier for the agent-produced head force."""
        self._beta = float(max(beta, 0.0))

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs: np.ndarray
        info: dict[str, Any]
        obs, info = self.env.reset(**kwargs)

        head_pos: np.ndarray = np.asarray(
            self._data.xpos[self._head_id], dtype=np.float64
        )

        self._initial_head_z = float(head_pos[2])
        self._forward_accum = 0.0

        self._data.xfrc_applied[:] = 0.0
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._initial_head_z is None:
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

        horiz: np.ndarray = np.array([mean_x - head_x, 0.0, 0.0], dtype=np.float64)

        ref_z: float = float(self.cfg.ref_height) * float(self._initial_head_z)
        if head_z < ref_z:
            vert: np.ndarray = np.array([0.0, 0.0, ref_z - head_z], dtype=np.float64)
        else:
            vert = np.zeros((3,), dtype=np.float64)

        u: np.ndarray = horiz + vert

        a: float = float(self.cfg.decay)
        x: float = float(self._forward_accum)
        f_decay: float = 1.0 if a <= 0.0 else float(1.0 / (1.0 + (x / a) ** 2))

        base_gravity_force: float = float(self._total_mass * self._g_mag)
        target_mag: float = (
            float(self.cfg.initial_intensity) * base_gravity_force * f_decay
        )

        f_head_agent: np.ndarray = np.asarray(
            self._data.cfrc_int[self._head_id, :3], dtype=np.float64
        )
        w: np.ndarray = float(self._beta) * f_head_agent
        w_mag: float = float(np.linalg.norm(w))

        alpha: float
        diff: np.ndarray
        if (target_mag <= 0.0) or (w_mag >= target_mag):
            alpha = 0.0
            diff = np.zeros((3,), dtype=np.float64)
        else:
            if float(np.linalg.norm(u)) <= 1e-12:
                alpha = 0.0
                diff = np.zeros((3,), dtype=np.float64)
            else:
                alpha = _solve_alpha_for_target_norm(u=u, w=w, target=target_mag)
                diff = alpha * u - w

        self._data.xfrc_applied[self._head_id, :3] = diff.astype(np.float64)

        # obs: np.ndarray
        # reward: float
        # terminated: bool
        # truncated: bool
        # info: dict[str, Any]
        # obs, reward, terminated, truncated, info = self.env.step(action)

        # forward_inc: float = float(info.get("reward_forward", 0.0))
        # self._forward_accum += forward_inc

        # denom: float = float(alpha + self._beta)
        # if denom > 0.0:
        #     reward = float(reward) * (float(self._beta) / denom)

        # return obs, float(reward), bool(terminated), bool(truncated), info

        obs: np.ndarray
        reward: float
        terminated: bool
        truncated: bool
        info: dict[str, Any]
        obs, reward, terminated, truncated, info = self.env.step(action)

        forward_inc: float = float(info.get("reward_forward", 0.0))
        self._forward_accum += forward_inc

        denom: float = float(alpha + self._beta)
        agent_contrib: float = float(self._beta / denom) if denom > 0.0 else 0.0

        real_reward: float = float(reward) * agent_contrib

        # ---------- monitoring ----------
        MONITOR.set_raw_reward(float(reward))
        MONITOR.set_head_height(head_z)
        MONITOR.set_acc_fw_reward(self._forward_accum)
        MONITOR.set_helper_intensity(
            target_mag / base_gravity_force if base_gravity_force > 0 else 0.0
        )
        MONITOR.set_agent_contrib(agent_contrib)
        MONITOR.set_real_reward(real_reward)
        # --------------------------------

        return obs, real_reward, bool(terminated), bool(truncated), info


def _resolve_body_id(
    model: Any,
    primary_name: str,
    fallback_name: str,
    fallback_body_id: int,
) -> int:
    """Resolve a MuJoCo body id robustly, without enumerating model names.

    Args:
        model: MuJoCo MjModel-like object from Gymnasium.
        primary_name: First name to try (e.g., "head").
        fallback_name: Second name to try (e.g., "torso").
        fallback_body_id: Final fallback id (usually 1 = first non-world body).

    Returns:
        Body id to use.
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


def _solve_alpha_for_target_norm(u: np.ndarray, w: np.ndarray, target: float) -> float:
    """Solve for alpha >= 0 such that ||alpha*u - w|| == target when possible."""
    A: float = float(np.dot(u, u))
    B: float = float(-2.0 * np.dot(u, w))
    C: float = float(np.dot(w, w) - target**2)

    disc: float = float(B * B - 4.0 * A * C)
    disc = max(disc, 0.0)
    sqrt_disc: float = float(np.sqrt(disc))

    alpha1: float = float((-B + sqrt_disc) / (2.0 * A))
    alpha2: float = float((-B - sqrt_disc) / (2.0 * A))

    candidates: list[float] = [a for a in (alpha1, alpha2) if a >= 0.0]
    return min(candidates) if candidates else 0.0


def stabilizer_config_from_yaml(cfg: dict[str, Any]) -> StabilizerConfig | None:
    """Parse `stabilizer:` section from a loaded YAML config."""
    raw: Any = cfg.get("stabilizer")
    if raw is None:
        return None
    s: dict[str, Any] = dict(raw)
    return StabilizerConfig(
        initial_intensity=float(s["initial_intensity"]),
        decay=float(s["decay"]),
        ref_height=float(s["ref_height"]),
    )
