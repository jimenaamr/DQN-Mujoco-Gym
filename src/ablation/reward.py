# src/rdqn/reward.py

from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.ablation.monitoring import MONITOR


def _monitor_set(name: str, value: float) -> None:
    """Set a monitor field, overwriting if the key already exists."""
    key: str = str(name)
    try:
        fields: dict[str, float] = MONITOR._fields  # type: ignore[assignment]
        fields[key] = float(value)
        return
    except Exception:
        pass

    try:
        MONITOR.add_field(name=key, value=float(value))
    except Exception:
        return


def _first_xy(info: dict, names: list[str]) -> tuple[float, float] | None:
    """Return (x,y) from the first available <name>_x/<name>_y pair."""
    for n in names:
        xk: str = f"{n}_x"
        yk: str = f"{n}_y"
        if xk in info and yk in info:
            return (float(info[xk]), float(info[yk]))
    return None


def _first_dx(info: dict, names: list[str]) -> float | None:
    """Return dx from the first available <name>_dx key."""
    for n in names:
        k: str = f"{n}_dx"
        if k in info:
            return float(info[k])
    return None


def _head_height(info: dict) -> float:
    """Return head height (z) using best-available convention."""
    if "head_z" in info:
        return float(info["head_z"])
    return float(info.get("z_torso", 0.0))


def walker2d_default_reward(
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict,
    env_reward: float,
    env: gym.Env,
) -> float:
    """Reproduce the default Walker2d reward, with extra shaping terms."""
    healthy_reward: float = float(info.get("reward_survive", 0.0))
    forward_reward: float = float(info.get("reward_forward", 0.0))
    ctrl_cost: float = float(info.get("reward_ctrl", 0.0))

    x_velocity_new: float | None = _first_dx(
        info=info,
        names=["rootx", "torso", "hips", "pelvis"],
    )
    x_velocity: float = float(info.get("x_velocity", 0.0))
    if x_velocity_new is not None:
        x_velocity = float(x_velocity_new)

    feet_dist: float = float(info.get("feet_dist_2d", 0.0))
    tip_xy: tuple[float, float] | None = _first_xy(info=info, names=["tiptoes"])
    tip_l_xy: tuple[float, float] | None = _first_xy(info=info, names=["tiptoes_left"])
    if tip_xy is not None and tip_l_xy is not None:
        dx_f: float = float(tip_xy[0] - tip_l_xy[0])
        dy_f: float = float(tip_xy[1] - tip_l_xy[1])
        feet_dist = float(np.sqrt(dx_f * dx_f + dy_f * dy_f))

    z_head: float = _head_height(info=info)

    r: float = 0.0
    r += healthy_reward
    r += forward_reward
    r -= ctrl_cost

    x_velocity_reward: float = 0.10 * float(
        np.clip(a=x_velocity, a_min=-4.0, a_max=4.0)
    )
    r += x_velocity_reward

    feet_dist_reward: float = 0.10 * float(
        np.clip(a=feet_dist, a_min=0.0, a_max=2.0) - 1.0
    )
    r += feet_dist_reward

    head_height_reward: float = 0.5 * float(
        np.clip(a=(z_head - 1.05), a_min=-1.0, a_max=0.0)
    )
    r += head_height_reward

    if not (terminated or truncated):
        r += 0.05

    _monitor_set(name="x_velocity", value=x_velocity)
    _monitor_set(name="x_velocity_reward", value=x_velocity_reward)
    _monitor_set(name="feet_dist", value=feet_dist)
    _monitor_set(name="feet_dist_reward", value=feet_dist_reward)
    _monitor_set(name="z_head", value=z_head)
    _monitor_set(name="head_height_reward", value=head_height_reward)

    return r
