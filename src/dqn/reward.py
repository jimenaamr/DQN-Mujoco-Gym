from __future__ import annotations

import gymnasium as gym
import numpy as np


def _first_float(info: dict, keys: list[str]) -> float | None:
    """Return the first available float from a list of keys."""
    for k in keys:
        if k in info:
            return float(info[k])
    return None


def _body_xy(info: dict, body: str) -> tuple[float, float] | None:
    """Return (x,y) from <body>_x/<body>_y if present."""
    xk: str = f"{body}_x"
    yk: str = f"{body}_y"
    if xk in info and yk in info:
        return (float(info[xk]), float(info[yk]))
    return None


def _head_height(info: dict) -> float:
    """Return head height using the new convention: y is vertical."""
    if "head_y" in info:
        return float(info["head_y"])
    torso_y: float | None = _first_float(info=info, keys=["torso_y"])
    if torso_y is not None:
        return float(torso_y)
    return 0.0


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
    """Reproduce the default Walker2d reward, with extra shaping terms.

    This version is equivalent to the previous one, but it only relies on the
    reduced info dict:
      - default env terms (reward_* and x_velocity)
      - body x/y and dx/dy
      - head x/y and dx/dy
      - heel x/y for both feet
    """
    healthy_reward: float = float(info.get("reward_survive", 0.0))
    forward_reward: float = float(info.get("reward_forward", 0.0))
    ctrl_cost: float = float(info.get("reward_ctrl", 0.0))

    # Prefer x_velocity from env; if absent, fall back to torso_dx.
    x_velocity: float = float(info.get("x_velocity", 0.0))
    torso_dx: float | None = _first_float(info=info, keys=["torso_dx"])
    if torso_dx is not None:
        x_velocity = float(torso_dx)

    # Feet distance in the (x, vertical) plane with the requested convention (y := z).
    feet_dist: float = 0.0
    fr: tuple[float, float] | None = _body_xy(info=info, body="foot_right")
    fl: tuple[float, float] | None = _body_xy(info=info, body="foot_left")
    if fr is not None and fl is not None:
        dx_f: float = float(fr[0] - fl[0])
        dy_f: float = float(fr[1] - fl[1])
        feet_dist = float(np.sqrt(dx_f * dx_f + dy_f * dy_f))

    # Feet max speed components (new shaping term).
    foot_right_dx: float = float(info.get("foot_right_dx", 0.0))
    foot_left_dx: float = float(info.get("foot_left_dx", 0.0))
    torso_dx_val: float = float(info.get("torso_dx", x_velocity))
    feet_max_speed: float = float(
        max(foot_right_dx, foot_left_dx) - max(0.0, torso_dx_val)
    )

    r: float = 0.0
    r += healthy_reward * 1.2
    r += forward_reward * 0.5
    r -= ctrl_cost

    heel_right_y: float = float(info.get("heel_right_y", 0.0))
    heel_left_y: float = float(info.get("heel_left_y", 0.0))
    feet_height_reward: float = 0.25 * min(heel_right_y, heel_left_y)
    # r += feet_height_reward
    from src.dqn.monitoring import MONITOR

    MONITOR.add_field(name="feet_height_reward", value=feet_height_reward)
    MONITOR.add_field(name="heel_y", value=(heel_right_y, heel_left_y))

    # x_velocity_reward: float
    # x_velocity_reward = 0.02 * float(np.clip(a=x_velocity, a_min=-30.0, a_max=30.0))
    # r += x_velocity_reward

    # feet_dist_reward: float
    # feet_dist_reward = 0.30 * float(np.clip(a=feet_dist, a_min=0.0, a_max=2.0) - 0.5)
    # r += feet_dist_reward

    # feet_max_speed_reward: float = 0.02 * float(
    #     np.clip(a=feet_max_speed, a_min=-30.0, a_max=30.0)
    # )
    # r += feet_max_speed_reward

    # # Same shape as before, but using y as vertical (previously z).
    # head_height_reward: float
    # head_height_reward = 0.75 * float(np.clip(a=(head_y - 1.25), a_min=-1.0, a_max=0.0))
    # r += head_height_reward

    # if not (terminated or truncated):
    #     r += 0.05

    # MONITOR.add_field(name="healthy_reward", value=healthy_reward)
    # MONITOR.add_field(name="forward_reward", value=forward_reward)
    # MONITOR.add_field(name="ctrl_cost", value=ctrl_cost)
    # MONITOR.add_field(name="x_velocity", value=x_velocity)
    # MONITOR.add_field(name="x_velocity_reward", value=x_velocity_reward)
    # MONITOR.add_field(name="feet_dist", value=feet_dist)
    # MONITOR.add_field(name="feet_dist_reward", value=feet_dist_reward)

    # MONITOR.add_field(name="torso dx", value=torso_dx_val)
    # MONITOR.add_field(name="feet dx", value=(foot_right_dx, foot_left_dx))

    # MONITOR.add_field(name="feet_max_speed", value=feet_max_speed)
    # MONITOR.add_field(name="feet_max_speed_reward", value=feet_max_speed_reward)
    # MONITOR.add_field(name="head_height", value=head_y)
    # MONITOR.add_field(name="head_height_reward", value=head_height_reward)

    # torso_speed = float(info.get("torso_speed", 0.0))
    # r += 0.10 * float(np.clip(a=torso_speed, a_min=0.0, a_max=4.0))  # weight small
    # # (2) Alive bonus (small)
    # if not (terminated or truncated):
    #     r += 0.05

    return r
