# src/rdqn/reward.py

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
    terminated: bool,
    truncated: bool,
    info: dict,
) -> float:

    healthy_reward = float(info.get("reward_survive", 0.0))
    forward_reward = float(info.get("reward_forward", 0.0))
    ctrl_cost = float(info.get("reward_ctrl", 0.0))

    r = 0.0
    r += healthy_reward * 1.2
    r += forward_reward * 0.5
    r -= ctrl_cost

    return r
