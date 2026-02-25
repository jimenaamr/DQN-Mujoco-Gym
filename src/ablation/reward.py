# src/ablation/reward.py

from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.rdqn.monitoring import MONITOR


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

    Total reward:
        reward = healthy_reward + forward_reward - ctrl_cost
                 + 0.10*clip(torso_speed,0,4)
                 + 0.20*clip(feet_dist,0,2)
                 + 0.10*(z_torso - torso_height_offset)
                 + alive_bonus
    """
    healthy_reward: float = float(info.get("reward_survive", 0.0))
    forward_reward: float = float(info.get("reward_forward", 0.0))
    ctrl_cost: float = float(info.get("reward_ctrl", 0.0))

    feet_dist: float = float(info.get("feet_dist_2d", 0.0))
    torso_speed: float = float(info.get("torso_speed", 0.0))

    r: float = 0.0
    r += healthy_reward
    r += forward_reward
    r -= ctrl_cost

    r += 0.10 * float(np.clip(torso_speed, 0.0, 4.0))
    r += 0.10 * float(np.clip(feet_dist, 0.0, 2.0))

    z_torso: float = float(info.get("z_torso", 0.0))
    torso_height_offset: float = 1.1
    torso_height_term: float = float(z_torso - torso_height_offset)
    r += 0.10 * torso_height_term

    if not (terminated or truncated):
        r += 0.05

    MONITOR.add_field(name="torso_speed", value=torso_speed)
    MONITOR.add_field(
        name="capped_weighted_torso_speed",
        value=0.10 * float(np.clip(torso_speed, 0.0, 4.0)),
    )
    MONITOR.add_field(name="feet_dist", value=feet_dist)
    MONITOR.add_field(
        name="capped_weighted_feet_dist",
        value=0.20 * float(np.clip(feet_dist, 0.0, 2.0)),
    )
    MONITOR.add_field(name="z_torso", value=z_torso)
    MONITOR.add_field(name="torso_height_offset", value=torso_height_offset)
    MONITOR.add_field(name="torso_height_term", value=torso_height_term)
    MONITOR.add_field(
        name="capped_weighted_torso_height_term",
        value=0.10 * torso_height_term,
    )

    return r
