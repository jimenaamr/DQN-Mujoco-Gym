# src/DQN_walker2d/reward.py

from __future__ import annotations

import gymnasium as gym
import numpy as np


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
    """Reproduce the default Walker2d reward, with an extra lean_forward term.

    Total reward:
        reward = healthy_reward + lean_forward + forward_reward - ctrl_cost
    """
    healthy_reward: float = float(info.get("reward_survive", 0.0))
    forward_reward: float = float(info.get("reward_forward", 0.0))
    ctrl_cost: float = float(info.get("reward_ctrl", 0.0))

    x_hips: float = float(info.get("x_hips", 0.0))
    x_foot1: float = float(info.get("x_foot1", 0.0))
    x_foot2: float = float(info.get("x_foot2", 0.0))

    feet_dist: float = float(info.get("feet_dist_2d", 0.0))

    # assert False

    lean_forward: float = x_hips - ((x_foot1 + x_foot2) / 2.0)
    lowest_foot_height: float = float(info.get("lowest_foot_height", 0.0))

    reward: float = 0.0
    reward += 0.5 * healthy_reward
    reward += forward_reward
    reward += lean_forward
    reward += lowest_foot_height
    reward += feet_dist
    reward -= ctrl_cost

    return reward
