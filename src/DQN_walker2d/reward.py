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
    """Reproduce the default Walker2d reward exactly.

    Total reward:
        reward = healthy_reward + forward_reward - ctrl_cost

    The individual terms are already provided inside `info`
    by the Gymnasium Walker2d environment.

    Args:
        obs: Previous observation.
        action: Discrete action index (unused here).
        next_obs: Next observation.
        terminated: Episode termination flag.
        truncated: Episode truncation flag.
        info: Info dict from env.step(), containing reward terms.
        env_reward: Original reward returned by the environment.
        env: Underlying Gym environment.

    Returns:
        The reconstructed default reward.
    """
    healthy_reward: float = float(info.get("reward_survive", 0.0))
    forward_reward: float = float(info.get("reward_forward", 0.0))
    ctrl_cost: float = float(info.get("reward_ctrl", 0.0))

    healthy_reward *= 0.1

    # print(
    #     f"Healthy reward: {healthy_reward}, Forward reward: {forward_reward}, Control cost: {ctrl_cost}"
    # )
    # time.sleep(2)
    return healthy_reward + forward_reward - ctrl_cost
