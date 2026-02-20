# # src/DQN_walker2d/reward.py

# from __future__ import annotations

# import gymnasium as gym
# import numpy as np


# def walker2d_default_reward(
#     obs: np.ndarray,
#     action: int,
#     next_obs: np.ndarray,
#     terminated: bool,
#     truncated: bool,
#     info: dict,
#     env_reward: float,
#     env: gym.Env,
# ) -> float:
#     """Reproduce the default Walker2d reward, with an extra lean_forward term.

#     Total reward:
#         reward = healthy_reward + lean_forward + forward_reward - ctrl_cost
#     """
#     healthy_reward: float = float(info.get("reward_survive", 0.0))
#     forward_reward: float = float(info.get("reward_forward", 0.0))
#     ctrl_cost: float = float(info.get("reward_ctrl", 0.0))

#     x_hips: float = float(info.get("x_hips", 0.0))
#     x_foot1: float = float(info.get("x_foot1", 0.0))
#     x_foot2: float = float(info.get("x_foot2", 0.0))

#     feet_dist: float = float(info.get("feet_dist_2d", 0.0))

#     # assert False

#     lean_forward: float = x_hips - ((x_foot1 + x_foot2) / 2.0)
#     lowest_foot_height: float = float(info.get("lowest_foot_height", 0.0))

#     reward: float = 0.0
#     reward += 0.5 * healthy_reward
#     reward += forward_reward
#     reward += lean_forward
#     reward += lowest_foot_height
#     reward += feet_dist
#     reward -= ctrl_cost

#     return reward


# # src/DQN_walker2d/reward.py

# from __future__ import annotations

# import gymnasium as gym
# import numpy as np


# def walker2d_default_reward(
#     obs: np.ndarray,
#     action: int,
#     next_obs: np.ndarray,
#     terminated: bool,
#     truncated: bool,
#     info: dict,
#     env_reward: float,
#     env: gym.Env,
# ) -> float:
#     """Reproduce the default Walker2d reward, with an extra lean_forward term.

#     Total reward:
#         reward = healthy_reward + lean_forward + forward_reward - ctrl_cost
#     """
#     healthy_reward: float = float(info.get("reward_survive", 0.0))
#     forward_reward: float = float(info.get("reward_forward", 0.0))
#     ctrl_cost: float = float(info.get("reward_ctrl", 0.0))

#     x_hips: float = float(info.get("x_hips", 0.0))
#     x_foot1: float = float(info.get("x_foot1", 0.0))
#     x_foot2: float = float(info.get("x_foot2", 0.0))

#     feet_dist: float = float(info.get("feet_dist_2d", 0.0))

#     # assert False

#     lean_forward: float = x_hips - ((x_foot1 + x_foot2) / 2.0)
#     lowest_foot_height: float = float(info.get("lowest_foot_height", 0.0))

#     reward: float = 0.0
#     reward += 0.5 * healthy_reward
#     reward += forward_reward
#     reward += lean_forward
#     reward += lowest_foot_height
#     reward += feet_dist
#     reward -= 2 * (ctrl_cost**2)

#     return reward


# src/DQN_walker2d/reward.py

from __future__ import annotations

import time

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
    """Reproduce the default Walker2d reward, with extra shaping terms.

    Total reward:
        reward = 0.5*healthy_reward + forward_reward + lean_forward
                 + lowest_foot_height + feet_dist + feet_rel_speed
                 - 2*(ctrl_cost^2)
    """
    healthy_reward: float = float(info.get("reward_survive", 0.0))
    forward_reward: float = float(info.get("reward_forward", 0.0))
    ctrl_cost: float = float(info.get("reward_ctrl", 0.0))

    x_hips: float = float(info.get("x_hips", 0.0))
    x_foot1: float = float(info.get("x_foot1", 0.0))
    x_foot2: float = float(info.get("x_foot2", 0.0))

    feet_dist: float = float(info.get("feet_dist_2d", 0.0))
    lowest_foot_height: float = float(info.get("lowest_foot_height", 0.0))

    foot1_speed2d: float = float(info.get("foot1_speed2d", 0.0))
    foot2_speed2d: float = float(info.get("foot2_speed2d", 0.0))
    torso_speed: float = float(info.get("torso_speed", 0.0))

    lean_forward: float = x_hips - ((x_foot1 + x_foot2) / 2.0)
    feet_torso_rel_speed: float = min(foot1_speed2d, foot2_speed2d) - torso_speed

    # # custom
    # reward: float = 0.0
    # reward += 0.5 * healthy_reward
    # reward += forward_reward
    # reward += lean_forward
    # reward += lowest_foot_height
    # reward += feet_dist
    # reward += 0.1 * max(0, feet_torso_rel_speed) ** 2
    # reward -= ctrl_cost
    # reward -= 2 * (ctrl_cost**2)

    # # default reward
    # reward: float = 0.0
    # reward += healthy_reward
    # reward += forward_reward
    # reward -= ctrl_cost

    # return reward

    # (1) Forward-speed shaping (small + clipped)
    r: float = 0.0
    r += healthy_reward
    r += forward_reward
    r -= ctrl_cost
    torso_speed = float(info.get("torso_speed", 0.0))
    r += 0.10 * float(np.clip(torso_speed, 0.0, 4.0))  # weight small
    # (2) Alive bonus (small)
    if not (terminated or truncated):
        r += 0.05

    print(r, torso_speed, healthy_reward, forward_reward, ctrl_cost, lean_forward)
    time.sleep(0.5)

    return r
