from __future__ import annotations

from dataclasses import dataclass
from typing import List
import cv2

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvSpec:
    env_id: str
    frame_stack: int
    action_repeat: int
    time_limit: int
    action_prototypes: List[List[float]]


class PixelObservationWrapper(gym.Wrapper):
    """
    Replace vector observation with rendered RGB image.
    """
    def __init__(self, env: gym.Env, height=84, width=84):
        super().__init__(env)
        self.height = height
        self.width = width

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, height, width),
            dtype=np.uint8,
        )

    def _get_obs(self):
        frame = self.env.render()  # (H, W, 3)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)  # (h,w,3)
        frame = np.transpose(frame, (2, 0, 1))  # (3, h, w)
        return frame.astype(np.uint8)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._get_obs(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(), reward, terminated, truncated, info


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        self.frames = None

        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c * k, h, w),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(self.frames, axis=0)


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, prototypes: np.ndarray):
        super().__init__(env)
        self.prototypes = prototypes.astype(np.float32)
        self.action_space = spaces.Discrete(len(prototypes))

    def action(self, act):
        return self.prototypes[act]


def make_env(spec: EnvSpec, seed: int):
    env = gym.make(spec.env_id, render_mode="rgb_array")
    env.reset(seed=seed)

    # Convert to pixel observation
    env = PixelObservationWrapper(env)

    # Discretize continuous actions
    prototypes = np.array(spec.action_prototypes, dtype=np.float32)
    cont_dim = env.action_space.shape[0]
    if prototypes.shape[1] != cont_dim:
        raise ValueError(
            f"action_prototypes dim mismatch: got {prototypes.shape[1]} but env action dim is {cont_dim}"
        )

    env = DiscreteActionWrapper(env, prototypes)

    # Frame stack
    if spec.frame_stack > 1:
        env = FrameStack(env, spec.frame_stack)

    return env