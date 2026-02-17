from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.util import info
from typing import List
import cv2
from gymnasium.wrappers import RecordVideo

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit


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
        _, info = self.env.reset(**kwargs)
        return self._get_obs(), info

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
    

class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        self.repeat = int(repeat)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None

        for _ in range(self.repeat):
            obs, r, term, trunc, info = self.env.step(action)
            total_reward += float(r)
            terminated = terminated or bool(term)
            truncated = truncated or bool(trunc)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, prototypes: np.ndarray):
        super().__init__(env)
        self.prototypes = prototypes.astype(np.float32)
        self.action_space = spaces.Discrete(len(prototypes))

    def action(self, act):
        return self.prototypes[act]


def make_env(spec: EnvSpec, seed: int):
    env = gym.make(spec.env_id, render_mode="rgb_array")

    # 1) Time limit (max steps per episode)
    env = TimeLimit(env, max_episode_steps=spec.time_limit)

    # 2) Seed
    env.reset(seed=seed)

    # 3) Action repeat (repeat same action k steps)
    if spec.action_repeat > 1:
        env = ActionRepeat(env, spec.action_repeat)

    # 4) Pixels
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

def make_eval_env(spec: EnvSpec, seed: int, video_dir: str):
    env = gym.make(spec.env_id, render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=spec.time_limit)
    env.reset(seed=seed)

    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,  # graba todos
        name_prefix="eval",
        disable_logger=True,
    )

    env = PixelObservationWrapper(env)
    prototypes = np.array(spec.action_prototypes, dtype=np.float32)
    env = DiscreteActionWrapper(env, prototypes)
    if spec.frame_stack > 1:
        env = FrameStack(env, spec.frame_stack)
    return env