from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


class DiscreteActionWrapper(gym.ActionWrapper):
    """Map discrete action index -> continuous action prototype."""
    def __init__(self, env: gym.Env, prototypes: np.ndarray):
        super().__init__(env)
        assert prototypes.ndim == 2
        self.prototypes = prototypes.astype(np.float32)
        self.action_space = spaces.Discrete(self.prototypes.shape[0])

    def action(self, act: int):
        return self.prototypes[int(act)]


class ActionRepeat(gym.Wrapper):
    """Repeat the same action K times and accumulate reward."""
    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        assert repeat >= 1
        self.repeat = int(repeat)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class FrameStack(gym.Wrapper):
    """Stack last K observations along the last axis (Gymnasium style)."""
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        assert k >= 1
        self.k = int(k)
        self.frames = None

        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 1  # classic mujoco obs is vector
        n = obs_space.shape[0]

        low = np.repeat(obs_space.low, self.k, axis=0)
        high = np.repeat(obs_space.high, self.k, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=obs_space.dtype)

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
        assert self.frames is not None
        return np.concatenate(self.frames, axis=0)


def make_env(spec: EnvSpec, seed: int) -> gym.Env:
    env = gym.make(spec.env_id)

    # Seed
    env.reset(seed=seed)

    # Time limit (override if needed)
    if spec.time_limit is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(spec.time_limit))

    # Action repeat
    if spec.action_repeat and int(spec.action_repeat) > 1:
        env = ActionRepeat(env, int(spec.action_repeat))

    # Discretize continuous actions via prototypes
    prototypes = np.array(spec.action_prototypes, dtype=np.float32)
    cont_dim = int(np.prod(env.action_space.shape))
    if prototypes.shape[1] != cont_dim:
        raise ValueError(
            f"action_prototypes dim mismatch: got {prototypes.shape[1]} but env action dim is {cont_dim}. "
            "Fix configs/dqn.yaml env.action_prototypes."
        )
    env = DiscreteActionWrapper(env, prototypes)

    # Frame stack (vector obs)
    if spec.frame_stack and int(spec.frame_stack) > 1:
        env = FrameStack(env, int(spec.frame_stack))

    return env