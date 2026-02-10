from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import gymnasium as gym

try:
    import dm_control2gymnasium  # type: ignore
except Exception as e:
    dm_control2gymnasium = None


@dataclass
class EnvSpec:
    domain: str
    task: str
    height: int
    width: int
    camera_id: int
    frame_stack: int
    action_repeat: int
    time_limit: int
    action_prototypes: List[List[float]]


class FrameStack(gym.Wrapper):
    """Stack last K frames along channel dimension (C becomes C*K)."""
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        assert k >= 1
        self.k = k
        self.frames = None

        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        assert len(obs_space.shape) == 3  # (C,H,W)
        c, h, w = obs_space.shape

        low = np.repeat(obs_space.low, k, axis=0)
        high = np.repeat(obs_space.high, k, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

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


class DiscreteActionWrapper(gym.ActionWrapper):
    """Map discrete action index -> continuous action prototype."""
    def __init__(self, env: gym.Env, prototypes: np.ndarray):
        super().__init__(env)
        assert prototypes.ndim == 2
        self.prototypes = prototypes.astype(np.float32)
        self.action_space = gym.spaces.Discrete(self.prototypes.shape[0])

    def action(self, act: int):
        return self.prototypes[int(act)]


def make_env(spec: EnvSpec, seed: int) -> gym.Env:
    if dm_control2gymnasium is None:
        raise ImportError(
            "dm_control2gymnasium not available. Install it or adapt make_env to your wrapper."
        )

    env = dm_control2gymnasium.make(
        domain_name=spec.domain,
        task_name=spec.task,
        seed=seed,
        height=spec.height,
        width=spec.width,
        camera_id=spec.camera_id,
        frame_skip=spec.action_repeat,
        time_limit=spec.time_limit,
        channels_first=True,     # IMPORTANT for PyTorch: (C,H,W)
        from_pixels=True,
    )

    # Discretize actions
    prototypes = np.array(spec.action_prototypes, dtype=np.float32)
    env = DiscreteActionWrapper(env, prototypes)

    # Frame stack
    if spec.frame_stack > 1:
        env = FrameStack(env, spec.frame_stack)

    return env