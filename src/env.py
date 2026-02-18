from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, TimeLimit


@dataclass
class EnvSpec:
    env_id: str
    frame_stack: int
    action_repeat: int
    time_limit: int
    action_prototypes: list[list[float]]
    use_pixels: bool = True  # NEW: allow disabling pixels for training


class PixelObservationWrapper(gym.Wrapper):
    """Replace vector observation with rendered RGB image."""

    def __init__(self, env: gym.Env, height: int = 84, width: int = 84) -> None:
        super().__init__(env=env)
        self.height: int = int(height)
        self.width: int = int(width)

        self.observation_space: spaces.Box = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.height, self.width),
            dtype=np.uint8,
        )

    def _get_obs(self) -> np.ndarray:
        frame: np.ndarray = self.env.render()  # (H, W, 3)
        frame = cv2.resize(
            src=frame,
            dsize=(self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        frame = np.transpose(a=frame, axes=(2, 0, 1))
        return frame.astype(dtype=np.uint8)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        _obs, info = self.env.reset(**kwargs)
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward: float
        terminated: bool
        truncated: bool
        _obs, reward, terminated, truncated, info = self.env.step(action=action)
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int) -> None:
        super().__init__(env=env)
        self.k: int = int(k)
        self.frames: list[np.ndarray] | None = None

        c: int
        h: int
        w: int
        c, h, w = env.observation_space.shape
        self.observation_space: spaces.Box = spaces.Box(
            low=0,
            high=255,
            shape=(c * self.k, h, w),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward: float
        terminated: bool
        truncated: bool
        obs, reward, terminated, truncated, info = self.env.step(action=action)
        assert self.frames is not None
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self) -> np.ndarray:
        assert self.frames is not None
        return np.concatenate(self.frames, axis=0)


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat: int) -> None:
        super().__init__(env=env)
        self.repeat: int = int(repeat)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward: float = 0.0
        terminated: bool = False
        truncated: bool = False
        info: dict = {}
        obs: np.ndarray | None = None

        for _ in range(self.repeat):
            r: SupportsFloat
            term: bool
            trunc: bool
            obs, r, term, trunc, info = self.env.step(action=action)
            total_reward += float(r)
            terminated = terminated or bool(term)
            truncated = truncated or bool(trunc)
            if terminated or truncated:
                break

        assert obs is not None
        return obs, total_reward, terminated, truncated, info


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, prototypes: np.ndarray) -> None:
        super().__init__(env=env)
        self.prototypes: np.ndarray = prototypes.astype(dtype=np.float32)
        self.action_space: spaces.Discrete = spaces.Discrete(n=len(prototypes))

    def action(self, act: int) -> np.ndarray:
        return self.prototypes[int(act)]


def make_env(spec: EnvSpec, seed: int) -> gym.Env:
    # Only request a render_mode if we will actually call env.render().
    render_mode: str | None = "rgb_array" if spec.use_pixels else None
    env: gym.Env = gym.make(id=spec.env_id, render_mode=render_mode)

    env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))
    env.reset(seed=int(seed))

    if spec.action_repeat > 1:
        env = ActionRepeat(env=env, repeat=spec.action_repeat)

    if spec.use_pixels:
        env = PixelObservationWrapper(env=env)

    prototypes: np.ndarray = np.array(object=spec.action_prototypes, dtype=np.float32)
    cont_dim: int = int(env.action_space.shape[0])
    if prototypes.shape[1] != cont_dim:
        raise ValueError(
            "action_prototypes dim mismatch: "
            f"got {prototypes.shape[1]} but env action dim is {cont_dim}"
        )

    env = DiscreteActionWrapper(env=env, prototypes=prototypes)

    if spec.frame_stack > 1 and spec.use_pixels:
        env = FrameStack(env=env, k=spec.frame_stack)

    return env


def make_eval_env(spec: EnvSpec, seed: int, video_dir: str) -> gym.Env:
    # For video/pixels we do need rgb_array.
    env: gym.Env = gym.make(id=spec.env_id, render_mode="rgb_array")
    env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))
    env.reset(seed=int(seed))

    env = RecordVideo(
        env=env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix="eval",
        disable_logger=True,
    )

    env = PixelObservationWrapper(env=env)

    prototypes: np.ndarray = np.array(object=spec.action_prototypes, dtype=np.float32)
    env = DiscreteActionWrapper(env=env, prototypes=prototypes)

    if spec.frame_stack > 1:
        env = FrameStack(env=env, k=spec.frame_stack)

    return env
