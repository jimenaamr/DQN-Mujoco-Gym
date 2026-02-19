# # src/DQN_walker2d/env.py

# from __future__ import annotations

# from collections.abc import Callable
# from dataclasses import dataclass

# import cv2
# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces
# from gymnasium.wrappers import RecordVideo, TimeLimit

# from src.DQN_walker2d.helper import HeadStabilizerWrapper, StabilizerConfig
# from src.DQN_walker2d.reward import walker2d_default_reward


# @dataclass
# class EnvSpec:
#     env_id: str
#     frame_stack: int
#     action_repeat: int
#     time_limit: int
#     action_prototypes: list[list[float]]
#     obs_h: int
#     obs_w: int


# class PixelObservationWrapper(gym.Wrapper):
#     """Replace vector observation with rendered RGB image."""

#     def __init__(self, env: gym.Env, height: int = 84, width: int = 84) -> None:
#         super().__init__(env=env)
#         self.height: int = int(height)
#         self.width: int = int(width)

#         self.observation_space: spaces.Box = spaces.Box(
#             low=0,
#             high=255,
#             shape=(3, self.height, self.width),
#             dtype=np.uint8,
#         )

#     def _get_obs(self) -> np.ndarray:
#         frame: np.ndarray = self.env.render()
#         frame = cv2.resize(
#             src=frame,
#             dsize=(self.width, self.height),
#             interpolation=cv2.INTER_AREA,
#         )
#         frame = np.transpose(a=frame, axes=(2, 0, 1))
#         return frame.astype(dtype=np.uint8)

#     def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
#         _obs, info = self.env.reset(**kwargs)
#         return self._get_obs(), info

#     def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
#         _obs, reward, terminated, truncated, info = self.env.step(action=action)

#         info = dict(info)
#         info["terminated"] = bool(terminated)
#         info["truncated"] = bool(truncated)

#         if "TimeLimit.truncated" in info:
#             info["time_limit_truncated"] = bool(info["TimeLimit.truncated"])

#         return self._get_obs(), float(reward), bool(terminated), bool(truncated), info


# class FrameStack(gym.Wrapper):
#     def __init__(self, env: gym.Env, k: int) -> None:
#         super().__init__(env=env)
#         self.k: int = int(k)
#         self.frames: list[np.ndarray] | None = None

#         c, h, w = env.observation_space.shape
#         self.observation_space: spaces.Box = spaces.Box(
#             low=0,
#             high=255,
#             shape=(int(c) * self.k, int(h), int(w)),
#             dtype=np.uint8,
#         )

#     def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
#         obs, info = self.env.reset(**kwargs)
#         self.frames = [obs] * self.k
#         return self._get_obs(), info

#     def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
#         obs, reward, terminated, truncated, info = self.env.step(action=action)
#         assert self.frames is not None
#         self.frames.pop(0)
#         self.frames.append(obs)
#         return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

#     def _get_obs(self) -> np.ndarray:
#         assert self.frames is not None
#         return np.concatenate(self.frames, axis=0)


# class ActionRepeat(gym.Wrapper):
#     def __init__(self, env: gym.Env, repeat: int) -> None:
#         super().__init__(env=env)
#         self.repeat: int = int(repeat)

#     def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
#         total_reward: float = 0.0
#         terminated: bool = False
#         truncated: bool = False
#         info: dict = {}
#         obs: np.ndarray | None = None

#         for _ in range(self.repeat):
#             obs, r, term, trunc, info = self.env.step(action=action)
#             total_reward += float(r)
#             terminated = terminated or bool(term)
#             truncated = truncated or bool(trunc)
#             if terminated or truncated:
#                 break

#         assert obs is not None
#         info = dict(info)
#         info["action_repeat"] = int(self.repeat)
#         return obs, float(total_reward), bool(terminated), bool(truncated), info


# RewardFn = Callable[
#     [np.ndarray, int, np.ndarray, bool, bool, dict, float, gym.Env],
#     float,
# ]


# class DiscreteActionWrapper(gym.Wrapper):
#     """Map discrete actions to continuous prototypes and optionally override reward."""

#     def __init__(
#         self,
#         env: gym.Env,
#         prototypes: np.ndarray,
#         reward_fn: RewardFn | None = None,
#     ) -> None:
#         super().__init__(env=env)

#         self.prototypes: np.ndarray = prototypes.astype(dtype=np.float32)
#         self.action_space: spaces.Discrete = spaces.Discrete(n=len(self.prototypes))

#         self._reward_fn: RewardFn | None = reward_fn
#         self._last_obs: np.ndarray | None = None

#     def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
#         obs, info = self.env.reset(**kwargs)
#         self._last_obs = obs
#         return obs, info

#     def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
#         if self._last_obs is None:
#             raise RuntimeError("reset() must be called before step().")

#         cont_action: np.ndarray = self.prototypes[int(action)]

#         next_obs, env_reward, terminated, truncated, info = self.env.step(
#             action=cont_action
#         )

#         if self._reward_fn is None:
#             self._last_obs = next_obs
#             return next_obs, float(env_reward), bool(terminated), bool(truncated), info

#         new_reward: float = float(
#             self._reward_fn(
#                 self._last_obs,
#                 int(action),
#                 next_obs,
#                 bool(terminated),
#                 bool(truncated),
#                 info,
#                 float(env_reward),
#                 self.env,
#             )
#         )

#         self._last_obs = next_obs
#         return next_obs, float(new_reward), bool(terminated), bool(truncated), info


# def make_env(
#     spec: EnvSpec, seed: int, stabilizer: StabilizerConfig | None = None
# ) -> gym.Env:
#     env: gym.Env = gym.make(id=spec.env_id, render_mode="rgb_array")

#     env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))
#     env.reset(seed=int(seed))

#     if spec.action_repeat > 1:
#         env = ActionRepeat(env=env, repeat=spec.action_repeat)

#     env = PixelObservationWrapper(
#         env=env, height=int(spec.obs_h), width=int(spec.obs_w)
#     )

#     prototypes: np.ndarray = np.array(object=spec.action_prototypes, dtype=np.float32)
#     cont_dim: int = int(env.action_space.shape[0])
#     if prototypes.shape[1] != cont_dim:
#         raise ValueError(
#             "action_prototypes dim mismatch: "
#             f"got {prototypes.shape[1]} but env action dim is {cont_dim}"
#         )

#     env = DiscreteActionWrapper(
#         env=env,
#         prototypes=prototypes,
#         reward_fn=walker2d_default_reward,
#     )

#     if stabilizer is not None:
#         env = HeadStabilizerWrapper(env=env, cfg=stabilizer)

#     if spec.frame_stack > 1:
#         env = FrameStack(env=env, k=spec.frame_stack)

#     return env


# def make_eval_env(spec: EnvSpec, seed: int, video_dir: str) -> gym.Env:
#     env: gym.Env = gym.make(id=spec.env_id, render_mode="rgb_array")
#     env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))
#     env.reset(seed=int(seed))

#     env = RecordVideo(
#         env=env,
#         video_folder=video_dir,
#         episode_trigger=lambda ep: True,
#         name_prefix="eval",
#         disable_logger=True,
#     )

#     env = PixelObservationWrapper(
#         env=env, height=int(spec.obs_h), width=int(spec.obs_w)
#     )

#     prototypes: np.ndarray = np.array(object=spec.action_prototypes, dtype=np.float32)
#     env = DiscreteActionWrapper(
#         env=env,
#         prototypes=prototypes,
#         reward_fn=walker2d_default_reward,
#     )

#     if spec.frame_stack > 1:
#         env = FrameStack(env=env, k=spec.frame_stack)

#     return env


# src/DQN_walker2d/env.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, TimeLimit

from src.DQN_walker2d.helper import HeadStabilizerWrapper, StabilizerConfig
from src.DQN_walker2d.reward import walker2d_default_reward

# CROPS = (top, bottom, left, right)  # fractions of each border to crop (e.g., 0.10 = 10%)
CROPS: tuple[float, float, float, float] = (0.25, 0.05, 0.1, 0.1)


@dataclass
class EnvSpec:
    env_id: str
    frame_stack: int
    action_repeat: int
    time_limit: int
    action_prototypes: list[list[float]]
    obs_h: int
    obs_w: int


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

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop borders from an HxWxC RGB frame using fractional crop ratios.

        Args:
            frame: Rendered RGB frame with shape (H, W, C).

        Returns:
            Cropped RGB frame with shape (H', W', C).
        """
        top_f: float
        bottom_f: float
        left_f: float
        right_f: float
        top_f, bottom_f, left_f, right_f = CROPS

        if not (0.0 <= top_f < 1.0 and 0.0 <= bottom_f < 1.0):
            raise ValueError(f"Invalid vertical CROPS={CROPS}; must be in [0,1).")
        if not (0.0 <= left_f < 1.0 and 0.0 <= right_f < 1.0):
            raise ValueError(f"Invalid horizontal CROPS={CROPS}; must be in [0,1).")

        h: int = int(frame.shape[0])
        w: int = int(frame.shape[1])

        top_px: int = int(round(float(h) * float(top_f)))
        bottom_px: int = int(round(float(h) * float(bottom_f)))
        left_px: int = int(round(float(w) * float(left_f)))
        right_px: int = int(round(float(w) * float(right_f)))

        y0: int = max(0, top_px)
        y1: int = min(h, h - max(0, bottom_px))
        x0: int = max(0, left_px)
        x1: int = min(w, w - max(0, right_px))

        if y1 <= y0 or x1 <= x0:
            raise ValueError(
                f"Invalid CROPS={CROPS} for frame shape (H,W)=({h},{w}) "
                f"-> crop px (top,bottom,left,right)=({top_px},{bottom_px},"
                f"{left_px},{right_px})"
            )

        return frame[y0:y1, x0:x1, :]

    def _get_obs(self) -> np.ndarray:
        frame: np.ndarray = self.env.render()
        frame = self._crop_frame(frame=frame)
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
        _obs, reward, terminated, truncated, info = self.env.step(action=action)

        info = dict(info)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)

        if "TimeLimit.truncated" in info:
            info["time_limit_truncated"] = bool(info["TimeLimit.truncated"])

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int) -> None:
        super().__init__(env=env)
        self.k: int = int(k)
        self.frames: list[np.ndarray] | None = None

        c, h, w = env.observation_space.shape
        self.observation_space: spaces.Box = spaces.Box(
            low=0,
            high=255,
            shape=(int(c) * self.k, int(h), int(w)),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
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
            obs, r, term, trunc, info = self.env.step(action=action)
            total_reward += float(r)
            terminated = terminated or bool(term)
            truncated = truncated or bool(trunc)
            if terminated or truncated:
                break

        assert obs is not None
        info = dict(info)
        info["action_repeat"] = int(self.repeat)
        return obs, float(total_reward), bool(terminated), bool(truncated), info


RewardFn = Callable[
    [np.ndarray, int, np.ndarray, bool, bool, dict, float, gym.Env],
    float,
]


class DiscreteActionWrapper(gym.Wrapper):
    """Map discrete actions to continuous prototypes and optionally override reward."""

    def __init__(
        self,
        env: gym.Env,
        prototypes: np.ndarray,
        reward_fn: RewardFn | None = None,
    ) -> None:
        super().__init__(env=env)

        self.prototypes: np.ndarray = prototypes.astype(dtype=np.float32)
        self.action_space: spaces.Discrete = spaces.Discrete(n=len(self.prototypes))

        self._reward_fn: RewardFn | None = reward_fn
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before step().")

        cont_action: np.ndarray = self.prototypes[int(action)]

        next_obs, env_reward, terminated, truncated, info = self.env.step(
            action=cont_action
        )

        if self._reward_fn is None:
            self._last_obs = next_obs
            return next_obs, float(env_reward), bool(terminated), bool(truncated), info

        new_reward: float = float(
            self._reward_fn(
                self._last_obs,
                int(action),
                next_obs,
                bool(terminated),
                bool(truncated),
                info,
                float(env_reward),
                self.env,
            )
        )

        self._last_obs = next_obs
        return next_obs, float(new_reward), bool(terminated), bool(truncated), info


def make_env(
    spec: EnvSpec, seed: int, stabilizer: StabilizerConfig | None = None
) -> gym.Env:
    env: gym.Env = gym.make(id=spec.env_id, render_mode="rgb_array")

    env = TimeLimit(env=env, max_episode_steps=int(spec.time_limit))
    env.reset(seed=int(seed))

    if spec.action_repeat > 1:
        env = ActionRepeat(env=env, repeat=spec.action_repeat)

    env = PixelObservationWrapper(
        env=env, height=int(spec.obs_h), width=int(spec.obs_w)
    )

    prototypes: np.ndarray = np.array(object=spec.action_prototypes, dtype=np.float32)
    cont_dim: int = int(env.action_space.shape[0])
    if prototypes.shape[1] != cont_dim:
        raise ValueError(
            "action_prototypes dim mismatch: "
            f"got {prototypes.shape[1]} but env action dim is {cont_dim}"
        )

    env = DiscreteActionWrapper(
        env=env,
        prototypes=prototypes,
        reward_fn=walker2d_default_reward,
    )

    if stabilizer is not None:
        env = HeadStabilizerWrapper(env=env, cfg=stabilizer)

    if spec.frame_stack > 1:
        env = FrameStack(env=env, k=spec.frame_stack)

    return env


def make_eval_env(spec: EnvSpec, seed: int, video_dir: str) -> gym.Env:
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

    env = PixelObservationWrapper(
        env=env, height=int(spec.obs_h), width=int(spec.obs_w)
    )

    prototypes: np.ndarray = np.array(object=spec.action_prototypes, dtype=np.float32)
    env = DiscreteActionWrapper(
        env=env,
        prototypes=prototypes,
        reward_fn=walker2d_default_reward,
    )

    if spec.frame_stack > 1:
        env = FrameStack(env=env, k=spec.frame_stack)

    return env
