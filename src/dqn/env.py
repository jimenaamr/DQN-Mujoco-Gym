# src/dqn/env.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, TimeLimit

from src.dqn.helper import HeadStabilizerWrapper, StabilizerConfig
from src.dqn.reward import walker2d_default_reward

# CROPS = (top, bottom, left, right)  # fractions of each border to crop (e.g., 0.10 = 10%)
CROPS: tuple[float, float, float, float] = (0.25, 0.05, 0.1, 0.1)


# ------------------------- MuJoCo model/data helpers -------------------------


def _mujoco_model_data(env: gym.Env) -> tuple[object, object] | None:
    """Return (model, data) from unwrapped env if MuJoCo-backed, else None."""
    base: gym.Env = env.unwrapped
    model: object | None = getattr(base, "model", None)
    data: object | None = getattr(base, "data", None)
    if model is None or data is None:
        return None
    return (model, data)


def _mujoco_body_id(model: object, name: str) -> int | None:
    """Resolve a MuJoCo body id by name, or None."""
    try:
        return int(model.body(name).id)  # type: ignore[misc]
    except Exception:
        name2id = getattr(model, "name2id", None)
        if name2id is None:
            return None
        try:
            return int(name2id(name, "body"))
        except Exception:
            return None


def _body_xz(data: object, body_id: int) -> tuple[float, float] | None:
    """Return (x,z) for body in world coordinates, or None."""
    try:
        p = data.xpos[body_id]
        return (float(p[0]), float(p[2]))
    except Exception:
        return None


def _body_vxvz(model: object, data: object, body_id: int) -> tuple[float, float] | None:
    """Return (vx,vz) for body in world coordinates, or None."""
    try:
        import mujoco  # type: ignore

        vel6: np.ndarray = np.zeros(shape=(6,), dtype=np.float64)
        mujoco.mj_objectVelocity(  # type: ignore[attr-defined]
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,  # type: ignore[attr-defined]
            int(body_id),
            vel6,
            0,
        )
        return (float(vel6[0]), float(vel6[2]))
    except Exception:
        pass

    # Fallback if available.
    try:
        v = data.xvelp[body_id]
        return (float(v[0]), float(v[2]))
    except Exception:
        return None


def _effective_dt(env: gym.Env, info: dict) -> float | None:
    """Return effective dt for this wrapper step (accounts for ActionRepeat)."""
    md: tuple[object, object] | None = _mujoco_model_data(env=env)
    if md is None:
        return None
    model, _data = md
    try:
        base_dt: float = float(model.opt.timestep)
    except Exception:
        return None
    repeat: int = int(info.get("action_repeat", 1))
    return float(base_dt * float(repeat))


# ------------------------- Head estimation via geom extrema -------------------------


def _geom_rot_z_axis(data: object, geom_id: int) -> np.ndarray | None:
    """Return geom local z-axis in world coordinates (3,), or None."""
    try:
        xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        return xmat[:, 2].copy()
    except Exception:
        return None


def _geom_center(data: object, geom_id: int) -> np.ndarray | None:
    """Return geom center in world coordinates (3,), or None."""
    try:
        return np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy()
    except Exception:
        return None


def _geom_size(model: object, geom_id: int) -> np.ndarray | None:
    """Return geom size (3,), or None."""
    try:
        return np.asarray(model.geom_size[geom_id], dtype=np.float64).copy()
    except Exception:
        return None


def _geom_type(model: object, geom_id: int) -> int | None:
    """Return geom type int, or None."""
    try:
        return int(model.geom_type[geom_id])
    except Exception:
        return None


def _geom_extreme_points(model: object, data: object, geom_id: int) -> list[np.ndarray]:
    """Return a small set of world points spanning the geom (best-effort)."""
    c: np.ndarray | None = _geom_center(data=data, geom_id=geom_id)
    if c is None:
        return []

    gsize: np.ndarray | None = _geom_size(model=model, geom_id=geom_id)
    gtype: int | None = _geom_type(model=model, geom_id=geom_id)

    if gsize is None or gtype is None:
        return [c]

    try:
        import mujoco  # type: ignore

        mj_capsule: int = int(mujoco.mjtGeom.mjGEOM_CAPSULE)  # type: ignore[attr-defined]
        mj_cyl: int = int(mujoco.mjtGeom.mjGEOM_CYLINDER)  # type: ignore[attr-defined]
        mj_box: int = int(mujoco.mjtGeom.mjGEOM_BOX)  # type: ignore[attr-defined]
        mj_sphere: int = int(mujoco.mjtGeom.mjGEOM_SPHERE)  # type: ignore[attr-defined]
        mj_plane: int = int(mujoco.mjtGeom.mjGEOM_PLANE)  # type: ignore[attr-defined]
    except Exception:
        mj_capsule, mj_cyl, mj_box, mj_sphere, mj_plane = -1, -1, -1, -1, -1

    if gtype == mj_plane:
        return []

    if gtype == mj_capsule or gtype == mj_cyl:
        axis_z: np.ndarray | None = _geom_rot_z_axis(data=data, geom_id=geom_id)
        if axis_z is None:
            return [c]

        radius: float = float(gsize[0])
        half_len: float = float(gsize[1])
        half_extent: float = float(half_len + (radius if gtype == mj_capsule else 0.0))
        d: np.ndarray = axis_z * float(half_extent)
        return [c - d, c + d]

    if gtype == mj_box:
        try:
            xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        except Exception:
            return [c]

        sx: float = float(gsize[0])
        sy: float = float(gsize[1])
        sz: float = float(gsize[2])

        pts: list[np.ndarray] = []
        for dx in (-sx, sx):
            for dy in (-sy, sy):
                for dz in (-sz, sz):
                    local: np.ndarray = np.array([dx, dy, dz], dtype=np.float64)
                    pts.append(c + xmat @ local)
        return pts

    if gtype == mj_sphere:
        r: float = float(gsize[0])
        return [
            c + np.array([r, 0.0, 0.0], dtype=np.float64),
            c + np.array([-r, 0.0, 0.0], dtype=np.float64),
            c + np.array([0.0, r, 0.0], dtype=np.float64),
            c + np.array([0.0, -r, 0.0], dtype=np.float64),
            c + np.array([0.0, 0.0, r], dtype=np.float64),
            c + np.array([0.0, 0.0, -r], dtype=np.float64),
        ]

    return [c]


def _argmax_point(points: list[np.ndarray], axis: int) -> np.ndarray | None:
    """Return point with maximum coordinate on axis, or None."""
    if not points:
        return None
    best: np.ndarray = points[0]
    best_v: float = float(best[axis])
    for p in points[1:]:
        v: float = float(p[axis])
        if v > best_v:
            best = p
            best_v = v
    return best


def _argmin_point(points: list[np.ndarray], axis: int) -> np.ndarray | None:
    """Return point with minimum coordinate on axis, or None."""
    if not points:
        return None
    best: np.ndarray = points[0]
    best_v: float = float(best[axis])
    for p in points[1:]:
        v: float = float(p[axis])
        if v < best_v:
            best = p
            best_v = v
    return best


def _geom_body_id(model: object, geom_id: int) -> int | None:
    """Return the body id attached to a geom, or None."""
    try:
        return int(model.geom_bodyid[geom_id])
    except Exception:
        return None


def _body_geom_extreme_point(
    model: object,
    data: object,
    body_id: int,
    axis: int,
    maximize: bool,
) -> np.ndarray | None:
    """Return an extreme world point across all geoms attached to a body."""
    points: list[np.ndarray] = []

    try:
        ngeom: int = int(model.ngeom)
    except Exception:
        ngeom = 0

    for geom_id in range(ngeom):
        geom_body_id: int | None = _geom_body_id(model=model, geom_id=geom_id)
        if geom_body_id != body_id:
            continue
        points.extend(_geom_extreme_points(model=model, data=data, geom_id=geom_id))

    if maximize:
        return _argmax_point(points=points, axis=axis)
    return _argmin_point(points=points, axis=axis)


def _heel_xy(model: object, data: object, body_name: str) -> tuple[float, float] | None:
    """Return heel (x,y) for a foot body, using the rearmost geom point.

    Convention:
      - x := MuJoCo x
      - y := MuJoCo z (vertical)

    For Walker2d, the heel is approximated as the point with minimum x among
    all extreme points of the geoms attached to the foot body.
    """
    body_id: int | None = _mujoco_body_id(model=model, name=body_name)
    if body_id is None:
        return None

    heel_xyz: np.ndarray | None = _body_geom_extreme_point(
        model=model,
        data=data,
        body_id=body_id,
        axis=0,
        maximize=False,
    )
    if heel_xyz is None:
        return None

    return (float(heel_xyz[0]), float(heel_xyz[2]))


def _head_xyz(model: object, data: object) -> np.ndarray | None:
    """Return head xyz as max-z point across all non-plane geoms (best-effort)."""
    head_candidates: list[np.ndarray] = []
    try:
        ngeom: int = int(model.ngeom)
    except Exception:
        ngeom = 0

    for gid in range(ngeom):
        head_candidates.extend(
            _geom_extreme_points(model=model, data=data, geom_id=gid)
        )
    return _argmax_point(points=head_candidates, axis=2)


# ------------------------- info shaping (ONLY allowed keys) -------------------------


_ALLOWED_DEFAULT_KEYS: tuple[str, ...] = (
    "x_position",
    "x_velocity",
    "reward_forward",
    "reward_ctrl",
    "reward_survive",
)


_BODY_KEY_MAP: dict[str, str] = {
    "torso": "torso",
    "thigh": "thigh_right",
    "leg": "leg_right",
    "foot": "foot_right",
    "thigh_left": "thigh_left",
    "leg_left": "leg_left",
    "foot_left": "foot_left",
}


def _build_info(
    raw_info: dict,
    env: gym.Env,
    terminated: bool,
    truncated: bool,
    prev_head_xy: tuple[float, float] | None,
    prev_body_xy: dict[str, tuple[float, float]],
) -> tuple[dict, tuple[float, float] | None, dict[str, tuple[float, float]]]:
    """Build a new info dict with ONLY the required fields.

    Coordinate convention:
      - x := MuJoCo x
      - y := MuJoCo z (vertical)

    Velocities:
      - dx, dy computed by finite differences using effective dt.
    """
    info: dict = {}

    for k in _ALLOWED_DEFAULT_KEYS:
        if k in raw_info:
            info[k] = raw_info[k]

    if "z_distance_from_origin" in raw_info:
        info["y_distance_from_origin"] = raw_info["z_distance_from_origin"]
    elif "y_distance_from_origin" in raw_info:
        info["y_distance_from_origin"] = raw_info["y_distance_from_origin"]

    info["terminated"] = bool(terminated)
    info["truncated"] = bool(truncated)

    md: tuple[object, object] | None = _mujoco_model_data(env=env)
    if md is None:
        return info, prev_head_xy, prev_body_xy

    model, data = md
    dt: float | None = _effective_dt(env=env, info=raw_info)

    new_prev_body_xy: dict[str, tuple[float, float]] = dict(prev_body_xy)

    # Bodies: positions and velocities via finite difference
    for mujoco_name, out_name in _BODY_KEY_MAP.items():
        bid: int | None = _mujoco_body_id(model=model, name=mujoco_name)
        if bid is None:
            continue

        xz: tuple[float, float] | None = _body_xz(data=data, body_id=bid)
        if xz is None:
            continue

        x: float = float(xz[0])
        y: float = float(xz[1])  # y := z

        info[f"{out_name}_x"] = x
        info[f"{out_name}_y"] = y

        if dt is not None:
            prev_xy: tuple[float, float] | None = prev_body_xy.get(out_name)
            if prev_xy is not None:
                info[f"{out_name}_dx"] = float((x - float(prev_xy[0])) / dt)
                info[f"{out_name}_dy"] = float((y - float(prev_xy[1])) / dt)
            else:
                info[f"{out_name}_dx"] = 0.0
                info[f"{out_name}_dy"] = 0.0
        else:
            info[f"{out_name}_dx"] = 0.0
            info[f"{out_name}_dy"] = 0.0

        new_prev_body_xy[out_name] = (x, y)

        # Make sure x_velocity is always consistent with torso_dx.
        if out_name == "torso":
            info["x_velocity"] = float(info.get("torso_dx", 0.0))

    # Heels: use rearmost geom point of each foot body instead of body origin.
    for foot_body_name, heel_prefix in (
        ("foot", "heel_right"),
        ("foot_left", "heel_left"),
    ):
        heel_xy: tuple[float, float] | None = _heel_xy(
            model=model,
            data=data,
            body_name=foot_body_name,
        )
        if heel_xy is None:
            continue

        info[f"{heel_prefix}_x"] = float(heel_xy[0])
        info[f"{heel_prefix}_y"] = float(heel_xy[1])

    # Head: position from geom extrema; velocity by finite differences
    head_xyz: np.ndarray | None = _head_xyz(model=model, data=data)
    if head_xyz is not None:
        head_x: float = float(head_xyz[0])
        head_y: float = float(head_xyz[2])  # y := z

        info["head_x"] = head_x
        info["head_y"] = head_y

        if dt is not None and prev_head_xy is not None:
            info["head_dx"] = float((head_x - float(prev_head_xy[0])) / dt)
            info["head_dy"] = float((head_y - float(prev_head_xy[1])) / dt)
        else:
            info["head_dx"] = 0.0
            info["head_dy"] = 0.0

        prev_head_xy = (head_x, head_y)

    return info, prev_head_xy, new_prev_body_xy


# -------------------------------------- Env code --------------------------------------


@dataclass
class EnvSpec:
    env_id: str
    frame_stack: int
    action_repeat: int
    time_limit: int
    action_prototypes: list[list[float]]
    obs_h: int
    obs_w: int
    grayscale: bool = False


class PixelObservationWrapper(gym.Wrapper):
    """Replace vector observation with rendered image (RGB or grayscale)."""

    def __init__(
        self,
        env: gym.Env,
        height: int = 84,
        width: int = 84,
        grayscale: bool = False,
    ) -> None:
        super().__init__(env=env)
        self.height: int = int(height)
        self.width: int = int(width)
        self.grayscale: bool = bool(grayscale)

        c_out: int = 1 if self.grayscale else 3
        self.observation_space: spaces.Box = spaces.Box(
            low=0,
            high=255,
            shape=(c_out, self.height, self.width),
            dtype=np.uint8,
        )

        self._prev_head_xy: tuple[float, float] | None = None
        self._prev_body_xy: dict[str, tuple[float, float]] = {}

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
        frame: np.ndarray = self.env.render()  # (H,W,3) RGB
        frame = self._crop_frame(frame=frame)
        frame = cv2.resize(
            src=frame,
            dsize=(self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

        if self.grayscale:
            gray: np.ndarray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
            gray_chw: np.ndarray = gray[None, :, :]
            return gray_chw.astype(dtype=np.uint8)

        frame_chw: np.ndarray = np.transpose(a=frame, axes=(2, 0, 1))
        return frame_chw.astype(dtype=np.uint8)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        _obs, info = self.env.reset(**kwargs)
        self._prev_head_xy = None
        self._prev_body_xy = {}
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        _obs, reward, terminated, truncated, raw_info = self.env.step(action=action)

        raw_info = dict(raw_info)
        raw_info["terminated"] = bool(terminated)
        raw_info["truncated"] = bool(truncated)

        if "TimeLimit.truncated" in raw_info:
            raw_info["time_limit_truncated"] = bool(raw_info["TimeLimit.truncated"])

        info, self._prev_head_xy, self._prev_body_xy = _build_info(
            raw_info=raw_info,
            env=self.env,
            terminated=bool(terminated),
            truncated=bool(truncated),
            prev_head_xy=self._prev_head_xy,
            prev_body_xy=self._prev_body_xy,
        )

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
        env=env,
        height=int(spec.obs_h),
        width=int(spec.obs_w),
        grayscale=bool(spec.grayscale),
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
        env=env,
        height=int(spec.obs_h),
        width=int(spec.obs_w),
        grayscale=bool(spec.grayscale),
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
