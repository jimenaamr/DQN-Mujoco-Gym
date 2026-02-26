# src/dqn/env.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo, TimeLimit

from src.dqn.helper import HeadStabilizerWrapper, StabilizerConfig
from src.dqn.reward import walker2d_default_reward

# CROPS = (top, bottom, left, right)  # fractions of each border to crop (e.g., 0.10 = 10%)
CROPS: tuple[float, float, float, float] = (0.25, 0.05, 0.1, 0.1)


# ------------------------- MuJoCo position/velocity retrieval -------------------------


def _mujoco_model_data(env: gym.Env) -> tuple[object, object] | None:
    """Return (model, data) from unwrapped env if MuJoCo-backed, else None."""
    base: gym.Env = env.unwrapped
    model: object | None = getattr(base, "model", None)
    data: object | None = getattr(base, "data", None)
    if model is None or data is None:
        return None
    return (model, data)


def _mujoco_id(model: object, name: str, kind: str) -> int | None:
    """Resolve a MuJoCo object id (body/geom/site/joint) by name, or None."""
    try:
        if kind == "body":
            return int(model.body(name).id)  # type: ignore[misc]
        if kind == "geom":
            return int(model.geom(name).id)  # type: ignore[misc]
        if kind == "site":
            return int(model.site(name).id)  # type: ignore[misc]
        if kind == "joint":
            return int(model.joint(name).id)  # type: ignore[misc]
        return None
    except Exception:
        name2id = getattr(model, "name2id", None)
        if name2id is None:
            return None
        try:
            return int(name2id(name, kind))
        except Exception:
            return None


def _first_id(model: object, names: Sequence[str], kind: str) -> int | None:
    """Return first resolvable id for the given MuJoCo object kind."""
    for n in names:
        i: int | None = _mujoco_id(model=model, name=str(n), kind=kind)
        if i is not None:
            return int(i)
    return None


def _pos3(arr: object, idx: int) -> tuple[float, float, float] | None:
    """Return (x,y,z) from arr[idx], or None if unavailable."""
    try:
        p = arr[idx]
        return (float(p[0]), float(p[1]), float(p[2]))
    except Exception:
        return None


def _body_x(data: object, body_id: int) -> float | None:
    """Return body x-position, or None if unavailable."""
    try:
        return float(data.xpos[body_id][0])
    except Exception:
        return None


def _obj_speed2d(model: object, data: object, kind: str, obj_id: int) -> float | None:
    """Return planar speed for MuJoCo body/geom, or None if not available."""
    try:
        import mujoco  # type: ignore

        vel6: np.ndarray = np.zeros(shape=(6,), dtype=np.float64)
        mjtobj = (
            mujoco.mjtObj.mjOBJ_BODY  # type: ignore[attr-defined]
            if kind == "body"
            else mujoco.mjtObj.mjOBJ_GEOM  # type: ignore[attr-defined]
        )
        mujoco.mj_objectVelocity(  # type: ignore[attr-defined]
            model,
            data,
            mjtobj,
            int(obj_id),
            vel6,
            0,
        )
        vx: float = float(vel6[0])
        vy: float = float(vel6[1])
        return float(sqrt(vx * vx + vy * vy))
    except Exception:
        pass

    try:
        if kind == "body":
            v = data.xvelp[obj_id]
        else:
            v = data.geom_xvelp[obj_id]
        vx_f: float = float(v[0])
        vy_f: float = float(v[1])
        return float(sqrt(vx_f * vx_f + vy_f * vy_f))
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


def _sanitize_name(name: str) -> str:
    """Make MuJoCo names safe for info keys."""
    return str(name).strip().replace(" ", "_")


def _joint_names(model: object) -> list[str]:
    """Return a list of joint names from the MuJoCo model (best-effort)."""
    names: list[str] = []
    try:
        njnt: int = int(model.njnt)
    except Exception:
        return names

    try:
        joint_fn = model.joint
        for j in range(njnt):
            try:
                nm: str = str(joint_fn(j).name)
                if nm:
                    names.append(_sanitize_name(name=nm))
            except Exception:
                continue
        if names:
            return names
    except Exception:
        pass

    try:
        import mujoco  # type: ignore

        for j in range(njnt):
            try:
                nm2: str | None = mujoco.mj_id2name(  # type: ignore[attr-defined]
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,  # type: ignore[attr-defined]
                    int(j),
                )
                if nm2:
                    names.append(_sanitize_name(name=str(nm2)))
            except Exception:
                continue
    except Exception:
        pass

    return names


# ------------------------- Terminal points via geom extrema -------------------------


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

    # If we can't reason about shape, use center only.
    if gsize is None or gtype is None:
        return [c]

    # Try to map mujoco enums; if unavailable, fall back to heuristic by value.
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

    # Capsule / cylinder: endpoints along local z axis.
    if gtype == mj_capsule or gtype == mj_cyl:
        axis_z: np.ndarray | None = _geom_rot_z_axis(data=data, geom_id=geom_id)
        if axis_z is None:
            return [c]

        radius: float = float(gsize[0])
        half_len: float = float(gsize[1])

        # Capsule extends by radius beyond the cylinder half-length.
        half_extent: float = float(half_len + (radius if gtype == mj_capsule else 0.0))
        d: np.ndarray = axis_z * float(half_extent)
        return [c - d, c + d]

    # Box: sample 8 corners (local frame) -> world via xmat.
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

    # Sphere: sample axis points in world frame.
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


def _terminal_points(
    model: object,
    data: object,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return (head_xyz, tiptoe1_xyz, tiptoe2_xyz) best-effort from geom extrema."""
    # Head: max z over all non-plane geoms.
    head_candidates: list[np.ndarray] = []
    try:
        ngeom: int = int(model.ngeom)
    except Exception:
        ngeom = 0

    for gid in range(ngeom):
        head_candidates.extend(
            _geom_extreme_points(model=model, data=data, geom_id=gid)
        )
    head_xyz: np.ndarray | None = _argmax_point(points=head_candidates, axis=2)

    # Tiptoes: max-x endpoint per foot geom (if names exist).
    foot1_geom_candidates: tuple[str, ...] = (
        "foot",
        "right_foot",
        "foot_right",
        "foot_geom",
        "foot_1",
    )
    foot2_geom_candidates: tuple[str, ...] = (
        "foot_left",
        "left_foot",
        "foot_l",
        "foot_left_geom",
        "foot_2",
    )

    foot1_id: int | None = _first_id(
        model=model, names=foot1_geom_candidates, kind="geom"
    )
    foot2_id: int | None = _first_id(
        model=model, names=foot2_geom_candidates, kind="geom"
    )

    tip1_xyz: np.ndarray | None = None
    tip2_xyz: np.ndarray | None = None

    if foot1_id is not None:
        pts1: list[np.ndarray] = _geom_extreme_points(
            model=model, data=data, geom_id=foot1_id
        )
        tip1_xyz = _argmax_point(points=pts1, axis=0)

    if foot2_id is not None:
        pts2: list[np.ndarray] = _geom_extreme_points(
            model=model, data=data, geom_id=foot2_id
        )
        tip2_xyz = _argmax_point(points=pts2, axis=0)

    return (head_xyz, tip1_xyz, tip2_xyz)


def _update_xy_fields(
    info: dict,
    env: gym.Env,
    prev_xy: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Populate info with <name>_x, <name>_y, <name>_dx, <name>_dy for all joints.

    Additionally populates terminal points:
      - head_x/head_y/head_z and head_dx/head_dy
      - tiptoes_x/tiptoes_y/tiptoes_z and tiptoes_dx/tiptoes_dy
      - tiptoes_left_x/tiptoes_left_y/tiptoes_left_z and corresponding d*
    """
    md: tuple[object, object] | None = _mujoco_model_data(env=env)
    if md is None:
        return prev_xy
    model, data = md

    dt: float | None = _effective_dt(env=env, info=info)
    new_prev: dict[str, tuple[float, float]] = dict(prev_xy)

    # Joints: use data.xanchor (world joint anchor positions).
    joint_names: list[str] = _joint_names(model=model)
    try:
        xanchor = data.xanchor
    except Exception:
        xanchor = None

    if xanchor is not None:
        for j, jname in enumerate(joint_names):
            p: tuple[float, float, float] | None = _pos3(arr=xanchor, idx=int(j))
            if p is None:
                continue
            x: float = float(p[0])
            y: float = float(p[1])
            info[f"{jname}_x"] = x
            info[f"{jname}_y"] = y

            if dt is not None:
                prev: tuple[float, float] | None = prev_xy.get(jname)
                if prev is not None:
                    info[f"{jname}_dx"] = float((x - float(prev[0])) / dt)
                    info[f"{jname}_dy"] = float((y - float(prev[1])) / dt)
                else:
                    info[f"{jname}_dx"] = 0.0
                    info[f"{jname}_dy"] = 0.0

            new_prev[jname] = (x, y)

    # Terminals: geom extrema (robust; does not rely on specific names existing).
    head_xyz, tip1_xyz, tip2_xyz = _terminal_points(model=model, data=data)

    def _put_term(name: str, xyz: np.ndarray | None) -> None:
        if xyz is None:
            return
        x: float = float(xyz[0])
        y: float = float(xyz[1])
        z: float = float(xyz[2])
        info[f"{name}_x"] = x
        info[f"{name}_y"] = y
        info[f"{name}_z"] = z

        if dt is not None:
            prev: tuple[float, float] | None = prev_xy.get(name)
            if prev is not None:
                info[f"{name}_dx"] = float((x - float(prev[0])) / dt)
                info[f"{name}_dy"] = float((y - float(prev[1])) / dt)
            else:
                info[f"{name}_dx"] = 0.0
                info[f"{name}_dy"] = 0.0

        new_prev[name] = (x, y)

    _put_term(name="head", xyz=head_xyz)

    # Keep a stable convention: "tiptoes" = forward-most of the two feet by x.
    if tip1_xyz is not None and tip2_xyz is not None:
        if float(tip1_xyz[0]) >= float(tip2_xyz[0]):
            _put_term(name="tiptoes", xyz=tip1_xyz)
            _put_term(name="tiptoes_left", xyz=tip2_xyz)
        else:
            _put_term(name="tiptoes", xyz=tip2_xyz)
            _put_term(name="tiptoes_left", xyz=tip1_xyz)
    else:
        _put_term(name="tiptoes", xyz=tip1_xyz)
        _put_term(name="tiptoes_left", xyz=tip2_xyz)

    return new_prev


def _mujoco_info_update(info: dict, env: gym.Env) -> None:
    """Populate info with MuJoCo-derived positions/velocities when available."""
    md: tuple[object, object] | None = _mujoco_model_data(env=env)
    if md is None:
        return
    model, data = md

    torso_names: tuple[str, ...] = ("torso", "hips", "pelvis")
    torso_body_id: int | None = _first_id(model=model, names=torso_names, kind="body")

    if torso_body_id is not None:
        x_hips: float | None = _body_x(data=data, body_id=torso_body_id)
        if x_hips is not None:
            info["x_hips"] = float(x_hips)

        torso_speed: float | None = _obj_speed2d(
            model=model, data=data, kind="body", obj_id=torso_body_id
        )
        if torso_speed is not None:
            info["torso_speed"] = float(torso_speed)

        torso_pos: tuple[float, float, float] | None = _pos3(
            arr=data.xpos,
            idx=torso_body_id,
        )
        if torso_pos is not None:
            info["z_torso"] = float(torso_pos[2])

    if "z_torso" not in info:
        torso_geom_candidates: tuple[str, ...] = (
            "torso",
            "torso_geom",
            "body",
            "trunk",
        )
        torso_geom_id: int | None = _first_id(
            model=model, names=torso_geom_candidates, kind="geom"
        )
        if torso_geom_id is not None:
            gpos: tuple[float, float, float] | None = _pos3(
                arr=data.geom_xpos,
                idx=torso_geom_id,
            )
            if gpos is not None:
                info["z_torso"] = float(gpos[2])

    foot1_geom_candidates: tuple[str, ...] = (
        "foot",
        "right_foot",
        "foot_right",
        "foot_geom",
        "foot_1",
    )
    foot2_geom_candidates: tuple[str, ...] = (
        "foot_left",
        "left_foot",
        "foot_l",
        "foot_left_geom",
        "foot_2",
    )

    foot1_id: int | None = _first_id(
        model=model, names=foot1_geom_candidates, kind="geom"
    )
    foot2_id: int | None = _first_id(
        model=model, names=foot2_geom_candidates, kind="geom"
    )

    foot1_pos: tuple[float, float, float] | None = None
    foot2_pos: tuple[float, float, float] | None = None

    if foot1_id is not None:
        foot1_pos = _pos3(arr=data.geom_xpos, idx=foot1_id)
        s1: float | None = _obj_speed2d(
            model=model, data=data, kind="geom", obj_id=foot1_id
        )
        if s1 is not None:
            info["foot1_speed2d"] = float(s1)

    if foot2_id is not None:
        foot2_pos = _pos3(arr=data.geom_xpos, idx=foot2_id)
        s2: float | None = _obj_speed2d(
            model=model, data=data, kind="geom", obj_id=foot2_id
        )
        if s2 is not None:
            info["foot2_speed2d"] = float(s2)

    if foot1_pos is not None:
        info["x_foot1"] = float(foot1_pos[0])
        info["y_foot1"] = float(foot1_pos[1])
        info["z_foot1"] = float(foot1_pos[2])

    if foot2_pos is not None:
        info["x_foot2"] = float(foot2_pos[0])
        info["y_foot2"] = float(foot2_pos[1])
        info["z_foot2"] = float(foot2_pos[2])

    if foot1_pos is not None and foot2_pos is not None:
        info["lowest_foot_height"] = float(min(foot1_pos[2], foot2_pos[2]))
        dx: float = float(foot1_pos[0] - foot2_pos[0])
        dy: float = float(foot1_pos[1] - foot2_pos[1])
        info["feet_dist_2d"] = float(sqrt(dx * dx + dy * dy))
    elif foot1_pos is not None:
        info["lowest_foot_height"] = float(foot1_pos[2])
    elif foot2_pos is not None:
        info["lowest_foot_height"] = float(foot2_pos[2])


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

        self._prev_xy: dict[str, tuple[float, float]] = {}

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
        self._prev_xy = {}
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        _obs, reward, terminated, truncated, info = self.env.step(action=action)

        info = dict(info)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)

        if "TimeLimit.truncated" in info:
            info["time_limit_truncated"] = bool(info["TimeLimit.truncated"])

        _mujoco_info_update(info=info, env=self.env)
        self._prev_xy = _update_xy_fields(
            info=info, env=self.env, prev_xy=self._prev_xy
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
