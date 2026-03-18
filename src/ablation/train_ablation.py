# src/ablation/train_ablation.py

from __future__ import annotations

import os

from src.utils.backends import auto_detect_mujoco_gl

# os.environ.setdefault(key="MUJOCO_GL", value="egl")  # DEFINE BEFORE IMPORTING GYMNASIUM

graphics_backend: str = auto_detect_mujoco_gl()
os.environ.setdefault(key="MUJOCO_GL", value=graphics_backend)

import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from argparse import Namespace
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.ablation.rdqn_ablation import RDQNAgent, RDQNConfig
from src.rdqn.env import EnvSpec, make_env
from src.rdqn.helper import (
    StabilizerConfig,
    resolve_device,
    stabilizer_config_from_yaml,
)
from src.rdqn.monitoring import MONITOR
from src.utils.tensorboard import launch_tensorboard

LIVE_CLEANUP_TIMER: float = 120.0
_STEP_RE: re.Pattern[str] = re.compile(pattern=r"^step_(\d+)_.*\.pt$")
_STEP_ANYWHERE_RE: re.Pattern[str] = re.compile(pattern=r"step_(\d+)_.*\.pt$")


def _ensure_rdqn_ablation_subdir(root: str) -> str:
    """Ensure a root directory is namespaced under 'rdqn-ablation'.

    Rules:
      - If 'rdqn-ablation' is already present, return unchanged.
      - Else if 'rdqn' is present as a path component, replace the last 'rdqn'
        component with 'rdqn-ablation'.
      - Else append 'rdqn-ablation'.

    Args:
        root: Base directory (e.g., "runs", "runs/rdqn", "checkpoints/rdqn").

    Returns:
        Directory under the rdqn-ablation namespace (e.g., "runs/rdqn-ablation").
    """
    p: Path = Path(root)

    parts: list[str] = list(p.parts)
    if "rdqn-ablation" in parts:
        return str(p)

    if "rdqn" in parts:
        idx: int = len(parts) - 1 - parts[::-1].index("rdqn")
        parts[idx] = "rdqn-ablation"
        return str(Path(*parts))

    return str(p / "rdqn-ablation")


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed YAML content.
    """
    with open(file=path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(stream=f)
    return data


def to_env_spec(cfg: dict[str, Any]) -> EnvSpec:
    """Convert the `env` section of the config into an EnvSpec.

    Args:
        cfg: Full experiment configuration.

    Returns:
        Parsed EnvSpec.
    """
    e: dict[str, Any] = cfg["env"]
    return EnvSpec(
        env_id=str(e["env_id"]),
        frame_stack=int(e["frame_stack"]),
        action_repeat=int(e["action_repeat"]),
        time_limit=int(e["time_limit"]),
        action_prototypes=e["action_prototypes"],
        obs_h=int(e.get("obs_h", 84)),
        obs_w=int(e.get("obs_w", 84)),
        grayscale=bool(e.get("grayscale")),
    )


def to_rdqn_cfg(cfg: dict[str, Any]) -> RDQNConfig:
    """Convert YAML config into RDQNConfig (including ablation toggles)."""

    t: dict[str, Any] = cfg["train"]
    r: dict[str, Any] = dict(cfg.get("rainbow", {}))

    device_name: str = str(cfg.get("device", "cpu"))
    device_resolved: str = resolve_device(name=device_name)

    use_per: bool = bool(r.get("use_per", True))
    use_noisy: bool = bool(r.get("use_noisy", True))

    return RDQNConfig(
        gamma=float(t["gamma"]),
        lr=float(t["lr"]),
        batch_size=int(t["batch_size"]),
        buffer_size=int(t["buffer_size"]),
        learning_starts=int(t["learning_starts"]),
        train_freq=int(t["train_freq"]),
        target_update_freq=int(t["target_update_freq"]),
        grad_clip_norm=float(t["grad_clip_norm"]),
        # Rainbow-specific
        noisy_sigma0=float(r.get("noisy_sigma0", 0.5)),
        n_step=int(r.get("n_step", 3)),
        prio_alpha=float(r.get("prio_alpha", 0.6)),
        prio_beta_start=float(r.get("prio_beta_start", 0.4)),
        prio_beta_end=float(r.get("prio_beta_end", 1.0)),
        prio_beta_steps=int(r.get("prio_beta_steps", 1_000_000)),
        prio_eps=float(r.get("prio_eps", 1.0e-6)),
        v_min=float(r.get("v_min", -10.0)),
        v_max=float(r.get("v_max", 10.0)),
        n_atoms=int(r.get("n_atoms", 51)),
        # Ablations
        use_per=bool(use_per),
        use_noisy=bool(use_noisy),
        device=str(device_resolved),
    )


def _extract_last_rgb_frame(obs: np.ndarray) -> np.ndarray:
    """Extract an RGB frame (H, W, 3) from an observation shaped like (C, H, W).

    For grayscale observations (C < 3), replicate the last channel into RGB.

    Args:
        obs: Observation array with shape (C, H, W) and C >= 1.

    Returns:
        RGB frame with shape (H, W, 3).

    Raises:
        ValueError: If obs does not have 3 dims or C < 1.
    """
    if obs.ndim != 3:
        raise ValueError(f"Expected obs with shape (C,H,W), got {obs.shape}")
    c: int = int(obs.shape[0])
    if c < 1:
        raise ValueError(f"Expected at least 1 channel, got C={c}")

    if c >= 3:
        frame_chw: np.ndarray = obs[c - 3 : c, :, :]
        frame_hwc: np.ndarray = np.transpose(frame_chw, (1, 2, 0))
        return frame_hwc

    gray_hw: np.ndarray = obs[c - 1, :, :]
    rgb_hwc: np.ndarray = np.stack((gray_hw, gray_hw, gray_hw), axis=-1)
    return rgb_hwc


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write a small text file.

    Args:
        path: Destination file path.
        text: Text content.
    """
    tmp: Path = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _ensure_dir(path: Path) -> None:
    """Create a directory if it doesn't exist.

    Args:
        path: Directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def _write_jpg_atomic(path: Path, frame_rgb: np.ndarray, quality: int = 90) -> None:
    """Write an RGB frame to JPG atomically.

    Args:
        path: Destination .jpg file.
        frame_rgb: RGB uint8 frame (H,W,3).
        quality: JPEG quality (0-100).
    """
    if frame_rgb.dtype != np.uint8:
        frame_rgb = frame_rgb.astype(np.uint8, copy=False)

    frame_bgr: np.ndarray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    ok: bool
    buf: np.ndarray
    ok, buf = cv2.imencode(
        ext=".jpg",
        img=frame_bgr,
        params=[int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        return

    tmp: Path = path.with_suffix(".jpg.tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)


def _write_metrics_snapshot_atomic(path: Path) -> None:
    """Write current MONITOR snapshot as a metrics_XXXXXX.txt atomically.

    Args:
        path: Destination path for metrics text.
    """
    text: str = MONITOR.to_text()
    tmp: Path = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _live_frames_enabled(cfg: dict[str, Any]) -> bool:
    """Decide whether filesystem live-frames export is enabled.

    Args:
        cfg: Full experiment config.

    Returns:
        True if enabled.
    """
    logging_cfg: dict[str, Any] = dict(cfg.get("logging", {}))
    if "live_frames" in logging_cfg:
        return bool(logging_cfg.get("live_frames"))
    return True


def _run_eval_in_subprocess(
    config_path: str,
    checkpoint_path: str,
    seed: int,
    episodes: int,
    video_dir: str,
) -> float:
    """Run evaluation in a separate process and return mean return.

    Args:
        config_path: YAML config path.
        checkpoint_path: Checkpoint to evaluate.
        seed: RNG seed for evaluation.
        episodes: Number of evaluation episodes.
        video_dir: Directory to store evaluation videos.

    Returns:
        Mean return over evaluation episodes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path: str = os.path.join(tmpdir, "eval_return.txt")

        cmd: list[str] = [
            "python",
            "-m",
            "src.rdqn.eval_worker",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--seed",
            str(int(seed)),
            "--episodes",
            str(int(episodes)),
            "--video-dir",
            str(video_dir),
            "--output-path",
            str(out_path),
        ]

        env: dict[str, str] = dict(os.environ)
        env["DQN_MONITOR_DISABLED"] = "1"
        subprocess.run(args=cmd, check=True, env=env)

        with open(file=out_path, encoding="utf-8") as f:
            raw: str = f.read().strip()
        return float(raw)


def _select_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Select the latest checkpoint inside a directory.

    Preference order:
        1) Largest step_XXXX_*.pt by XXXX.
        2) Most recently modified best_*.pt.

    Args:
        ckpt_dir: Directory containing checkpoint files.

    Returns:
        Path to selected checkpoint, or None if no checkpoints exist.
    """
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return None

    step_files: list[tuple[int, Path]] = []
    best_files: list[Path] = []

    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix != ".pt":
            continue

        m: re.Match[str] | None = _STEP_RE.match(p.name)
        if m is not None:
            step_files.append((int(m.group(1)), p))
        elif p.name.startswith("best_"):
            best_files.append(p)

    if step_files:
        step_files.sort(key=lambda t: t[0])
        return step_files[-1][1]

    if best_files:
        best_files.sort(key=lambda p: p.stat().st_mtime)
        return best_files[-1]

    return None


def resolve_resume_path(resume: str | None) -> Path | None:
    """Resolve a resume argument into a concrete checkpoint file path.

    Args:
        resume: Either a checkpoint file path, a directory path, or None.

    Returns:
        A checkpoint file path, or None if resume is None / invalid / empty.
    """
    if resume is None:
        return None

    p: Path = Path(resume).expanduser()
    if p.is_file():
        return p

    if p.is_dir():
        return _select_latest_checkpoint(ckpt_dir=p)

    return None


def _infer_run_name_from_checkpoint(ckpt_file: Path) -> str | None:
    """Infer run name from a checkpoint path.

    Expected layout:
        checkpoints/rdqn-ablation/<run_name>/<file>.pt

    Args:
        ckpt_file: Checkpoint file path.

    Returns:
        run_name if pattern matches, else None.
    """
    parent: Path = ckpt_file.parent
    if parent.name == "":
        return None
    return str(parent.name)


def _infer_step_from_checkpoint_filename(ckpt_file: Path) -> int | None:
    """Infer global step from a checkpoint filename like step_5000000_XX-YY-ZZ.pt.

    Args:
        ckpt_file: Checkpoint file.

    Returns:
        Parsed step integer, or None if pattern doesn't match.
    """
    m: re.Match[str] | None = _STEP_ANYWHERE_RE.search(ckpt_file.name)
    if m is None:
        return None
    return int(m.group(1))


def _is_listening(host: str, port: int) -> bool:
    """Check whether a TCP host:port has a listener.

    Args:
        host: Host to connect to.
        port: TCP port.

    Returns:
        True if connection succeeds, else False.
    """
    s: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.25)
        return s.connect_ex((host, int(port))) == 0
    finally:
        s.close()


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a subprocess (and its process group) best-effort.

    Args:
        proc: Process handle.
    """
    try:
        if proc.poll() is not None:
            return
        pgid: int = int(os.getpgid(proc.pid))
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=3.0)
    except Exception:
        try:
            pgid = int(os.getpgid(proc.pid))
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass


def _write_text_atomic(path: Path, text: str) -> None:
    """Write text to a file atomically.

    Args:
        path: Destination file path.
        text: Text to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _maybe_env_dt_s(env: Any) -> float | None:
    """Best-effort extract env dt (seconds per env.step())."""
    base: Any = getattr(env, "unwrapped", env)
    dt: Any = getattr(base, "dt", None)
    try:
        if dt is None:
            return None
        return float(dt)
    except Exception:
        return None


def main(
    config_path: str, resume_path: str | None = None, new_run: bool = False
) -> None:
    """Entry point for training."""
    cfg: dict[str, Any] = load_yaml(path=config_path)
    seed: int = int(cfg["seed"])

    env_spec: EnvSpec = to_env_spec(cfg=cfg)
    stabilizer_cfg: StabilizerConfig | None = stabilizer_config_from_yaml(cfg=cfg)

    train_env = make_env(spec=env_spec, seed=seed, stabilizer=stabilizer_cfg)
    live_ep_queue: deque[tuple[float, int, Path]] = deque()

    obs_shape_raw: tuple[int, ...] | None = train_env.observation_space.shape
    if obs_shape_raw is None or len(obs_shape_raw) != 3:
        raise ValueError(
            f"Expected pixel obs shape (C,H,W); got {obs_shape_raw}. "
            "If you want vector observations, you must also update the RDQN model."
        )
    obs_shape: tuple[int, int, int] = (
        int(obs_shape_raw[0]),
        int(obs_shape_raw[1]),
        int(obs_shape_raw[2]),
    )
    n_actions: int = int(train_env.action_space.n)

    rdqn_cfg: RDQNConfig = to_rdqn_cfg(cfg=cfg)
    agent: RDQNAgent = RDQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=rdqn_cfg)

    device_raw: str = str(cfg.get("device", "cpu"))
    print(
        f"[train] device config: {device_raw} -> resolved: {rdqn_cfg.device}",
        flush=True,
    )

    run_dir_raw: str = str(cfg["logging"]["run_dir"])
    ckpt_dir_raw: str = str(cfg["logging"]["ckpt_dir"])
    run_dir: str = _ensure_rdqn_ablation_subdir(root=run_dir_raw)
    ckpt_dir: str = _ensure_rdqn_ablation_subdir(root=ckpt_dir_raw)
    os.makedirs(name=run_dir, exist_ok=True)
    os.makedirs(name=ckpt_dir, exist_ok=True)

    resume_ckpt: Path | None = resolve_resume_path(resume=resume_path)

    if resume_ckpt is not None and (not new_run):
        inferred: str | None = _infer_run_name_from_checkpoint(ckpt_file=resume_ckpt)
        run_name: str = inferred or (
            f"{env_spec.env_id}_resume_"
            f"{datetime.now().strftime(format='%Y%m%d_%H%M%S')}"
        )
    else:
        run_name = (
            f"{env_spec.env_id}_{datetime.now().strftime(format='%Y%m%d_%H%M%S')}"
        )

    run_name = f"{run_name}_per{int(rdqn_cfg.use_per)}_noisy{int(rdqn_cfg.use_noisy)}"

    run_path: str = os.path.join(run_dir, run_name)
    ckpt_path: str = os.path.join(ckpt_dir, run_name)
    os.makedirs(name=run_path, exist_ok=True)
    os.makedirs(name=ckpt_path, exist_ok=True)

    MONITOR.set_run_path(run_path=Path(run_path))

    pid_file: Path = Path(run_path) / "train.pid"
    meta_file: Path = Path(run_path) / "train_meta.yaml"
    created_at_epoch_s: float = float(time.time())
    _write_text_atomic(path=pid_file, text=f"{int(os.getpid())}\n")
    _write_text_atomic(
        path=meta_file,
        text=yaml.safe_dump(
            {
                "pid": int(os.getpid()),
                "created_at_epoch_s": float(created_at_epoch_s),
                "run_path": str(run_path),
                "run_name": str(run_name),
                "env_dt_s": _maybe_env_dt_s(env=train_env),
                "ablation_use_per": bool(rdqn_cfg.use_per),
                "ablation_use_noisy": bool(rdqn_cfg.use_noisy),
            }
        ),
    )

    # --- live_frames export (used by live_visualization.py / live_monitoring.py) ---
    live_enabled: bool = bool(_live_frames_enabled(cfg=cfg))
    live_dir: Path = Path(run_path) / "live_frames"
    current_ep_file: Path = live_dir / "current_episode.txt"
    episode_frame_idx: int = 0
    episode_dir: Path | None = None
    # ---------------------------------------------------------------------------

    tb_proc: subprocess.Popen | None = None
    tb_log_path: str = str(Path(run_path) / "tensorboard.log")
    tb_proc = launch_tensorboard(
        run_dir=str(run_dir),
        run_name=str(run_name),
        tb_log_path=str(tb_log_path),
    )

    print(
        f"[train] Live tracking (launch all): "
        f'python -m src.utils.live_tracking "{run_path}"',
        flush=True,
    )
    print(
        f'[train]     Live metrics: python -m src.utils.live_metrics "{run_path}"',
        flush=True,
    )
    print(
        f"[train]     Live visualization: python -m src.utils.live_visualization "
        f'"{run_path}"',
        flush=True,
    )
    print(
        f'[train]     Live monitoring: python -m src.utils.live_monitoring "{run_path}"',
        flush=True,
    )

    writer: SummaryWriter = SummaryWriter(log_dir=run_path)

    with open(
        file=os.path.join(run_path, "config.yaml"), mode="w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(data=cfg, stream=f)

    total_steps: int = int(cfg["train"]["total_steps"])
    log_every: int = int(cfg["logging"]["log_every"])
    eval_every: int = int(cfg["eval"]["every_steps"])
    eval_episodes: int = int(cfg["eval"]["episodes"])

    start_step: int = 0
    skip_eval_step: int | None = None
    if resume_ckpt is not None:
        loaded_step: int = int(agent.load(path=str(resume_ckpt)))
        inferred_step: int | None = _infer_step_from_checkpoint_filename(
            ckpt_file=resume_ckpt
        )

        if loaded_step <= 0 and inferred_step is not None:
            loaded_step = int(inferred_step)
            agent.global_step = int(loaded_step)

        if new_run:
            start_step = 0
            agent.global_step = 0
            agent.updates = 0
            print(
                f"[train] Resuming weights from: {resume_ckpt} "
                f"(loaded_step={loaded_step}) -> new run (start_step=0)",
                flush=True,
            )
        else:
            start_step = int(loaded_step)
            skip_eval_step = int(start_step)
            print(
                f"[train] Resuming from: {resume_ckpt} (start_step={start_step})",
                flush=True,
            )

    obs, _ = train_env.reset()
    ep_ret: float = 0.0
    ep_len: int = 0
    best_eval: float = -1e18

    returns_since_log: list[float] = []
    lengths_since_log: list[int] = []

    try:
        episode_idx: int = 0

        if live_enabled:
            _ensure_dir(live_dir)
            _atomic_write_text(current_ep_file, f"{int(episode_idx)}\n")
            episode_dir = live_dir / f"ep_{int(episode_idx):06d}"
            _ensure_dir(episode_dir)
            episode_frame_idx = 0

        for step in trange(
            start_step,
            total_steps,
            desc="train",
            total=int(total_steps),
            initial=int(start_step),
        ):
            agent.global_step = int(step)

            MONITOR.begin_step(
                episode=int(episode_idx),
                global_step=int(step),
                inner_step=int(ep_len),
            )

            a: int = agent.act(obs=obs, eval_mode=False)
            next_obs, r, terminated, truncated, _ = train_env.step(a)
            done: bool = bool(terminated or truncated)

            if live_enabled and (episode_dir is not None):
                frame_rgb_live: np.ndarray = np.ascontiguousarray(
                    _extract_last_rgb_frame(obs=next_obs)
                )
                out_file: Path = episode_dir / f"frame_{int(episode_frame_idx):06d}.jpg"
                _write_jpg_atomic(path=out_file, frame_rgb=frame_rgb_live, quality=90)

                metrics_file: Path = episode_dir / (
                    f"metrics_{int(episode_frame_idx):06d}.txt"
                )
                _write_metrics_snapshot_atomic(path=metrics_file)

                episode_frame_idx += 1

            agent.store(
                obs=obs, action=a, reward=float(r), next_obs=next_obs, done=done
            )

            ep_ret += float(r)
            ep_len += 1
            obs = next_obs

            if agent.can_update():
                metrics: dict[str, float] = agent.update()
                if (step % log_every) == 0:
                    for k, v in metrics.items():
                        writer.add_scalar(
                            tag=f"train/{k}",
                            scalar_value=float(v),
                            global_step=int(step),
                        )

                    writer.add_scalar(
                        tag="train/episode_return_mean_since_last_log",
                        scalar_value=float(np.mean(returns_since_log))
                        if returns_since_log
                        else float("nan"),
                        global_step=int(step),
                    )
                    writer.add_scalar(
                        tag="train/episode_length_mean_since_last_log",
                        scalar_value=float(np.mean(lengths_since_log))
                        if lengths_since_log
                        else float("nan"),
                        global_step=int(step),
                    )

                    returns_since_log.clear()
                    lengths_since_log.clear()

            if done:
                episode_idx += 1

                writer.add_scalar(
                    tag="train/episode_return",
                    scalar_value=float(ep_ret),
                    global_step=int(step),
                )
                writer.add_scalar(
                    tag="train/episode_length",
                    scalar_value=float(ep_len),
                    global_step=int(step),
                )

                returns_since_log.append(float(ep_ret))
                lengths_since_log.append(int(ep_len))

                obs, _ = train_env.reset()
                ep_ret = 0.0
                ep_len = 0

                if live_enabled:
                    _atomic_write_text(current_ep_file, f"{int(episode_idx)}\n")
                    episode_dir = live_dir / f"ep_{int(episode_idx):06d}"
                    _ensure_dir(episode_dir)
                    episode_frame_idx = 0

                    # Remove episode dirs that are older than LIVE_CLEANUP_TIMER seconds
                    live_ep_queue.append(
                        (
                            float(time.time()),
                            int(episode_idx),
                            episode_dir,
                        )
                    )
                    now_s: float = float(time.time())
                    cutoff_s: float = now_s - float(LIVE_CLEANUP_TIMER)

                    while live_ep_queue and live_ep_queue[0][0] < cutoff_s:
                        _, old_ep_idx, old_ep_dir = live_ep_queue.popleft()

                        # Never delete the currently active episode directory.
                        if int(old_ep_idx) == int(episode_idx):
                            continue

                        shutil.rmtree(path=old_ep_dir, ignore_errors=True)

            if (
                (step > 0)
                and ((step % eval_every) == 0)
                and (skip_eval_step is None or step != skip_eval_step)
            ):
                timestamp: str = datetime.now().strftime(format="%H-%M-%S")
                ckpt_file: str = os.path.join(ckpt_path, f"step_{step}_{timestamp}.pt")
                agent.save(path=ckpt_file)

                video_dir: str = str(
                    Path("videos")
                    / "rdqn-ablation"
                    / str(run_name)
                    / f"step_{int(step)}"
                )

                MONITOR.deactivate()
                try:
                    eval_ret: float = _run_eval_in_subprocess(
                        config_path=str(config_path),
                        checkpoint_path=str(ckpt_file),
                        seed=int(seed + 123),
                        episodes=int(eval_episodes),
                        video_dir=str(video_dir),
                    )
                finally:
                    MONITOR.activate()

                writer.add_scalar(
                    tag="eval/return_mean",
                    scalar_value=float(eval_ret),
                    global_step=int(step),
                )

                if eval_ret > best_eval:
                    best_eval = float(eval_ret)
                    timestamp = datetime.now().strftime(format="%H-%M-%S")
                    agent.save(path=os.path.join(ckpt_path, f"best_{timestamp}.pt"))

    finally:
        train_env.close()
        writer.close()
        if tb_proc is not None:
            _terminate_process_tree(proc=tb_proc)

        for p in (pid_file, meta_file):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    import argparse

    graphics_backend = auto_detect_mujoco_gl()
    print("[train] graphics_backend:", graphics_backend)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rdqn.yaml")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint .pt file or directory to resume from (default: None).",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="If set and --resume is used, write outputs to a new run directory "
        "and restart step counters from 0.",
    )
    args: Namespace = parser.parse_args()
    main(
        config_path=str(args.config),
        resume_path=args.resume,
        new_run=bool(args.new_run),
    )
