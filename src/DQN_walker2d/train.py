# src/DQN_walker2d/train.py

from __future__ import annotations

import os

os.environ.setdefault(key="MUJOCO_GL", value="egl")  # DEFINE BEFORE IMPORTING GYMNASIUM

import re
import subprocess
import tempfile
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.DQN_walker2d.dqn import DQNAgent, DQNConfig
from src.DQN_walker2d.env import EnvSpec, make_env
from src.DQN_walker2d.gui_viewer import TkLiveViewer
from src.DQN_walker2d.helper import (
    StabilizerConfig,
    resolve_device,
    stabilizer_config_from_yaml,
)
from src.DQN_walker2d.monitoring import MONITOR

_STEP_RE: re.Pattern[str] = re.compile(pattern=r"^step_(\d+)_.*\.pt$")
_STEP_ANYWHERE_RE: re.Pattern[str] = re.compile(pattern=r"step_(\d+)_.*\.pt$")


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
    )


def to_dqn_cfg(cfg: dict[str, Any]) -> DQNConfig:
    """Convert the config into a DQNConfig.

    Notes:
        This reads the top-level `device` field and normalizes it so:
        - "cpu" -> "cpu"
        - "gpu" -> "cuda" if available else "cpu"

    Args:
        cfg: Full experiment configuration.

    Returns:
        Parsed DQNConfig.
    """
    t: dict[str, Any] = cfg["train"]
    ex: dict[str, Any] = cfg["exploration"]
    device_name: str = str(cfg.get("device", "cpu"))
    device_resolved: str = resolve_device(name=device_name)

    return DQNConfig(
        gamma=float(t["gamma"]),
        lr=float(t["lr"]),
        batch_size=int(t["batch_size"]),
        buffer_size=int(t["buffer_size"]),
        learning_starts=int(t["learning_starts"]),
        train_freq=int(t["train_freq"]),
        target_update_freq=int(t["target_update_freq"]),
        grad_clip_norm=float(t["grad_clip_norm"]),
        eps_start=float(ex["eps_start"]),
        eps_end=float(ex["eps_end"]),
        eps_decay_steps=int(ex["eps_decay_steps"]),
        device=str(device_resolved),
    )


def _extract_last_rgb_frame(obs: np.ndarray) -> np.ndarray:
    """Extract an RGB frame (H, W, 3) from an observation shaped like (C, H, W).

    Args:
        obs: Observation array with shape (C, H, W) and C >= 3.

    Returns:
        RGB frame with shape (H, W, 3).

    Raises:
        ValueError: If obs does not have 3 dims or C < 3.
    """
    if obs.ndim != 3:
        raise ValueError(f"Expected obs with shape (C,H,W), got {obs.shape}")
    c: int = int(obs.shape[0])
    if c < 3:
        raise ValueError(f"Expected at least 3 channels, got C={c}")
    frame_chw: np.ndarray = obs[c - 3 : c, :, :]
    frame_hwc: np.ndarray = np.transpose(frame_chw, (1, 2, 0))
    return frame_hwc


def _run_eval_in_subprocess(
    config_path: str,
    checkpoint_path: str,
    seed: int,
    episodes: int,
    video_dir: str,
) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path: str = os.path.join(tmpdir, "eval_return.txt")

        cmd: list[str] = [
            "python",
            "-m",
            "src.DQN_walker2d.eval_worker",
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
        env["CUDA_VISIBLE_DEVICES"] = ""  # eval on CPU only
        subprocess.run(args=cmd, check=True, env=env)

        with open(file=out_path, encoding="utf-8") as f:
            raw: str = f.read().strip()
        return float(raw)


def _format_legend_text() -> str:
    """Build legend text from the global MONITOR.

    Returns:
        Human-readable debug text for the live viewer overlay.
    """
    return (
        f"episode: {MONITOR.episode_index}\n"
        f"raw reward: {MONITOR.raw_reward:.3f}\n"
        f"head height: {MONITOR.head_height:.3f}\n"
        f"acc fw reward: {MONITOR.acc_fw_reward:.3f}\n"
        f"helper intensity: {MONITOR.helper_intensity:.3f}\n"
        f"agent contrib: {MONITOR.agent_contrib:.3f}\n"
        f"real reward: {MONITOR.real_reward:.3f}\n"
    )


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
    """Infer run name from a checkpoint path: checkpoints/<run_name>/<file>.pt.

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


def main(
    config_path: str, resume_path: str | None = None, new_run: bool = False
) -> None:
    """Entry point for training.

    Args:
        config_path: Path to YAML config file.
        resume_path: Optional checkpoint file or directory to resume from.
        new_run: If True, start a new output directory even when resuming.
    """
    cfg: dict[str, Any] = load_yaml(path=config_path)
    seed: int = int(cfg["seed"])

    env_spec: EnvSpec = to_env_spec(cfg=cfg)
    stabilizer_cfg: StabilizerConfig | None = stabilizer_config_from_yaml(cfg=cfg)

    train_env = make_env(spec=env_spec, seed=seed, stabilizer=stabilizer_cfg)

    obs_shape_raw: tuple[int, ...] | None = train_env.observation_space.shape
    # assert False, obs_shape_raw
    if obs_shape_raw is None or len(obs_shape_raw) != 3:
        raise ValueError(
            f"Expected pixel obs shape (C,H,W); got {obs_shape_raw}. "
            "If you want vector observations, you must also update the DQN model."
        )
    obs_shape: tuple[int, int, int] = (
        int(obs_shape_raw[0]),
        int(obs_shape_raw[1]),
        int(obs_shape_raw[2]),
    )
    n_actions: int = int(train_env.action_space.n)

    dqn_cfg: DQNConfig = to_dqn_cfg(cfg=cfg)
    agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=dqn_cfg)

    device_raw: str = str(cfg.get("device", "cpu"))
    print(
        f"[train] device config: {device_raw} -> resolved: {dqn_cfg.device}", flush=True
    )

    run_dir: str = str(cfg["logging"]["run_dir"])
    ckpt_dir: str = str(cfg["logging"]["ckpt_dir"])
    os.makedirs(name=run_dir, exist_ok=True)
    os.makedirs(name=ckpt_dir, exist_ok=True)

    resume_ckpt: Path | None = resolve_resume_path(resume=resume_path)

    # Directory naming policy:
    # - If not resuming: always new run.
    # - If resuming:
    #   - new_run=True -> always new run name
    #   - new_run=False -> keep run_name inferred from checkpoint dir if possible
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

    run_path: str = os.path.join(run_dir, run_name)
    ckpt_path: str = os.path.join(ckpt_dir, run_name)
    os.makedirs(name=run_path, exist_ok=True)
    os.makedirs(name=ckpt_path, exist_ok=True)

    writer: SummaryWriter = SummaryWriter(log_dir=run_path)

    with open(
        file=os.path.join(run_path, "config.yaml"), mode="w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(data=cfg, stream=f)

    total_steps: int = int(cfg["train"]["total_steps"])
    log_every: int = int(cfg["logging"]["log_every"])
    eval_every: int = int(cfg["eval"]["every_steps"])
    eval_episodes: int = int(cfg["eval"]["episodes"])

    logging_cfg: dict[str, Any] = dict(cfg.get("logging", {}))
    render_live: bool = bool(logging_cfg.get("render_live", True))
    render_every: int = int(logging_cfg.get("render_every", 1))
    has_display: bool = bool(os.environ.get("DISPLAY"))

    viewer: TkLiveViewer = TkLiveViewer(
        enabled=bool(render_live and has_display),
        title="training_live",
    )
    viewer.start()

    start_step: int = 0
    if resume_ckpt is not None:
        loaded_step: int = int(agent.load(path=str(resume_ckpt)))
        inferred_step: int | None = _infer_step_from_checkpoint_filename(
            ckpt_file=resume_ckpt
        )

        # If checkpoint doesn't contain global_step (old format), infer from filename.
        if loaded_step <= 0 and inferred_step is not None:
            loaded_step = int(inferred_step)
            agent.global_step = int(loaded_step)

        # new_run=True means: load weights but reset counters like a fresh run.
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
        MONITOR.set_episode(episode_idx)

        for step in trange(start_step, total_steps, desc="train"):
            agent.global_step = int(step)

            if (render_every > 0) and ((step % render_every) == 0):
                frame_rgb: np.ndarray = np.ascontiguousarray(
                    _extract_last_rgb_frame(obs=obs)
                )
                viewer.push(frame_rgb=frame_rgb, legend_text=_format_legend_text())

            a: int = agent.act(obs=obs, eval_mode=False)
            next_obs, r, terminated, truncated, _ = train_env.step(a)
            done: bool = bool(terminated or truncated)

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
                MONITOR.set_episode(episode_idx)

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

            if (step > 0) and ((step % eval_every) == 0):
                timestamp: str = datetime.now().strftime(format="%H-%M-%S")
                ckpt_file: str = os.path.join(ckpt_path, f"step_{step}_{timestamp}.pt")
                agent.save(path=ckpt_file)

                video_dir: str = os.path.join("videos", run_name, f"step_{step}")

                eval_ret: float = _run_eval_in_subprocess(
                    config_path=str(config_path),
                    checkpoint_path=str(ckpt_file),
                    seed=int(seed + 123),
                    episodes=int(eval_episodes),
                    video_dir=str(video_dir),
                )

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
        viewer.close()
        train_env.close()
        writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dqn.yaml")
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
