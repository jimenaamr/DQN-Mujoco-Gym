# src/DQN_walker2d/train.py

from __future__ import annotations

import os
import subprocess
import tempfile
from argparse import Namespace
from datetime import datetime
from typing import Any

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.DQN_walker2d.dqn import DQNAgent, DQNConfig
from src.DQN_walker2d.env import EnvSpec, make_env
from src.DQN_walker2d.gui_viewer import TkLiveViewer
from src.DQN_walker2d.helper import StabilizerConfig, stabilizer_config_from_yaml
from src.DQN_walker2d.monitoring import MONITOR

os.environ.setdefault(key="MUJOCO_GL", value="egl")


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed YAML content as a dictionary.
    """
    with open(file=path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(stream=f)
    return data


def to_env_spec(cfg: dict[str, Any]) -> EnvSpec:
    """Convert the `env` section of the config into an EnvSpec.

    Args:
        cfg: Full experiment config.

    Returns:
        Environment specification parsed from config.
    """
    e: dict[str, Any] = cfg["env"]
    return EnvSpec(
        env_id=str(e["env_id"]),
        frame_stack=int(e["frame_stack"]),
        action_repeat=int(e["action_repeat"]),
        time_limit=int(e["time_limit"]),
        action_prototypes=e["action_prototypes"],
        use_pixels=bool(e.get("use_pixels", True)),
    )


def to_dqn_cfg(cfg: dict[str, Any]) -> DQNConfig:
    """Convert the config into a DQNConfig.

    Args:
        cfg: Full experiment config.

    Returns:
        DQN hyperparameters parsed from config.
    """
    t: dict[str, Any] = cfg["train"]
    ex: dict[str, Any] = cfg["exploration"]
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
        device=str(cfg["device"]),
    )


def _extract_last_rgb_frame(obs: np.ndarray) -> np.ndarray:
    """Extract an RGB frame (H, W, 3) from an observation shaped like (C, H, W).

    Args:
        obs: Observation with shape (C, H, W) where C >= 3.

    Returns:
        RGB frame with shape (H, W, 3).

    Raises:
        ValueError: If obs is not (C,H,W) or has fewer than 3 channels.
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
    """Run evaluation + video recording in a separate process to isolate EGL.

    Args:
        config_path: Path to the YAML config.
        checkpoint_path: Path to a saved agent checkpoint.
        seed: RNG seed for evaluation.
        episodes: Number of episodes to evaluate.
        video_dir: Directory where videos will be written.

    Returns:
        Mean episode return over evaluation episodes.
    """
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
        subprocess.run(args=cmd, check=True)

        with open(file=out_path, encoding="utf-8") as f:
            raw: str = f.read().strip()
        return float(raw)


def _format_legend_text() -> str:
    """Build legend text from the global MONITOR.

    Returns:
        A multi-line string summarizing the current episode metrics.
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


def main(config_path: str) -> None:
    """Train a DQN agent on the configured environment.

    Args:
        config_path: Path to YAML config file.
    """
    cfg: dict[str, Any] = load_yaml(path=config_path)
    seed: int = int(cfg["seed"])

    env_spec: EnvSpec = to_env_spec(cfg=cfg)
    stabilizer_cfg: StabilizerConfig | None = stabilizer_config_from_yaml(cfg=cfg)

    train_env = make_env(spec=env_spec, seed=seed, stabilizer=stabilizer_cfg)

    obs_shape_raw: tuple[int, ...] | None = train_env.observation_space.shape
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

    run_dir: str = str(cfg["logging"]["run_dir"])
    ckpt_dir: str = str(cfg["logging"]["ckpt_dir"])
    os.makedirs(name=run_dir, exist_ok=True)
    os.makedirs(name=ckpt_dir, exist_ok=True)

    run_name: str = (
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

    obs, _ = train_env.reset()
    ep_ret: float = 0.0
    ep_len: int = 0
    best_eval: float = -1e18

    returns_since_log: list[float] = []
    lengths_since_log: list[int] = []

    try:
        episode_idx: int = 0
        MONITOR.set_episode(episode_idx)

        for step in trange(total_steps, desc="train"):
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
    args: Namespace = parser.parse_args()
    main(config_path=str(args.config))
