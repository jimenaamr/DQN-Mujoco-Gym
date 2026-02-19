# src/DQN_walker2d/eval_worker.py

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import yaml

from src.DQN_walker2d.dqn import DQNAgent, DQNConfig
from src.DQN_walker2d.env import EnvSpec, make_eval_env
from src.DQN_walker2d.helper import resolve_device


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
    """Convert the config into a DQNConfig for evaluation.

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


def run_eval(
    config_path: str,
    checkpoint_path: str,
    seed: int,
    episodes: int,
    video_dir: str,
) -> float:
    """Evaluate a checkpoint for a fixed number of episodes.

    Args:
        config_path: Path to YAML config.
        checkpoint_path: Path to saved checkpoint.
        seed: Base evaluation seed.
        episodes: Number of episodes.
        video_dir: Directory to store videos.

    Returns:
        Mean return.
    """
    cfg: dict[str, Any] = load_yaml(path=config_path)
    env_spec: EnvSpec = to_env_spec(cfg=cfg)

    # os.makedirs(name=video_dir, exist_ok=True) EVAL ENV ALREADY CREATES VIDEO DIR
    env = make_eval_env(spec=env_spec, seed=int(seed), video_dir=str(video_dir))

    obs_shape_raw: tuple[int, ...] | None = env.observation_space.shape
    if obs_shape_raw is None or len(obs_shape_raw) != 3:
        raise ValueError(f"Expected pixel obs shape (C,H,W); got {obs_shape_raw}.")
    obs_shape: tuple[int, int, int] = (
        int(obs_shape_raw[0]),
        int(obs_shape_raw[1]),
        int(obs_shape_raw[2]),
    )
    n_actions: int = int(env.action_space.n)

    dqn_cfg: DQNConfig = to_dqn_cfg(cfg=cfg)
    agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=dqn_cfg)
    agent.load(path=str(checkpoint_path))

    returns: list[float] = []
    try:
        for ep in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed + ep))
            done: bool = False
            ep_ret: float = 0.0

            while not done:
                a: int = agent.act(obs=obs, eval_mode=True)
                obs, r, terminated, truncated, _ = env.step(a)
                done = bool(terminated or truncated)
                ep_ret += float(r)

            returns.append(float(ep_ret))
    finally:
        env.close()

    return float(np.mean(returns)) if returns else float("nan")


def main() -> None:
    """CLI entry point for subprocess evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    mean_ret: float = run_eval(
        config_path=str(args.config),
        checkpoint_path=str(args.checkpoint),
        seed=int(args.seed),
        episodes=int(args.episodes),
        video_dir=str(args.video_dir),
    )

    with open(file=str(args.output_path), mode="w", encoding="utf-8") as f:
        f.write(f"{float(mean_ret)}")


if __name__ == "__main__":
    main()
