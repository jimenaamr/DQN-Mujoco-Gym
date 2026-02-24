# src/ablation/monitoring.py

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import yaml

from src.rdqn.env import EnvSpec, make_eval_env
from src.rdqn.helper import resolve_device
from src.rdqn.rdqn import RDQNAgent, RDQNConfig

os.environ.setdefault("DQN_MONITOR_DISABLED", "1")


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


def _infer_run_name_from_checkpoint(ckpt_path: str) -> str:
    """Infer run name as the checkpoint parent directory name.

    Args:
        ckpt_path: Path to checkpoint file.

    Returns:
        Inferred run name (fallbacks to "unknown_run" if empty).
    """
    p: Path = Path(ckpt_path).expanduser()
    name: str = p.parent.name
    return name if name else "unknown_run"


def main(config_path: str, ckpt_path: str, episodes: int) -> None:
    """Evaluate a checkpoint and write videos under videos/rdqn/<run_name>/manual_eval.

    Args:
        config_path: YAML config file.
        ckpt_path: Checkpoint .pt file.
        episodes: Number of episodes.
    """
    cfg: dict[str, Any] = load_yaml(path=config_path)
    seed: int = int(cfg["seed"])

    e: dict[str, Any] = cfg["env"]
    env_spec: EnvSpec = EnvSpec(
        env_id=str(e["env_id"]),
        frame_stack=int(e["frame_stack"]),
        action_repeat=int(e["action_repeat"]),
        time_limit=int(e["time_limit"]),
        action_prototypes=e["action_prototypes"],
        obs_h=int(e.get("obs_h", 84)),
        obs_w=int(e.get("obs_w", 84)),
        grayscale=bool(e.get("grayscale")),
    )

    run_name: str = _infer_run_name_from_checkpoint(ckpt_path=str(ckpt_path))
    video_dir: str = str(Path("videos") / "rdqn" / run_name / "manual_eval")
    env = make_eval_env(spec=env_spec, seed=int(seed + 999), video_dir=str(video_dir))

    t: dict[str, Any] = cfg["train"]
    ex: dict[str, Any] = dict(cfg.get("exploration", {}))
    rb: dict[str, Any] = dict(cfg.get("rainbow", {}))

    device_name: str = str(cfg.get("device", "cpu"))
    device_resolved: str = resolve_device(name=device_name)
    noisy: bool = bool(rb.get("noisy", True))

    rdqn_cfg: RDQNConfig = RDQNConfig(
        gamma=float(t["gamma"]),
        lr=float(t["lr"]),
        batch_size=int(t["batch_size"]),
        buffer_size=int(t["buffer_size"]),
        learning_starts=int(t["learning_starts"]),
        train_freq=int(t["train_freq"]),
        target_update_freq=int(t["target_update_freq"]),
        grad_clip_norm=float(t["grad_clip_norm"]),
        device=str(device_resolved),
        eps_start=float(ex.get("eps_start", 1.0)),
        eps_end=float(ex.get("eps_end", 0.05)),
        eps_decay_steps=int(ex.get("eps_decay_steps", 200_000)),
        n_step=int(rb.get("n_step", 3)),
        per_alpha=float(rb.get("per_alpha", 0.6)),
        per_beta_start=float(rb.get("per_beta_start", 0.4)),
        per_beta_frames=int(rb.get("per_beta_frames", 200_000)),
        use_noisy=bool(noisy),
        noisy_std_init=float(rb.get("noisy_std_init", 0.5)),
        atoms=int(rb.get("atoms", 51)),
        v_min=float(rb.get("v_min", -10.0)),
        v_max=float(rb.get("v_max", 10.0)),
    )

    obs_shape_raw: tuple[int, ...] | None = env.observation_space.shape
    if obs_shape_raw is None or len(obs_shape_raw) != 3:
        raise ValueError(f"Expected pixel obs shape (C,H,W); got {obs_shape_raw}.")
    obs_shape: tuple[int, int, int] = (
        int(obs_shape_raw[0]),
        int(obs_shape_raw[1]),
        int(obs_shape_raw[2]),
    )

    n_actions: int = int(env.action_space.n)
    agent: RDQNAgent = RDQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=rdqn_cfg)
    agent.load(path=str(ckpt_path))

    returns: list[float] = []
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done: bool = False
        ep_ret: float = 0.0
        while not done:
            a: int = agent.act(obs=obs, eval_mode=True)
            r: SupportsFloat
            terminated: bool
            truncated: bool
            obs, r, terminated, truncated, _ = env.step(action=a)
            done = bool(terminated or truncated)
            ep_ret += float(r)
        returns.append(float(ep_ret))

    print(
        f"episodes={int(episodes)} "
        f"mean_return={float(np.mean(a=returns)):.3f} "
        f"std_return={float(np.std(a=returns)):.3f} "
        f"video_dir={video_dir}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rdqn.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args: argparse.Namespace = parser.parse_args()
    main(
        config_path=str(args.config),
        ckpt_path=str(args.ckpt),
        episodes=int(args.episodes),
    )
