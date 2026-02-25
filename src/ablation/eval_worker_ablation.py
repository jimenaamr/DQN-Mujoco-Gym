# src/ablation/eval_worker_ablation.py

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import yaml

from src.rdqn.env import EnvSpec, make_eval_env
from src.rdqn.helper import resolve_device
from src.rdqn.rdqn import RDQNAgent, RDQNConfig


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

    rdqn_cfg: RDQNConfig = to_rdqn_cfg(cfg=cfg)
    agent = RDQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=rdqn_cfg)
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
