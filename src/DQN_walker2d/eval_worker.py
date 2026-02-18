# src/DQN_walker2d/eval_worker.py

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml

from src.DQN_walker2d.dqn import DQNAgent, DQNConfig
from src.DQN_walker2d.env import EnvSpec, make_eval_env


def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary."""
    with open(file=path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(stream=f)
    return data


def _to_env_spec(cfg: dict[str, Any]) -> EnvSpec:
    """Convert the `env` section of the config into an EnvSpec."""
    e: dict[str, Any] = cfg["env"]
    return EnvSpec(
        env_id=str(e["env_id"]),
        frame_stack=int(e["frame_stack"]),
        action_repeat=int(e["action_repeat"]),
        time_limit=int(e["time_limit"]),
        action_prototypes=e["action_prototypes"],
        use_pixels=True,
    )


def _to_dqn_cfg(cfg_dict: dict[str, Any]) -> DQNConfig:
    """Rebuild DQNConfig from a checkpoint-serialized dict."""
    return DQNConfig(
        gamma=float(cfg_dict["gamma"]),
        lr=float(cfg_dict["lr"]),
        batch_size=int(cfg_dict["batch_size"]),
        buffer_size=int(cfg_dict["buffer_size"]),
        learning_starts=int(cfg_dict["learning_starts"]),
        train_freq=int(cfg_dict["train_freq"]),
        target_update_freq=int(cfg_dict["target_update_freq"]),
        grad_clip_norm=float(cfg_dict["grad_clip_norm"]),
        eps_start=float(cfg_dict["eps_start"]),
        eps_end=float(cfg_dict["eps_end"]),
        eps_decay_steps=int(cfg_dict["eps_decay_steps"]),
        device=str(cfg_dict["device"]),
    )


@torch.no_grad()
def _evaluate(agent: DQNAgent, env: Any, episodes: int) -> float:
    """Run evaluation episodes and return mean episodic return."""
    returns: list[float] = []
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done: bool = False
        ep_ret: float = 0.0
        while not done:
            a: int = agent.act(obs=obs, eval_mode=True)
            obs, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            ep_ret += float(r)
        returns.append(ep_ret)
    return float(np.mean(returns)) if returns else float("nan")


@dataclass(frozen=True)
class EvalArgs:
    """Arguments for the evaluation worker."""

    config: str
    checkpoint: str
    seed: int
    episodes: int
    video_dir: str
    output_path: str


def run(args: EvalArgs) -> None:
    """Entry point for subprocess evaluation with video recording."""
    cfg: dict[str, Any] = _load_yaml(path=args.config)
    env_spec: EnvSpec = _to_env_spec(cfg=cfg)

    payload: dict[str, Any] = torch.load(
        f=args.checkpoint, map_location=torch.device("cpu")
    )
    dqn_cfg: DQNConfig = _to_dqn_cfg(cfg_dict=dict(payload["cfg"]))

    eval_env = make_eval_env(
        spec=env_spec, seed=int(args.seed), video_dir=args.video_dir
    )
    try:
        obs_shape_raw: tuple[int, ...] | None = eval_env.observation_space.shape
        if obs_shape_raw is None or len(obs_shape_raw) != 3:
            raise ValueError(f"Expected obs shape (C,H,W), got {obs_shape_raw}")
        obs_shape: tuple[int, int, int] = (
            int(obs_shape_raw[0]),
            int(obs_shape_raw[1]),
            int(obs_shape_raw[2]),
        )
        n_actions: int = int(eval_env.action_space.n)

        agent: DQNAgent = DQNAgent(
            obs_shape=obs_shape, n_actions=n_actions, cfg=dqn_cfg
        )
        agent.load(path=args.checkpoint)

        mean_ret: float = _evaluate(
            agent=agent, env=eval_env, episodes=int(args.episodes)
        )
    finally:
        eval_env.close()

    os.makedirs(name=os.path.dirname(args.output_path), exist_ok=True)
    with open(file=args.output_path, mode="w", encoding="utf-8") as f:
        f.write(f"{mean_ret:.10f}\n")


def _parse_args() -> EvalArgs:
    """Parse CLI args for the eval worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    ns = parser.parse_args()
    return EvalArgs(
        config=str(ns.config),
        checkpoint=str(ns.checkpoint),
        seed=int(ns.seed),
        episodes=int(ns.episodes),
        video_dir=str(ns.video_dir),
        output_path=str(ns.output_path),
    )


if __name__ == "__main__":
    run(args=_parse_args())
