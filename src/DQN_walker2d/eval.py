# src/DQN_walker2d/eval.py

from __future__ import annotations

import argparse
from typing import SupportsFloat

import numpy as np
import yaml
from src.dqn import DQNAgent, DQNConfig
from src.env import EnvSpec, make_eval_env


def load_yaml(path: str):
    with open(file=path, encoding="utf-8") as f:
        return yaml.safe_load(stream=f)


def main(config_path: str, ckpt_path: str, episodes: int) -> None:
    cfg = load_yaml(path=config_path)
    seed = int(cfg["seed"])

    e = cfg["env"]
    env_spec = EnvSpec(
        env_id=e["env_id"],
        frame_stack=int(e["frame_stack"]),
        action_repeat=int(e["action_repeat"]),
        time_limit=int(e["time_limit"]),
        action_prototypes=e["action_prototypes"],
    )

    video_dir = "videos"
    env = make_eval_env(spec=env_spec, seed=seed + 999, video_dir=video_dir)

    t = cfg["train"]
    ex = cfg["exploration"]
    dqn_cfg = DQNConfig(
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

    obs_shape: tuple[int, ...] | None = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=dqn_cfg)
    agent.load(path=ckpt_path)

    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a: int = agent.act(obs=obs, eval_mode=True)
            r: SupportsFloat
            terminated: bool
            truncated: bool
            obs, r, terminated, truncated, _ = env.step(action=a)
            done: bool = terminated or truncated
            ep_ret += float(r)
        returns.append(ep_ret)

    print(
        f"episodes={episodes} "
        f"mean_return={np.mean(a=returns):.3f} "
        f"std_return={np.std(a=returns):.3f}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dqn.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args: argparse.Namespace = parser.parse_args()
    main(config_path=args.config, ckpt_path=args.ckpt, episodes=args.episodes)
