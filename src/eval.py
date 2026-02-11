from __future__ import annotations

import argparse
import yaml
import numpy as np

from env import EnvSpec, make_env
from dqn import DQNAgent, DQNConfig


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str, ckpt_path: str, episodes: int) -> None:
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])

    e = cfg["env"]
    env_spec = EnvSpec(
        env_id=e["env_id"],
        frame_stack=int(e["frame_stack"]),
        action_repeat=int(e["action_repeat"]),
        time_limit=int(e["time_limit"]),
        action_prototypes=e["action_prototypes"],
    )

    env = make_env(env_spec, seed=seed + 999)

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

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DQNAgent(obs_shape, n_actions, dqn_cfg)
    agent.load(ckpt_path)

    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(obs, eval_mode=True)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_ret += float(r)
        returns.append(ep_ret)

    print(f"episodes={episodes} mean_return={np.mean(returns):.3f} std_return={np.std(returns):.3f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dqn.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    main(args.config, args.ckpt, args.episodes)