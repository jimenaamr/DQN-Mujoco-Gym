from __future__ import annotations

import os
import time
from argparse import Namespace
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.dqn import DQNAgent, DQNConfig
from src.env import DiscreteActionWrapper, EnvSpec, FrameStack, make_env


def load_yaml(path: str) -> dict[str, Any]:
    with open(file=path, encoding="utf-8") as f:
        return yaml.safe_load(stream=f)


def to_env_spec(cfg: dict[str, Any]) -> EnvSpec:
    e = cfg["env"]
    return EnvSpec(
        env_id=e["env_id"],
        frame_stack=int(e["frame_stack"]),
        action_repeat=int(e["action_repeat"]),
        time_limit=int(e["time_limit"]),
        action_prototypes=e["action_prototypes"],
    )


def to_dqn_cfg(cfg: dict[str, Any]) -> DQNConfig:
    t = cfg["train"]
    ex = cfg["exploration"]
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


@torch.no_grad()
def evaluate(agent: DQNAgent, env, episodes: int) -> float:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a: int = agent.act(obs=obs, eval_mode=True)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_ret += float(r)
        returns.append(ep_ret)
    return float(np.mean(returns))


def main(config_path: str) -> None:
    cfg = load_yaml(path=config_path)
    seed = int(cfg["seed"])

    env_spec: EnvSpec = to_env_spec(cfg=cfg)
    train_env: FrameStack | DiscreteActionWrapper = make_env(spec=env_spec, seed=seed)
    eval_env: FrameStack | DiscreteActionWrapper = make_env(
        spec=env_spec, seed=seed + 123
    )

    obs_shape = train_env.observation_space.shape  # vector (n,) or stacked (k*n,)
    n_actions = train_env.action_space.n

    dqn_cfg: DQNConfig = to_dqn_cfg(cfg=cfg)
    agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=dqn_cfg)

    run_dir = cfg["logging"]["run_dir"]
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(name=run_dir, exist_ok=True)
    os.makedirs(name=ckpt_dir, exist_ok=True)

    run_name: str = f"{env_spec.env_id}_seed{seed}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(run_dir, run_name))

    # Save resolved config
    os.makedirs(name=os.path.join(run_dir, run_name), exist_ok=True)
    with open(
        file=os.path.join(run_dir, run_name, "config.yaml"), mode="w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(data=cfg, stream=f)

    total_steps = int(cfg["train"]["total_steps"])
    log_every = int(cfg["logging"]["log_every"])
    eval_every = int(cfg["eval"]["every_steps"])
    eval_episodes = int(cfg["eval"]["episodes"])

    obs, _ = train_env.reset()
    ep_ret = 0.0
    ep_len = 0
    best_eval: float = -1e18

    for step in trange(total_steps, desc="train"):
        agent.global_step = step

        a: int = agent.act(obs=obs, eval_mode=False)
        next_obs, r, terminated, truncated, _ = train_env.step(a)
        done = bool(terminated or truncated)

        agent.store(obs=obs, action=a, reward=float(r), next_obs=next_obs, done=done)

        ep_ret += float(r)
        ep_len += 1
        obs = next_obs

        if agent.can_update():
            metrics: dict[str, float] = agent.update()
            if step % log_every == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)

        if done:
            writer.add_scalar("train/episode_return", ep_ret, step)
            writer.add_scalar("train/episode_length", ep_len, step)
            obs, _ = train_env.reset()
            ep_ret = 0.0
            ep_len = 0

        if (step > 0) and (step % eval_every == 0):
            eval_ret: float = evaluate(
                agent=agent, env=eval_env, episodes=eval_episodes
            )
            writer.add_scalar("eval/return_mean", eval_ret, step)

            ckpt_path: str = os.path.join(ckpt_dir, f"{run_name}_step{step}.pt")
            agent.save(path=ckpt_path)

            if eval_ret > best_eval:
                best_eval = eval_ret
                agent.save(path=os.path.join(ckpt_dir, f"{run_name}_best.pt"))

    writer.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dqn.yaml")
    args: Namespace = parser.parse_args()
    main(config_path=args.config)
