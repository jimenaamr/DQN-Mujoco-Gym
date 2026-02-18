from __future__ import annotations

import os

os.environ.setdefault(key="MUJOCO_GL", value="egl")

from argparse import Namespace
from datetime import datetime
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.DQN_walker2d.dqn import DQNAgent, DQNConfig
from src.DQN_walker2d.env import (
    EnvSpec,
    make_env,
    make_eval_env,
)
from src.DQN_walker2d.helper import StabilizerConfig, stabilizer_config_from_yaml


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


def to_stabilizer_cfg(cfg: dict[str, Any]) -> StabilizerConfig | None:
    s_raw: Any = cfg.get("stabilizer")
    if s_raw is None:
        return None
    s: dict[str, Any] = dict(s_raw)
    return StabilizerConfig(
        initial_intensity=float(s["initial_intensity"]),
        decay=float(s["decay"]),
        ref_height=float(s["ref_height"]),
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
    # train_env: FrameStack | DiscreteActionWrapper = make_env(spec=env_spec, seed=seed)
    # eval_env: FrameStack | DiscreteActionWrapper = make_env(
    #     spec=env_spec, seed=seed + 123
    # )

    stabilizer_cfg: StabilizerConfig | None = stabilizer_config_from_yaml(cfg=cfg)
    train_env = make_env(
        spec=env_spec,
        seed=seed,
        stabilizer=stabilizer_cfg,
    )

    obs_shape: tuple[int, ...] | None = (
        train_env.observation_space.shape
    )  # vector (n,) or stacked (k*n,)
    n_actions = train_env.action_space.n

    dqn_cfg: DQNConfig = to_dqn_cfg(cfg=cfg)
    agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=dqn_cfg)

    run_dir = cfg["logging"]["run_dir"]
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(name=run_dir, exist_ok=True)
    os.makedirs(name=ckpt_dir, exist_ok=True)

    # run_name: str = f"{env_spec.env_id}_seed{seed}_{int(time.time())}"
    run_name: str = (
        f"{env_spec.env_id}_{datetime.now().strftime(format='%Y%m%d_%H%M%S')}"
    )
    run_path: str = os.path.join(run_dir, run_name)
    ckpt_path: str = os.path.join(ckpt_dir, run_name)

    os.makedirs(name=run_path, exist_ok=True)
    os.makedirs(name=ckpt_path, exist_ok=True)

    writer = SummaryWriter(log_dir=run_path)

    # Save resolved config
    with open(
        file=os.path.join(run_path, "config.yaml"), mode="w", encoding="utf-8"
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
        r: float
        terminated: bool
        truncated: bool
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
                    writer.add_scalar(
                        tag=f"train/{k}", scalar_value=v, global_step=step
                    )

        if done:
            writer.add_scalar(
                tag="train/episode_return", scalar_value=ep_ret, global_step=step
            )
            writer.add_scalar(
                tag="train/episode_length", scalar_value=ep_len, global_step=step
            )
            obs, _ = train_env.reset()
            ep_ret = 0.0
            ep_len = 0

        if (step > 0) and (step % eval_every == 0):
            video_dir: str = os.path.join("videos", run_name, f"step_{step}")

            eval_env = make_eval_env(
                spec=env_spec,
                seed=seed + 123,
                video_dir=video_dir,
            )
            try:
                eval_ret: float = evaluate(
                    agent=agent,
                    env=eval_env,
                    episodes=eval_episodes,
                )
            finally:
                eval_env.close()

            writer.add_scalar(
                tag="eval/return_mean",
                scalar_value=eval_ret,
                global_step=step,
            )

            timestamp: str = datetime.now().strftime(format="%H-%M-%S")
            ckpt_file: str = os.path.join(ckpt_path, f"step_{step}_{timestamp}.pt")
            agent.save(path=ckpt_file)

            if eval_ret > best_eval:
                best_eval = eval_ret
                timestamp = datetime.now().strftime(format="%H-%M-%S")
                agent.save(path=os.path.join(ckpt_path, f"best_{timestamp}.pt"))

            eval_env.close()
            writer.close()

    train_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dqn.yaml")
    args: Namespace = parser.parse_args()
    main(config_path=args.config)
