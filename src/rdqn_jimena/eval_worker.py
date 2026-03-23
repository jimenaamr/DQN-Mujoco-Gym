# src/rdqn/eval_worker.py
# src/rdqn_jimena/eval_worker.py

from __future__ import annotations
import argparse
from typing import Any
import numpy as np
import yaml
import torch

from src.rdqn_jimena.env import EnvSpec, make_eval_env
from src.rdqn_jimena.rdqn import RDQNAgent, RDQNConfig

# Función local para resolver dispositivo y evitar dependencia de helper
def resolve_device(name: str) -> str:
    if name in ["gpu", "cuda"]:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"

def load_yaml(path: str) -> dict[str, Any]:
    with open(file=path, encoding="utf-8") as f:
        return yaml.safe_load(stream=f)

def to_env_spec(cfg: dict[str, Any]) -> EnvSpec:
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
    t: dict[str, Any] = cfg["train"]
    r: dict[str, Any] = cfg["rainbow"]
    device_resolved = resolve_device(str(cfg.get("device", "cpu")))

    return RDQNConfig(
        gamma=float(t["gamma"]), lr=float(t["lr"]),
        batch_size=int(t["batch_size"]), buffer_size=int(t["buffer_size"]),
        learning_starts=int(t["learning_starts"]), train_freq=int(t["train_freq"]),
        target_update_freq=int(t["target_update_freq"]), grad_clip_norm=float(t["grad_clip_norm"]),
        device=str(device_resolved), noisy_sigma0=float(r["noisy_sigma0"]),
        n_step=int(r["n_step"]), prio_alpha=float(r["prio_alpha"]),
        prio_beta_start=float(r["prio_beta_start"]), prio_beta_end=float(r["prio_beta_end"]),
        prio_beta_steps=int(r["prio_beta_steps"]), prio_eps=float(r["prio_eps"]),
        v_min=float(r["v_min"]), v_max=float(r["v_max"]), n_atoms=int(r["n_atoms"]),
    )

def run_eval(config_path: str, checkpoint_path: str, seed: int, episodes: int, video_dir: str) -> float:
    cfg = load_yaml(path=config_path)
    env_spec = to_env_spec(cfg=cfg)
    env = make_eval_env(spec=env_spec, seed=int(seed), video_dir=str(video_dir))

    obs_shape_raw = env.observation_space.shape
    obs_shape = (int(obs_shape_raw[0]), int(obs_shape_raw[1]), int(obs_shape_raw[2]))
    n_actions = int(env.action_space.n)

    rdqn_cfg = to_rdqn_cfg(cfg=cfg)
    agent = RDQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=rdqn_cfg)

    agent.load(path=str(checkpoint_path))
    agent.q.eval() # IMPORTANTE para Noisy Networks

    returns = []
    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=int(seed + ep))
        done, ep_ret = False, 0.0
        while not done:
            a = agent.act(obs=obs, eval_mode=True)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_ret += float(r)
        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    mean_ret = run_eval(args.config, args.checkpoint, args.seed, args.episodes, args.video_dir)
    with open(args.output_path, "w") as f: f.write(f"{mean_ret}")

if __name__ == "__main__":
    main()