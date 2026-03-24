
# src/rdqn_jimena/train.py

from __future__ import annotations
import os
from src.utils.backends import auto_detect_mujoco_gl

# Configuración de gráficos para MuJoCo antes de importar gymnasium
graphics_backend: str = auto_detect_mujoco_gl()
os.environ.setdefault(key="MUJOCO_GL", value=graphics_backend)

import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.rdqn_jimena.env import EnvSpec, make_env
from src.rdqn_jimena.monitoring import MONITOR
from src.rdqn_jimena.rdqn import RDQNAgent, RDQNConfig

LIVE_CLEANUP_TIMER: float = 120.0
_STEP_RE: re.Pattern[str] = re.compile(pattern=r"^step_(\d+)_.*\.pt$")
_STEP_ANYWHERE_RE: re.Pattern[str] = re.compile(pattern=r"step_(\d+)_.*\.pt$")


def resolve_device(name: str) -> str:
    """Resolve a user-facing device option into a torch device string.

    Args:
        name: User-provided device name. Allowed: "cpu", "gpu".

    Returns:
        "cpu" or "cuda" (if available) depending on `name`.

    Raises:
        ValueError: If `name` is not one of {"cpu", "gpu"}.
    """

    normalized: str = str(name).strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "gpu":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Invalid device '{name}'. Use 'cpu' or 'gpu'.")


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
    device_resolved = resolve_device(name=str(cfg.get("device", "cpu")))

    return RDQNConfig(
        gamma=float(t["gamma"]),
        lr=float(t["lr"]),
        batch_size=int(t["batch_size"]),
        buffer_size=int(t["buffer_size"]),
        learning_starts=int(t["learning_starts"]),
        train_freq=int(t["train_freq"]),
        target_update_freq=int(t["target_update_freq"]),
        grad_clip_norm=float(t["grad_clip_norm"]),
        device=str(device_resolved),
        noisy_sigma0=float(r["noisy_sigma0"]),
        n_step=int(r["n_step"]),
        prio_alpha=float(r["prio_alpha"]),
        prio_beta_start=float(r["prio_beta_start"]),
        prio_beta_end=float(r["prio_beta_end"]),
        prio_beta_steps=int(r["prio_beta_steps"]),
        prio_eps=float(r["prio_eps"]),
        v_min=float(r["v_min"]),
        v_max=float(r["v_max"]),
        n_atoms=int(r["n_atoms"]),
    )


def resolve_resume_path(resume: str | None) -> Path | None:
    if resume is None:
        return None
    p = Path(resume).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        ckpt_files = list(p.glob("*.pt"))
        if not ckpt_files:
            return None
        return max(ckpt_files, key=os.path.getmtime)
    return None


def _infer_run_name_from_checkpoint(ckpt_file: Path) -> str | None:
    parent = ckpt_file.parent
    return str(parent.name) if parent.name else None


# --- FUNCIONES AUXILIARES ---


def _ensure_rdqn_subdir(root: str) -> str:
    p: Path = Path(root)
    return str(p) if "rdqn" in p.parts else str(p / "rdqn")


def _run_eval_in_subprocess(config_path, checkpoint_path, seed, episodes, video_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "eval_return.txt")
        cmd = [
            "python",
            "-m",
            "src.rdqn_jimena.eval_worker",
            "--config",
            config_path,
            "--checkpoint",
            checkpoint_path,
            "--seed",
            str(seed),
            "--episodes",
            str(episodes),
            "--video-dir",
            video_dir,
            "--output-path",
            out_path,
        ]
        env = dict(os.environ)
        env["DQN_MONITOR_DISABLED"] = "1"
        subprocess.run(cmd, check=True, env=env)
        with open(out_path, "r") as f:
            return float(f.read().strip())


# --- MAIN ---


def main(
    config_path: str, resume_path: str | None = None, new_run: bool = False
) -> None:
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])

    env_spec = to_env_spec(cfg)
    # Se elimina la dependencia de stabilizer_config_from_yaml
    train_env = make_env(spec=env_spec, seed=seed, stabilizer=None)

    obs_shape = tuple(train_env.observation_space.shape)
    n_actions = int(train_env.action_space.n)

    rdqn_cfg = to_rdqn_cfg(cfg)
    agent = RDQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=rdqn_cfg)

    # Asegurar nombres de redes correctos para RDQNAgent optimizado
    agent.q.train()
    agent.q_target.eval()

    run_dir = _ensure_rdqn_subdir(str(cfg["logging"]["run_dir"]))
    ckpt_dir = _ensure_rdqn_subdir(str(cfg["logging"]["ckpt_dir"]))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_ckpt = resolve_resume_path(resume_path)
    if resume_ckpt and not new_run:
        run_name = (
            _infer_run_name_from_checkpoint(resume_ckpt) or f"{env_spec.env_id}_resume"
        )
    else:
        run_name = f"{env_spec.env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_path = os.path.join(run_dir, run_name)
    ckpt_path = os.path.join(ckpt_dir, run_name)
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    writer = SummaryWriter(log_dir=run_path)
    MONITOR.set_run_path(Path(run_path))

    total_steps = int(cfg["train"]["total_steps"])
    log_every = int(cfg["logging"]["log_every"])
    eval_every = int(cfg["eval"]["every_steps"])

    start_step = 0
    if resume_ckpt:
        start_step = agent.load(str(resume_ckpt))
        if new_run:
            start_step = 0
            agent.global_step = 0

    obs, _ = train_env.reset()
    ep_ret, ep_len = 0.0, 0
    best_eval = -1e18
    episode_idx = 0

    for step in trange(start_step, total_steps, desc="Rainbow Training"):
        agent.global_step = step

        action = agent.act(obs, eval_mode=False)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)

        ep_ret += reward
        ep_len += 1
        obs = next_obs

        if agent.can_update():
            metrics = agent.update()
            if step % log_every == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)

        if done:
            writer.add_scalar("train/episode_return", ep_ret, step)
            writer.add_scalar("train/episode_length", ep_len, step)
            obs, _ = train_env.reset()
            ep_ret, ep_len = 0.0, 0
            episode_idx += 1

        if step > 0 and step % eval_every == 0:
            ckpt_file = os.path.join(ckpt_path, f"step_{step}.pt")
            agent.save(ckpt_file)

            video_dir = os.path.join("videos", "rdqn", run_name, f"step_{step}")
            eval_ret = _run_eval_in_subprocess(
                config_path,
                ckpt_file,
                seed + 100,
                int(cfg["eval"]["episodes"]),
                video_dir,
            )

            writer.add_scalar("eval/return_mean", eval_ret, step)
            if eval_ret > best_eval:
                best_eval = eval_ret
                agent.save(os.path.join(ckpt_path, "best_model.pt"))

    train_env.close()
    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rdqn-jimena.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--new-run", action="store_true")
    args = parser.parse_args()
    main(args.config, args.resume, args.new_run)
