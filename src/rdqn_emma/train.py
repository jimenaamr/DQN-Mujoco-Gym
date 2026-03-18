# src/rdqn/train.py

from __future__ import annotations
import os
from src.utils.backends import auto_detect_mujoco_gl

# Configuración de gráficos para MuJoCo antes de importar gymnasium
graphics_backend: str = auto_detect_mujoco_gl()
os.environ.setdefault(key="MUJOCO_GL", value=graphics_backend)

import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from argparse import Namespace
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
import torch # Añadido para verificaciones de dispositivo
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.rdqn.env import EnvSpec, make_env
from src.rdqn.helper import (
    StabilizerConfig,
    resolve_device,
    stabilizer_config_from_yaml,
)
from src.rdqn.monitoring import MONITOR
from src.rdqn.rdqn import RDQNAgent, RDQNConfig # Importa la clase optimizada
from src.utils.tensorboard import launch_tensorboard

LIVE_CLEANUP_TIMER: float = 120.0
_STEP_RE: re.Pattern[str] = re.compile(pattern=r"^step_(\d+)_.*\.pt$")
_STEP_ANYWHERE_RE: re.Pattern[str] = re.compile(pattern=r"step_(\d+)_.*\.pt$")

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
    """Mapea el YAML a la RDQNConfig optimizada."""
    t: dict[str, Any] = cfg["train"]
    r: dict[str, Any] = cfg["rainbow"]
    device_resolved: str = resolve_device(name=str(cfg.get("device", "cpu")))

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
        # Rainbow specific - nombres sincronizados con el código optimizado
        noisy_sigma0=float(r["noisy_sigma0"]),
        n_step=int(r["n_step"]),
        prio_alpha=float(r["prio_alpha"]),
        prio_beta_start=float(r["prio_beta_start"]),
        prio_beta_end=float(r["prio_beta_end"]),
        prio_beta_steps=int(r["prio_beta_steps"]),
        prio_eps=float(r["prio_eps"]),
        v_min=float(r["v_min"]),
        v_max=float(r["v_max"]),
        n_atoms=int(r["n_atoms"])
    )

# ... (Mantenemos las funciones auxiliares de guardado de imágenes y frames igual) ...
def _ensure_rdqn_subdir(root: str) -> str:
    p: Path = Path(root)
    return str(p) if "rdqn" in p.parts else str(p / "rdqn")

def _extract_last_rgb_frame(obs: np.ndarray) -> np.ndarray:
    c = int(obs.shape[0])
    if c >= 3:
        return np.transpose(obs[c-3:c, :, :], (1, 2, 0))
    gray = obs[c-1, :, :]
    return np.stack((gray, gray, gray), axis=-1)

def _write_jpg_atomic(path: Path, frame_rgb: np.ndarray, quality: int = 90) -> None:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        tmp = path.with_suffix(".jpg.tmp")
        tmp.write_bytes(buf.tobytes())
        tmp.replace(path)

def _run_eval_in_subprocess(config_path, checkpoint_path, seed, episodes, video_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "eval_return.txt")
        cmd = ["python", "-m", "src.rdqn.eval_worker", "--config", config_path, 
               "--checkpoint", checkpoint_path, "--seed", str(seed), 
               "--episodes", str(episodes), "--video-dir", video_dir, "--output-path", out_path]
        env = dict(os.environ)
        env["DQN_MONITOR_DISABLED"] = "1"
        subprocess.run(cmd, check=True, env=env)
        with open(out_path, "r") as f:
            return float(f.read().strip())

def main(config_path: str, resume_path: str | None = None, new_run: bool = False) -> None:
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    
    env_spec = to_env_spec(cfg)
    stabilizer_cfg = stabilizer_config_from_yaml(cfg)
    train_env = make_env(spec=env_spec, seed=seed, stabilizer=stabilizer_cfg)
    
    obs_shape = tuple(train_env.observation_space.shape) # (C, H, W)
    n_actions = int(train_env.action_space.n)
    
    rdqn_cfg = to_rdqn_cfg(cfg)
    agent = RDQNAgent(obs_shape=obs_shape, n_actions=n_actions, cfg=rdqn_cfg)

    # --- CORRECCIÓN DE NOMBRES DE REDES ---
    agent.q.train()
    agent.q_target.eval() # Cambiado de q_targ a q_target para RDQN optimizado
    # --------------------------------------

    # Gestión de directorios y nombres de ejecución
    run_dir = _ensure_rdqn_subdir(str(cfg["logging"]["run_dir"]))
    ckpt_dir = _ensure_rdqn_subdir(str(cfg["logging"]["ckpt_dir"]))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Lógica de Resume corregida
    from src.rdqn.train import resolve_resume_path, _infer_run_name_from_checkpoint
    resume_ckpt = resolve_resume_path(resume_path)
    if resume_ckpt and not new_run:
        run_name = _infer_run_name_from_checkpoint(resume_ckpt) or f"{env_spec.env_id}_resume"
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

    # Cargar checkpoint si existe
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

    # Bucle de entrenamiento principal
    for step in trange(start_step, total_steps, desc="Rainbow Training"):
        agent.global_step = step
        
        # Selección de acción (Noisy Net maneja exploración)
        action = agent.act(obs, eval_mode=False)
        
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        
        # Guardar en buffer (n-step y PER se gestionan dentro del agente/buffer)
        agent.store(obs, action, reward, next_obs, done)
        
        ep_ret += reward
        ep_len += 1
        obs = next_obs

        # Actualización del modelo
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

        # Evaluación periódica
        if step > 0 and step % eval_every == 0:
            ckpt_file = os.path.join(ckpt_path, f"step_{step}.pt")
            agent.save(ckpt_file)
            
            video_dir = os.path.join("videos", "rdqn", run_name, f"step_{step}")
            eval_ret = _run_eval_in_subprocess(config_path, ckpt_file, seed+100, int(cfg["eval"]["episodes"]), video_dir)
            
            writer.add_scalar("eval/return_mean", eval_ret, step)
            if eval_ret > best_eval:
                best_eval = eval_ret
                agent.save(os.path.join(ckpt_path, "best_model.pt"))

    train_env.close()
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rdqn.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--new-run", action="store_true")
    args = parser.parse_args()
    main(args.config, args.resume, args.new_run)