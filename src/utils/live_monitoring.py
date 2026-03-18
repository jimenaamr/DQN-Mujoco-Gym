# src/utils/live_monitoring.py

from __future__ import annotations

import argparse
import os
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LISTEN_HZ: float = 30.0


@dataclass(frozen=True)
class TrainProcInfo:
    """Training process identification info."""

    pid: int
    created_at_epoch_s: float


def _read_text(path: Path) -> str:
    """Read UTF-8 text from a file.

    Args:
        path: File path.

    Returns:
        File content.
    """
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file if it exists, else return empty dict.

    Args:
        path: YAML path.

    Returns:
        Loaded dict or empty dict.
    """
    try:
        if not path.exists():
            return {}
        data: Any = yaml.safe_load(_read_text(path))
        if isinstance(data, dict):
            return dict(data)
        return {}
    except Exception:
        return {}


def _resolve_run_path(arg: str) -> Path:
    """Resolve and validate a run directory path.

    Args:
        arg: User-provided path.

    Returns:
        Resolved directory path.

    Raises:
        FileNotFoundError: If path doesn't exist.
        NotADirectoryError: If path isn't a directory.
    """
    p: Path = Path(arg).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_dir():
        raise NotADirectoryError(str(p))
    return p


def _load_train_proc_info(run_path: Path) -> TrainProcInfo | None:
    """Load training process info from run metadata files.

    Args:
        run_path: Run directory.

    Returns:
        TrainProcInfo or None if missing/invalid.
    """
    meta_file: Path = run_path / "train_meta.yaml"
    pid_file: Path = run_path / "train.pid"

    if meta_file.exists():
        try:
            meta: dict[str, Any] = _load_yaml(path=meta_file)
            return TrainProcInfo(
                pid=int(meta["pid"]),
                created_at_epoch_s=float(meta["created_at_epoch_s"]),
            )
        except Exception:
            return None

    if pid_file.exists():
        try:
            return TrainProcInfo(
                pid=int(_read_text(pid_file).strip()),
                created_at_epoch_s=0.0,
            )
        except Exception:
            return None

    return None


def _proc_exists(pid: int) -> bool:
    """Check whether a Linux PID exists.

    Args:
        pid: Process id.

    Returns:
        True if /proc/<pid> exists.
    """
    return Path(f"/proc/{int(pid)}").exists()


def _linux_proc_start_epoch_s(pid: int) -> float | None:
    """Read Linux process start time in epoch seconds.

    Args:
        pid: Process id.

    Returns:
        Start time in epoch seconds, or None on failure.
    """
    stat_path: Path = Path(f"/proc/{int(pid)}/stat")
    if not stat_path.exists():
        return None

    try:
        parts: list[str] = _read_text(stat_path).split()
        start_ticks: int = int(parts[21])
        clk_tck: int = int(os.sysconf("SC_CLK_TCK"))

        btime: int | None = None
        for line in _read_text(Path("/proc/stat")).splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
                break
        if btime is None:
            return None

        return float(btime) + float(start_ticks) / float(clk_tck)
    except Exception:
        return None


def _is_correct_train_process(info: TrainProcInfo) -> bool:
    """Check that the PID still exists and (optionally) matches creation time.

    Args:
        info: Training process info.

    Returns:
        True if process appears valid.
    """
    if not _proc_exists(info.pid):
        return False
    if info.created_at_epoch_s <= 0.0:
        return True
    start = _linux_proc_start_epoch_s(info.pid)
    if start is None:
        return True
    return start >= (info.created_at_epoch_s - 5.0)


def _monitor_mode() -> str:
    """Read live monitor mode from configs/live.yaml.

    Expected:
        monitor: "visualization"  # "visualization" / "training"

    Returns:
        "visualization" or "training" (defaults to "visualization").
    """
    cfg_file: Path = Path("configs") / "live.yaml"
    cfg: dict[str, Any] = _load_yaml(path=cfg_file)
    mode_raw: str = str(cfg.get("monitor", "visualization")).strip().lower()
    if mode_raw not in ("visualization", "training"):
        return "visualization"
    return mode_raw


def _read_monitor_text_training(run_path: Path) -> str:
    """Read monitoring text for training-sync mode.

    Args:
        run_path: Run directory.

    Returns:
        Text content.
    """
    monitor_file: Path = run_path / "live_monitoring" / "monitor.txt"
    try:
        if not monitor_file.exists():
            return ""
        return _read_text(monitor_file).rstrip("\n")
    except Exception:
        return ""


def _parse_visualization_cursor(text: str) -> tuple[int, int] | None:
    """Parse visualization cursor file.

    Format:
        episode=<int>
        frame=<int>

    Args:
        text: File content.

    Returns:
        (episode, frame) or None if invalid.
    """
    ep: int | None = None
    fr: int | None = None
    for line in text.splitlines():
        s: str = line.strip()
        if s.startswith("episode="):
            try:
                ep = int(s.split("=", 1)[1])
            except Exception:
                ep = None
        elif s.startswith("frame="):
            try:
                fr = int(s.split("=", 1)[1])
            except Exception:
                fr = None

    if ep is None or fr is None:
        return None
    return (ep, fr)


def _read_visualization_cursor(run_path: Path) -> tuple[int, int] | None:
    """Read episode/frame currently displayed by live_visualization.

    Args:
        run_path: Run directory.

    Returns:
        (episode, frame) or None.
    """
    cursor_file: Path = run_path / "live_frames" / "visualization_cursor.txt"
    try:
        if not cursor_file.exists():
            return None
        parsed = _parse_visualization_cursor(_read_text(cursor_file))
        return parsed
    except Exception:
        return None


def _read_metrics_for_frame(run_path: Path, episode: int, frame_idx: int) -> str:
    """Read metrics snapshot for a given episode/frame.

    Expects:
        live_frames/ep_XXXXXX/metrics_000123.txt

    Args:
        run_path: Run directory.
        episode: Episode index.
        frame_idx: Frame index.

    Returns:
        Text content (empty if missing).
    """
    ep_dir: Path = run_path / "live_frames" / f"ep_{int(episode):06d}"
    p: Path = ep_dir / f"metrics_{int(frame_idx):06d}.txt"
    try:
        if not p.exists():
            return ""
        return _read_text(p).rstrip("\n")
    except Exception:
        return ""


def _read_current_episode(run_path: Path) -> int | None:
    """Read current episode index from live_frames/current_episode.txt.

    Args:
        run_path: Run directory.

    Returns:
        Episode index if available.
    """
    p: Path = run_path / "live_frames" / "current_episode.txt"
    try:
        if not p.exists():
            return None
        return int(_read_text(p).strip())
    except Exception:
        return None


def _find_latest_episode_dir(run_path: Path) -> Path | None:
    """Find the latest ep_XXXXXX directory in live_frames.

    Args:
        run_path: Run directory.

    Returns:
        Episode directory or None.
    """
    base: Path = run_path / "live_frames"
    if not base.exists():
        return None

    best: Path | None = None
    best_num: int = -1
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name: str = p.name
        if not name.startswith("ep_"):
            continue
        suffix: str = name[len("ep_") :]
        if not suffix.isdigit():
            continue
        n: int = int(suffix)
        if n > best_num:
            best_num = n
            best = p
    return best


def _max_frame_index(ep_dir: Path) -> int:
    """Return the maximum frame index for frame_*.jpg in an episode dir.

    Args:
        ep_dir: Episode directory.

    Returns:
        Max frame index, or -1 if none.
    """
    best: int = -1
    try:
        for p in ep_dir.iterdir():
            if not p.is_file():
                continue
            name: str = p.name
            if not (name.startswith("frame_") and name.endswith(".jpg")):
                continue
            mid: str = name[len("frame_") : -len(".jpg")]
            if not mid.isdigit():
                continue
            best = max(best, int(mid))
    except Exception:
        return -1
    return best


def _read_metrics_for_frame_fallback(ep_dir: Path, frame_idx: int) -> str:
    """Fallback: read metrics for ep_dir + frame_idx without cursor.

    Args:
        ep_dir: Episode directory.
        frame_idx: Frame index.

    Returns:
        Text content (empty if missing).
    """
    p: Path = ep_dir / f"metrics_{int(frame_idx):06d}.txt"
    try:
        if not p.exists():
            return ""
        return _read_text(p).rstrip("\n")
    except Exception:
        return ""


def main(run_path_str: str) -> None:
    run_path: Path = _resolve_run_path(arg=run_path_str)

    deadline: float = time.time() + 10.0
    info: TrainProcInfo | None = None
    while time.time() < deadline:
        info = _load_train_proc_info(run_path=run_path)
        if info and _is_correct_train_process(info):
            break
        time.sleep(0.05)

    if info is None or not _is_correct_train_process(info):
        raise RuntimeError("No valid training process found.")

    mode: str = _monitor_mode()
    poll_dt_s: float = 1.0 / float(LISTEN_HZ)

    root: tk.Tk = tk.Tk()
    root.title(f"live_monitoring ({mode})")

    text_label: tk.Label = tk.Label(
        root,
        justify="left",
        anchor="nw",
        font=("TkFixedFont", 11),
        padx=12,
        pady=10,
    )
    text_label.pack(side="top", fill="both", expand=True)

    closing: bool = False

    def _ui_alive() -> bool:
        """Return True if the Tk UI and label still exist."""
        if closing:
            return False
        try:
            if int(root.winfo_exists()) != 1:
                return False
            if int(text_label.winfo_exists()) != 1:
                return False
            return True
        except Exception:
            return False

    def _safe_set_text(text: str) -> bool:
        """Best-effort label update.

        Args:
            text: New label text.

        Returns:
            True if applied, False if UI is already closed.
        """
        if not _ui_alive():
            return False
        try:
            text_label.configure(text=text)
            return True
        except tk.TclError:
            return False

    def _on_close() -> None:
        nonlocal closing
        closing = True
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.minsize(width=420, height=260)

    last_text: str = ""
    last_cursor: tuple[int, int] | None = None

    try:
        if mode == "training":
            while _is_correct_train_process(info):
                try:
                    root.update_idletasks()
                    root.update()
                except tk.TclError:
                    return

                if not _ui_alive():
                    return

                txt: str = _read_monitor_text_training(run_path=run_path)
                if txt != last_text:
                    if not _safe_set_text(text=txt):
                        return
                    last_text = txt

                time.sleep(poll_dt_s)
            return

        # mode == "visualization": hard-sync to live_visualization cursor when present
        while _is_correct_train_process(info):
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                return

            if not _ui_alive():
                return

            cursor: tuple[int, int] | None = _read_visualization_cursor(
                run_path=run_path
            )
            if cursor is not None:
                if cursor != last_cursor:
                    ep, fr = cursor
                    txt = _read_metrics_for_frame(
                        run_path=run_path, episode=int(ep), frame_idx=int(fr)
                    )
                    if txt != "" and txt != last_text:
                        if not _safe_set_text(text=txt):
                            return
                        last_text = txt
                    last_cursor = cursor

                time.sleep(poll_dt_s)
                continue

            # Fallback if cursor isn't available: best-effort follow live_frames.
            ep: int | None = _read_current_episode(run_path=run_path)
            if ep is None:
                ep_dir: Path | None = _find_latest_episode_dir(run_path=run_path)
                if ep_dir is None:
                    time.sleep(poll_dt_s)
                    continue
            else:
                ep_dir = run_path / "live_frames" / f"ep_{int(ep):06d}"
                if not ep_dir.exists():
                    ep_dir = _find_latest_episode_dir(run_path=run_path)
                    if ep_dir is None:
                        time.sleep(poll_dt_s)
                        continue

            max_idx: int = _max_frame_index(ep_dir=ep_dir)
            if max_idx < 0:
                time.sleep(poll_dt_s)
                continue

            txt2: str = _read_metrics_for_frame_fallback(
                ep_dir=ep_dir, frame_idx=max_idx
            )
            if txt2 != "" and txt2 != last_text:
                if not _safe_set_text(text=txt2):
                    return
                last_text = txt2

            time.sleep(poll_dt_s)

    finally:
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str)
    args = parser.parse_args()
    main(run_path_str=str(args.run_path))
