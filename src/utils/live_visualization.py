# src/utils/live_visualization.py

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_FRAME_RE: re.Pattern[str] = re.compile(pattern=r"^frame_(\d+)\.jpg$")

LISTEN_HZ: float = 30.0
MAX_FRAMES_PER_EPISODE: int = 200_000

LIVE_CONFIG_PATH: Path = Path("configs/live.yaml")
DEFAULT_LIVE_DT: float = 0.01


@dataclass(frozen=True)
class TrainProcInfo:
    pid: int
    created_at_epoch_s: float


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write a small UTF-8 text file.

    Args:
        path: Destination path.
        text: File content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_live_dt() -> float:
    """
    Load live_dt from configs/live.yaml.

    Returns:
        live_dt if available, otherwise DEFAULT_LIVE_DT.
    """
    if not LIVE_CONFIG_PATH.exists():
        return DEFAULT_LIVE_DT

    try:
        cfg: dict[str, Any] = yaml.safe_load(_read_text(LIVE_CONFIG_PATH)) or {}
        return float(cfg.get("live_dt", DEFAULT_LIVE_DT))
    except Exception:
        return DEFAULT_LIVE_DT


def _load_train_proc_info(run_path: Path) -> TrainProcInfo | None:
    meta_file: Path = run_path / "train_meta.yaml"
    pid_file: Path = run_path / "train.pid"

    if meta_file.exists():
        try:
            meta: dict[str, Any] = yaml.safe_load(_read_text(meta_file)) or {}
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
    return Path(f"/proc/{int(pid)}").exists()


def _linux_proc_start_epoch_s(pid: int) -> float | None:
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
    if not _proc_exists(info.pid):
        return False
    if info.created_at_epoch_s <= 0.0:
        return True
    start = _linux_proc_start_epoch_s(info.pid)
    if start is None:
        return True
    return start >= (info.created_at_epoch_s - 5.0)


def _resolve_run_path(arg: str) -> Path:
    p: Path = Path(arg).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_dir():
        raise NotADirectoryError(str(p))
    return p


def _ensure_cv2_qt_fonts_dir() -> None:
    spec = importlib.util.find_spec("cv2")
    if spec is None or spec.origin is None:
        return

    cv2_pkg_dir: Path = Path(spec.origin).resolve().parent
    fonts_dir: Path = cv2_pkg_dir / "qt" / "fonts"

    try:
        fonts_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    if any(fonts_dir.glob("*.ttf")):
        return

    candidates = [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
        Path("/usr/share/fonts/truetype"),
    ]

    for base in candidates:
        if not base.exists():
            continue
        for name in [
            "DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf",
            "DejaVuSansMono.ttf",
            "DejaVuSerif.ttf",
        ]:
            src: Path = base / name
            if src.exists():
                try:
                    (fonts_dir / name).symlink_to(src)
                except Exception:
                    try:
                        (fonts_dir / name).write_bytes(src.read_bytes())
                    except Exception:
                        pass
        break


def _read_current_episode(path: Path) -> int | None:
    try:
        if not path.exists():
            return None
        return int(_read_text(path).strip())
    except Exception:
        return None


def _frame_path(ep_dir: Path, idx: int) -> Path:
    return ep_dir / f"frame_{int(idx):06d}.jpg"


def _infer_step(frame_path: Path) -> int | None:
    m: re.Match[str] | None = _FRAME_RE.match(frame_path.name)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _wait_stable(path: Path) -> bool:
    last: int | None = None
    for _ in range(10):
        try:
            size: int = path.stat().st_size
        except Exception:
            return False
        if last is not None and size == last and size > 0:
            return True
        last = size
        time.sleep(0.01)
    return False


def main(run_path_str: str) -> None:
    run_path: Path = _resolve_run_path(run_path_str)

    deadline: float = time.time() + 10.0
    info: TrainProcInfo | None = None
    while time.time() < deadline:
        info = _load_train_proc_info(run_path)
        if info and _is_correct_train_process(info):
            break
        time.sleep(0.05)

    if info is None or not _is_correct_train_process(info):
        raise RuntimeError("No valid training process found.")

    _ensure_cv2_qt_fonts_dir()
    import cv2

    # Silence OpenCV warnings like "findDecoder imread_ ... can't open/read file".
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        try:
            cv2.setLogLevel(3)  # 3 == LOG_LEVEL_ERROR on many builds
        except Exception:
            pass

    def _window_is_alive(name: str) -> bool:
        """Return True if an OpenCV window still exists (Qt/X close safe)."""
        try:
            visible: float = float(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE))
            return visible > 0.0
        except Exception:
            return False

    live_dt: float = _load_live_dt()
    poll_dt_s: float = 1.0 / LISTEN_HZ

    live_dir: Path = run_path / "live_frames"
    current_ep_file: Path = live_dir / "current_episode.txt"
    cursor_file: Path = live_dir / "visualization_cursor.txt"

    window_name: str = "train_visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    mode: str = "WAIT"
    record_ep: int | None = None
    record_ep_dir: Path | None = None
    next_idx: int = 0
    buffer: list[Path] = []
    wait_from: int | None = None

    try:
        cur_ep: int | None = _read_current_episode(current_ep_file)
        if cur_ep is not None:
            record_ep = cur_ep
            record_ep_dir = live_dir / f"ep_{record_ep:06d}"
            mode = "RECORD"
        else:
            mode = "WAIT"

        while _is_correct_train_process(info):
            if not _window_is_alive(window_name):
                break

            cur_ep = _read_current_episode(current_ep_file)

            key: int = int(cv2.waitKey(1)) & 0xFF
            if key == 27 or key == ord("q"):
                break

            if mode == "WAIT":
                if cur_ep is None:
                    time.sleep(poll_dt_s)
                    continue
                if wait_from is None:
                    wait_from = cur_ep
                    time.sleep(poll_dt_s)
                    continue
                if cur_ep == wait_from:
                    time.sleep(poll_dt_s)
                    continue
                record_ep = cur_ep
                record_ep_dir = live_dir / f"ep_{record_ep:06d}"
                buffer = []
                next_idx = 0
                wait_from = None
                mode = "RECORD"
                continue

            if mode == "RECORD":
                if record_ep_dir is None or not record_ep_dir.exists():
                    # Treat removed episode dir like "episode finished displaying":
                    # reset state and wait for the next episode.
                    wait_from = cur_ep
                    buffer = []
                    record_ep = None
                    record_ep_dir = None
                    next_idx = 0
                    mode = "WAIT"
                    continue

                while len(buffer) < MAX_FRAMES_PER_EPISODE:
                    fp: Path = _frame_path(record_ep_dir, next_idx)
                    if not fp.exists():
                        break
                    if not _wait_stable(fp):
                        break
                    buffer.append(fp)
                    next_idx += 1

                if cur_ep is not None and cur_ep != record_ep:
                    mode = "PLAYBACK"
                else:
                    time.sleep(poll_dt_s)
                continue

            if mode == "PLAYBACK":
                if record_ep is None:
                    mode = "WAIT"
                    continue

                for i, fp in enumerate(buffer):
                    if not _window_is_alive(window_name):
                        return

                    img = cv2.imread(str(fp))
                    if img is None:
                        continue
                    step = _infer_step(fp)
                    if step is None:
                        step = i

                    try:
                        cv2.setWindowTitle(
                            window_name,
                            f"episode={record_ep} step={step}",
                        )
                    except Exception:
                        return

                    # Write cursor for other tools to sync precisely.
                    _atomic_write_text(
                        cursor_file,
                        f"episode={int(record_ep)}\nframe={int(step)}\n",
                    )

                    cv2.imshow(window_name, img)

                    key2: int = int(cv2.waitKey(1)) & 0xFF
                    if key2 == 27 or key2 == ord("q"):
                        return

                    time.sleep(live_dt)

                cur_ep_now = _read_current_episode(current_ep_file)
                wait_from = cur_ep_now
                buffer = []
                record_ep = None
                record_ep_dir = None
                next_idx = 0
                mode = "WAIT"
                continue

            time.sleep(poll_dt_s)

    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str)
    args = parser.parse_args()
    main(args.run_path)
