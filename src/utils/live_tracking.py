# src/utils/live_tracking.py

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChildProc:
    """A managed child process."""

    name: str
    proc: subprocess.Popen[str]


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float) -> None:
    """Terminate a process best-effort.

    Args:
        proc: Process to terminate.
        timeout_s: Seconds to wait before killing.
    """
    if proc.poll() is not None:
        return

    try:
        proc.terminate()
        proc.wait(timeout=timeout_s)
        return
    except Exception:
        pass

    try:
        proc.kill()
    except Exception:
        pass


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


def _start_child(module: str, run_path: Path) -> ChildProc:
    """Start a child process (python -m <module> <run_path>).

    Args:
        module: Python module path to execute.
        run_path: Run directory passed as positional argument.

    Returns:
        ChildProc record.
    """
    cmd: list[str] = [str(sys.executable), "-m", module, str(run_path)]
    proc: subprocess.Popen[str] = subprocess.Popen(args=cmd)
    return ChildProc(name=module, proc=proc)


def main(run_path_str: str) -> None:
    """Launch live tracking tools for a run directory.

    Tools:
      - src.utils.live_metrics
      - src.utils.live_visualization
      - src.utils.live_monitoring

    Args:
        run_path_str: Path to the run directory.
    """
    run_path: Path = _resolve_run_path(arg=run_path_str)

    children: list[ChildProc] = [
        _start_child(module="src.utils.live_metrics", run_path=run_path),
        _start_child(module="src.utils.live_visualization", run_path=run_path),
        _start_child(module="src.utils.live_monitoring", run_path=run_path),
    ]

    try:
        while children:
            alive: list[ChildProc] = []
            for c in children:
                if c.proc.poll() is None:
                    alive.append(c)
            children = alive
            time.sleep(0.10)
    except KeyboardInterrupt:
        pass
    finally:
        for c in children:
            _terminate_process(proc=c.proc, timeout_s=2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str)
    args = parser.parse_args()
    main(run_path_str=str(args.run_path))
