from __future__ import annotations

import os
import socket
import subprocess
import sys
import time


def _is_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    """Return True if something is accepting TCP connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex((host, port)) == 0


def launch_tensorboard(
    run_dir: str,
    run_name: str,
    tb_log_path: str,
    *,
    host: str = "127.0.0.1",
    base_port: int = 6006,
    max_tries: int = 50,
    startup_timeout_s: float = 3.0,
) -> subprocess.Popen | None:
    """Launch TensorBoard on the first available port starting from base_port.

    Args:
        run_dir: Root directory containing all runs (TensorBoard logdir).
        run_name: Current run name (for display only).
        tb_log_path: File path to write TensorBoard stdout/stderr.
        host: Host used to probe port availability.
        base_port: First port to try (default 6006). Can be overridden via
            TENSORBOARD_PORT env var if it parses as an int.
        max_tries: Maximum number of consecutive ports to try.
        startup_timeout_s: Time to wait for TensorBoard to start listening.

    Returns:
        The Popen handle if started, else None.
    """
    port_env: str = os.environ.get("TENSORBOARD_PORT", "")
    if port_env:
        try:
            base_port = int(port_env)
        except ValueError:
            base_port = 6006

    for port in range(base_port, base_port + max_tries):
        # Fast pre-check: skip ports already in use.
        if _is_listening(host=host, port=port):
            continue

        cmd: list[str] = [
            str(sys.executable),
            "-m",
            "tensorboard.main",
            "--logdir",
            str(run_dir),
            "--port",
            str(port),
            "--bind_all",
        ]

        tb_log_file = open(file=tb_log_path, mode="w", encoding="utf-8")

        try:
            proc: subprocess.Popen = subprocess.Popen(
                args=cmd,
                stdout=tb_log_file,
                stderr=subprocess.STDOUT,
                env=dict(os.environ),
                start_new_session=True,
            )
        except Exception:
            tb_log_file.close()
            continue

        deadline_s: float = time.time() + startup_timeout_s
        started: bool = False
        while time.time() < deadline_s:
            if proc.poll() is not None:
                break
            if _is_listening(host=host, port=port):
                started = True
                break
            time.sleep(0.05)

        if started:
            url: str = f"http://localhost:{port}/"
            print(f"[train] TensorBoard: {url} (run: {run_name})", flush=True)
            return proc

        # Not started on this port: clean up and try the next.
        try:
            proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=0.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        try:
            tb_log_file.close()
        except Exception:
            pass

    print(
        f"[train] WARNING: TensorBoard failed to start after trying ports "
        f"{base_port}..{base_port + max_tries - 1}. See log: {tb_log_path}",
        flush=True,
    )
    return None
