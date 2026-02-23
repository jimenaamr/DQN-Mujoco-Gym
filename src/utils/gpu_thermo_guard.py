#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

try:
    import pynvml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: pynvml. Install with: pip install nvidia-ml-py3"
    ) from e


@dataclass(frozen=True)
class GuardConfig:
    """Configuration for GPU thermal guard based on power limiting."""

    gpu_index: int
    poll_s: float
    high_c: int
    low_c: int
    power_normal_w: int
    power_throttle_w: int
    throttle_min_s: float
    require_sudo: bool


def _require_cmd(name: str) -> str:
    """Return path to a required executable or exit with a clear error."""
    path: str | None = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required command not found in PATH: {name}")
    return path


def _run_cmd(args: list[str]) -> None:
    """Run a command and raise a helpful error if it fails."""
    try:
        subprocess.run(
            args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        stderr: str = e.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{stderr}") from e


def _get_gpu_temp_c(gpu_index: int) -> int:
    """Read current GPU temperature in Celsius using NVML."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        temp: int = int(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        )
        return temp
    finally:
        pynvml.nvmlShutdown()


def _set_power_limit_w(
    nvidia_smi_path: str, gpu_index: int, watts: int, require_sudo: bool
) -> None:
    """Set GPU power limit (W) via nvidia-smi."""
    base_args: list[str] = [
        nvidia_smi_path,
        f"--id={gpu_index}",
        f"--power-limit={watts}",
    ]
    args: list[str] = ["sudo", "-n", *base_args] if require_sudo else base_args
    _run_cmd(args=args)


def _parse_args() -> GuardConfig:
    """Parse CLI arguments into a GuardConfig."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Thermal guard: reduces GPU power limit when temperature is too high, "
            "restores it after cooling down (with hysteresis)."
        )
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0).")
    parser.add_argument(
        "--poll-s", type=float, default=1.0, help="Polling interval in seconds."
    )
    parser.add_argument(
        "--high-c",
        type=int,
        default=84,
        help="Throttle when temp >= this (C).",
    )
    parser.add_argument(
        "--low-c",
        type=int,
        default=78,
        help="Restore normal power when temp <= this (C). Must be < high-c.",
    )
    parser.add_argument(
        "--power-normal-w",
        type=int,
        default=220,
        help="Normal power limit in watts to restore to.",
    )
    parser.add_argument(
        "--power-throttle-w",
        type=int,
        default=160,
        help="Throttled power limit in watts while hot.",
    )
    parser.add_argument(
        "--throttle-min-s",
        type=float,
        default=30.0,
        help="Minimum time (s) to stay throttled once engaged.",
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Do not prefix nvidia-smi commands with sudo -n.",
    )

    ns: argparse.Namespace = parser.parse_args()

    if ns.low_c >= ns.high_c:
        raise ValueError("--low-c must be strictly less than --high-c")
    if ns.power_throttle_w >= ns.power_normal_w:
        raise ValueError("--power-throttle-w must be less than --power-normal-w")
    if ns.poll_s <= 0.0:
        raise ValueError("--poll-s must be > 0")
    if ns.throttle_min_s < 0.0:
        raise ValueError("--throttle-min-s must be >= 0")

    cfg: GuardConfig = GuardConfig(
        gpu_index=int(ns.gpu),
        poll_s=float(ns.poll_s),
        high_c=int(ns.high_c),
        low_c=int(ns.low_c),
        power_normal_w=int(ns.power_normal_w),
        power_throttle_w=int(ns.power_throttle_w),
        throttle_min_s=float(ns.throttle_min_s),
        require_sudo=not bool(ns.no_sudo),
    )
    return cfg


def main() -> int:
    """Run the temperature monitoring loop and apply power limits."""
    nvidia_smi_path: str = _require_cmd(name="nvidia-smi")
    cfg: GuardConfig = _parse_args()

    throttled: bool = False
    throttle_since_s: float = 0.0

    print(
        "gpu_thermo_guard running with:\n"
        f"  gpu_index={cfg.gpu_index}\n"
        f"  high_c={cfg.high_c}, low_c={cfg.low_c}\n"
        f"  power_normal_w={cfg.power_normal_w}, power_throttle_w={cfg.power_throttle_w}\n"
        f"  poll_s={cfg.poll_s}, throttle_min_s={cfg.throttle_min_s}\n"
        f"  require_sudo={cfg.require_sudo}\n"
    )
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            temp_c: int = _get_gpu_temp_c(gpu_index=cfg.gpu_index)
            now_s: float = time.time()

            if not throttled and temp_c >= cfg.high_c:
                _set_power_limit_w(
                    nvidia_smi_path=nvidia_smi_path,
                    gpu_index=cfg.gpu_index,
                    watts=cfg.power_throttle_w,
                    require_sudo=cfg.require_sudo,
                )
                throttled = True
                throttle_since_s = now_s
                print(
                    f"[HOT] temp={temp_c}C >= {cfg.high_c}C -> "
                    f"set power limit {cfg.power_throttle_w}W"
                )

            elif throttled:
                throttled_for_s: float = now_s - throttle_since_s
                can_restore: bool = throttled_for_s >= cfg.throttle_min_s
                if can_restore and temp_c <= cfg.low_c:
                    _set_power_limit_w(
                        nvidia_smi_path=nvidia_smi_path,
                        gpu_index=cfg.gpu_index,
                        watts=cfg.power_normal_w,
                        require_sudo=cfg.require_sudo,
                    )
                    throttled = False
                    print(
                        f"[COOL] temp={temp_c}C <= {cfg.low_c}C and "
                        f"throttled_for={throttled_for_s:.1f}s -> "
                        f"restore power limit {cfg.power_normal_w}W"
                    )
                else:
                    state: str = "throttled"
                    print(
                        f"[{state}] temp={temp_c}C, throttled_for={throttled_for_s:.1f}s"
                    )
            else:
                print(f"[ok] temp={temp_c}C")

            time.sleep(cfg.poll_s)

    except KeyboardInterrupt:
        print("\nStopping...")
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
