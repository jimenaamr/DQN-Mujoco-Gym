# src/rdqn/monitoring.py

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

MonitorValue = Union[int, float, str]


def _write_text_atomic(path: Path, text: str) -> None:
    """Write text to a file atomically.

    Args:
        path: Destination file path.
        text: Text to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _format_float(v: float) -> str:
    """Format a float with up to 3 decimals.

    Notes:
        Uses rounding to 3 decimals, then strips trailing zeros.

    Args:
        v: Float value.

    Returns:
        A compact string with at most 3 decimal digits.
    """
    s: str = f"{float(v):.3f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _format_value(v: MonitorValue) -> str:
    """Format a monitor value for display.

    Args:
        v: Value to format.

    Returns:
        Display string.
    """
    if isinstance(v, float):
        return _format_float(v)
    return str(v)


def _env_monitor_disabled() -> bool:
    """Check whether monitoring is disabled via env var.

    Returns:
        True if DQN_MONITOR_DISABLED is set to a truthy value.
    """
    raw: str = str(os.environ.get("DQN_MONITOR_DISABLED", "")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


@dataclass
class MonitoringState:
    """Mutable monitoring state written by training and read by live_monitoring.

    This is a debugging-oriented key/value store. Training is expected to:

      1) Call begin_step(...) at the beginning of each train loop iteration.
      2) Call add_field(...) to register additional metrics for that step.
      3) Optionally call flush(...) to force-write the current state.

    Live monitoring reads a single text file updated atomically each step.
    """

    _fields: dict[str, MonitorValue] = field(default_factory=dict)
    _out_path: Path | None = None
    _active: bool = True

    def __post_init__(self) -> None:
        """Initialize active state based on environment configuration."""
        if _env_monitor_disabled():
            self._active = False

    def activate(self) -> None:
        """Enable monitoring (add_field/begin_step will write again)."""
        self._active = True

    def deactivate(self) -> None:
        """Disable monitoring (add_field/begin_step become no-ops)."""
        self._active = False

    def set_run_path(self, run_path: Path) -> None:
        """Configure the output path under a run directory.

        Args:
            run_path: Path to the run directory.
        """
        p: Path = Path(run_path).expanduser().resolve()
        self._out_path = p / "live_monitoring" / "monitor.txt"

    def clear(self) -> None:
        """Clear all fields for the next step."""
        self._fields.clear()

    def add_field(self, name: str, value: MonitorValue) -> None:
        """Add a new field.

        Args:
            name: Field name.
            value: Field value.

        Raises:
            AssertionError: If the field name already exists in the state.
        """
        if not self._active:
            return

        key: str = str(name)
        assert key not in self._fields, f"Field already exists: {key}"
        self._fields[key] = value

    def begin_step(self, episode: int, global_step: int, inner_step: int) -> None:
        """Reset fields and seed the standard step identifiers.

        Args:
            episode: Episode index.
            global_step: Global environment step.
            inner_step: Step within the current episode.
        """
        if not self._active:
            return

        self.clear()
        self.add_field(name="episode", value=int(episode))
        self.add_field(name="global_step", value=int(global_step))
        self.add_field(name="inner_step", value=int(inner_step))
        self.flush()

    def to_text(self) -> str:
        """Render the current fields as text lines.

        Returns:
            Multi-line text "name: value" in insertion order.
        """
        lines: list[str] = []
        for k, v in self._fields.items():
            lines.append(f"{k}: {_format_value(v)}")
        return "\n".join(lines) + "\n"

    def flush(self) -> None:
        """Atomically write the current state to disk (if configured)."""
        if not self._active:
            return
        if self._out_path is None:
            return
        _write_text_atomic(path=self._out_path, text=self.to_text())


def add_field(name: str, value: MonitorValue) -> None:
    """Module-level helper to add a field to the global MONITOR.

    Args:
        name: Field name.
        value: Field value.
    """
    MONITOR.add_field(name=name, value=value)
    MONITOR.flush()


def begin_step(episode: int, global_step: int, inner_step: int) -> None:
    """Module-level helper to reset and seed the global MONITOR.

    Args:
        episode: Episode index.
        global_step: Global environment step.
        inner_step: Step within episode.
    """
    MONITOR.begin_step(
        episode=int(episode),
        global_step=int(global_step),
        inner_step=int(inner_step),
    )


MONITOR: MonitoringState = MonitoringState()
