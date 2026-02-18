# src/DQN_walker2d/monitoring.py

from __future__ import annotations

from dataclasses import dataclass


def _t(x: float) -> float:
    """Truncate to 3 decimals (not round)."""
    return float(int(x * 1000.0) / 1000.0)


@dataclass
class MonitoringState:
    """Mutable runtime monitoring values shared across modules."""

    episode_index: int = 0

    raw_reward: float = 0.0
    head_height: float = 0.0
    acc_fw_reward: float = 0.0
    helper_intensity: float = 0.0
    agent_contrib: float = 0.0
    real_reward: float = 0.0

    # ---------- setters with truncation ----------

    def set_episode(self, idx: int) -> None:
        self.episode_index = int(idx)

    def set_raw_reward(self, v: float) -> None:
        self.raw_reward = _t(v)

    def set_head_height(self, v: float) -> None:
        self.head_height = _t(v)

    def set_acc_fw_reward(self, v: float) -> None:
        self.acc_fw_reward = _t(v)

    def set_helper_intensity(self, v: float) -> None:
        self.helper_intensity = _t(v)

    def set_agent_contrib(self, v: float) -> None:
        self.agent_contrib = _t(v)

    def set_real_reward(self, v: float) -> None:
        self.real_reward = _t(v)


# single shared instance (minimal global state)
MONITOR: MonitoringState = MonitoringState()
