"""Public latency facade; old imports remain supported."""

from .latency_monitor import LatencyMonitor, LatencyTrace

__all__ = ["LatencyMonitor", "LatencyTrace"]
