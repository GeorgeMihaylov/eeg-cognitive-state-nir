from __future__ import annotations

import csv
import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .buffer import Window


@dataclass
class LatencyTrace:

    window_start: float
    window_end: float
    device_id: Optional[str] = None
    _t0: float = field(default_factory=time.perf_counter)
    _marks: Dict[str, float] = field(default_factory=dict)
    _finished_at: Optional[float] = None

    def mark(self, stage: str) -> None:
        """Зафиксировать момент завершения стадии `stage`."""
        self._marks[stage] = time.perf_counter()

    def finish(self) -> None:
        self._finished_at = time.perf_counter()

    def stage_latencies_ms(self) -> Dict[str, float]:
        latencies = {}
        prev_t = self._t0
        for stage, t in self._marks.items():
            latencies[stage] = (t - prev_t) * 1000
            prev_t = t
        return latencies

    def total_latency_ms(self) -> float:
        end = self._finished_at if self._finished_at is not None else time.perf_counter()
        return (end - self._t0) * 1000


class LatencyMonitor:

    def __init__(self, max_history: int = 10_000):
        self._max_history = max_history
        self._records: List[LatencyTrace] = []
        self._lock = threading.Lock()

    def start_trace(self, window: Window, device_id: Optional[str] = None) -> LatencyTrace:
        return LatencyTrace(
            window_start=window.start_time,
            window_end=window.end_time,
            device_id=device_id,
        )

    def record(self, trace: LatencyTrace) -> None:
        with self._lock:
            self._records.append(trace)
            if len(self._records) > self._max_history:
                self._records.pop(0)

    def summary(self) -> Dict[str, float]:
        with self._lock:
            totals = [r.total_latency_ms() for r in self._records]

        if not totals:
            return {}

        totals_sorted = sorted(totals)
        return {
            "count": len(totals_sorted),
            "mean_ms": statistics.mean(totals_sorted),
            "median_ms": statistics.median(totals_sorted),
            "p95_ms": _percentile(totals_sorted, 0.95),
            "p99_ms": _percentile(totals_sorted, 0.99),
            "max_ms": totals_sorted[-1],
            "min_ms": totals_sorted[0],
        }

    def summary_by_device(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            by_device: Dict[str, List[float]] = {}
            for r in self._records:
                key = r.device_id or "unknown"
                by_device.setdefault(key, []).append(r.total_latency_ms())

        result = {}
        for device, totals in by_device.items():
            totals_sorted = sorted(totals)
            result[device] = {
                "count": len(totals_sorted),
                "mean_ms": statistics.mean(totals_sorted),
                "p95_ms": _percentile(totals_sorted, 0.95),
            }
        return result

    def export_csv(self, path: str | Path) -> None:
        path = Path(path)
        with self._lock:
            records = list(self._records)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["window_start", "window_end", "device_id", "total_latency_ms"])
            for r in records:
                writer.writerow([r.window_start, r.window_end, r.device_id or "", r.total_latency_ms()])

    def export_json_summary(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "overall": self.summary(),
            "by_device": self.summary_by_device(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[idx]
