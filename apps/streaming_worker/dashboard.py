"""Live terminal dashboard for streaming PM predictions."""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cogstate.protocol import PM_METRICS


_STATE_STYLE = {"low": "blue", "medium": "yellow", "high": "green"}


def _bar(value: float, width: int = 18) -> Text:
    probability = min(1.0, max(0.0, float(value)))
    filled = round(probability * width)
    return Text("█" * filled + "░" * (width - filled), style="cyan")


class StreamingDashboardSink:
    """A sink that renders the latest window without changing inference output."""

    def __init__(
        self,
        *,
        source_name: str,
        sample_rate: float,
        channels: int,
        window_seconds: float,
        refresh_rate: float = 5.0,
        history_size: int = 60,
        console: Console | None = None,
    ) -> None:
        if refresh_rate <= 0:
            raise ValueError("refresh_rate must be positive")
        self.source_name = source_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_seconds = window_seconds
        self.console = console or Console(stderr=True)
        self.history: deque[dict[str, str]] = deque(maxlen=max(1, history_size))
        self.latest: Any | None = None
        self.received = 0
        self.rejected = 0
        self._lock = threading.RLock()
        self._live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=refresh_rate,
            transient=False,
        )
        self._started = False

    def start(self) -> None:
        with self._lock:
            if not self._started:
                self._live.start(refresh=True)
                self._started = True

    def publish(self, result: object) -> None:
        with self._lock:
            self.latest = result
            self.received += 1
            prediction = getattr(result, "prediction", None)
            if prediction is None:
                self.rejected += 1
            else:
                labels = getattr(prediction, "target_labels", None) or {
                    "attention": prediction.label
                }
                self.history.append(dict(labels))
            if not self._started:
                self.start()
            self._live.update(self.render(), refresh=True)

    def _header(self) -> Panel:
        if self.latest is None:
            model = "loading"
            mode = "—"
            diagnostic = ""
        else:
            model = f"{self.latest.model_type} · {self.latest.model_version}"
            mode = self.latest.input_mode
            if self.latest.model_version == "demo-synthetic-v1":
                diagnostic = " · [bold magenta]DEMO · SYNTHETIC DATA[/bold magenta]"
            else:
                diagnostic = " · DIAGNOSTIC" if self.latest.diagnostic_model else ""
        return Panel(
            f"[bold cyan]COGSTATE STREAM[/bold cyan]  {model}{diagnostic}\n"
            f"source: [white]{self.source_name}[/white]  ·  {self.channels} ch  ·  "
            f"{self.sample_rate:g} Hz  ·  {self.window_seconds:g} s  ·  input: {mode}",
            border_style="cyan",
        )

    def _predictions(self) -> Panel:
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("PM metric", style="bold")
        table.add_column("State", width=9)
        table.add_column("Confidence", justify="right", width=10)
        table.add_column("Probability", ratio=1)
        prediction = getattr(self.latest, "prediction", None)
        labels = getattr(prediction, "target_labels", None) or {}
        probabilities = getattr(prediction, "target_probabilities", None) or {}
        for metric in PM_METRICS:
            state = labels.get(metric, "—")
            confidence = float(probabilities.get(metric, {}).get(state, 0.0))
            style = _STATE_STYLE.get(state, "dim")
            table.add_row(
                metric,
                Text(state.upper(), style=f"bold {style}"),
                f"{confidence:6.1%}" if state != "—" else "—",
                _bar(confidence),
            )
        return Panel(table, title="Seven PM predictions", border_style="blue")

    def _history(self) -> Panel:
        rows: list[RenderableType] = []
        symbols = {"low": "▁", "medium": "▄", "high": "█"}
        for metric in PM_METRICS:
            series = "".join(symbols.get(item.get(metric, ""), "·") for item in self.history)
            rows.append(Text(f"{metric:<11} {series or 'waiting for windows…'}", style="cyan"))
        return Panel(Group(*rows), title="Recent states  ▁ low  ▄ medium  █ high")

    def _telemetry(self) -> Columns:
        quality = getattr(self.latest, "quality", None)
        status = getattr(quality, "status", "waiting")
        valid = bool(getattr(quality, "valid", False))
        quality_style = "green" if valid else ("yellow" if status == "waiting" else "red")
        reasons = ", ".join(getattr(quality, "reasons", ())) or "none"
        quality_panel = Panel(
            f"status: [{quality_style}]{status.upper()}[/{quality_style}]\n"
            f"finite: {getattr(quality, 'finite_ratio', 0.0):.1%}  ·  reasons: {reasons}",
            title="Signal quality",
        )
        latencies = getattr(self.latest, "stage_latencies_ms", {}) or {}
        prediction = getattr(self.latest, "prediction", None)
        inference_ms = getattr(prediction, "inference_time_ms", 0.0) if prediction else 0.0
        stages = "  ".join(f"{key}: {value:.1f} ms" for key, value in latencies.items())
        latency_panel = Panel(
            f"inference: {inference_ms:.1f} ms\n{stages or 'waiting for a valid window'}",
            title="Latency",
        )
        counters = Panel(
            f"windows: {self.received}\naccepted: {self.received - self.rejected}  ·  rejected: {self.rejected}",
            title="Runtime",
        )
        return Columns([quality_panel, latency_panel, counters], expand=True, equal=True)

    def render(self) -> RenderableType:
        with self._lock:
            return Group(self._header(), self._predictions(), self._history(), self._telemetry())

    def close(self) -> None:
        with self._lock:
            if self._started:
                self._live.update(self.render(), refresh=True)
                self._live.stop()
                self._started = False


__all__ = ["StreamingDashboardSink"]
