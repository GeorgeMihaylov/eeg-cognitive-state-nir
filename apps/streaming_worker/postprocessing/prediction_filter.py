"""Causal probability smoothing and class-switch hysteresis."""
from __future__ import annotations

from dataclasses import replace

from cogstate.streaming.inference import PredictionResult

from ..config import PostprocessingConfig


class PredictionFilter:
    def __init__(self, config: PostprocessingConfig) -> None:
        self.config = config
        self._probabilities: dict[str, dict[str, float]] = {}
        self._labels: dict[str, str] = {}
        self._pending: dict[str, tuple[str, int]] = {}

    def reset(self) -> None:
        self._probabilities.clear()
        self._labels.clear()
        self._pending.clear()

    def _update_metric(self, metric: str, incoming: dict[str, float]) -> tuple[str, dict[str, float]]:
        alpha = self.config.probability_ema_alpha
        previous = self._probabilities.get(metric)
        if previous is None:
            smoothed = {name: float(value) for name, value in incoming.items()}
        else:
            names = set(previous) | set(incoming)
            smoothed = {
                name: alpha * float(incoming.get(name, 0.0))
                + (1.0 - alpha) * float(previous.get(name, 0.0))
                for name in names
            }
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {name: value / total for name, value in smoothed.items()}
        self._probabilities[metric] = smoothed

        candidate = max(smoothed, key=smoothed.get) if smoothed else "unknown"
        confidence = smoothed.get(candidate, 0.0)
        if confidence < self.config.minimum_confidence:
            candidate = "unknown"

        current = self._labels.get(metric, "unknown")
        if candidate == current:
            self._pending.pop(metric, None)
        else:
            pending_label, count = self._pending.get(metric, (candidate, 0))
            count = count + 1 if pending_label == candidate else 1
            self._pending[metric] = (candidate, count)
            if count >= self.config.confirmation_windows:
                current = candidate
                self._labels[metric] = candidate
                self._pending.pop(metric, None)
        return current, smoothed

    def apply(self, prediction: PredictionResult) -> PredictionResult:
        targets = prediction.target_probabilities
        if targets is None:
            targets = {"primary": prediction.probabilities}

        labels: dict[str, str] = {}
        probabilities: dict[str, dict[str, float]] = {}
        for metric, values in targets.items():
            labels[metric], probabilities[metric] = self._update_metric(metric, values)

        primary_metric = "attention" if "attention" in labels else next(iter(labels), "primary")
        primary_label = labels.get(primary_metric, "unknown")
        primary_probabilities = probabilities.get(primary_metric, {})
        return replace(
            prediction,
            label=primary_label,
            probabilities=primary_probabilities,
            target_labels=labels if prediction.target_probabilities is not None else None,
            target_probabilities=probabilities if prediction.target_probabilities is not None else None,
        )
