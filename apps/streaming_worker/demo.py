"""Synthetic output generator for demonstrating the terminal dashboard."""
from __future__ import annotations

import math
import time
from collections.abc import Iterator

import numpy as np

from cogstate.protocol import PM_METRICS
from cogstate.streaming.inference import PredictionResult

from .quality import QualityReport
from .runtime import StreamingOutput


CLASS_NAMES = ("low", "medium", "high")


def iter_demo_outputs(
    *,
    seed: int = 42,
    sample_rate: float = 256.0,
    window_seconds: float = 10.0,
    step_seconds: float = 1.0,
) -> Iterator[StreamingOutput]:
    """Yield deterministic, smoothly changing seven-target demo predictions."""
    rng = np.random.default_rng(seed)
    expected_samples = int(round(sample_rate * window_seconds))
    index = 0
    while True:
        targets: dict[str, dict[str, float]] = {}
        labels: dict[str, str] = {}
        for metric_index, metric in enumerate(PM_METRICS):
            phase = index * 0.24 + metric_index * 0.71
            center = math.sin(phase)
            logits = np.asarray(
                [
                    -1.3 * center,
                    0.7 - abs(center),
                    1.3 * center,
                ],
                dtype=float,
            )
            logits += rng.normal(0.0, 0.08, size=3)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            targets[metric] = dict(zip(CLASS_NAMES, map(float, probabilities)))
            labels[metric] = CLASS_NAMES[int(np.argmax(probabilities))]

        inference_ms = float(3.5 + 1.2 * abs(math.sin(index * 0.31)))
        window_start = index * step_seconds
        yield StreamingOutput(
            window_start=window_start,
            window_end=window_start + window_seconds,
            quality=QualityReport(
                status="good",
                valid=True,
                reasons=(),
                sample_count=expected_samples,
                expected_sample_count=expected_samples,
                finite_ratio=1.0,
                estimated_sample_rate=sample_rate,
                missing_ratio=0.0,
            ),
            prediction=PredictionResult(
                label=labels["attention"],
                probabilities=targets["attention"],
                model_version="demo-synthetic-v1",
                is_calibrated=False,
                inference_time_ms=inference_ms,
                target_labels=labels,
                target_probabilities=targets,
            ),
            stage_latencies_ms={
                "preprocessing": 0.8,
                "feature_extraction": 0.4,
                "inference": inference_ms,
            },
            model_version="demo-synthetic-v1",
            model_type="torch_shallow_convnet_multitask",
            input_mode="raw_eeg",
            class_names=CLASS_NAMES,
            diagnostic_model=True,
        )
        index += 1


def run_dashboard_demo(
    sink: object,
    *,
    seed: int,
    speed: float,
    sample_rate: float,
    window_seconds: float,
    step_seconds: float,
) -> None:
    """Publish synthetic windows until interrupted by the user."""
    if speed <= 0:
        raise ValueError("demo speed must be positive")
    interval_seconds = 0.75 / speed
    for output in iter_demo_outputs(
        seed=seed,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    ):
        sink.publish(output)
        time.sleep(interval_seconds)


__all__ = ["iter_demo_outputs", "run_dashboard_demo"]
