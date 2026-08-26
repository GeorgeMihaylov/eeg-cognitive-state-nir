"""Application latency metrics.

Classification and regression metrics belong exclusively to
``bench.validation.metrics``.
"""

from __future__ import annotations

import numpy as np


def latency_metrics(latencies_ms: object) -> dict[str, float]:
    """Summarize application latency; latency is not a scientific target metric."""
    values = np.asarray(latencies_ms, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("latencies_ms must be a non-empty finite one-dimensional array")
    return {
        "mean_ms": float(values.mean()),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }
