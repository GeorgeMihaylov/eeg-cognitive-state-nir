"""Append-only log of per-subject AutoML decisions.

This is the substrate for a future algorithm-selection meta-layer: once
enough records exist, any tabular model can be trained on
`meta.tags -> selected_candidate` to pre-narrow future shortlists. That
training is intentionally out of scope for this module — it records,
it does not learn. Keeping it dataset-agnostic (tags instead of named
EEG/PM fields) means the same store works for a completely different
project without modification.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .meta_features import SubjectMetaFeatures


@dataclass(frozen=True)
class PortfolioRecord:
    run_id: str
    subject_id: str
    meta: SubjectMetaFeatures
    selected_candidate: str
    selected_status: str
    inner_metric: float
    holdout_metric: Optional[float]
    shadow_metrics: Dict[str, Any]
    timestamp: float

    def to_json(self) -> str:
        payload = {
            "run_id": self.run_id,
            "subject_id": self.subject_id,
            "tags": dict(self.meta.tags),
            "n_calibration_samples": self.meta.n_calibration_samples,
            "quality_score": self.meta.quality_score,
            "selected_candidate": self.selected_candidate,
            "selected_status": self.selected_status,
            "inner_metric": self.inner_metric,
            "holdout_metric": self.holdout_metric,
            "shadow_metrics": self.shadow_metrics,
            "timestamp": self.timestamp,
        }
        return json.dumps(payload, sort_keys=True)


class PortfolioStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._records: List[PortfolioRecord] = []

    def append(self, record: PortfolioRecord) -> None:
        self._records.append(record)
        if self._path is not None:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(record.to_json() + "\n")

    @property
    def records(self) -> Tuple[PortfolioRecord, ...]:
        return tuple(self._records)


def new_run_id() -> str:
    return str(uuid.uuid4())


def now() -> float:
    return time.time()
