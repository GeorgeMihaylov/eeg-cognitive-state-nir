from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class SubjectMetaFeatures:
    subject_id: str
    n_calibration_samples: int
    quality_score: float
    tags: Mapping[str, str] = field(default_factory=dict)
    reference_tags: Mapping[str, str] = field(default_factory=dict)

    def tag_shift(self, key: str) -> bool:
        """Whether this subject's tag differs from the reference population's.

        Returns False (no shift detected) if the key isn't tracked in
        `reference_tags` — absence of a reference is not evidence of shift.
        """
        if key not in self.reference_tags:
            return False
        return self.tags.get(key) != self.reference_tags[key]

    def matches_constraints(self, constraints: Mapping[str, str]) -> bool:
        """Whether every constraint key/value is present and equal in tags."""
        return all(self.tags.get(key) == value for key, value in constraints.items())


def extract_meta_features(
    subject_id: str,
    calibration_samples: np.ndarray,
    tags: Mapping[str, str],
    reference_tags: Optional[Mapping[str, str]] = None,
    quality_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> SubjectMetaFeatures:
    """Build cheap, target-free meta-features used only for shortlisting."""
    if calibration_samples.ndim < 1 or calibration_samples.shape[0] == 0:
        raise ValueError("calibration_samples must be non-empty with a leading batch dim")
    quality = float(quality_fn(calibration_samples)) if quality_fn is not None else 1.0
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality_fn must return a value in [0, 1]")
    return SubjectMetaFeatures(
        subject_id=str(subject_id),
        n_calibration_samples=int(calibration_samples.shape[0]),
        quality_score=quality,
        tags=dict(tags),
        reference_tags=dict(reference_tags or {}),
    )
