from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .meta_features import SubjectMetaFeatures
from .protocols import ConditionFn, ModelAdapter, ModelBuilder


class CandidateStatus(str, enum.Enum):
    APPROVED_DEFAULT = "approved-default"
    APPROVED_FALLBACK = "approved-fallback"
    DEPRIORITIZED = "deprioritized"
    EXPERIMENTAL_CONDITIONAL = "experimental-conditional"
    EXPERIMENTAL_GATED = "experimental-gated"
    DISABLED = "disabled"

    @property
    def is_selectable(self) -> bool:
        """Whether this status may ever be chosen as the active model."""
        return self in {
            CandidateStatus.APPROVED_DEFAULT,
            CandidateStatus.APPROVED_FALLBACK,
            CandidateStatus.DEPRIORITIZED,
            CandidateStatus.EXPERIMENTAL_CONDITIONAL,
        }

    @property
    def is_shadow_only(self) -> bool:
        """Runs and logs diagnostics, but is never returned to a user."""
        return self is CandidateStatus.EXPERIMENTAL_GATED


class AdaptationMethod(str, enum.Enum):
    ZERO_SHOT = "zero_shot"
    HEAD_ONLY = "head_only"
    FULL_FINETUNE = "full_finetune"
    REFIT = "refit"
    SHADOW_ONLY = "shadow_only"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    builder: ModelBuilder
    builder_params: Mapping[str, Any]
    adaptation_method: AdaptationMethod
    status: CandidateStatus
    min_calibration_samples: int = 0
    required_tags: Mapping[str, str] = field(default_factory=dict)
    condition: Optional[ConditionFn] = None
    notes: str = ""

    def build(self, input_shape: Sequence[int], num_outputs: int) -> ModelAdapter:
        return self.builder(input_shape, num_outputs, dict(self.builder_params))

    def is_eligible(self, meta: SubjectMetaFeatures) -> bool:
        if not meta.matches_constraints(self.required_tags):
            return False
        if self.condition is not None and not self.condition(meta):
            return False
        if meta.n_calibration_samples < self.min_calibration_samples:
            return False
        return True


class CandidateRegistry:
    """An explicit, human-curated list of candidates."""

    def __init__(self, candidates: Sequence[CandidateSpec]) -> None:
        names = [c.name for c in candidates]
        if len(names) != len(set(names)):
            raise ValueError("CandidateSpec names must be unique within a registry")
        self._candidates: Tuple[CandidateSpec, ...] = tuple(candidates)

    def __iter__(self):
        return iter(self._candidates)

    def __len__(self) -> int:
        return len(self._candidates)

    def get(self, name: str) -> CandidateSpec:
        for candidate in self._candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(f"No candidate named {name!r} in registry")

    def shortlist(
        self,
        meta: SubjectMetaFeatures,
        *,
        max_selectable: int = 4,
        include_shadow: bool = True,
    ) -> Tuple[List[CandidateSpec], List[CandidateSpec]]:
        """Return (selectable_shortlist, shadow_candidates) for one subject.

        Filtering rules (conservative, reversible only via registry edits,
        never learned):
          * DISABLED candidates are always dropped.
          * EXPERIMENTAL_GATED candidates go to the shadow list only, and
            only if `is_eligible` (required_tags + condition) holds; they
            never enter the selectable shortlist.
          * Any candidate whose `required_tags`, `condition`, or
            `min_calibration_samples` isn't satisfied is dropped.
        """
        selectable: List[CandidateSpec] = []
        shadow: List[CandidateSpec] = []

        for candidate in self._candidates:
            if candidate.status is CandidateStatus.DISABLED:
                continue
            if not candidate.is_eligible(meta):
                continue
            if candidate.status.is_shadow_only:
                if include_shadow:
                    shadow.append(candidate)
                continue
            selectable.append(candidate)

        rank = {
            CandidateStatus.APPROVED_DEFAULT: 0,
            CandidateStatus.APPROVED_FALLBACK: 1,
            CandidateStatus.EXPERIMENTAL_CONDITIONAL: 2,
            CandidateStatus.DEPRIORITIZED: 3,
        }
        selectable.sort(key=lambda c: rank.get(c.status, 99))
        if not selectable:
            raise RuntimeError(
                f"No selectable candidate for subject {meta.subject_id!r}; "
                "check registry required_tags / min_calibration_samples / condition."
            )
        return selectable[:max_selectable], shadow
