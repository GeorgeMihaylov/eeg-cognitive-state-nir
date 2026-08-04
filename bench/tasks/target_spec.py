"""Canonical target contracts shared by feature and raw EEG datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORTED_TARGET_TYPES = frozenset(
    {
        "continuous_regression",
        "multioutput_regression",
        "binary_proxy",
        "multilabel_proxy",
        "derived_ordinal_classification",
        "legacy_classification",
    }
)
SUPPORTED_EXECUTION_STATUSES = frozenset({"executable", "disabled"})


@dataclass(frozen=True)
class TargetSpec:
    """Executable description of one target and its scientific contract."""

    target_id: str
    display_name: str
    target_type: str
    processed_columns: tuple[str, ...]
    output_names: tuple[str, ...]
    output_dim: int
    missing_value_policy: str
    cohort_policy: str
    transform_policy: str
    fit_scope: str
    recommended_metrics: tuple[str, ...]
    allowed_feature_inputs: tuple[str, ...]
    raw_input_supported: bool
    execution_status: str
    registry_status: str

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id must be non-empty")
        if self.target_type not in SUPPORTED_TARGET_TYPES:
            raise ValueError(f"Unsupported target_type: {self.target_type!r}")
        if self.execution_status not in SUPPORTED_EXECUTION_STATUSES:
            raise ValueError(
                f"Unsupported execution_status: {self.execution_status!r}"
            )
        if not self.processed_columns:
            raise ValueError("processed_columns must be non-empty")
        if len(set(self.processed_columns)) != len(self.processed_columns):
            raise ValueError("processed_columns must be unique")
        if self.output_dim <= 0 or self.output_dim != len(self.output_names):
            raise ValueError("output_dim must equal len(output_names) and be positive")
        if self.target_type in {"multioutput_regression", "multilabel_proxy"} and (
            len(self.processed_columns) != self.output_dim
        ):
            raise ValueError(
                "multi-output targets require one processed column per output"
            )
        if self.target_type not in {
            "multioutput_regression",
            "multilabel_proxy",
        } and len(self.processed_columns) != 1:
            raise ValueError("single-output targets require one processed column")
        invalid_inputs = set(self.allowed_feature_inputs) - {
            "eeg",
            "pow",
            "eeg_pow",
        }
        if invalid_inputs:
            raise ValueError(f"Unsupported feature inputs: {sorted(invalid_inputs)}")

    @property
    def is_executable(self) -> bool:
        return self.execution_status == "executable"

    @property
    def is_classification(self) -> bool:
        return self.target_type in {
            "binary_proxy",
            "multilabel_proxy",
            "derived_ordinal_classification",
            "legacy_classification",
        }

    @property
    def is_regression(self) -> bool:
        return self.target_type in {
            "continuous_regression",
            "multioutput_regression",
        }

    @property
    def task_type(self) -> str:
        return "classification" if self.is_classification else "regression"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "processed_columns",
            "output_names",
            "recommended_metrics",
            "allowed_feature_inputs",
        ):
            payload[name] = list(payload[name])
        return payload


@dataclass(frozen=True)
class TargetCohort:
    """Target-specific availability inside an already fixed outer cohort."""

    target_id: str
    availability_mask: np.ndarray
    selected_positions: np.ndarray
    n_outer_rows: int
    n_available_rows: int

    def __post_init__(self) -> None:
        mask = np.asarray(self.availability_mask, dtype=bool)
        positions = np.asarray(self.selected_positions, dtype=np.int64)
        if mask.ndim != 1:
            raise ValueError("availability_mask must be one-dimensional")
        if len(mask) != int(self.n_outer_rows):
            raise ValueError("availability_mask length must equal n_outer_rows")
        expected = np.flatnonzero(mask)
        if not np.array_equal(positions, expected):
            raise ValueError("selected_positions must equal flatnonzero(mask)")
        if int(self.n_available_rows) != len(positions):
            raise ValueError("n_available_rows does not match selected_positions")
        object.__setattr__(self, "availability_mask", mask)
        object.__setattr__(self, "selected_positions", positions)


@dataclass(frozen=True)
class TargetView:
    """Targets and identifiers selected without changing canonical row order."""

    spec: TargetSpec
    targets: np.ndarray
    cohort: TargetCohort
    sample_ids: np.ndarray
    subject_ids: np.ndarray
    record_ids: np.ndarray

    def __post_init__(self) -> None:
        n = self.cohort.n_available_rows
        for name in ("targets", "sample_ids", "subject_ids", "record_ids"):
            value = np.asarray(getattr(self, name))
            if len(value) != n:
                raise ValueError(f"{name} length does not match target cohort")
            object.__setattr__(self, name, value)


def normalize_string_tuple(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{field} must contain non-empty strings")
    return normalized


def target_spec_from_mapping(payload: Mapping[str, Any]) -> TargetSpec:
    """Construct a validated spec from YAML/JSON-compatible data."""

    values = dict(payload)
    for name in (
        "processed_columns",
        "output_names",
        "recommended_metrics",
        "allowed_feature_inputs",
    ):
        values[name] = normalize_string_tuple(values[name], field=name)
    return TargetSpec(**values)
