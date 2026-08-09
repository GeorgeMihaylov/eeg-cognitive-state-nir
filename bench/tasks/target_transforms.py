"""Leakage-safe target transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from typing import Any

import numpy as np

from .target_spec import TargetSpec


TARGET_TRANSFORM_MANIFEST_SCHEMA_VERSION = "fold-local-target-transform-v1"


class TargetTransform(ABC):
    """Minimal fit/transform protocol for target-only transformations."""

    @abstractmethod
    def fit(self, y_outer_train: np.ndarray) -> "TargetTransform":
        """Fit using outer-train targets only."""

    @abstractmethod
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform a train, validation, or test partition."""

    @abstractmethod
    def manifest(self) -> dict[str, Any]:
        """Return deterministic transform provenance."""


class IdentityTargetTransform(TargetTransform):
    def __init__(self) -> None:
        self._fitted = False

    def fit(self, y_outer_train: np.ndarray) -> "IdentityTargetTransform":
        values = _as_single_output_float(y_outer_train)
        if not np.isfinite(values).any():
            raise ValueError("Cannot fit target transform without finite targets")
        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Target transform must be fit before transform")
        return np.asarray(y).copy()

    def manifest(self) -> dict[str, Any]:
        return {"transform": "identity", "fitted": self._fitted}


class FoldLocalQuantileTargetTransform(TargetTransform):
    """Discretize a continuous target with outer-train-only quantiles."""

    def __init__(self, q: int, *, duplicates: str = "drop") -> None:
        if q not in {3, 5}:
            raise ValueError("Fold-local quantile transform supports q=3 or q=5")
        if duplicates not in {"drop", "raise"}:
            raise ValueError("duplicates must be 'drop' or 'raise'")
        self.q = int(q)
        self.duplicates = duplicates
        self._boundaries: np.ndarray | None = None
        self._fit_n = 0

    def fit(self, y_outer_train: np.ndarray) -> "FoldLocalQuantileTargetTransform":
        values = _as_single_output_float(y_outer_train)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("Cannot fit quantile boundaries without finite targets")
        probabilities = np.linspace(0.0, 1.0, self.q + 1)
        try:
            boundaries = np.quantile(finite, probabilities, method="linear")
        except TypeError:  # NumPy < 1.22
            boundaries = np.quantile(finite, probabilities, interpolation="linear")
        unique_boundaries = np.unique(boundaries)
        if self.duplicates == "raise" and len(unique_boundaries) != len(boundaries):
            raise ValueError(
                "Quantile boundaries are not unique; choose duplicates='drop' "
                "or change the target"
            )
        self._boundaries = (
            unique_boundaries if self.duplicates == "drop" else boundaries
        ).astype(float, copy=False)
        self._fit_n = int(finite.size)
        return self

    @property
    def actual_class_count(self) -> int:
        if self._boundaries is None:
            raise RuntimeError("Target transform has not been fit")
        return max(1, len(self._boundaries) - 1)

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self._boundaries is None:
            raise RuntimeError("Target transform must be fit before transform")
        values = _as_single_output_float(y)
        result = np.full(values.shape, np.nan, dtype=float)
        valid = np.isfinite(values)
        # np.digitize with right=True matches right-closed qcut intervals.
        result[valid] = np.digitize(
            values[valid], self._boundaries[1:-1], right=True
        ).astype(float)
        return result

    def manifest(self) -> dict[str, Any]:
        if self._boundaries is None:
            raise RuntimeError("Target transform has not been fit")
        return {
            "transform": "fold_local_quantile",
            "fit_scope": "outer_train_only",
            "requested_quantiles": self.q,
            "duplicates": self.duplicates,
            "actual_class_count": self.actual_class_count,
            "boundaries": self._boundaries.tolist(),
            "fit_sample_count": self._fit_n,
        }


def build_fold_local_target_transform(
    spec: TargetSpec,
) -> FoldLocalQuantileTargetTransform:
    """Build the registered outer-train target transform for one target spec."""
    if not spec.requires_fold_local_transform:
        raise ValueError(
            f"Target {spec.target_id!r} does not require a fold-local transform"
        )
    prefix = "fold_local_quantile_q"
    try:
        q = int(spec.transform_policy.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(
            f"Invalid quantile transform policy {spec.transform_policy!r}"
        ) from exc
    return FoldLocalQuantileTargetTransform(q=q, duplicates="drop")


def stable_target_transform_hash(payload: dict[str, Any]) -> str:
    """Hash a transform manifest while excluding its self-referential hash."""
    canonical = {
        str(key): value
        for key, value in payload.items()
        if key != "transform_hash"
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_target_transform_manifest(
    spec: TargetSpec,
    transform: TargetTransform,
    *,
    outer_fold: int,
    outer_train_sample_ids: np.ndarray,
    outer_train_targets: np.ndarray,
) -> dict[str, Any]:
    """Create deterministic provenance for one frozen outer-fold transform."""
    base = transform.manifest()
    sample_ids = np.asarray(outer_train_sample_ids).astype(str)
    target_values = _as_single_output_float(outer_train_targets)
    if len(sample_ids) != len(target_values):
        raise ValueError("Outer-train sample IDs and targets must have equal length")
    sample_payload = json.dumps(
        sorted(sample_ids.tolist()),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    paired_payload = json.dumps(
        sorted(
            (sample_id, format(float(value), ".17g"))
            for sample_id, value in zip(sample_ids.tolist(), target_values)
        ),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    payload: dict[str, Any] = {
        "schema_version": TARGET_TRANSFORM_MANIFEST_SCHEMA_VERSION,
        "target_id": spec.target_id,
        "source_continuous_target": spec.processed_columns[0],
        "task_type": spec.task_type,
        "outer_fold": int(outer_fold),
        "q": int(base["requested_quantiles"]),
        "requested_quantiles": int(base["requested_quantiles"]),
        "boundaries": [float(value) for value in base["boundaries"]],
        "fit_scope": spec.fit_scope,
        "fit_sample_count": int(base["fit_sample_count"]),
        "outer_train_sample_hash": hashlib.sha256(sample_payload).hexdigest(),
        "outer_train_target_hash": hashlib.sha256(paired_payload).hexdigest(),
        "transform_policy": spec.transform_policy,
        "duplicate_boundary_policy": str(base["duplicates"]),
        "actual_class_count": int(base["actual_class_count"]),
    }
    payload["transform_hash"] = stable_target_transform_hash(payload)
    return payload


def validate_target_transform_manifest(
    manifest: dict[str, Any],
    *,
    expected_hash: str | None = None,
) -> str:
    """Validate self-integrity and, when supplied, resume compatibility."""
    stored_hash = str(manifest.get("transform_hash", ""))
    actual_hash = stable_target_transform_hash(manifest)
    if not stored_hash or stored_hash != actual_hash:
        raise ValueError(
            "Target transform manifest hash mismatch: "
            f"stored={stored_hash or '<missing>'}, actual={actual_hash}"
        )
    if expected_hash is not None and stored_hash != str(expected_hash):
        raise ValueError(
            "Incompatible target transform for resume: "
            f"expected={expected_hash}, actual={stored_hash}"
        )
    return stored_hash


def _as_single_output_float(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"Fold-local target transform expects shape [n_samples], got {values.shape}"
        )
    return values
