"""Order-preserving target and feature views over canonical EEG tables."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.tasks.target_spec import TargetCohort, TargetSpec, TargetView

from .base_eeg_data_loader import resolve_feature_columns


def sample_id_filter_hash(sample_ids: Iterable[object]) -> str:
    """Return an order-independent hash for an explicit sample cohort."""

    values = sorted(str(value) for value in sample_ids)
    payload = json.dumps(
        values, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_sample_id_filter(path: str | Path) -> np.ndarray:
    """Load one unique, target-free sample-ID cohort from Parquet or CSV."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Sample-ID filter not found: {source}")
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("Sample-ID filter must be a Parquet or CSV table")
    target_like = [
        column for column in frame.columns
        if str(column).startswith(("target_", "label_", "PM."))
    ]
    if target_like:
        raise ValueError(
            f"Sample-ID filter must be target-free; found {target_like}"
        )
    column = (
        "sample_id" if "sample_id" in frame.columns
        else "endpoint_sample_id" if "endpoint_sample_id" in frame.columns
        else None
    )
    if column is None:
        raise ValueError(
            "Sample-ID filter requires sample_id or endpoint_sample_id"
        )
    values = frame[column]
    if values.isna().any() or values.duplicated().any():
        raise ValueError("Sample-ID filter values must be unique and non-null")
    return values.to_numpy(copy=True)


def sample_id_filter_positions(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    strict: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Resolve filter IDs to source-order row positions with an identity audit."""

    _require_columns(frame, ("sample_id",))
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Source table sample_id values must be unique and non-null")
    requested = load_sample_id_filter(path)
    source_keys = frame["sample_id"].astype(str)
    requested_keys = pd.Index(pd.Series(requested).astype(str))
    missing = sorted(set(requested_keys) - set(source_keys))
    if strict and missing:
        raise ValueError(
            "Sample-ID filter contains IDs absent from the source table: "
            f"{missing[:20]}"
        )
    positions = np.flatnonzero(
        source_keys.isin(set(requested_keys)).to_numpy()
    )
    selected_ids = frame.iloc[positions]["sample_id"]
    audit: dict[str, object] = {
        "path": str(Path(path)),
        "requested_count": int(len(requested)),
        "selected_count": int(len(positions)),
        "missing_count": int(len(missing)),
        "sample_id_filter_hash": sample_id_filter_hash(requested),
        "selected_sample_id_hash": sample_id_filter_hash(selected_ids),
        "exact_match": bool(not missing and len(positions) == len(requested)),
    }
    if strict and not audit["exact_match"]:
        raise RuntimeError(f"Sample-ID filter identity mismatch: {audit}")
    return positions, audit


@dataclass(frozen=True)
class FeatureTargetView:
    """A target-specific supervised feature view without dataframe mutation."""

    features: np.ndarray
    feature_names: tuple[str, ...]
    target_view: TargetView

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got {features.shape}")
        if features.shape[0] != self.target_view.cohort.n_available_rows:
            raise ValueError("Feature and target row counts differ")
        if features.shape[1] != len(self.feature_names):
            raise ValueError("Feature width does not match feature_names")
        object.__setattr__(self, "features", features)

    @property
    def availability_mask(self) -> np.ndarray:
        return self.target_view.cohort.availability_mask


def build_target_view(
    frame: pd.DataFrame,
    spec: TargetSpec,
    *,
    additional_availability: np.ndarray | None = None,
    sample_id_column: str = "sample_id",
    subject_id_column: str = "subject_id",
    record_id_column: str = "record_id",
) -> TargetView:
    """Extract one target while preserving the source table's row order."""

    _require_columns(frame, spec.processed_columns)
    numeric = frame.loc[:, list(spec.processed_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    target_matrix = numeric.to_numpy(dtype=np.float64)
    availability = np.isfinite(target_matrix).all(axis=1)
    if additional_availability is not None:
        extra = np.asarray(additional_availability, dtype=bool)
        if extra.shape != availability.shape:
            raise ValueError("additional_availability must match dataframe rows")
        availability &= extra

    if spec.is_classification and not spec.requires_fold_local_transform:
        valid_values = target_matrix[availability]
        if not np.allclose(valid_values, np.round(valid_values)):
            raise ValueError(
                f"Classification target {spec.target_id!r} contains non-integer values"
            )
        if spec.output_dim == 1:
            targets = np.round(target_matrix[availability, 0]).astype(np.int64)
        else:
            targets = np.round(target_matrix[availability]).astype(np.int64)
    elif spec.output_dim == 1:
        targets = target_matrix[availability, 0].astype(np.float32)
    else:
        targets = target_matrix[availability].astype(np.float32)

    positions = np.flatnonzero(availability)
    cohort = TargetCohort(
        target_id=spec.target_id,
        availability_mask=availability,
        selected_positions=positions,
        n_outer_rows=len(frame),
        n_available_rows=len(positions),
    )
    sample_ids = _identifier_values(frame, sample_id_column, fallback_index=True)
    subject_ids = _identifier_values(frame, subject_id_column)
    record_ids = _identifier_values(frame, record_id_column)
    return TargetView(
        spec=spec,
        targets=targets,
        cohort=cohort,
        sample_ids=sample_ids[availability],
        subject_ids=subject_ids[availability],
        record_ids=record_ids[availability],
    )


def build_feature_target_view(
    frame: pd.DataFrame,
    spec: TargetSpec,
    feature_set: str,
) -> FeatureTargetView:
    """Build an EEG/POW view with target and identifier columns excluded."""

    canonical_feature_set = normalize_feature_input(feature_set)
    if canonical_feature_set not in spec.allowed_feature_inputs:
        raise ValueError(
            f"Target {spec.target_id!r} does not allow feature input "
            f"{canonical_feature_set!r}"
        )
    feature_names = tuple(
        resolve_feature_columns(frame.columns.tolist(), canonical_feature_set)
    )
    if not feature_names:
        raise ValueError(f"Feature set {feature_set!r} selected no columns")
    forbidden = [
        column
        for column in feature_names
        if column.startswith(("target_", "label_", "PM."))
        or column
        in {
            "sample_id",
            "subject_id",
            "record_id",
            "record_group_id",
            "source",
            "window_id",
        }
    ]
    if forbidden:
        raise ValueError(f"Target or identifier columns entered features: {forbidden}")
    feature_values = frame.loc[:, list(feature_names)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    feature_available = np.isfinite(feature_values).all(axis=1)
    target_view = build_target_view(
        frame, spec, additional_availability=feature_available
    )
    return FeatureTargetView(
        features=feature_values[target_view.cohort.availability_mask],
        feature_names=feature_names,
        target_view=target_view,
    )


def attach_targets_by_sample_id(
    window_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    spec: TargetSpec,
    *,
    validate_identifiers: bool = True,
) -> pd.DataFrame:
    """Attach canonical processed targets to a raw-window manifest."""

    _require_columns(window_frame, ("sample_id", "subject_id", "record_id"))
    _require_columns(
        target_frame,
        ("sample_id", "subject_id", "record_id", *spec.processed_columns),
    )
    if target_frame["sample_id"].duplicated().any():
        duplicates = target_frame.loc[
            target_frame["sample_id"].duplicated(keep=False), "sample_id"
        ].head(10).tolist()
        raise ValueError(f"Canonical target table has duplicate sample_id: {duplicates}")
    target_columns = [
        "sample_id",
        "subject_id",
        "record_id",
        *spec.processed_columns,
    ]
    right = target_frame.loc[:, target_columns].copy()
    right = right.rename(
        columns={"subject_id": "_target_subject_id", "record_id": "_target_record_id"}
    )
    merged = window_frame.merge(
        right,
        on="sample_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if not np.array_equal(
        merged["sample_id"].to_numpy(), window_frame["sample_id"].to_numpy()
    ):
        raise RuntimeError("Target attachment changed raw-window row order")
    if validate_identifiers:
        available = merged["_target_subject_id"].notna()
        subject_mismatch = available & (
            merged["subject_id"].astype(str)
            != merged["_target_subject_id"].astype(str)
        )
        record_mismatch = available & (
            merged["record_id"].astype(str)
            != merged["_target_record_id"].astype(str)
        )
        if subject_mismatch.any() or record_mismatch.any():
            raise ValueError(
                "Raw-window sample identifiers disagree with the canonical target table"
            )
    return merged.drop(columns=["_target_subject_id", "_target_record_id"])


def target_cohort_manifest(
    frame: pd.DataFrame,
    spec: TargetSpec,
    outer_fold_by_subject: Mapping[str, int],
) -> pd.DataFrame:
    """Summarize a target-specific cohort inside immutable subject folds."""

    view = build_target_view(frame, spec)
    selected = frame.iloc[view.cohort.selected_positions].copy()
    selected["outer_fold"] = selected["subject_id"].astype(str).map(
        {str(key): int(value) for key, value in outer_fold_by_subject.items()}
    )
    if selected["outer_fold"].isna().any():
        missing = sorted(
            selected.loc[selected["outer_fold"].isna(), "subject_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"Subjects missing from fixed outer folds: {missing}")
    rows = []
    for fold, partition in selected.groupby("outer_fold", sort=True):
        rows.append(
            {
                "target_id": spec.target_id,
                "outer_fold": int(fold),
                "n_samples": int(len(partition)),
                "n_subjects": int(partition["subject_id"].nunique()),
                "n_records": int(partition["record_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def normalize_feature_input(feature_set: str) -> str:
    aliases = {
        "eeg": "eeg",
        "eeg_only": "eeg",
        "pow": "pow",
        "pow_only": "pow",
        "eeg_pow": "eeg_pow",
        "pow_plus_eeg": "eeg_pow",
        "all": "eeg_pow",
    }
    try:
        return aliases[str(feature_set).lower()]
    except KeyError as exc:
        raise ValueError(
            "feature_set must select one of eeg, pow, or eeg_pow"
        ) from exc


def _identifier_values(
    frame: pd.DataFrame, column: str, *, fallback_index: bool = False
) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy(copy=True)
    if fallback_index:
        return np.asarray(frame.index)
    return np.full(len(frame), "unknown", dtype=object)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Required columns are absent: {missing}")
