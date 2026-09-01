import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


GROUP_COLUMNS = ("source", "subject_id", "record_group_id")
TIME_COLUMN_PRIORITY = ("t_start", "t_center", "window_id")
SEQUENCE_INDEX_COLUMNS = (
    "sequence_id",
    "fold",
    "source",
    "subject_id",
    "record_id",
    "target_sample_id",
    "target_time",
    "y_true",
)


@dataclass(frozen=True)
class SequenceBuildResult:
    """Sequence-to-one tensors plus row-level metadata and build statistics."""

    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    stats: Dict[str, Any]


def sequence_index_sha256(
    metadata: pd.DataFrame,
    columns: tuple[str, ...] = SEQUENCE_INDEX_COLUMNS,
) -> str:
    """Hash a canonical, order-independent sequence index.

    Rows are sorted by ``sequence_id`` and serialized as compact UTF-8 JSON
    arrays with one trailing newline per row. This keeps the hash independent
    of feature values and DataFrame row order while retaining fold and target
    assignments in the semantic identity.
    """

    frame = pd.DataFrame(metadata).copy()
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Sequence index is missing required columns: {missing}")
    if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
        raise ValueError("Sequence index requires unique, non-null sequence_id values")
    frame = frame.loc[:, list(columns)].sort_values(
        "sequence_id", kind="mergesort"
    )
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False, name=None):
        values = [value.item() if isinstance(value, np.generic) else value for value in row]
        digest.update(
            json.dumps(
                values,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _select_time_column(metadata: pd.DataFrame) -> str:
    for column in TIME_COLUMN_PRIORITY:
        if column in metadata.columns:
            if metadata[column].isna().any():
                raise ValueError(f"Sequence time column {column!r} contains missing values")
            return column
    raise ValueError(
        "Sequence metadata requires a temporal column. Expected one of "
        f"{list(TIME_COLUMN_PRIORITY)}"
    )


def _ensure_record_group_id(metadata: pd.DataFrame) -> str:
    """Keep modern logical groups, with a deterministic legacy fallback."""
    if "record_group_id" in metadata.columns:
        return "provided"
    fallback_columns = ("source", "subject_id", "record_id")
    missing = sorted(set(fallback_columns) - set(metadata.columns))
    if missing:
        raise ValueError(
            "Sequence metadata without record_group_id requires fallback "
            f"columns {list(fallback_columns)}; missing={missing}"
        )
    if metadata.loc[:, list(fallback_columns)].isna().any().any():
        raise ValueError(
            "Sequence record-group fallback columns contain missing values"
        )
    # GROUP_COLUMNS already include source and subject_id. Reusing record_id as
    # the fallback preserves historical sequence IDs and immutable experiment
    # hashes while still preventing a sequence from crossing a physical record.
    metadata["record_group_id"] = metadata["record_id"].astype(str)
    return "derived_from_source_subject_record"


def _validate_inputs(
    X: Any,
    y: Any,
    metadata: pd.DataFrame,
    sequence_length: int,
    stride: int,
    target_position: str,
    expected_step_seconds: Optional[float],
    max_gap_seconds: Optional[float],
    endpoint_targets: Optional[Dict[Any, Any]],
) -> tuple[np.ndarray, np.ndarray, str]:
    features = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y)
    if features.ndim != 2:
        raise ValueError(
            f"Sequence builder expects X=[windows, features], got {features.shape}"
        )
    if endpoint_targets is None and labels.ndim != 1:
        raise ValueError(f"Sequence builder expects one-dimensional y, got {labels.shape}")
    if len(features) != len(metadata) or (
        endpoint_targets is None and len(features) != len(labels)
    ):
        raise ValueError(
            "X, y and metadata must have the same number of rows: "
            f"{len(features)}, {len(labels)}, {len(metadata)}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Sequence features contain NaN or infinite values")
    values_to_validate = labels if endpoint_targets is None else np.asarray(
        list(endpoint_targets.values())
    )
    if values_to_validate.ndim != 1 or not np.issubdtype(values_to_validate.dtype, np.number):
        raise ValueError("Sequence labels must be numeric")
    if not np.isfinite(values_to_validate.astype(np.float64, copy=False)).all():
        raise ValueError("Sequence labels contain NaN or infinite values")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if target_position != "last":
        raise ValueError("Only target_position='last' is currently supported")
    if (expected_step_seconds is None) != (max_gap_seconds is None):
        raise ValueError(
            "expected_step_seconds and max_gap_seconds must be configured together"
        )
    if expected_step_seconds is not None:
        if expected_step_seconds <= 0:
            raise ValueError("expected_step_seconds must be positive")
        if max_gap_seconds is None or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if max_gap_seconds < expected_step_seconds:
            raise ValueError(
                "max_gap_seconds must be at least expected_step_seconds"
            )

    required_columns = set(GROUP_COLUMNS) | {"record_id", "sample_id"}
    missing = sorted(required_columns - set(metadata.columns))
    if missing:
        raise ValueError(f"Sequence metadata is missing required columns: {missing}")
    for column in required_columns:
        if metadata[column].isna().any():
            raise ValueError(f"Sequence metadata column {column!r} contains missing values")
    time_column = _select_time_column(metadata)
    try:
        time_values = metadata[time_column].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sequence time column {time_column!r} must be numeric"
        ) from exc
    if not np.isfinite(time_values).all():
        raise ValueError(
            f"Sequence time column {time_column!r} contains NaN or infinite values"
        )
    return np.ascontiguousarray(features), labels, time_column


def _sequence_count(n_windows: int, sequence_length: int, stride: int) -> int:
    if n_windows < sequence_length:
        return 0
    return ((n_windows - sequence_length) // stride) + 1


def build_sequences(
    X: Any,
    y: Any,
    metadata: pd.DataFrame,
    sequence_length: int = 10,
    stride: int = 1,
    target_position: str = "last",
    expected_step_seconds: Optional[float] = None,
    max_gap_seconds: Optional[float] = None,
    endpoint_targets: Optional[Dict[Any, Any]] = None,
) -> SequenceBuildResult:
    """Build sequence-to-one samples inside logical records.

    When ``endpoint_targets`` is supplied, all rows in ``X`` remain eligible as
    target-free context while only mapping keys may become sequence endpoints.
    This preserves one outer-fold target transform fitted before sequence
    filtering and avoids requiring labels on the preceding context windows.
    """
    if not isinstance(metadata, pd.DataFrame):
        metadata = pd.DataFrame(metadata)
    metadata = metadata.reset_index(drop=True).copy()
    record_group_id_source = _ensure_record_group_id(metadata)
    features, labels, time_column = _validate_inputs(
        X,
        y,
        metadata,
        sequence_length,
        stride,
        target_position,
        expected_step_seconds,
        max_gap_seconds,
        endpoint_targets,
    )
    endpoint_lookup = None if endpoint_targets is None else dict(endpoint_targets)

    sequences = []
    sequence_labels = []
    sequence_rows = []
    records_total = 0
    records_used = 0
    records_skipped_short = 0
    continuous_segments_total = 0
    records_with_gaps = 0
    gaps_detected = 0
    largest_observed_gap: Optional[float] = None
    windows_excluded_due_to_gaps = 0
    candidate_sequences_without_gap_check = 0
    candidate_endpoint_ids_without_gap: set[Any] = set()
    gap_check_enabled = max_gap_seconds is not None

    grouped = metadata.groupby(list(GROUP_COLUMNS), sort=True, dropna=False)
    for (source, subject_id, record_group_id), group in grouped:
        records_total += 1
        ordered = group.sort_values(
            [time_column, "sample_id"], kind="mergesort"
        )
        ordered_indices = ordered.index.to_numpy(dtype=np.int64)
        ordered_times = ordered[time_column].to_numpy(dtype=np.float64)
        time_deltas = np.diff(ordered_times)
        if len(time_deltas):
            record_largest_gap = float(np.max(time_deltas))
            largest_observed_gap = (
                record_largest_gap
                if largest_observed_gap is None
                else max(largest_observed_gap, record_largest_gap)
            )

        candidate_sequences_without_gap_check += _sequence_count(
            len(ordered_indices), sequence_length, stride
        )
        for start in range(0, max(0, len(ordered_indices) - sequence_length + 1), stride):
            candidate_endpoint_ids_without_gap.add(
                metadata.at[int(ordered_indices[start + sequence_length - 1]), "sample_id"]
            )
        if gap_check_enabled:
            break_mask = (time_deltas <= 0) | (time_deltas > max_gap_seconds)
            break_positions = np.flatnonzero(break_mask) + 1
        else:
            break_positions = np.empty((0,), dtype=np.int64)
        record_gap_count = int(len(break_positions))
        if record_gap_count:
            records_with_gaps += 1
            gaps_detected += record_gap_count
        segments = np.split(ordered_indices, break_positions)
        continuous_segments_total += len(segments)
        if record_gap_count:
            windows_excluded_due_to_gaps += sum(
                len(segment)
                for segment in segments
                if len(segment) < sequence_length
            )

        record_sequences = 0
        for segment_id, segment_indices in enumerate(segments):
            segment_sequence_count = _sequence_count(
                len(segment_indices), sequence_length, stride
            )
            if segment_sequence_count == 0:
                continue
            for start in range(
                0, len(segment_indices) - sequence_length + 1, stride
            ):
                indices = segment_indices[start:start + sequence_length]
                sequence_times = metadata.loc[indices, time_column].to_numpy(
                    dtype=np.float64
                )
                internal_deltas = np.diff(sequence_times)
                max_internal_gap = (
                    float(np.max(internal_deltas))
                    if len(internal_deltas)
                    else 0.0
                )
                if gap_check_enabled and (
                    np.any(internal_deltas <= 0)
                    or max_internal_gap > max_gap_seconds
                ):
                    raise RuntimeError(
                        "Gap-aware segmentation produced a discontinuous sequence"
                    )
                target_index = int(indices[-1])
                first_index = int(indices[0])
                target_sample_id = metadata.at[target_index, "sample_id"]
                if endpoint_lookup is not None and target_sample_id not in endpoint_lookup:
                    continue
                record_id = metadata.at[target_index, "record_id"]
                sequence_id = (
                    f"{source}|{record_group_id}|{target_sample_id}|"
                    f"len={sequence_length}|stride={stride}"
                )
                sequences.append(features[indices])
                sequence_labels.append(
                    labels[target_index]
                    if endpoint_lookup is None
                    else endpoint_lookup[target_sample_id]
                )
                sequence_rows.append({
                    "sequence_id": sequence_id,
                    "source": source,
                    "subject_id": subject_id,
                    "record_id": record_id,
                    "record_group_id": record_group_id,
                    "segment_id": segment_id,
                    "sequence_length": sequence_length,
                    "sequence_start_sample_id": metadata.at[
                        first_index, "sample_id"
                    ],
                    "sequence_end_sample_id": target_sample_id,
                    "sequence_start_time": float(sequence_times[0]),
                    "sequence_end_time": float(sequence_times[-1]),
                    "max_internal_gap": max_internal_gap,
                    "target_sample_id": target_sample_id,
                    "target_time": float(sequence_times[-1]),
                })
                record_sequences += 1
        if record_sequences:
            records_used += 1
        else:
            records_skipped_short += 1

    if sequences:
        sequence_X = np.ascontiguousarray(np.stack(sequences), dtype=np.float32)
        sequence_y = np.asarray(sequence_labels, dtype=labels.dtype)
    else:
        sequence_X = np.empty(
            (0, sequence_length, features.shape[1]), dtype=np.float32
        )
        sequence_y = np.empty((0,), dtype=labels.dtype)
    sequence_metadata = pd.DataFrame(sequence_rows, columns=[
        "sequence_id",
        "source",
        "subject_id",
        "record_id",
        "record_group_id",
        "segment_id",
        "sequence_length",
        "sequence_start_sample_id",
        "sequence_end_sample_id",
        "sequence_start_time",
        "sequence_end_time",
        "max_internal_gap",
        "target_sample_id",
        "target_time",
    ])
    if sequence_metadata["sequence_id"].duplicated().any():
        duplicates = sequence_metadata.loc[
            sequence_metadata["sequence_id"].duplicated(), "sequence_id"
        ].head(10).tolist()
        raise ValueError(f"Generated non-unique sequence_id values: {duplicates}")

    unique_classes, class_counts = np.unique(sequence_y, return_counts=True)
    generated_endpoint_ids = set(sequence_metadata["target_sample_id"].tolist())
    eligible_endpoint_ids = (
        set(metadata["sample_id"].tolist())
        if endpoint_lookup is None
        else set(endpoint_lookup)
    )
    eligible_candidates_without_gap = eligible_endpoint_ids & candidate_endpoint_ids_without_gap
    stats: Dict[str, Any] = {
        "window_rows": int(len(features)),
        "sequences_created": int(len(sequence_X)),
        "records_total": int(records_total),
        "records_used": int(records_used),
        "records_skipped_short": int(records_skipped_short),
        "subjects": int(metadata["subject_id"].nunique()),
        "class_distribution": {
            str(class_label): int(count)
            for class_label, count in zip(unique_classes, class_counts)
        },
        "sequence_length": int(sequence_length),
        "stride": int(stride),
        "target_position": target_position,
        "time_column": time_column,
        "record_group_id_source": record_group_id_source,
        "expected_step_seconds_config": expected_step_seconds,
        "max_gap_seconds_config": max_gap_seconds,
        "continuous_segments_total": int(continuous_segments_total),
        "records_with_gaps": int(records_with_gaps),
        "gaps_detected": int(gaps_detected),
        "largest_observed_gap": largest_observed_gap,
        "windows_excluded_due_to_gaps": int(windows_excluded_due_to_gaps),
        "candidate_sequences_without_gap_check": int(
            candidate_sequences_without_gap_check
        ),
        "valid_sequences_after_gap_check": int(len(sequence_X)),
        "sequences_rejected_due_to_gaps": int(
            candidate_sequences_without_gap_check - len(sequence_X)
        ),
        "full_target_count": int(len(eligible_endpoint_ids)),
        "sequence_endpoint_count": int(len(generated_endpoint_ids)),
        "dropped_no_history": int(
            len(eligible_endpoint_ids - candidate_endpoint_ids_without_gap)
        ),
        "dropped_gap": int(
            len(eligible_candidates_without_gap - generated_endpoint_ids)
        ),
        "dropped_other": int(
            len(
                eligible_endpoint_ids
                - generated_endpoint_ids
                - (eligible_endpoint_ids - candidate_endpoint_ids_without_gap)
                - (eligible_candidates_without_gap - generated_endpoint_ids)
            )
        ),
    }
    return SequenceBuildResult(
        X=sequence_X,
        y=sequence_y,
        metadata=sequence_metadata,
        stats=stats,
    )
