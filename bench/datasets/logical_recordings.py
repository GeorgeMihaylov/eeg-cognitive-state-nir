"""Logical-record identities and deterministic source-record deduplication."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


RAW_ALL_SOURCE_RECORDS = "raw_all_source_records"
RAW_DEDUPLICATED_LOGICAL_RECORDS = "raw_deduplicated_logical_records"
RAW_DATASET_MODES = {
    RAW_ALL_SOURCE_RECORDS,
    RAW_DEDUPLICATED_LOGICAL_RECORDS,
}
DEFAULT_SOURCE_PRIORITY = ("gpn_data", "Old_EEG")


def infer_record_group_id(record_id: str) -> str:
    """Strip only the source prefix from the canonical source-specific id."""
    normalized = str(record_id)
    if "__" not in normalized:
        return normalized
    source, logical_id = normalized.split("__", 1)
    if not source or not logical_id:
        raise ValueError(f"Invalid source-specific record_id {record_id!r}")
    return logical_id


def ensure_record_group_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate existing logical ids or derive them from source-specific ids."""
    if "record_id" not in frame:
        raise ValueError("Logical-record operations require record_id")
    result = frame.copy()
    inferred = result["record_id"].astype(str).map(infer_record_group_id)
    if "record_group_id" in result:
        existing = result["record_group_id"].astype(str)
        mismatch = existing != inferred
        if mismatch.any():
            examples = result.loc[
                mismatch, ["record_id", "record_group_id"]
            ].head(10).to_dict("records")
            raise ValueError(
                "record_group_id does not match canonical source-prefix stripping: "
                f"{examples}"
            )
    result["record_group_id"] = inferred
    return result


def build_deduplication_selection(
    manifest: pd.DataFrame,
    *,
    record_schema: Mapping[str, Mapping[str, Any]] | None = None,
    source_priority: Sequence[str] = DEFAULT_SOURCE_PRIORITY,
) -> pd.DataFrame:
    """Rank one source record per logical recording using a stable QC rule.

    Ranking is, in order: accepted-window fraction (higher), available raw EEG
    samples (higher), accepted-window missing fraction (lower), fixed source
    priority, then lexical ``record_id``.
    """
    frame = ensure_record_group_ids(manifest)
    required = {"record_id", "record_group_id", "source", "subject_id", "status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Deduplication manifest is missing columns: {missing}")
    source_rank = {
        str(source): rank for rank, source in enumerate(source_priority)
    }
    schema = record_schema or {}
    rows: list[dict[str, Any]] = []
    for record_id, record_rows in frame.groupby("record_id", sort=True):
        sources = record_rows["source"].astype(str).unique()
        subjects = record_rows["subject_id"].astype(str).unique()
        groups = record_rows["record_group_id"].astype(str).unique()
        if len(sources) != 1 or len(subjects) != 1 or len(groups) != 1:
            raise ValueError(f"Inconsistent metadata within record {record_id!r}")
        accepted = record_rows["status"].astype(str).eq("ok")
        accepted_count = int(accepted.sum())
        supervised_count = int(len(record_rows))
        missing_values = (
            pd.to_numeric(
                record_rows.loc[accepted, "missing_fraction"], errors="coerce"
            ).dropna()
            if "missing_fraction" in record_rows
            else pd.Series(dtype=float)
        )
        schema_row = schema.get(str(record_id), {})
        if "raw_n_rows" in record_rows:
            available_samples = pd.to_numeric(
                record_rows["raw_n_rows"], errors="coerce"
            ).max()
        else:
            available_samples = schema_row.get("n_rows", np.nan)
        rows.append({
            "record_group_id": str(groups[0]),
            "record_id": str(record_id),
            "source": str(sources[0]),
            "subject_id": str(subjects[0]),
            "supervised_windows": supervised_count,
            "accepted_raw_windows": accepted_count,
            "accepted_fraction": (
                accepted_count / supervised_count if supervised_count else 0.0
            ),
            "available_eeg_samples": (
                int(available_samples) if pd.notna(available_samples) else -1
            ),
            "mean_missing_fraction": (
                float(missing_values.mean()) if len(missing_values) else float("inf")
            ),
            "source_priority_rank": source_rank.get(
                str(sources[0]), len(source_rank)
            ),
        })
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("Cannot deduplicate an empty manifest")

    invalid_subjects = candidates.groupby("record_group_id")["subject_id"].nunique()
    invalid_subjects = invalid_subjects[invalid_subjects != 1]
    if len(invalid_subjects):
        raise ValueError(
            "Logical recordings map to multiple subjects: "
            f"{invalid_subjects.index.astype(str).tolist()}"
        )
    candidates = candidates.sort_values(
        [
            "record_group_id",
            "accepted_fraction",
            "available_eeg_samples",
            "mean_missing_fraction",
            "source_priority_rank",
            "record_id",
        ],
        ascending=[True, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    candidates["selection_rank"] = (
        candidates.groupby("record_group_id", sort=False).cumcount() + 1
    )
    candidates["selected"] = candidates["selection_rank"].eq(1)
    candidates["selection_reason"] = candidates.apply(
        lambda row: json.dumps(
            {
                "rule": [
                    "accepted_fraction_desc",
                    "available_eeg_samples_desc",
                    "mean_missing_fraction_asc",
                    "source_priority_asc",
                    "record_id_lexical_asc",
                ],
                "accepted_fraction": float(row["accepted_fraction"]),
                "available_eeg_samples": int(row["available_eeg_samples"]),
                "mean_missing_fraction": (
                    None
                    if not np.isfinite(row["mean_missing_fraction"])
                    else float(row["mean_missing_fraction"])
                ),
                "source_priority_rank": int(row["source_priority_rank"]),
                "record_id": str(row["record_id"]),
            },
            sort_keys=True,
        ),
        axis=1,
    )
    return candidates


def build_logical_recording_map(
    manifest: pd.DataFrame,
    *,
    record_schema: Mapping[str, Mapping[str, Any]] | None = None,
    source_priority: Sequence[str] = DEFAULT_SOURCE_PRIORITY,
) -> pd.DataFrame:
    """Collapse ranked source records to one auditable row per logical record."""
    frame = ensure_record_group_ids(manifest)
    selection = build_deduplication_selection(
        frame, record_schema=record_schema, source_priority=source_priority
    )
    records = frame.drop_duplicates("record_id").set_index("record_id")
    rows: list[dict[str, Any]] = []
    for group_id, candidates in selection.groupby("record_group_id", sort=True):
        group_rows = frame.loc[frame["record_group_id"].astype(str) == group_id]
        selected = candidates.loc[candidates["selected"]].iloc[0]
        label_counts = group_rows["label_q5"].value_counts().sort_index()
        source_records = candidates["record_id"].astype(str).tolist()
        sources = candidates["source"].astype(str).tolist()
        folds = sorted(
            pd.to_numeric(group_rows.get("outer_fold"), errors="coerce")
            .dropna().astype(int).unique().tolist()
        ) if "outer_fold" in group_rows else []
        schema_rows = [
            (record_schema or {}).get(record_id, {}) for record_id in source_records
        ]
        starts = [item.get("timestamp_min") for item in schema_rows]
        durations = [item.get("duration_seconds") for item in schema_rows]
        rows.append({
            "record_group_id": str(group_id),
            "source_record_ids": source_records,
            "sources": sources,
            "subject_id": str(candidates["subject_id"].iloc[0]),
            "start_datetimes": [
                pd.to_datetime(value, unit="s", utc=True).isoformat()
                if value is not None else None
                for value in starts
            ],
            "duration_seconds": [
                float(value) if value is not None else None for value in durations
            ],
            "supervised_windows": int(len(group_rows)),
            "accepted_raw_windows": int(group_rows["status"].eq("ok").sum()),
            "label_distribution": json.dumps({
                str(int(label)): int(count) for label, count in label_counts.items()
            }, sort_keys=True),
            "outer_folds": folds,
            "selected_record_id": str(selected["record_id"]),
            "selected_source": str(selected["source"]),
            "selection_reason": str(selected["selection_reason"]),
            "source_record_count": int(len(candidates)),
            "present_in_both_sources": bool(len(set(sources)) > 1),
        })
    return pd.DataFrame(rows)
