"""Canonical PM-union raw EEG manifest and composite-cache contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.tasks.target_registry import PM_TARGET_COLUMNS

from .logical_recordings import ensure_record_group_ids
from .raw_eeg_window_dataset import (
    CANONICAL_EEG_CHANNELS,
    infer_record_id,
)


PM_UNION_SCHEMA_VERSION = "raw-eeg-pm-union-composite-v1"


def pm_union_availability(
    frame: pd.DataFrame,
    target_columns: Sequence[str] = PM_TARGET_COLUMNS,
) -> np.ndarray:
    """Return rows with at least one finite canonical PM target."""
    missing = sorted(set(target_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"PM-union input is missing target columns: {missing}")
    values = frame.loc[:, list(target_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    return np.isfinite(values).any(axis=1)


def stable_sample_manifest_hash(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> str:
    """Hash an ordered manifest projection without local filesystem paths."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest hash columns are missing: {missing}")
    ordered = frame.loc[:, list(columns)].sort_values(
        list(columns), kind="stable"
    )
    payload = ordered.to_json(
        orient="records", date_format="iso", double_precision=15
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fixed_outer_fold_mapping(historical_manifest: pd.DataFrame) -> dict[str, int]:
    """Extract and validate the immutable subject-to-fold assignment."""
    required = {"subject_id", "outer_fold"}
    missing = sorted(required - set(historical_manifest.columns))
    if missing:
        raise ValueError(f"Historical manifest is missing fold columns: {missing}")
    pairs = historical_manifest.loc[:, ["subject_id", "outer_fold"]].copy()
    counts = pairs.groupby("subject_id")["outer_fold"].nunique()
    invalid = counts[counts != 1]
    if len(invalid):
        raise ValueError(
            "Historical subjects map to multiple outer folds: "
            f"{invalid.index.astype(str).tolist()}"
        )
    mapping = (
        pairs.drop_duplicates("subject_id")
        .set_index("subject_id")["outer_fold"]
        .astype(int)
        .to_dict()
    )
    return {str(subject): int(fold) for subject, fold in mapping.items()}


def _catalog_with_record_ids(catalog_path: Path | str) -> pd.DataFrame:
    path = Path(catalog_path)
    catalog = (
        pd.read_parquet(path)
        if path.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(path)
    )
    catalog = catalog.copy()
    catalog["record_id"] = catalog.apply(infer_record_id, axis=1)
    duplicates = catalog.loc[catalog["record_id"].duplicated(False), "record_id"]
    if len(duplicates):
        raise ValueError(
            "Ambiguous catalog record ids: "
            f"{sorted(duplicates.astype(str).unique().tolist())}"
        )
    return catalog


def _audit_records(audit_schema_path: Path | str | None) -> dict[str, Mapping[str, Any]]:
    if audit_schema_path is None:
        return {}
    path = Path(audit_schema_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw EEG audit schema not found: {path}")
    with path.open(encoding="utf-8") as input_file:
        document = json.load(input_file)
    return {
        str(row["record_id"]): row for row in document.get("records", [])
    }


def _delta_metadata(
    delta: pd.DataFrame,
    *,
    catalog: pd.DataFrame,
    audit_records: Mapping[str, Mapping[str, Any]],
    target_sfreq: float,
    preprocessing_hash: str,
    preprocessing_variant: str,
) -> pd.DataFrame:
    catalog_columns = [
        "record_id", "main_path", "main_rel_path", "header_row", "separator",
        "time_columns", "eeg_columns",
    ]
    missing_catalog_columns = sorted(set(catalog_columns) - set(catalog.columns))
    if missing_catalog_columns:
        raise ValueError(
            f"Raw EEG catalog is missing columns: {missing_catalog_columns}"
        )
    joined = delta.merge(
        catalog.loc[:, catalog_columns],
        on="record_id",
        how="left",
        validate="many_to_one",
    )
    joined["raw_file_path"] = joined["main_rel_path"].fillna(joined["main_path"])
    joined["sfreq_target"] = float(target_sfreq)
    joined["n_channels"] = len(CANONICAL_EEG_CHANNELS)
    joined["n_samples_expected"] = np.rint(
        (joined["t_end"] - joined["t_start"]) * float(target_sfreq)
    ).astype(np.int64)
    for column in (
        "sfreq_original", "raw_n_rows", "raw_file_size_bytes",
        "raw_timestamp_min", "raw_timestamp_max", "raw_duration_seconds",
        "raw_gap_count", "absolute_t_start", "absolute_t_end",
        "missing_fraction", "max_abs_amplitude", "max_flat_fraction",
    ):
        joined[column] = np.nan
    joined["status"] = np.where(
        joined["main_path"].notna() | joined["main_rel_path"].notna(),
        "pending",
        "unmatched",
    )
    joined["rejection_reason"] = ""
    joined["cache_file"] = ""
    joined["cache_offset"] = -1
    joined["preprocessing_hash"] = str(preprocessing_hash)
    joined["preprocessing_variant"] = str(preprocessing_variant)

    for record_id, indices in joined.groupby("record_id", sort=False).groups.items():
        audit = audit_records.get(str(record_id))
        if audit is None:
            continue
        index = np.asarray(list(indices))
        sfreq = float(audit["sampling_rate_hz"])
        origin = float(audit["window_origin_abs"])
        raw_min = float(audit["timestamp_min"])
        raw_max = float(audit["timestamp_max"])
        joined.loc[index, "sfreq_original"] = sfreq
        joined.loc[index, "raw_n_rows"] = int(audit["n_rows"])
        joined.loc[index, "raw_file_size_bytes"] = int(audit["file_size_bytes"])
        joined.loc[index, "raw_timestamp_min"] = raw_min
        joined.loc[index, "raw_timestamp_max"] = raw_max
        joined.loc[index, "raw_duration_seconds"] = float(
            audit["duration_seconds"]
        )
        joined.loc[index, "raw_gap_count"] = int(
            audit["gap_count_gt_1_5_nominal"]
        )
        joined.loc[index, "absolute_t_start"] = (
            origin + joined.loc[index, "t_start"]
        )
        joined.loc[index, "absolute_t_end"] = origin + joined.loc[index, "t_end"]
        out_of_range = (
            (joined.loc[index, "absolute_t_start"] < raw_min)
            | (joined.loc[index, "absolute_t_end"] > raw_max + 1.0 / sfreq)
        )
        rejected = index[np.asarray(out_of_range)]
        joined.loc[rejected, "status"] = "rejected"
        joined.loc[rejected, "rejection_reason"] = "out_of_range"
    return joined


def plan_pm_union_composite(
    processed_path: Path | str,
    catalog_path: Path | str,
    historical_manifest_path: Path | str,
    logical_recording_map_path: Path | str,
    *,
    audit_schema_path: Path | str | None = None,
    target_sfreq: float = 256.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build a read-only PM-union plan while preserving historical identities."""
    processed = pd.read_parquet(processed_path)
    required = {
        "record_id", "source", "subject_id", "t_start", "t_end",
        *PM_TARGET_COLUMNS,
    }
    missing = sorted(required - set(processed.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {missing}")
    if "sample_id" in processed.columns:
        sample_ids = pd.to_numeric(processed["sample_id"], errors="raise")
    else:
        sample_ids = pd.Series(processed.index.to_numpy(), index=processed.index)
    if sample_ids.duplicated().any():
        raise ValueError("Processed dataset contains duplicate canonical sample_id")
    union = processed.loc[pm_union_availability(processed)].copy()
    canonical_ids = sample_ids.loc[union.index].to_numpy(np.int64)
    if "sample_id" in union.columns:
        union.loc[:, "sample_id"] = canonical_ids
    else:
        union.insert(0, "sample_id", canonical_ids)
    union = ensure_record_group_ids(union)

    historical = ensure_record_group_ids(
        pd.read_parquet(historical_manifest_path)
    )
    if historical["sample_id"].duplicated().any():
        raise ValueError("Historical raw-v3 manifest contains duplicate sample_id")
    historical_ids = set(historical["sample_id"].astype(np.int64))
    union_ids = set(union["sample_id"].astype(np.int64))
    outside = sorted(historical_ids - union_ids)
    if outside:
        raise ValueError(
            "Historical raw-v3 contains sample_id outside PM-union: "
            f"{outside[:20]}"
        )

    folds = fixed_outer_fold_mapping(historical)
    union["outer_fold"] = union["subject_id"].astype(str).map(folds)
    if union["outer_fold"].isna().any():
        missing_subjects = sorted(
            union.loc[union["outer_fold"].isna(), "subject_id"]
            .astype(str).unique().tolist()
        )
        raise ValueError(
            f"PM-union subjects are missing fixed outer folds: {missing_subjects}"
        )
    union["outer_fold"] = union["outer_fold"].astype(int)

    logical_map = pd.read_parquet(logical_recording_map_path)
    required_map = {"record_group_id", "selected_record_id"}
    missing_map = sorted(required_map - set(logical_map.columns))
    if missing_map:
        raise ValueError(f"Logical recording map is missing columns: {missing_map}")
    if logical_map["record_group_id"].astype(str).duplicated().any():
        raise ValueError("Logical recording map contains duplicate record_group_id")
    selected_by_group = dict(zip(
        logical_map["record_group_id"].astype(str),
        logical_map["selected_record_id"].astype(str),
    ))
    missing_groups = sorted(set(union["record_group_id"].astype(str)) - set(selected_by_group))
    if missing_groups:
        raise ValueError(
            "PM-union contains logical records absent from fixed selection: "
            f"{missing_groups}"
        )

    preprocessing_hashes = historical["preprocessing_hash"].dropna().astype(str).unique()
    variants = historical["preprocessing_variant"].dropna().astype(str).unique()
    if len(preprocessing_hashes) != 1 or len(variants) != 1:
        raise ValueError("Historical raw-v3 must have one preprocessing identity")

    delta = union.loc[~union["sample_id"].isin(historical_ids)].copy()
    keep = [
        "sample_id", "source", "subject_id", "record_id", "record_group_id",
        "t_start", "t_end", "outer_fold",
    ]
    delta = _delta_metadata(
        delta.loc[:, keep],
        catalog=_catalog_with_record_ids(catalog_path),
        audit_records=_audit_records(audit_schema_path),
        target_sfreq=target_sfreq,
        preprocessing_hash=str(preprocessing_hashes[0]),
        preprocessing_variant=str(variants[0]),
    )

    historical_copy = historical.copy()
    all_columns = list(historical_copy.columns)
    all_columns.extend(column for column in delta.columns if column not in all_columns)
    for column in all_columns:
        if column not in historical_copy:
            historical_copy[column] = np.nan
        if column not in delta:
            delta[column] = np.nan
    composite_plan = pd.concat(
        [historical_copy.loc[:, all_columns], delta.loc[:, all_columns]],
        ignore_index=True,
    ).sort_values("sample_id", kind="stable").reset_index(drop=True)
    if composite_plan["sample_id"].duplicated().any():
        raise RuntimeError("Composite PM-union plan contains duplicate sample_id")

    selected = composite_plan["record_id"].astype(str).eq(
        composite_plan["record_group_id"].astype(str).map(selected_by_group)
    )
    selected_union = union["record_id"].astype(str).eq(
        union["record_group_id"].astype(str).map(selected_by_group)
    )
    selected_delta = delta["record_id"].astype(str).eq(
        delta["record_group_id"].astype(str).map(selected_by_group)
    )
    summary = {
        "schema_version": PM_UNION_SCHEMA_VERSION,
        "processed_rows": int(len(processed)),
        "candidate_source_specific_rows": int(len(union)),
        "historical_rows_reused": int(len(historical_copy)),
        "delta_candidate_rows": int(len(delta)),
        "delta_metadata_buildable_rows": int(delta["status"].eq("pending").sum()),
        "delta_rejected_rows": int(delta["status"].eq("rejected").sum()),
        "delta_rejection_reasons": {
            str(key): int(value)
            for key, value in delta.loc[delta["status"] != "pending", "rejection_reason"]
            .value_counts().items()
        },
        "delta_deduplicated_candidate_rows": int(selected_delta.sum()),
        "delta_deduplicated_metadata_buildable_rows": int(
            (selected_delta & delta["status"].eq("pending")).sum()
        ),
        "delta_deduplicated_rejected_rows": int(
            (selected_delta & delta["status"].eq("rejected")).sum()
        ),
        "candidate_deduplicated_rows": int(selected.sum()),
        "target_candidate_deduplicated_rows": {
            column: int(
                (
                    selected_union
                    & np.isfinite(
                        pd.to_numeric(union[column], errors="coerce")
                        .to_numpy(dtype=float)
                    )
                ).sum()
            )
            for column in PM_TARGET_COLUMNS
        },
        "subjects": int(composite_plan["subject_id"].nunique()),
        "source_records": int(composite_plan["record_id"].nunique()),
        "logical_records": int(composite_plan["record_group_id"].nunique()),
        "subject_counts_by_outer_fold": {
            str(int(key)): int(value)
            for key, value in (
                composite_plan.drop_duplicates("subject_id")["outer_fold"]
                .value_counts().sort_index().items()
            )
        },
        "preprocessing_hash": str(preprocessing_hashes[0]),
        "preprocessing_variant": str(variants[0]),
        "selected_record_mapping_hash": stable_sample_manifest_hash(
            logical_map,
            columns=("record_group_id", "selected_record_id"),
        ),
        "candidate_manifest_hash": stable_sample_manifest_hash(
            composite_plan,
            columns=(
                "sample_id", "subject_id", "record_id", "record_group_id",
                "outer_fold", "t_start", "t_end",
            ),
        ),
    }
    return composite_plan, delta, summary


def finalize_pm_union_composite(
    historical_manifest: pd.DataFrame,
    built_delta: pd.DataFrame,
    *,
    expected_preprocessing_hash: str,
) -> pd.DataFrame:
    """Combine immutable historical rows with independently built delta rows."""
    historical = historical_manifest.copy()
    delta = built_delta.copy()
    overlap = np.intersect1d(historical["sample_id"], delta["sample_id"])
    if len(overlap):
        raise ValueError(
            f"Historical and delta manifests overlap sample_id: {overlap[:20].tolist()}"
        )
    for name, frame in (("historical", historical), ("delta", delta)):
        hashes = frame["preprocessing_hash"].dropna().astype(str).unique()
        if hashes.tolist() != [str(expected_preprocessing_hash)]:
            raise ValueError(
                f"{name} preprocessing hash is incompatible: {hashes.tolist()}"
            )
    columns = list(historical.columns)
    columns.extend(column for column in delta.columns if column not in columns)
    for column in columns:
        if column not in historical:
            historical[column] = np.nan
        if column not in delta:
            delta[column] = np.nan
    result = pd.concat(
        [historical.loc[:, columns], delta.loc[:, columns]], ignore_index=True
    ).sort_values("sample_id", kind="stable").reset_index(drop=True)
    if result["sample_id"].duplicated().any():
        raise RuntimeError("Final composite manifest contains duplicate sample_id")
    return result
