"""Timestamp-aligned raw EEG windows and a lazy record-shard dataset."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.model_selection import GroupKFold

from ..core.abstract_dataset import BaseDataset, EEGData
from ..tasks.target_registry import resolve_target_spec
from .logical_recordings import (
    DEFAULT_SOURCE_PRIORITY,
    RAW_ALL_SOURCE_RECORDS,
    RAW_DATASET_MODES,
    RAW_DEDUPLICATED_LOGICAL_RECORDS,
    build_deduplication_selection,
    ensure_record_group_ids,
    infer_record_group_id,
)
from .channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
from .raw_preprocessing import (
    apply_raw_preprocessing,
    normalize_raw_preprocessing,
    preprocessing_variant_name,
    raw_preprocessing_hash,
    raw_window_artifact_metrics,
)
from .target_view import attach_targets_by_sample_id, build_target_view


# Backward-compatible public name; the canonical value lives in one shared
# production contract used by both raw-window and cross-dataset selection.
CANONICAL_EEG_CHANNELS = PROJECT_EMOTIV_CHANNEL_ORDER
RAW_LOADER_VERSION = "raw-eeg-window-v3"


class RawEEGWindowError(ValueError):
    """A rejected window with a machine-readable reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a serialized list, got {value!r}")
    return [str(item) for item in parsed]


def infer_record_id(row: Mapping[str, Any]) -> str:
    """Match the record-id algorithm used by the processed dataset builders."""
    values = [
        row.get("source"),
        row.get("subject_id"),
        row.get("day"),
        row.get("part"),
        row.get("datetime_from_name"),
    ]
    normalized = [
        "" if value is None or pd.isna(value) else str(value)
        for value in values
    ]
    record_id = "__".join(normalized)
    record_id = re.sub(r"[\s:/\\+]+", "p", record_id)
    return record_id.strip("_")


def resolve_raw_path(record: Mapping[str, Any], repo_root: Path | str = ".") -> Path:
    """Resolve catalog paths without baking a machine-specific root into configs."""
    absolute = record.get("main_path")
    if absolute is not None and not pd.isna(absolute):
        path = Path(str(absolute))
        if path.exists():
            return path
    relative = record.get("main_rel_path")
    if relative is None or pd.isna(relative):
        raise FileNotFoundError("Catalog record has neither a usable main_path nor main_rel_path")
    path = Path(repo_root) / Path(str(relative))
    if not path.exists():
        raise FileNotFoundError(f"Raw EEG file does not exist: {path}")
    return path


@dataclass(frozen=True)
class RawEEGRecord:
    """One raw recording kept in memory only while its shards are built."""

    timestamps: np.ndarray
    signals: np.ndarray
    channels: tuple[str, ...]
    sampling_rate: float
    window_origin_abs: float
    raw_path: Path


def _estimate_sampling_rate(timestamps: np.ndarray) -> float:
    unique = np.unique(timestamps[np.isfinite(timestamps)])
    positive_deltas = np.diff(unique)
    positive_deltas = positive_deltas[positive_deltas > 0]
    if len(positive_deltas) == 0:
        raise RawEEGWindowError(
            "invalid_timestamps", "Cannot estimate sampling rate from timestamps"
        )
    median_delta = float(np.median(positive_deltas))
    sampling_rate = 1.0 / median_delta
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise RawEEGWindowError(
            "invalid_sampling_rate", f"Invalid estimated sampling rate {sampling_rate}"
        )
    return sampling_rate


def load_raw_eeg_record(
    record: Mapping[str, Any],
    *,
    channels: Sequence[str] = CANONICAL_EEG_CHANNELS,
    repo_root: Path | str = ".",
) -> RawEEGRecord:
    """Read one catalog record using only timestamp and canonical EEG columns."""
    raw_path = resolve_raw_path(record, repo_root=repo_root)
    catalog_eeg = set(_parse_list(record.get("eeg_columns")))
    missing_channels = [channel for channel in channels if channel not in catalog_eeg]
    if missing_channels:
        raise RawEEGWindowError(
            "missing_channels",
            f"{raw_path.name} is missing canonical channels {missing_channels}",
        )
    time_columns = _parse_list(record.get("time_columns"))
    if "Timestamp" not in time_columns:
        raise RawEEGWindowError(
            "missing_timestamp", f"{raw_path.name} does not contain Timestamp"
        )
    columns = ["Timestamp", *channels]
    frame = pd.read_csv(
        raw_path,
        header=int(record.get("header_row", 0)),
        sep=str(record.get("separator", ",")),
        usecols=columns,
        low_memory=False,
    )
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    timestamps = numeric["Timestamp"].to_numpy(dtype=np.float64)
    signals = numeric[list(channels)].to_numpy(dtype=np.float32)
    finite_time = np.isfinite(timestamps)
    timestamps = timestamps[finite_time]
    signals = signals[finite_time]
    if len(timestamps) < 2:
        raise RawEEGWindowError(
            "insufficient_samples", f"{raw_path.name} has fewer than two timestamped rows"
        )
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    signals = signals[order]
    sampling_rate = _estimate_sampling_rate(timestamps)
    first_window_id = math.floor(float(timestamps[0]) / 10.0)
    return RawEEGRecord(
        timestamps=timestamps,
        signals=signals,
        channels=tuple(channels),
        sampling_rate=sampling_rate,
        window_origin_abs=(first_window_id + 0.5) * 10.0,
        raw_path=raw_path,
    )


def _deduplicate_samples(
    timestamps: np.ndarray, signals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse, counts = np.unique(
        timestamps, return_inverse=True, return_counts=True
    )
    if len(unique) == len(timestamps):
        return timestamps, signals
    sums = np.zeros((len(unique), signals.shape[1]), dtype=np.float64)
    finite_counts = np.zeros_like(sums, dtype=np.int64)
    for channel_index in range(signals.shape[1]):
        values = signals[:, channel_index]
        finite = np.isfinite(values)
        np.add.at(sums[:, channel_index], inverse[finite], values[finite])
        np.add.at(finite_counts[:, channel_index], inverse[finite], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = sums / finite_counts
    averaged[finite_counts == 0] = np.nan
    return unique, averaged.astype(np.float32)


def extract_raw_eeg_window(
    raw_record: RawEEGRecord,
    t_start: float,
    t_end: float,
    *,
    target_sfreq: float,
    max_missing_fraction: float = 0.02,
    raw_preprocessing: Optional[Mapping[str, Any]] = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Extract a fixed-grid window, filtering an interval with two-sided padding."""
    if not np.isfinite([t_start, t_end, target_sfreq]).all():
        raise RawEEGWindowError("invalid_request", "Window bounds and sfreq must be finite")
    if t_end <= t_start:
        raise RawEEGWindowError("invalid_request", "t_end must be greater than t_start")
    if target_sfreq <= 0:
        raise RawEEGWindowError("invalid_request", "target_sfreq must be positive")
    if not 0 <= max_missing_fraction < 1:
        raise RawEEGWindowError(
            "invalid_request", "max_missing_fraction must be in [0, 1)"
        )

    preprocessing = normalize_raw_preprocessing(
        raw_preprocessing, default_resample_hz=target_sfreq
    )
    if not math.isclose(
        preprocessing["resample_hz"], target_sfreq, rel_tol=0, abs_tol=1e-9
    ):
        raise RawEEGWindowError(
            "invalid_request",
            "target_sfreq must equal raw_preprocessing.resample_hz",
        )

    absolute_start = raw_record.window_origin_abs + float(t_start)
    absolute_end = raw_record.window_origin_abs + float(t_end)
    duration = float(t_end - t_start)
    filtering_enabled = bool(
        preprocessing["bandpass"]["enabled"]
        or preprocessing["notch"]["enabled"]
    )
    filter_padding_seconds = 2.0 if filtering_enabled else 0.0
    interval_start = absolute_start - filter_padding_seconds
    interval_end = absolute_end + filter_padding_seconds
    margin = 2.0 / min(raw_record.sampling_rate, target_sfreq)
    left = int(np.searchsorted(
        raw_record.timestamps, interval_start - margin, side="left"
    ))
    right = int(np.searchsorted(
        raw_record.timestamps, interval_end + margin, side="right"
    ))
    timestamps = raw_record.timestamps[left:right]
    signals = raw_record.signals[left:right]
    if len(timestamps) < 2:
        raise RawEEGWindowError(
            "out_of_range", f"No usable samples in [{absolute_start}, {absolute_end})"
        )
    timestamps, signals = _deduplicate_samples(timestamps, signals)

    original_sfreq = float(raw_record.sampling_rate)
    nearest_nominal = min(
        (128.0, 256.0), key=lambda rate: abs(rate - original_sfreq)
    )
    grid_sfreq = (
        nearest_nominal
        if math.isclose(original_sfreq, nearest_nominal, rel_tol=0.005)
        else original_sfreq
    )
    rates_match = math.isclose(grid_sfreq, target_sfreq, rel_tol=0.005)
    source_count = int(round(duration * grid_sfreq))
    target_count = int(round(duration * target_sfreq))
    if source_count <= 1 or target_count <= 1:
        raise RawEEGWindowError("insufficient_samples", "Window expects fewer than two samples")
    source_grid = absolute_start + np.arange(
        source_count, dtype=np.float64
    ) / grid_sfreq
    tolerance = 0.75 / grid_sfreq
    nearest = np.searchsorted(timestamps, source_grid, side="left")
    nearest = np.clip(nearest, 0, len(timestamps) - 1)
    previous = np.maximum(nearest - 1, 0)
    use_previous = (
        np.abs(timestamps[previous] - source_grid)
        < np.abs(timestamps[nearest] - source_grid)
    )
    nearest[use_previous] = previous[use_previous]
    time_present = np.abs(timestamps[nearest] - source_grid) <= tolerance
    missing_by_channel = []
    for channel_index in range(signals.shape[1]):
        channel_present = time_present & np.isfinite(signals[nearest, channel_index])
        missing_by_channel.append(1.0 - float(channel_present.mean()))
    missing_fraction = max(missing_by_channel, default=1.0)
    if missing_fraction > max_missing_fraction:
        raise RawEEGWindowError(
            "missing_fraction_exceeded",
            f"Missing fraction {missing_fraction:.6f} exceeds {max_missing_fraction:.6f}",
        )

    interval_duration = duration + 2.0 * filter_padding_seconds
    interval_source_count = int(round(interval_duration * grid_sfreq))
    interval_grid = interval_start + np.arange(
        interval_source_count, dtype=np.float64
    ) / grid_sfreq
    regularized = np.empty(
        (signals.shape[1], interval_source_count), dtype=np.float32
    )
    for channel_index in range(signals.shape[1]):
        values = signals[:, channel_index]
        finite = np.isfinite(values)
        if finite.sum() < 2:
            raise RawEEGWindowError(
                "insufficient_channel_samples",
                f"Channel {raw_record.channels[channel_index]} has fewer than two samples",
            )
        regularized[channel_index] = np.interp(
            interval_grid,
            timestamps[finite],
            values[finite],
        ).astype(np.float32)

    resampled = not rates_match
    if resampled:
        ratio = Fraction(target_sfreq / grid_sfreq).limit_denominator(1000)
        padded_window = resample_poly(
            regularized,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=1,
        ).astype(np.float32, copy=False)
    else:
        padded_window = regularized
    padded_target_count = int(round(interval_duration * target_sfreq))
    if padded_window.shape[1] < padded_target_count:
        pad = padded_target_count - padded_window.shape[1]
        padded_window = np.pad(padded_window, ((0, 0), (0, pad)), mode="edge")
    elif padded_window.shape[1] > padded_target_count:
        padded_window = padded_window[:, :padded_target_count]
    try:
        padded_window = apply_raw_preprocessing(
            padded_window,
            sampling_rate=target_sfreq,
            config=preprocessing,
        )
    except ValueError as exc:
        raise RawEEGWindowError("preprocessing_error", str(exc)) from exc
    trim_start = int(round(filter_padding_seconds * target_sfreq))
    window = padded_window[:, trim_start:trim_start + target_count]
    window = np.ascontiguousarray(window, dtype=np.float32)
    if window.shape != (len(raw_record.channels), target_count):
        raise RuntimeError(f"Unexpected raw window shape {window.shape}")
    if not np.isfinite(window).all():
        raise RawEEGWindowError(
            "non_finite_output", "Window reconstruction produced NaN or Inf"
        )
    artifact_metrics = raw_window_artifact_metrics(window)
    rejection = preprocessing["artifact_rejection"]
    if rejection["enabled"]:
        max_abs = rejection["max_abs_amplitude"]
        if max_abs is not None and artifact_metrics["max_abs_amplitude"] > max_abs:
            raise RawEEGWindowError(
                "artifact_max_abs_amplitude",
                f"Window max absolute amplitude "
                f"{artifact_metrics['max_abs_amplitude']:.6f} exceeds {max_abs:.6f}",
            )
        max_flat = rejection["max_flat_fraction"]
        if max_flat is not None and artifact_metrics["max_flat_fraction"] > max_flat:
            raise RawEEGWindowError(
                "artifact_flat_fraction",
                f"Window max flat fraction "
                f"{artifact_metrics['max_flat_fraction']:.6f} exceeds {max_flat:.6f}",
            )
    return window, {
        "absolute_start": absolute_start,
        "absolute_end": absolute_end,
        "missing_fraction": missing_fraction,
        "sfreq_original": original_sfreq,
        "sfreq_regularized": grid_sfreq,
        "sfreq_target": float(target_sfreq),
        "resampled": resampled,
        "n_samples_source_grid": source_count,
        "n_samples_target": target_count,
        "filter_padding_seconds": filter_padding_seconds,
        "raw_preprocessing": preprocessing,
        **artifact_metrics,
    }


def load_raw_eeg_window(
    record: Mapping[str, Any],
    t_start: float,
    t_end: float,
    target_sfreq: float,
    channels: Sequence[str] = CANONICAL_EEG_CHANNELS,
    *,
    max_missing_fraction: float = 0.02,
    repo_root: Path | str = ".",
    raw_preprocessing: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Public one-window loader returning float32 ``[channel, time]``."""
    raw_record = load_raw_eeg_record(record, channels=channels, repo_root=repo_root)
    window, _ = extract_raw_eeg_window(
        raw_record,
        t_start,
        t_end,
        target_sfreq=target_sfreq,
        max_missing_fraction=max_missing_fraction,
        raw_preprocessing=raw_preprocessing,
    )
    return window


def assign_outer_group_folds(
    frame: pd.DataFrame,
    *,
    group_column: str = "subject_id",
    n_splits: int = 5,
) -> np.ndarray:
    """Assign folds before raw QC so subject lists match the feature benchmarks."""
    groups = frame[group_column].astype(str).to_numpy()
    folds = np.zeros(len(frame), dtype=np.int16)
    splitter = GroupKFold(n_splits=n_splits)
    for fold_index, (_, test_idx) in enumerate(
        splitter.split(np.zeros((len(frame), 1)), frame["label_q5"], groups), start=1
    ):
        folds[test_idx] = fold_index
    if np.any(folds == 0):
        raise RuntimeError("Outer fold assignment left unassigned supervised rows")
    return folds


def build_raw_window_index(
    processed_path: Path | str,
    catalog_path: Path | str,
    *,
    audit_schema_path: Optional[Path | str] = None,
    target_sfreq: float = 256.0,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Join label_q5 windows to catalog records without reading raw signals."""
    processed = pd.read_parquet(processed_path)
    required = {
        "record_id", "source", "subject_id", "t_start", "t_end", "label_q5"
    }
    missing = sorted(required - set(processed.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {missing}")
    supervised = processed.loc[processed["label_q5"].notna()].copy()
    supervised.insert(0, "sample_id", supervised.index.to_numpy(dtype=np.int64))
    supervised["label_q5"] = supervised["label_q5"].astype(np.int64)
    supervised["outer_fold"] = assign_outer_group_folds(
        supervised, n_splits=n_splits
    )

    catalog = pd.read_csv(catalog_path)
    catalog["record_id"] = catalog.apply(infer_record_id, axis=1)
    duplicate_ids = catalog.loc[catalog["record_id"].duplicated(False), "record_id"]
    if len(duplicate_ids):
        raise ValueError(
            f"Ambiguous catalog record ids: {sorted(duplicate_ids.unique().tolist())}"
        )
    catalog_columns = [
        "record_id", "main_path", "main_rel_path", "header_row", "separator",
        "time_columns", "eeg_columns",
    ]
    joined = supervised.merge(
        catalog[catalog_columns], on="record_id", how="left", validate="many_to_one"
    )
    joined["record_group_id"] = joined["record_id"].astype(str).map(
        infer_record_group_id
    )
    joined["raw_file_path"] = joined["main_rel_path"].fillna(joined["main_path"])
    joined["sfreq_target"] = float(target_sfreq)
    joined["n_channels"] = len(CANONICAL_EEG_CHANNELS)
    joined["n_samples_expected"] = np.rint(
        (joined["t_end"] - joined["t_start"]) * target_sfreq
    ).astype(np.int64)
    joined["sfreq_original"] = np.nan
    joined["raw_n_rows"] = np.nan
    joined["raw_file_size_bytes"] = np.nan
    joined["raw_timestamp_min"] = np.nan
    joined["raw_timestamp_max"] = np.nan
    joined["raw_duration_seconds"] = np.nan
    joined["raw_gap_count"] = np.nan
    joined["absolute_t_start"] = np.nan
    joined["absolute_t_end"] = np.nan
    joined["status"] = np.where(joined["main_path"].notna(), "pending", "unmatched")
    joined["rejection_reason"] = ""

    audit_records: Dict[str, Mapping[str, Any]] = {}
    if audit_schema_path is not None and Path(audit_schema_path).exists():
        with open(audit_schema_path, encoding="utf-8") as input_file:
            schema = json.load(input_file)
        audit_records = {
            str(item["record_id"]): item for item in schema.get("records", [])
        }
    for record_id, indices in joined.groupby("record_id", sort=False).groups.items():
        audit = audit_records.get(str(record_id))
        if not audit:
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
        joined.loc[index, "raw_duration_seconds"] = float(audit["duration_seconds"])
        joined.loc[index, "raw_gap_count"] = int(audit["gap_count_gt_1_5_nominal"])
        joined.loc[index, "absolute_t_start"] = origin + joined.loc[index, "t_start"]
        joined.loc[index, "absolute_t_end"] = origin + joined.loc[index, "t_end"]
        out_of_range = (
            (joined.loc[index, "absolute_t_start"] < raw_min)
            | (joined.loc[index, "absolute_t_end"] > raw_max + 1.0 / sfreq)
        )
        rejected_index = index[np.asarray(out_of_range)]
        joined.loc[rejected_index, "status"] = "rejected"
        joined.loc[rejected_index, "rejection_reason"] = "out_of_range"

    output_columns = [
        "sample_id", "source", "subject_id", "record_id", "record_group_id",
        "raw_file_path",
        "t_start", "t_end", "absolute_t_start", "absolute_t_end", "label_q5",
        "sfreq_original", "sfreq_target", "n_channels", "n_samples_expected",
        "raw_n_rows", "raw_file_size_bytes", "raw_timestamp_min",
        "raw_timestamp_max", "raw_duration_seconds", "raw_gap_count",
        "outer_fold", "status", "rejection_reason", "main_path", "main_rel_path",
        "header_row", "separator", "time_columns", "eeg_columns",
    ]
    index = joined[output_columns].copy()
    matching = {
        "supervised_windows": int(len(supervised)),
        "supervised_records": int(supervised["record_id"].nunique()),
        "catalog_records": int(len(catalog)),
        "matched_records": int(supervised["record_id"].isin(catalog["record_id"]).groupby(
            supervised["record_id"]
        ).any().sum()),
        "unmatched_records": sorted(
            set(supervised["record_id"].astype(str)) - set(catalog["record_id"].astype(str))
        ),
        "ambiguous_records": sorted(duplicate_ids.astype(str).unique().tolist()),
        "status_counts": {
            str(key): int(value) for key, value in index["status"].value_counts().items()
        },
    }
    return index, matching


def _cache_key(record_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id)[:80]
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable}_{digest}"


def _cache_config_hash(
    record: Mapping[str, Any],
    raw_path: Path,
    channels: Sequence[str],
    target_sfreq: float,
    max_missing_fraction: float,
    raw_preprocessing: Optional[Mapping[str, Any]] = None,
) -> str:
    stat = raw_path.stat()
    payload = {
        "version": RAW_LOADER_VERSION,
        "record_id": str(record["record_id"]),
        "raw_path": str(raw_path.resolve()),
        "raw_size": stat.st_size,
        "raw_mtime_ns": stat.st_mtime_ns,
        "channels": list(channels),
        "target_sfreq": float(target_sfreq),
        "max_missing_fraction": float(max_missing_fraction),
        "raw_preprocessing": normalize_raw_preprocessing(
            raw_preprocessing, default_resample_hz=target_sfreq
        ),
        "windows": [
            [int(row.sample_id), float(row.t_start), float(row.t_end)]
            for row in pd.DataFrame(record["windows"]).itertuples(index=False)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_cache_shard(
    array_path: Path,
    metadata_path: Path,
    expected_hash: str,
    expected_shape_tail: tuple[int, int],
) -> Optional[Dict[str, Any]]:
    if not array_path.exists() or not metadata_path.exists():
        return None
    try:
        with open(metadata_path, encoding="utf-8") as input_file:
            metadata = json.load(input_file)
        if metadata.get("config_hash") != expected_hash:
            return None
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(metadata["accepted_windows"]), *expected_shape_tail)
        if array.dtype != np.float32 or array.shape != expected_shape:
            return None
        if array.size and not np.isfinite(array[0]).all():
            return None
        return metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def build_raw_eeg_cache(
    index: pd.DataFrame,
    cache_dir: Path | str,
    *,
    channels: Sequence[str] = CANONICAL_EEG_CHANNELS,
    target_sfreq: float = 256.0,
    max_missing_fraction: float = 0.02,
    repo_root: Path | str = ".",
    record_limit: Optional[int] = None,
    raw_preprocessing: Optional[Mapping[str, Any]] = None,
    use_hash_subdirectory: bool = True,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Build or reuse memory-mappable record shards and update window statuses."""
    result = index.copy()
    result["cache_file"] = ""
    result["cache_offset"] = -1
    result["missing_fraction"] = np.nan
    result["max_abs_amplitude"] = np.nan
    result["max_flat_fraction"] = np.nan
    preprocessing = normalize_raw_preprocessing(
        raw_preprocessing, default_resample_hz=target_sfreq
    )
    target_sfreq = float(preprocessing["resample_hz"])
    preprocessing_hash = raw_preprocessing_hash(
        preprocessing, channels=channels, default_resample_hz=target_sfreq
    )
    result["sfreq_target"] = target_sfreq
    result["preprocessing_hash"] = preprocessing_hash
    result["preprocessing_variant"] = preprocessing_variant_name(preprocessing)
    cache_root = Path(cache_dir)
    if use_hash_subdirectory:
        namespace = (
            f"{preprocessing_variant_name(preprocessing)}-"
            f"{preprocessing_hash[:16]}"
        )
        cache_root = cache_root / namespace
    cache_root.mkdir(parents=True, exist_ok=True)
    groups = list(result.groupby("record_id", sort=False))
    if record_limit is not None:
        groups = groups[: int(record_limit)]
    reused_records = 0
    built_records = 0
    for record_id, rows in groups:
        pending = rows.loc[rows["status"] == "pending"]
        if pending.empty:
            continue
        catalog_record = pending.iloc[0].to_dict()
        raw_path = resolve_raw_path(catalog_record, repo_root=repo_root)
        key = _cache_key(str(record_id))
        array_path = cache_root / f"{key}.npy"
        metadata_path = cache_root / f"{key}.json"
        record_payload = dict(catalog_record)
        record_payload["record_id"] = str(record_id)
        record_payload["windows"] = pending[["sample_id", "t_start", "t_end"]].to_dict("records")
        config_hash = _cache_config_hash(
            record_payload,
            raw_path,
            channels,
            target_sfreq,
            max_missing_fraction,
            preprocessing,
        )
        target_count = int(round(
            float((pending["t_end"] - pending["t_start"]).iloc[0]) * target_sfreq
        ))
        cached = _valid_cache_shard(
            array_path,
            metadata_path,
            config_hash,
            (len(channels), target_count),
        )
        if cached is not None:
            window_results = cached.get("window_results", [])
            reused_records += 1
        else:
            raw_record = load_raw_eeg_record(
                catalog_record, channels=channels, repo_root=repo_root
            )
            accepted: list[np.ndarray] = []
            window_results = []
            for row in pending.itertuples():
                try:
                    window, diagnostics = extract_raw_eeg_window(
                        raw_record,
                        float(row.t_start),
                        float(row.t_end),
                        target_sfreq=target_sfreq,
                        max_missing_fraction=max_missing_fraction,
                        raw_preprocessing=preprocessing,
                    )
                    offset = len(accepted)
                    accepted.append(window)
                    window_results.append({
                        "sample_id": int(row.sample_id),
                        "status": "ok",
                        "rejection_reason": "",
                        "cache_offset": offset,
                        "missing_fraction": float(diagnostics["missing_fraction"]),
                        "sfreq_original": float(diagnostics["sfreq_original"]),
                        "max_abs_amplitude": float(
                            diagnostics["max_abs_amplitude"]
                        ),
                        "max_flat_fraction": float(
                            diagnostics["max_flat_fraction"]
                        ),
                    })
                except RawEEGWindowError as exc:
                    window_results.append({
                        "sample_id": int(row.sample_id),
                        "status": "rejected",
                        "rejection_reason": exc.reason,
                        "cache_offset": -1,
                        "missing_fraction": None,
                        "sfreq_original": float(raw_record.sampling_rate),
                    })
            array = (
                np.stack(accepted).astype(np.float32, copy=False)
                if accepted
                else np.empty((0, len(channels), target_count), dtype=np.float32)
            )
            np.save(array_path, array, allow_pickle=False)
            metadata = {
                "config_hash": config_hash,
                "loader_version": RAW_LOADER_VERSION,
                "record_id": str(record_id),
                "raw_file_path": str(raw_path),
                "channels": list(channels),
                "sfreq_original": float(raw_record.sampling_rate),
                "sfreq_target": float(target_sfreq),
                "raw_preprocessing": preprocessing,
                "preprocessing_hash": preprocessing_hash,
                "accepted_windows": int(len(accepted)),
                "window_results": window_results,
            }
            with open(metadata_path, "w", encoding="utf-8") as output_file:
                json.dump(metadata, output_file, indent=2)
            built_records += 1
        by_sample = {int(item["sample_id"]): item for item in window_results}
        for row_index in pending.index:
            sample_id = int(result.at[row_index, "sample_id"])
            item = by_sample.get(sample_id)
            if item is None:
                result.at[row_index, "status"] = "rejected"
                result.at[row_index, "rejection_reason"] = "cache_manifest_missing"
                continue
            result.at[row_index, "status"] = item["status"]
            result.at[row_index, "rejection_reason"] = item["rejection_reason"]
            result.at[row_index, "cache_offset"] = int(item["cache_offset"])
            result.at[row_index, "sfreq_original"] = float(item["sfreq_original"])
            if item["missing_fraction"] is not None:
                result.at[row_index, "missing_fraction"] = float(item["missing_fraction"])
            if item.get("max_abs_amplitude") is not None:
                result.at[row_index, "max_abs_amplitude"] = float(
                    item["max_abs_amplitude"]
                )
            if item.get("max_flat_fraction") is not None:
                result.at[row_index, "max_flat_fraction"] = float(
                    item["max_flat_fraction"]
                )
            if item["status"] == "ok":
                result.at[row_index, "cache_file"] = str(array_path)

    remaining = result["status"] == "pending"
    if remaining.any():
        result.loc[remaining, "status"] = "not_built"
        result.loc[remaining, "rejection_reason"] = "record_limit"
    status_counts = {
        str(key): int(value) for key, value in result["status"].value_counts().items()
    }
    rejection_counts = {
        str(key): int(value)
        for key, value in result.loc[result["status"] != "ok", "rejection_reason"]
        .value_counts().items()
    }
    return result, {
        "built_records": built_records,
        "reused_records": reused_records,
        "status_counts": status_counts,
        "rejection_counts": rejection_counts,
        "cache_dir": str(cache_root),
        "cache_namespace_hash": preprocessing_hash,
        "raw_preprocessing": preprocessing,
        "loader_version": RAW_LOADER_VERSION,
        "cache_size_bytes": int(sum(
            path.stat().st_size for path in cache_root.glob("*") if path.is_file()
        )),
    }


class RawEEGWindowArrayView:
    """NumPy-shaped lazy view over memory-mapped record shards."""

    is_lazy_raw_eeg = True

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        channel_mean: Optional[np.ndarray] = None,
        channel_scale: Optional[np.ndarray] = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        if len(self.manifest) == 0:
            raise ValueError("RawEEGWindowArrayView cannot be empty")
        required = {"cache_file", "cache_offset", "n_channels", "n_samples_expected"}
        missing = sorted(required - set(self.manifest.columns))
        if missing:
            raise ValueError(f"Raw manifest is missing columns: {missing}")
        if (self.manifest["status"] != "ok").any():
            raise ValueError("RawEEGWindowArrayView accepts status='ok' rows only")
        channels = self.manifest["n_channels"].astype(int).unique()
        samples = self.manifest["n_samples_expected"].astype(int).unique()
        if len(channels) != 1 or len(samples) != 1:
            raise ValueError("All raw windows must have one fixed channel/time shape")
        self.shape = (len(self.manifest), 1, int(channels[0]), int(samples[0]))
        self.ndim = 4
        self.dtype = np.dtype(np.float32)
        self.channel_mean = None if channel_mean is None else np.asarray(channel_mean, dtype=np.float32)
        self.channel_scale = None if channel_scale is None else np.asarray(channel_scale, dtype=np.float32)
        if (self.channel_mean is None) != (self.channel_scale is None):
            raise ValueError("channel_mean and channel_scale must be set together")
        if self.channel_mean is not None:
            expected = (self.shape[2],)
            if self.channel_mean.shape != expected or self.channel_scale.shape != expected:
                raise ValueError(f"Channel normalization must have shape {expected}")
        self._mapped_arrays: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.shape[0]

    def _read_scalar(self, index: int) -> np.ndarray:
        row = self.manifest.iloc[index]
        path = str(row["cache_file"])
        if path not in self._mapped_arrays:
            self._mapped_arrays[path] = np.load(
                path, mmap_mode="r", allow_pickle=False
            )
        mapped_array = self._mapped_arrays[path]
        window = np.asarray(
            mapped_array[int(row["cache_offset"])], dtype=np.float32
        )[None, :, :]
        if self.channel_mean is not None and self.channel_scale is not None:
            window = (
                window - self.channel_mean[None, :, None]
            ) / self.channel_scale[None, :, None]
        if window.shape != self.shape[1:] or not np.isfinite(window).all():
            raise ValueError(
                f"Cached raw window {row['sample_id']} is invalid: shape={window.shape}"
            )
        return np.ascontiguousarray(window, dtype=np.float32)

    def __getitem__(self, index: Any) -> Any:
        if np.isscalar(index):
            scalar = int(index)
            if scalar < 0:
                scalar += len(self)
            if scalar < 0 or scalar >= len(self):
                raise IndexError(scalar)
            return self._read_scalar(scalar)
        indices = np.arange(len(self))[index]
        return RawEEGWindowArrayView(
            self.manifest.iloc[np.asarray(indices, dtype=np.int64)],
            channel_mean=self.channel_mean,
            channel_scale=self.channel_scale,
        )

    def with_channel_normalization(
        self, mean: np.ndarray, scale: np.ndarray
    ) -> "RawEEGWindowArrayView":
        return RawEEGWindowArrayView(
            self.manifest, channel_mean=mean, channel_scale=scale
        )

    def compute_channel_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros(self.shape[2], dtype=np.float64)
        total_squares = np.zeros(self.shape[2], dtype=np.float64)
        count = 0
        for index in range(len(self)):
            window = self._read_scalar(index)[0].astype(np.float64, copy=False)
            total += window.sum(axis=1)
            total_squares += np.square(window).sum(axis=1)
            count += window.shape[1]
        mean = total / count
        variance = np.maximum(total_squares / count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-8] = 1.0
        return mean.astype(np.float32), scale.astype(np.float32)


class RawEEGWindowDataset(BaseDataset):
    """Benchmark dataset backed by a QC-filtered raw-window manifest."""

    def load(self) -> EEGData:
        manifest_path = Path(self.config["data_path"])
        manifest = ensure_record_group_ids(pd.read_parquet(manifest_path))
        target_spec = resolve_target_spec(
            self.config, default_target_id="label_focus_q5_legacy"
        )
        if not target_spec.raw_input_supported:
            raise ValueError(
                f"Target {target_spec.target_id!r} is not approved for raw EEG input"
            )
        mode = str(
            self.config.get("dataset_mode", RAW_ALL_SOURCE_RECORDS)
        ).strip()
        if mode not in RAW_DATASET_MODES:
            raise ValueError(
                f"Unknown raw dataset_mode {mode!r}; available={sorted(RAW_DATASET_MODES)}"
            )
        accepted_all = manifest.loc[manifest["status"] == "ok"].copy()
        accepted = accepted_all
        source_priority = tuple(
            str(value) for value in self.config.get(
                "source_priority", DEFAULT_SOURCE_PRIORITY
            )
        )
        selection_rows: list[dict[str, Any]] = []
        logical_map_path_value = self.config.get("logical_recording_map_path")
        logical_map_path = (
            None if logical_map_path_value is None else Path(logical_map_path_value)
        )
        if mode == RAW_DEDUPLICATED_LOGICAL_RECORDS:
            if logical_map_path is not None:
                logical_map = pd.read_parquet(logical_map_path)
                required_map = {"record_group_id", "selected_record_id"}
                missing_map = sorted(required_map - set(logical_map.columns))
                if missing_map:
                    raise ValueError(
                        f"Logical recording map is missing columns: {missing_map}"
                    )
                if logical_map["record_group_id"].astype(str).duplicated().any():
                    raise ValueError("Logical recording map has duplicate group rows")
                selected_record_ids = set(
                    logical_map["selected_record_id"].astype(str)
                )
                selection_rows = logical_map.to_dict("records")
            else:
                candidates = build_deduplication_selection(
                    manifest, source_priority=source_priority
                )
                selected = candidates.loc[candidates["selected"]]
                selected_record_ids = set(selected["record_id"].astype(str))
                selection_rows = selected.to_dict("records")
            available_record_ids = set(manifest["record_id"].astype(str))
            unavailable = sorted(selected_record_ids - available_record_ids)
            if unavailable:
                raise ValueError(
                    "Logical recording map selects records absent from manifest: "
                    f"{unavailable}"
                )
            accepted = accepted_all.loc[
                accepted_all["record_id"].astype(str).isin(selected_record_ids)
            ].copy()
            duplicated_logical = accepted.drop_duplicates("record_id")[
                "record_group_id"
            ].astype(str).duplicated()
            if duplicated_logical.any():
                raise RuntimeError(
                    "Deduplicated raw mode retained multiple source records for a "
                    "logical recording"
                )
            before_subjects = set(accepted_all["subject_id"].astype(str))
            after_subjects = set(accepted["subject_id"].astype(str))
            if before_subjects != after_subjects:
                raise RuntimeError(
                    "Deduplication changed the outer subject universe: "
                    f"removed={sorted(before_subjects - after_subjects)}, "
                    f"added={sorted(after_subjects - before_subjects)}"
                )
        else:
            selected_record_ids = set(accepted["record_id"].astype(str))
        accepted_for_mode = accepted.copy()
        n_before_target_filter = len(accepted)
        missing_target_columns = [
            column
            for column in target_spec.processed_columns
            if column not in accepted.columns
        ]
        target_data_path: Optional[Path] = None
        if missing_target_columns:
            target_data_value = self.config.get("target_data_path")
            if target_data_value is None:
                raise ValueError(
                    f"Raw manifest lacks target columns {missing_target_columns}; "
                    "configure target_data_path with the canonical processed table"
                )
            target_data_path = Path(str(target_data_value))
            if not target_data_path.is_file():
                raise FileNotFoundError(
                    f"Canonical target table not found: {target_data_path}"
                )
            target_columns = [
                "subject_id",
                "record_id",
                *target_spec.processed_columns,
            ]
            target_frame = pd.read_parquet(target_data_path, columns=target_columns)
            if "sample_id" not in target_frame.columns:
                target_frame = target_frame.copy()
                target_frame.insert(0, "sample_id", target_frame.index.to_numpy())
            accepted = attach_targets_by_sample_id(
                accepted, target_frame, target_spec, validate_identifiers=True
            )
        target_view = build_target_view(accepted, target_spec)
        accepted = accepted.iloc[target_view.cohort.selected_positions].copy()
        n_after_target_filter = len(accepted)
        max_windows = self.config.get("max_windows")
        if max_windows is not None:
            limit = int(max_windows)
            if limit <= 0:
                raise ValueError("max_windows must be positive")
            mandatory = accepted.groupby("subject_id", sort=False).head(1)
            if len(mandatory) > limit:
                raise ValueError(
                    "max_windows must be large enough to retain every subject; "
                    f"need at least {len(mandatory)}, got {limit}"
                )
            remaining = accepted.loc[~accepted.index.isin(mandatory.index)]
            balance_columns = ["outer_fold"]
            if target_spec.is_classification:
                balance_columns.extend(target_spec.processed_columns)
            group_count = max(
                1, remaining.groupby(balance_columns, observed=True).ngroups
            )
            per_group = int(math.ceil((limit - len(mandatory)) / group_count))
            balanced = remaining.groupby(
                balance_columns, sort=False, observed=True
            ).head(per_group)
            extra = balanced.head(limit - len(mandatory))
            accepted = pd.concat([mandatory, extra]).drop_duplicates(
                subset="sample_id"
            ).sort_values("sample_id")
        if accepted.empty:
            raise ValueError(f"No accepted raw EEG windows in {manifest_path}")
        configured_preprocessing = normalize_raw_preprocessing(
            self.config.get("raw_preprocessing"),
            default_resample_hz=float(accepted["sfreq_target"].iloc[0]),
        )
        manifest_hashes = (
            sorted(accepted["preprocessing_hash"].dropna().astype(str).unique())
            if "preprocessing_hash" in accepted
            else []
        )
        configured_channels_for_hash = self.config.get(
            "channel_names", CANONICAL_EEG_CHANNELS
        )
        if manifest_hashes:
            expected_hash = raw_preprocessing_hash(
                configured_preprocessing,
                channels=[str(value) for value in configured_channels_for_hash],
                default_resample_hz=float(accepted["sfreq_target"].iloc[0]),
            )
            if manifest_hashes != [expected_hash]:
                raise ValueError(
                    "Raw manifest cache preprocessing hash does not match dataset "
                    f"configuration: manifest={manifest_hashes}, expected={expected_hash}"
                )
        view = RawEEGWindowArrayView(accepted)
        configured_channels = self.config.get("channel_names")
        if configured_channels is not None:
            channel_names = [str(channel) for channel in configured_channels]
        elif view.shape[2] == len(CANONICAL_EEG_CHANNELS):
            channel_names = list(CANONICAL_EEG_CHANNELS)
        else:
            channel_names = [f"channel_{index}" for index in range(view.shape[2])]
        if len(channel_names) != view.shape[2]:
            raise ValueError(
                f"Configured channel_names has {len(channel_names)} entries for "
                f"{view.shape[2]} channels"
            )
        selected_target_view = build_target_view(accepted, target_spec)
        if selected_target_view.cohort.n_available_rows != len(accepted):
            raise RuntimeError("Target availability changed after raw-window selection")
        labels = selected_target_view.targets
        row_metadata = {
            column: accepted[column].to_numpy()
            for column in (
                "source", "record_group_id", "t_start", "t_end", "outer_fold",
                "raw_file_path", "sfreq_original", "sfreq_target",
                "missing_fraction", "preprocessing_hash", "preprocessing_variant",
                "max_abs_amplitude", "max_flat_fraction",
            )
            if column in accepted.columns
        }
        self._data = EEGData(
            data=view,  # type: ignore[arg-type]
            labels=labels,
            subject_ids=accepted["subject_id"].astype(str).to_numpy(),
            feature_names=channel_names,
            sampling_rate=float(accepted["sfreq_target"].iloc[0]),
            sample_ids=accepted["sample_id"].to_numpy(dtype=np.int64),
            record_ids=accepted["record_id"].astype(str).to_numpy(),
            row_metadata=row_metadata,
            metadata={
                "observation_unit": "raw_eeg_window",
                "manifest_path": str(manifest_path),
                "target_id": target_spec.target_id,
                "target_type": target_spec.target_type,
                "target_registry_status": target_spec.registry_status,
                "target_col": (
                    target_spec.processed_columns[0]
                    if target_spec.output_dim == 1
                    else None
                ),
                "target_cols": (
                    list(target_spec.processed_columns)
                    if target_spec.output_dim > 1
                    else None
                ),
                "target_output_names": list(target_spec.output_names),
                "n_outputs": target_spec.output_dim,
                "task_type": target_spec.task_type,
                "target_data_path": (
                    None if target_data_path is None else str(target_data_path)
                ),
                "n_samples_before_target_filter": int(n_before_target_filter),
                "n_samples_after_target_filter": int(n_after_target_filter),
                "dropped_target_rows": int(
                    n_before_target_filter - n_after_target_filter
                ),
                "input_shape": list(view.shape[1:]),
                "n_rejected_windows": int((manifest["status"] != "ok").sum()),
                "precomputed_outer_folds": True,
                "dataset_mode": mode,
                "logical_recording_map_path": (
                    None if logical_map_path is None else str(logical_map_path)
                ),
                "raw_preprocessing": configured_preprocessing,
                "preprocessing_hashes": manifest_hashes,
                "cache_roots": sorted({
                    str(Path(value).parent)
                    for value in accepted["cache_file"].astype(str).unique()
                }),
                "source_specific_records_before": int(
                    accepted_all["record_id"].nunique()
                ),
                "source_specific_records_after": int(
                    accepted_for_mode["record_id"].nunique()
                ),
                "logical_recordings": int(
                    accepted_for_mode["record_group_id"].nunique()
                ),
                "removed_source_records": int(
                    accepted_all["record_id"].nunique()
                    - accepted_for_mode["record_id"].nunique()
                ),
                "accepted_windows_before_deduplication": int(len(accepted_all)),
                "accepted_windows_after_deduplication": int(
                    len(accepted_for_mode)
                ),
                "windows_loaded": int(len(accepted)),
                "selected_record_ids": sorted(selected_record_ids),
                "logical_recording_selection": selection_rows,
            },
        )
        return self._data

    def get_description(self) -> Dict[str, Any]:
        data = self.data
        return {
            "name": "raw_eeg_window",
            "n_samples": data.n_samples,
            "n_subjects": data.n_subjects,
            "input_shape": list(data.data.shape[1:]),
            "sampling_rate": data.sampling_rate,
        }
