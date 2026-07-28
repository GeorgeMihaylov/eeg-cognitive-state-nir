"""Record-safe COG-BCI raw-window materialization and cache verification."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .channel_contracts import channel_contract_json
from .cog_bci_dataset import COGBCIDataset, COGBCIRecord, DATASET_VERSION
from .raw_preprocessing import (
    PreprocessingSpec,
    apply_preprocessing_spec,
)


CACHE_SCHEMA_VERSION = 1
BUILDER_VERSION = "cog-bci-window-cache-v1"
DATASET_NAME = "cog_bci"
ALLOWED_CHANNEL_POLICIES = {"cog_bci_common", "emotiv_common"}
ALLOWED_PREPROCESSING = {"none", "bandpass", "notch", "bandpass_notch"}
ALLOWED_SEGMENTATION_MODES = {
    "record_full",
    "task_interval",
    "event_interval",
}
WINDOW_STATUSES = {
    "accepted",
    "rejected_nonfinite",
    "rejected_constant",
    "rejected_incomplete",
    "rejected_invalid_range",
}

START_EVENTS: dict[str, set[str]] = {
    "n_back": {"600", "610", "620"},
    "flanker": {"20"},
    "pvt": {"10"},
    "resting_state": {"40", "42", "50", "52"},
}
END_EVENTS: dict[str, set[str]] = {
    "n_back": {"601", "611", "621"},
    "matb": {"MATBeasyend", "MATBmedend", "MATBdiffend"},
    "flanker": {"21"},
    "pvt": {"15"},
    "resting_state": {"41", "43", "51", "53"},
}

WINDOW_COLUMNS = [
    "sample_id",
    "dataset",
    "source",
    "subject_id",
    "session_id",
    "record_id",
    "record_group_id",
    "task_family",
    "task_variant",
    "condition",
    "window_index",
    "segment_index",
    "start_sample",
    "stop_sample",
    "valid_stop_sample",
    "start_time_seconds",
    "stop_time_seconds",
    "sampling_rate_hz",
    "channel_policy_name",
    "channel_order",
    "preprocessing_name",
    "event_count_in_window",
    "event_types_in_window",
    "nearest_previous_event",
    "nearest_next_event",
    "contains_task_start",
    "contains_task_end",
    "status",
    "rejection_reason",
    "cache_offset",
]

QC_COLUMNS = [
    "sample_id",
    "record_id",
    "window_index",
    "status",
    "rejection_reason",
    "valid_sample_fraction",
    "has_nan",
    "has_inf",
    "constant_channel_count",
    "near_zero_variance_channel_count",
    "absolute_max",
    "absolute_mean",
]

RECORD_COLUMNS = [
    "record_id",
    "subject_id",
    "session_id",
    "task_family",
    "task_variant",
    "condition",
    "input_relative_path",
    "output_relative_path",
    "manifest_relative_path",
    "window_count",
    "accepted_count",
    "rejected_count",
    "array_shape",
    "checksum",
    "dtype",
    "channel_order",
    "sampling_rate_hz",
    "window_samples",
    "source_record_fingerprint",
    "source_filter_status",
    "reader_highpass_hz",
    "reader_lowpass_hz",
    "config_hash",
]


class COGBCIWindowCacheError(ValueError):
    """Raised when materialization or cache verification violates a contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output_file:
        np.save(output_file, array, allow_pickle=False)
    temporary.replace(path)


def _relative_path(path: Path, root: Path) -> str:
    value = path.relative_to(root).as_posix()
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise COGBCIWindowCacheError(f"Cache path must be relative: {value}")
    return value


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class RawWindowSpec:
    """Semantic window, preprocessing and quality-control configuration."""

    window_duration_seconds: float = 5.12
    window_stride_seconds: float = 5.12
    drop_incomplete_window: bool = True
    minimum_valid_fraction: float = 1.0
    segmentation_mode: str = "record_full"
    preprocessing: str = "none"
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 45.0
    bandpass_order: int = 4
    notch_frequency_hz: float = 50.0
    notch_q: float = 30.0
    target_sampling_rate_hz: float | None = None
    constant_variance_threshold: float = 0.0
    near_zero_variance_threshold: float = 1e-12
    reject_nonfinite: bool = True
    reject_constant_channels: bool = True
    allow_filtering_when_source_status_unknown: bool = False

    def __post_init__(self) -> None:
        for name in ("window_duration_seconds", "window_stride_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < float(self.minimum_valid_fraction) <= 1:
            raise ValueError("minimum_valid_fraction must be in (0, 1]")
        if self.segmentation_mode not in ALLOWED_SEGMENTATION_MODES:
            raise ValueError(
                f"Unknown segmentation_mode {self.segmentation_mode!r}"
            )
        if self.segmentation_mode != "record_full":
            raise NotImplementedError(
                f"{self.segmentation_mode!r} is not enabled: exact COG-BCI "
                "task/event interval boundaries are not uniformly confirmed"
            )
        if self.preprocessing not in ALLOWED_PREPROCESSING:
            raise ValueError(
                f"Unknown preprocessing {self.preprocessing!r}; "
                f"available={sorted(ALLOWED_PREPROCESSING)}"
            )
        if int(self.bandpass_order) != 4:
            raise ValueError(
                "The shared raw preprocessing implementation currently "
                "requires bandpass_order=4"
            )
        if not (
            math.isfinite(self.constant_variance_threshold)
            and self.constant_variance_threshold >= 0
        ):
            raise ValueError(
                "constant_variance_threshold must be non-negative and finite"
            )
        if not (
            math.isfinite(self.near_zero_variance_threshold)
            and self.near_zero_variance_threshold
            >= self.constant_variance_threshold
        ):
            raise ValueError(
                "near_zero_variance_threshold must be finite and at least "
                "constant_variance_threshold"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def samples_per_window(self, sampling_rate_hz: float) -> int:
        return _duration_to_samples(
            self.window_duration_seconds, sampling_rate_hz
        )

    def samples_per_stride(self, sampling_rate_hz: float) -> int:
        return _duration_to_samples(
            self.window_stride_seconds, sampling_rate_hz
        )


def _duration_to_samples(duration_seconds: float, sampling_rate_hz: float) -> int:
    exact = float(duration_seconds) * float(sampling_rate_hz)
    samples = int(round(exact))
    if samples <= 0:
        raise ValueError("Window duration/stride rounds to zero samples")
    error_seconds = abs(samples / float(sampling_rate_hz) - duration_seconds)
    if error_seconds > 0.5 / float(sampling_rate_hz) + 1e-12:
        raise ValueError("Window sample rounding exceeds half a sample")
    return samples


def build_preprocessing_spec(
    spec: RawWindowSpec,
    *,
    sampling_rate_hz: float,
) -> PreprocessingSpec:
    """Translate a COG materialization profile to the shared typed registry."""

    target_rate = (
        float(sampling_rate_hz)
        if spec.target_sampling_rate_hz is None
        else float(spec.target_sampling_rate_hz)
    )
    if not math.isclose(target_rate, float(sampling_rate_hz), abs_tol=1e-9):
        raise NotImplementedError(
            "COG-BCI window materialization does not resample; "
            "target_sampling_rate_hz must equal the source rate"
        )
    return PreprocessingSpec.from_dict(
        {
            "target_sampling_rate": target_rate,
            "padding_seconds": 0.0,
            "window_seconds": spec.window_duration_seconds,
            "output_dtype": "float32",
            "bandpass": {
                "enabled": spec.preprocessing in {"bandpass", "bandpass_notch"},
                "low_hz": spec.bandpass_low_hz,
                "high_hz": spec.bandpass_high_hz,
                "order": spec.bandpass_order,
            },
            "notch": {
                "enabled": spec.preprocessing in {"notch", "bandpass_notch"},
                "frequency_hz": spec.notch_frequency_hz,
                "q": spec.notch_q,
            },
            "car": {"enabled": False},
        }
    )


def enumerate_record_windows(
    n_samples: int,
    sampling_rate_hz: float,
    spec: RawWindowSpec,
    *,
    segments: Sequence[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate fixed-grid windows independently inside continuous segments."""

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    window_samples = spec.samples_per_window(sampling_rate_hz)
    stride_samples = spec.samples_per_stride(sampling_rate_hz)
    spans = [(0, int(n_samples))] if segments is None else list(segments)
    rows: list[dict[str, Any]] = []
    window_index = 0
    for segment_index, (segment_start, segment_stop) in enumerate(spans):
        if not 0 <= segment_start <= segment_stop <= n_samples:
            raise ValueError(
                f"Invalid continuous segment {(segment_start, segment_stop)}"
            )
        start = int(segment_start)
        while start < segment_stop:
            valid_stop = min(start + window_samples, segment_stop)
            complete = valid_stop - start == window_samples
            rows.append(
                {
                    "window_index": window_index,
                    "segment_index": segment_index,
                    "start_sample": start,
                    "stop_sample": start + window_samples,
                    "valid_stop_sample": valid_stop,
                    "complete": complete,
                }
            )
            window_index += 1
            if not complete:
                break
            start += stride_samples
    return rows


def stable_sample_id(
    *,
    record_id: str,
    start_sample: int,
    stop_sample: int,
    spec: RawWindowSpec,
    channel_policy_name: str,
    preprocessing_hash: str,
) -> str:
    payload = {
        "dataset": DATASET_NAME,
        "record_id": record_id,
        "window_specification": spec.to_dict(),
        "channel_policy": channel_policy_name,
        "start_sample": int(start_sample),
        "stop_sample": int(stop_sample),
        "preprocessing_hash": preprocessing_hash,
    }
    return "cog-window-" + _stable_hash(payload)[:24]


def _continuous_segments(raw: Any, sampling_rate_hz: float) -> list[tuple[int, int]]:
    boundaries = sorted(
        {
            int(round(float(onset) * sampling_rate_hz))
            for onset, description in zip(
                raw.annotations.onset, raw.annotations.description
            )
            if str(description).strip().casefold() == "boundary"
            and 0 < int(round(float(onset) * sampling_rate_hz)) < raw.n_times
        }
    )
    edges = [0, *boundaries, int(raw.n_times)]
    return [
        (left, right)
        for left, right in zip(edges[:-1], edges[1:])
        if right > left
    ]


def _event_rows(raw: Any, record: COGBCIRecord) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_index, (onset, duration, description) in enumerate(
        zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )
    ):
        description_text = str(description)
        rows.append(
            {
                "event_id": "cog-event-"
                + _stable_hash(
                    {
                        "record_id": record.record_id,
                        "event_index": event_index,
                        "onset_seconds": float(onset),
                        "duration_seconds": float(duration),
                        "description": description_text,
                    }
                )[:24],
                "record_id": record.record_id,
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "task_family": record.task_family,
                "task_variant": record.task_variant,
                "event_index": event_index,
                "onset_seconds": float(onset),
                "duration_seconds": float(duration),
                "description": description_text,
                "is_boundary": description_text.strip().casefold() == "boundary",
                "is_task_start": description_text
                in START_EVENTS.get(record.task_family, set()),
                "is_task_end": description_text
                in END_EVENTS.get(record.task_family, set()),
            }
        )
    return rows


def _window_event_metadata(
    events: Sequence[Mapping[str, Any]],
    *,
    start_seconds: float,
    stop_seconds: float,
) -> dict[str, Any]:
    inside = [
        event
        for event in events
        if start_seconds
        <= float(event["onset_seconds"])
        < stop_seconds
    ]
    previous = [
        event
        for event in events
        if float(event["onset_seconds"]) < start_seconds
    ]
    following = [
        event
        for event in events
        if float(event["onset_seconds"]) >= stop_seconds
    ]
    return {
        "event_count_in_window": len(inside),
        "event_types_in_window": json.dumps(
            [str(event["description"]) for event in inside],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "nearest_previous_event": (
            None if not previous else str(previous[-1]["description"])
        ),
        "nearest_next_event": (
            None if not following else str(following[0]["description"])
        ),
        "contains_task_start": any(bool(event["is_task_start"]) for event in inside),
        "contains_task_end": any(bool(event["is_task_end"]) for event in inside),
    }


def compute_window_qc(
    window: np.ndarray,
    *,
    valid_samples: int,
    expected_samples: int,
    spec: RawWindowSpec,
) -> dict[str, Any]:
    """Compute QC on actual samples, excluding deterministic zero padding."""

    array = np.asarray(window)
    if array.ndim != 2:
        raise ValueError(f"Expected [channels, time], got {array.shape}")
    if array.shape[1] != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} time samples, got {array.shape[1]}"
        )
    if not 0 <= valid_samples <= expected_samples:
        raise ValueError("valid_samples is outside the window")
    valid = array[:, :valid_samples]
    finite = np.isfinite(valid)
    has_nan = bool(np.isnan(valid).any())
    has_inf = bool(np.isinf(valid).any())
    valid_fraction = (
        valid_samples / expected_samples if expected_samples else 0.0
    )
    safe = np.where(finite, valid, np.nan)
    with np.errstate(invalid="ignore"):
        variances = np.nanvar(safe, axis=1) if valid_samples else np.full(
            array.shape[0], np.nan
        )
        absolute = np.abs(safe)
        max_abs = float(np.nanmax(absolute)) if np.isfinite(absolute).any() else math.nan
        mean_abs = (
            float(np.nanmean(absolute)) if np.isfinite(absolute).any() else math.nan
        )
    constant_count = int(
        np.sum(np.isfinite(variances) & (variances <= spec.constant_variance_threshold))
    )
    near_zero_count = int(
        np.sum(
            np.isfinite(variances)
            & (variances <= spec.near_zero_variance_threshold)
        )
    )
    status = "accepted"
    if valid_fraction < spec.minimum_valid_fraction:
        status = "rejected_incomplete"
    elif spec.reject_nonfinite and (has_nan or has_inf):
        status = "rejected_nonfinite"
    elif spec.reject_constant_channels and constant_count:
        status = "rejected_constant"
    return {
        "status": status,
        "rejection_reason": None if status == "accepted" else status,
        "valid_sample_fraction": valid_fraction,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "constant_channel_count": constant_count,
        "near_zero_variance_channel_count": near_zero_count,
        "absolute_max": max_abs,
        "absolute_mean": mean_abs,
    }


def _source_record_fingerprint(
    root: Path, record: COGBCIRecord
) -> str:
    entries = []
    for relative in (record.set_relative_path, record.fdt_relative_path):
        path = root / relative
        stat = path.stat()
        entries.append(
            {
                "relative_path": Path(relative).as_posix(),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return _stable_hash(entries)


def _reader_filter_metadata(raw: Any) -> dict[str, Any]:
    highpass = float(raw.info.get("highpass", 0.0))
    lowpass = float(raw.info.get("lowpass", raw.info["sfreq"] / 2.0))
    return {
        "source_filter_status": "unknown_eeglab_processing_history",
        "reader_highpass_hz": highpass,
        "reader_lowpass_hz": lowpass,
        "reader_reports_full_nyquist_band": bool(
            math.isclose(highpass, 0.0, abs_tol=1e-9)
            and math.isclose(
                lowpass, float(raw.info["sfreq"]) / 2.0, abs_tol=1e-9
            )
        ),
    }


def _shard_stem(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:24]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _validate_record_manifest(
    manifest_path: Path,
    array_path: Path,
    *,
    config_hash: str,
    source_record_fingerprint: str,
) -> dict[str, Any]:
    if not manifest_path.is_file() or not array_path.is_file():
        raise COGBCIWindowCacheError(
            f"Incomplete shard: {manifest_path.name}/{array_path.name}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("config_hash") != config_hash:
        raise COGBCIWindowCacheError(
            f"Incompatible config hash for {manifest.get('record_id')}"
        )
    if manifest.get("source_record_fingerprint") != source_record_fingerprint:
        raise COGBCIWindowCacheError(
            f"Source record changed for {manifest.get('record_id')}"
        )
    actual_checksum = _file_sha256(array_path)
    if actual_checksum != manifest.get("checksum"):
        raise COGBCIWindowCacheError(
            f"Checksum mismatch for {manifest.get('record_id')}"
        )
    expected_shape = tuple(int(value) for value in manifest["array_shape"])
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != expected_shape or array.dtype != np.float32:
        raise COGBCIWindowCacheError(
            f"Array contract mismatch for {manifest.get('record_id')}: "
            f"shape={array.shape}, dtype={array.dtype}"
        )
    return manifest


def _normalize_cached_metadata(
    windows: pd.DataFrame,
    qc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Upgrade v1 metadata names/statuses without touching signal shards."""

    status_aliases = {
        "rejected_constant_channel": "rejected_constant",
    }
    for frame in (windows, qc):
        if frame.empty or "status" not in frame:
            continue
        old_generic = frame["status"].eq("rejected")
        if "rejection_reason" in frame:
            replacement = frame["rejection_reason"].replace(status_aliases)
            frame.loc[old_generic, "status"] = replacement.loc[old_generic]
            frame["rejection_reason"] = frame["rejection_reason"].replace(
                status_aliases
            )
        frame["status"] = frame["status"].replace(status_aliases)
        unknown = sorted(set(frame["status"].dropna()) - WINDOW_STATUSES)
        if unknown:
            raise COGBCIWindowCacheError(
                f"Unknown cached window statuses: {unknown}"
            )
    amplitude_aliases = {
        "max_abs_amplitude": "absolute_max",
        "mean_abs_amplitude": "absolute_mean",
    }
    for old_name, new_name in amplitude_aliases.items():
        if new_name not in qc and old_name in qc:
            qc[new_name] = qc[old_name]
    return windows, qc


class COGBCIWindowBuilder:
    """Materialize fixed-shape record shards through existing dataset contracts."""

    def __init__(
        self,
        dataset: COGBCIDataset,
        *,
        output_dir: Path | str,
        channel_policy_name: str,
        spec: RawWindowSpec,
    ) -> None:
        if channel_policy_name not in ALLOWED_CHANNEL_POLICIES:
            raise ValueError(
                "Batch materialization requires a fixed channel policy; "
                f"got {channel_policy_name!r}"
            )
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.channel_policy_name = channel_policy_name
        self.channel_policy = dataset.get_channel_policy(channel_policy_name)
        self.channel_order = tuple(self.channel_policy.required_names)
        self.channel_mapping_hash = _stable_hash(
            json.loads(channel_contract_json(self.channel_policy))
        )
        self.spec = spec
        rates = {float(record.sampling_rate_hz) for record in dataset.records}
        if len(rates) != 1:
            raise COGBCIWindowCacheError(
                f"A fixed-shape cache requires one sampling rate, got {sorted(rates)}"
            )
        self.sampling_rate_hz = next(iter(rates))
        self.preprocessing_spec = build_preprocessing_spec(
            spec, sampling_rate_hz=self.sampling_rate_hz
        )
        self.preprocessing_hash = self.preprocessing_spec.stable_hash(
            channels=self.channel_order,
            loader_schema_version=BUILDER_VERSION,
        )
        self.config_hash = _stable_hash(
            {
                "builder_version": BUILDER_VERSION,
                "dataset_version": DATASET_VERSION,
                "channel_policy": json.loads(
                    channel_contract_json(self.channel_policy)
                ),
                "window_spec": spec.to_dict(),
                "preprocessing": self.preprocessing_spec.to_dict(),
                "event_contract": {
                    "task_start": {
                        name: sorted(values)
                        for name, values in sorted(START_EVENTS.items())
                    },
                    "task_end": {
                        name: sorted(values)
                        for name, values in sorted(END_EVENTS.items())
                    },
                },
            }
        )

    def select_records(
        self,
        *,
        subjects: Sequence[str] | None = None,
        sessions: Sequence[str] | None = None,
        task_families: Sequence[str] | None = None,
        task_variants: Sequence[str] | None = None,
        max_records: int | None = None,
        one_per_subject_family: bool = False,
    ) -> tuple[COGBCIRecord, ...]:
        records = self.dataset.query(
            subject_ids=subjects,
            session_ids=sessions,
            task_families=task_families,
            task_variants=task_variants,
        )
        if one_per_subject_family:
            selected: dict[tuple[str, str], COGBCIRecord] = {}
            for record in records:
                selected.setdefault(
                    (record.subject_id, record.task_family), record
                )
            records = tuple(selected[key] for key in sorted(selected))
        if max_records is not None:
            if max_records < 1:
                raise ValueError("max_records must be positive")
            records = records[:max_records]
        if not records:
            raise ValueError("COG-BCI record selection is empty")
        return records

    def _dataset_manifest(self, *, created_at: str) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "dataset": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "result_status": "diagnostic",
            "source_root_fingerprint": self.dataset.index.source_root_fingerprint,
            "builder_version": BUILDER_VERSION,
            "channel_policy": self.channel_policy.to_dict(),
            "channel_policy_name": self.channel_policy_name,
            "channel_policy_schema_version": 1,
            "channel_mapping_hash": self.channel_mapping_hash,
            "channel_order": list(self.channel_order),
            "channel_count": len(self.channel_order),
            "has_physical_cz": any(
                bool(getattr(record, "has_cz", False))
                for record in self.dataset.records
            ),
            "uses_cz": "Cz" in self.channel_order,
            "auxiliary_channels_excluded": True,
            "preprocessing": self.preprocessing_spec.to_dict(),
            "preprocessing_name": self.spec.preprocessing,
            "source_filter_status": "unknown_eeglab_processing_history",
            "event_contract": {
                "task_start": {
                    name: sorted(values)
                    for name, values in sorted(START_EVENTS.items())
                },
                "task_end": {
                    name: sorted(values)
                    for name, values in sorted(END_EVENTS.items())
                },
            },
            "window_duration_seconds": self.spec.window_duration_seconds,
            "window_stride_seconds": self.spec.window_stride_seconds,
            "segmentation_mode": self.spec.segmentation_mode,
            "sampling_rate_hz": self.sampling_rate_hz,
            "samples_per_window": self.spec.samples_per_window(
                self.sampling_rate_hz
            ),
            "dtype": "float32",
            "created_at": created_at,
            "commit": _git_commit(),
            "config_hash": self.config_hash,
        }

    def _existing_global_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        def read(name: str, columns: Sequence[str]) -> pd.DataFrame:
            path = self.output_dir / name
            return (
                pd.read_parquet(path)
                if path.is_file()
                else pd.DataFrame(columns=list(columns))
            )

        return (
            read("window_index.parquet", WINDOW_COLUMNS),
            read("qc_windows.parquet", QC_COLUMNS),
            read(
                "events.parquet",
                [
                    "event_id",
                    "record_id",
                    "subject_id",
                    "session_id",
                    "task_family",
                    "task_variant",
                    "event_index",
                    "onset_seconds",
                    "duration_seconds",
                    "description",
                    "is_boundary",
                    "is_task_start",
                    "is_task_end",
                ],
            ),
        )

    def _materialize_record(
        self, record: COGBCIRecord
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        selection = self.dataset.select_raw_channels(
            record.record_id,
            self.channel_policy,
            preload=False,
            copy=True,
        )
        raw = selection.raw
        try:
            if tuple(raw.ch_names) != self.channel_order:
                raise COGBCIWindowCacheError(
                    f"Channel order mismatch for {record.record_id}"
                )
            if not math.isclose(
                float(raw.info["sfreq"]), self.sampling_rate_hz, abs_tol=1e-9
            ):
                raise COGBCIWindowCacheError(
                    f"Sampling rate mismatch for {record.record_id}"
                )
            filter_metadata = _reader_filter_metadata(raw)
            filtering_requested = self.spec.preprocessing != "none"
            if (
                filtering_requested
                and filter_metadata["source_filter_status"]
                == "unknown_eeglab_processing_history"
                and not self.spec.allow_filtering_when_source_status_unknown
            ):
                raise COGBCIWindowCacheError(
                    "Source filtering history is unknown; set "
                    "allow_filtering_when_source_status_unknown=true to apply "
                    "an additional filter explicitly"
                )
            signal = np.asarray(raw.get_data(), dtype=np.float32)
            if signal.shape != (len(self.channel_order), raw.n_times):
                raise COGBCIWindowCacheError(
                    f"Unexpected raw shape for {record.record_id}: {signal.shape}"
                )
            if filtering_requested:
                signal = apply_preprocessing_spec(
                    signal,
                    sampling_rate=self.sampling_rate_hz,
                    spec=self.preprocessing_spec,
                )
            else:
                signal = np.ascontiguousarray(signal, dtype=np.float32)
            events = _event_rows(raw, record)
            segments = _continuous_segments(raw, self.sampling_rate_hz)
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()

        window_specs = enumerate_record_windows(
            signal.shape[1],
            self.sampling_rate_hz,
            self.spec,
            segments=segments,
        )
        samples_per_window = self.spec.samples_per_window(self.sampling_rate_hz)
        accepted: list[np.ndarray] = []
        window_rows: list[dict[str, Any]] = []
        qc_rows: list[dict[str, Any]] = []
        for window_spec in window_specs:
            start = int(window_spec["start_sample"])
            valid_stop = int(window_spec["valid_stop_sample"])
            stop = int(window_spec["stop_sample"])
            valid_samples = valid_stop - start
            window = np.zeros(
                (len(self.channel_order), samples_per_window), dtype=np.float32
            )
            window[:, :valid_samples] = signal[:, start:valid_stop]
            qc = compute_window_qc(
                window,
                valid_samples=valid_samples,
                expected_samples=samples_per_window,
                spec=self.spec,
            )
            if not window_spec["complete"] and self.spec.drop_incomplete_window:
                qc["status"] = "rejected_incomplete"
                qc["rejection_reason"] = "rejected_incomplete"
            sample_id = stable_sample_id(
                record_id=record.record_id,
                start_sample=start,
                stop_sample=stop,
                spec=self.spec,
                channel_policy_name=self.channel_policy_name,
                preprocessing_hash=self.preprocessing_hash,
            )
            cache_offset = -1
            if qc["status"] == "accepted":
                cache_offset = len(accepted)
                accepted.append(window)
            start_seconds = start / self.sampling_rate_hz
            stop_seconds = valid_stop / self.sampling_rate_hz
            event_metadata = _window_event_metadata(
                events,
                start_seconds=start_seconds,
                stop_seconds=stop_seconds,
            )
            row = {
                "sample_id": sample_id,
                "dataset": DATASET_NAME,
                "source": "COG-BCI",
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "record_id": record.record_id,
                "record_group_id": record.record_id,
                "task_family": record.task_family,
                "task_variant": record.task_variant,
                "condition": record.condition,
                "window_index": int(window_spec["window_index"]),
                "segment_index": int(window_spec["segment_index"]),
                "start_sample": start,
                "stop_sample": stop,
                "valid_stop_sample": valid_stop,
                "start_time_seconds": start_seconds,
                "stop_time_seconds": stop_seconds,
                "sampling_rate_hz": self.sampling_rate_hz,
                "channel_policy_name": self.channel_policy_name,
                "channel_order": json.dumps(
                    list(self.channel_order), separators=(",", ":")
                ),
                "preprocessing_name": self.spec.preprocessing,
                **event_metadata,
                "status": qc["status"],
                "rejection_reason": qc["rejection_reason"],
                "cache_offset": cache_offset,
            }
            window_rows.append(row)
            qc_rows.append(
                {
                    "sample_id": sample_id,
                    "record_id": record.record_id,
                    "window_index": int(window_spec["window_index"]),
                    **qc,
                }
            )

        array = (
            np.stack(accepted).astype(np.float32, copy=False)
            if accepted
            else np.empty(
                (0, len(self.channel_order), samples_per_window),
                dtype=np.float32,
            )
        )
        shards = self.output_dir / "shards"
        stem = _shard_stem(record.record_id)
        array_path = shards / f"{stem}.npy"
        manifest_path = shards / f"{stem}.json"
        _atomic_npy(array_path, array)
        checksum = _file_sha256(array_path)
        source_fingerprint = _source_record_fingerprint(
            self.dataset.root, record
        )
        record_manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "record_id": record.record_id,
            "subject_id": record.subject_id,
            "session_id": record.session_id,
            "task_family": record.task_family,
            "task_variant": record.task_variant,
            "condition": record.condition,
            "input_relative_path": record.set_relative_path,
            "output_relative_path": _relative_path(array_path, self.output_dir),
            "manifest_relative_path": _relative_path(
                manifest_path, self.output_dir
            ),
            "window_count": len(window_rows),
            "accepted_count": len(accepted),
            "rejected_count": len(window_rows) - len(accepted),
            "array_shape": list(array.shape),
            "dtype": "float32",
            "channel_order": list(self.channel_order),
            "sampling_rate_hz": self.sampling_rate_hz,
            "window_samples": samples_per_window,
            "checksum": checksum,
            "source_record_fingerprint": source_fingerprint,
            **filter_metadata,
            "config_hash": self.config_hash,
        }
        _atomic_json(manifest_path, record_manifest)
        return record_manifest, window_rows, qc_rows, events

    def run(
        self,
        records: Iterable[COGBCIRecord],
        *,
        resume: bool = False,
        overwrite: bool = False,
        verify_only: bool = False,
    ) -> dict[str, Any]:
        if resume and overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        records = tuple(sorted(records, key=lambda record: record.record_id))
        if not records:
            raise ValueError("No records selected")
        if verify_only and not self.output_dir.is_dir():
            raise COGBCIWindowCacheError(
                f"Cache does not exist: {self.output_dir}"
            )
        if not verify_only:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "shards").mkdir(parents=True, exist_ok=True)

        dataset_manifest_path = self.output_dir / "dataset_manifest.json"
        if dataset_manifest_path.is_file():
            existing_dataset_manifest = _read_json(dataset_manifest_path)
            if (
                existing_dataset_manifest.get("config_hash") != self.config_hash
                and not overwrite
            ):
                raise COGBCIWindowCacheError(
                    "Existing dataset cache has an incompatible config hash; "
                    "use --overwrite or a separate output directory"
                )
        elif verify_only:
            raise COGBCIWindowCacheError("dataset_manifest.json is missing")

        existing_windows, existing_qc, existing_events = (
            self._existing_global_frames()
        )
        existing_windows, existing_qc = _normalize_cached_metadata(
            existing_windows, existing_qc
        )
        selected_ids = {record.record_id for record in records}
        record_documents: dict[str, dict[str, Any]] = {}
        rebuilt_ids: set[str] = set()
        skipped_ids: set[str] = set()
        errors: list[dict[str, Any]] = []
        for record in records:
            stem = _shard_stem(record.record_id)
            array_path = self.output_dir / "shards" / f"{stem}.npy"
            manifest_path = self.output_dir / "shards" / f"{stem}.json"
            source_fingerprint = _source_record_fingerprint(
                self.dataset.root, record
            )
            exists = array_path.exists() or manifest_path.exists()
            try:
                if exists and not overwrite:
                    if not (resume or verify_only):
                        raise COGBCIWindowCacheError(
                            f"Shard already exists for {record.record_id}; "
                            "use --resume or --overwrite"
                        )
                    document = _validate_record_manifest(
                        manifest_path,
                        array_path,
                        config_hash=self.config_hash,
                        source_record_fingerprint=source_fingerprint,
                    )
                    array_contract = {
                        "dtype": "float32",
                        "channel_order": list(self.channel_order),
                        "sampling_rate_hz": self.sampling_rate_hz,
                        "window_samples": self.spec.samples_per_window(
                            self.sampling_rate_hz
                        ),
                    }
                    if any(
                        document.get(name) != value
                        for name, value in array_contract.items()
                    ):
                        if verify_only:
                            raise COGBCIWindowCacheError(
                                f"Incomplete array contract for "
                                f"{record.record_id}; run --resume to migrate "
                                "metadata"
                            )
                        document.update(array_contract)
                        _atomic_json(manifest_path, document)
                    record_documents[record.record_id] = document
                    skipped_ids.add(record.record_id)
                    continue
                if verify_only:
                    raise COGBCIWindowCacheError(
                        f"Missing shard for {record.record_id}"
                    )
                document, window_rows, qc_rows, event_rows = (
                    self._materialize_record(record)
                )
                record_documents[record.record_id] = document
                rebuilt_ids.add(record.record_id)
                existing_windows = existing_windows[
                    existing_windows["record_id"] != record.record_id
                ]
                existing_qc = existing_qc[
                    existing_qc["record_id"] != record.record_id
                ]
                existing_events = existing_events[
                    existing_events["record_id"] != record.record_id
                ]
                existing_windows = pd.concat(
                    [existing_windows, pd.DataFrame(window_rows)],
                    ignore_index=True,
                )
                existing_qc = pd.concat(
                    [existing_qc, pd.DataFrame(qc_rows)],
                    ignore_index=True,
                )
                existing_events = pd.concat(
                    [existing_events, pd.DataFrame(event_rows)],
                    ignore_index=True,
                )
            except Exception as error:
                errors.append(
                    {
                        "record_id": record.record_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if verify_only or exists:
                    raise

        if verify_only:
            missing_index = selected_ids - set(existing_windows["record_id"])
            if missing_index:
                raise COGBCIWindowCacheError(
                    "Global window index is missing selected records: "
                    f"{sorted(missing_index)[:5]}"
                )
            return {
                "verified_records": len(skipped_ids),
                "rebuilt_records": 0,
                "errors": 0,
                "config_hash": self.config_hash,
            }

        existing_windows = existing_windows.sort_values(
            ["record_id", "window_index"], kind="stable"
        ).reset_index(drop=True)
        existing_qc = existing_qc.sort_values(
            ["record_id", "window_index"], kind="stable"
        ).reset_index(drop=True)
        existing_events = existing_events.sort_values(
            ["record_id", "event_index"], kind="stable"
        ).reset_index(drop=True)
        all_record_manifests = []
        for path in sorted((self.output_dir / "shards").glob("*.json")):
            document = _read_json(path)
            if document.get("config_hash") == self.config_hash:
                all_record_manifests.append(document)
        record_frame = pd.DataFrame(all_record_manifests)
        if record_frame.empty:
            record_frame = pd.DataFrame(columns=RECORD_COLUMNS)
        else:
            record_frame = record_frame.sort_values("record_id").reset_index(
                drop=True
            )
            record_frame["array_shape"] = record_frame["array_shape"].map(
                lambda value: json.dumps(value, separators=(",", ":"))
            )

        _atomic_parquet(
            self.output_dir / "record_manifest.parquet", record_frame
        )
        _atomic_parquet(
            self.output_dir / "window_index.parquet",
            existing_windows[WINDOW_COLUMNS],
        )
        _atomic_parquet(
            self.output_dir / "qc_windows.parquet",
            existing_qc[QC_COLUMNS],
        )
        _atomic_parquet(
            self.output_dir / "events.parquet", existing_events
        )
        error_frame = pd.DataFrame(
            errors, columns=["record_id", "error_type", "error"]
        )
        _atomic_csv(self.output_dir / "errors.csv", error_frame)

        accepted = existing_windows["status"].eq("accepted")
        qc_summary = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "result_status": "diagnostic",
            "record_count": int(existing_windows["record_id"].nunique()),
            "window_count": int(len(existing_windows)),
            "accepted_count": int(accepted.sum()),
            "rejected_count": int((~accepted).sum()),
            "rejection_reasons": {
                str(key): int(value)
                for key, value in existing_windows.loc[
                    ~accepted, "rejection_reason"
                ]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            },
            "has_nan_windows": int(existing_qc["has_nan"].sum()),
            "has_inf_windows": int(existing_qc["has_inf"].sum()),
            "constant_channel_windows": int(
                existing_qc["constant_channel_count"].gt(0).sum()
            ),
            "error_count": len(errors),
        }
        _atomic_json(self.output_dir / "qc_summary.json", qc_summary)
        created_at = datetime.now(timezone.utc).isoformat()
        _atomic_json(
            dataset_manifest_path,
            self._dataset_manifest(created_at=created_at),
        )
        report = (
            "# COG-BCI raw-window cache\n\n"
            "Status: `diagnostic`\n\n"
            f"- channel policy: `{self.channel_policy_name}` "
            f"({len(self.channel_order)} channels)\n"
            f"- preprocessing: `{self.spec.preprocessing}`\n"
            f"- source filter history: `unknown_eeglab_processing_history`\n"
            f"- segmentation: `{self.spec.segmentation_mode}`\n"
            f"- window/stride: {self.spec.window_duration_seconds}/"
            f"{self.spec.window_stride_seconds} seconds\n"
            f"- sampling rate: {self.sampling_rate_hz} Hz\n"
            f"- window samples: {self.spec.samples_per_window(self.sampling_rate_hz)}\n"
            f"- records: {qc_summary['record_count']}\n"
            f"- accepted windows: {qc_summary['accepted_count']}\n"
            f"- rejected windows: {qc_summary['rejected_count']}\n\n"
            "Windows are generated independently inside each physical EEGLAB "
            "record and each internal continuous segment. No split or target "
            "construction is performed.\n"
        )
        temporary_report = (self.output_dir / "cache_report.md.tmp")
        temporary_report.write_text(report, encoding="utf-8")
        temporary_report.replace(self.output_dir / "cache_report.md")
        summary = {
            **qc_summary,
            "rebuilt_records": len(rebuilt_ids),
            "skipped_records": len(skipped_ids),
            "config_hash": self.config_hash,
        }
        if errors:
            first = errors[0]
            raise COGBCIWindowCacheError(
                f"Materialization failed for {len(errors)} record(s); "
                f"first error: {first['error_type']}: {first['error']}; "
                f"see {self.output_dir / 'errors.csv'}"
            )
        return summary


def audit_window_index(frame: pd.DataFrame) -> dict[str, Any]:
    """Check the leakage-relevant identity and boundary invariants."""

    required = {
        "sample_id",
        "subject_id",
        "session_id",
        "record_id",
        "record_group_id",
        "start_sample",
        "valid_stop_sample",
        "status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Window index is missing columns: {missing}")
    accepted = frame.loc[frame["status"] == "accepted"].copy()
    duplicate_sample_ids = int(accepted["sample_id"].duplicated().sum())
    invalid_record_groups = int(
        (
            accepted.groupby("sample_id")["record_group_id"].nunique()
            > 1
        ).sum()
    )
    invalid_bounds = int(
        (
            accepted["start_sample"].astype(int)
            >= accepted["valid_stop_sample"].astype(int)
        ).sum()
    )
    return {
        "accepted_windows": len(accepted),
        "subjects": int(accepted["subject_id"].nunique()),
        "sessions": int(accepted["session_id"].nunique()),
        "records": int(accepted["record_id"].nunique()),
        "record_groups": int(accepted["record_group_id"].nunique()),
        "duplicate_sample_ids": duplicate_sample_ids,
        "invalid_record_group_assignments": invalid_record_groups,
        "invalid_bounds": invalid_bounds,
        "leakage_safe": (
            duplicate_sample_ids == 0
            and invalid_record_groups == 0
            and invalid_bounds == 0
        ),
    }
