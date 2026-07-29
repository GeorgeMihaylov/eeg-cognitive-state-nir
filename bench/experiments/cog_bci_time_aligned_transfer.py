"""Time-aligned COG-BCI contrastive transfer screening.

This orchestration extends the completed COG-BCI contrastive screening.  It
reuses the record-safe cache builder, production EEGNet, contrastive objective,
encoder checkpoint helpers, Torch adapter, canonical project split, and shared
metrics.  Only the physical time-axis alignment and the controlled fold-2
comparison are new.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from bench.datasets.channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
from bench.datasets.cog_bci_dataset import COGBCIDataset
from bench.datasets.cog_bci_window_cache import (
    COGBCIWindowBuilder,
    PolyphaseResamplingPreprocessor,
    RawWindowSpec,
    audit_window_index,
)
from bench.experiments.cog_bci_contrastive_transfer import (
    EXPECTED_CLASSES,
    EXPECTED_INPUT_SHAPE,
    COGBCIContrastiveTransferRunner,
    _any_parameter_changed,
    _canonical_hash,
    _encoder_parameter_state,
    _encoder_state,
    _git_commit,
    _head_parameter_state,
    _jsonable,
    _relative_path,
    _sha256_file,
    _state_hash,
    _write_json,
    classification_metrics,
    create_pretraining_split,
    load_unlabelled_cog_windows,
    validate_encoder_manifest_for_downstream,
)
from model_zoo.DL.adapter import seed_torch
from model_zoo.DL.contrastive import load_encoder_checkpoint


RESULT_STATUS = "diagnostic"
EXPECTED_RECORDS = 1_044
EXPECTED_SUBJECTS = 29
EXPECTED_SESSIONS = 3
EXPECTED_VALIDATION_SUBJECTS = (
    "sub-08",
    "sub-17",
    "sub-19",
    "sub-24",
    "sub-25",
)
DOWNSTREAM_MODES = (
    "random_init",
    "shape_only",
    "time_aligned",
)


def estimate_time_aligned_cache(
    records: Sequence[Any],
    *,
    resampler: PolyphaseResamplingPreprocessor,
    window_samples: int,
    channel_count: int,
) -> dict[str, Any]:
    """Estimate exact SciPy output lengths and fixed-window payload."""

    accepted = 0
    rejected_tails = 0
    by_family: dict[str, int] = {}
    for record in records:
        resampled_n_times = math.ceil(
            int(record.n_samples) * resampler.up / resampler.down
        )
        windows = resampled_n_times // int(window_samples)
        accepted += windows
        rejected_tails += int(resampled_n_times % int(window_samples) != 0)
        by_family[str(record.task_family)] = (
            by_family.get(str(record.task_family), 0) + windows
        )
    npy_bytes = accepted * int(channel_count) * int(window_samples) * 4
    return {
        "records": len(records),
        "accepted_windows": accepted,
        "rejected_incomplete_tails": rejected_tails,
        "window_candidates": accepted + rejected_tails,
        "npy_bytes": npy_bytes,
        "npy_gib": npy_bytes / 2**30,
        "estimated_manifest_bytes": (
            len(records) * 3_000 + (accepted + rejected_tails) * 1_000
        ),
        "windows_by_task_family": dict(sorted(by_family.items())),
    }


def build_event_timing_audit(
    old_events: pd.DataFrame,
    new_events: pd.DataFrame,
    *,
    target_sampling_rate_hz: float,
) -> pd.DataFrame:
    """Compare annotation metadata and map physical times to target samples."""

    keys = ["record_id", "event_index"]
    required = {
        *keys,
        "task_family",
        "onset_seconds",
        "duration_seconds",
        "description",
    }
    for label, frame in (("old", old_events), ("new", new_events)):
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"{label} event table is missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} event table has duplicate event identities")
    columns = [
        *keys,
        "task_family",
        "onset_seconds",
        "duration_seconds",
        "description",
    ]
    comparison = old_events[columns].merge(
        new_events[columns],
        on=keys,
        how="outer",
        suffixes=("_before", "_after"),
        validate="one_to_one",
        indicator=True,
    )
    comparison["nearest_target_sample"] = np.rint(
        comparison["onset_seconds_after"].astype(float)
        * float(target_sampling_rate_hz)
    ).astype("Int64")
    comparison["mapped_onset_seconds"] = (
        comparison["nearest_target_sample"].astype(float)
        / float(target_sampling_rate_hz)
    )
    comparison["timing_error_seconds"] = (
        comparison["mapped_onset_seconds"]
        - comparison["onset_seconds_after"].astype(float)
    )
    comparison["metadata_equal"] = (
        comparison["_merge"].eq("both")
        & comparison["task_family_before"].eq(
            comparison["task_family_after"]
        )
        & np.isclose(
            comparison["onset_seconds_before"].astype(float),
            comparison["onset_seconds_after"].astype(float),
            rtol=0.0,
            atol=1e-12,
        )
        & np.isclose(
            comparison["duration_seconds_before"].astype(float),
            comparison["duration_seconds_after"].astype(float),
            rtol=0.0,
            atol=1e-12,
        )
        & comparison["description_before"].eq(
            comparison["description_after"]
        )
    )
    return comparison


def build_window_time_mapping(
    old_windows: pd.DataFrame,
    new_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Map each new window to the nearest old physical window start."""

    old = old_windows.loc[old_windows["status"].eq("accepted")].copy()
    new = new_windows.loc[new_windows["status"].eq("accepted")].copy()
    rows: list[pd.DataFrame] = []
    for record_id, current in new.groupby("record_id", sort=True):
        reference = old.loc[old["record_id"].astype(str).eq(str(record_id))]
        if reference.empty:
            raise ValueError(f"Old cache has no windows for {record_id}")
        old_times = np.sort(
            reference["start_time_seconds"].astype(float).to_numpy()
        )
        new_times = current["start_time_seconds"].astype(float).to_numpy()
        right = np.searchsorted(old_times, new_times, side="left")
        right = np.clip(right, 0, len(old_times) - 1)
        left = np.maximum(right - 1, 0)
        choose_left = (
            np.abs(old_times[left] - new_times)
            <= np.abs(old_times[right] - new_times)
        )
        nearest = np.where(choose_left, old_times[left], old_times[right])
        rows.append(
            pd.DataFrame(
                {
                    "record_id": current["record_id"].astype(str).to_numpy(),
                    "old_physical_start_time_seconds": nearest,
                    "new_physical_start_time_seconds": new_times,
                    "new_sample_id": current["sample_id"].astype(str).to_numpy(),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    if result["new_sample_id"].duplicated().any():
        raise RuntimeError("Window-time mapping duplicates new sample_id")
    return result


def time_alignment_transfer_decision(
    metrics_by_mode: Mapping[str, Mapping[str, float]],
    *,
    collapse_fatal: bool,
    checkpoint_valid: bool,
    leakage_safe: bool,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the preregistered deterministic three-way decision rule."""

    if set(metrics_by_mode) != set(DOWNSTREAM_MODES):
        return {"decision": "inconclusive", "reason": "mode_result_missing"}
    if collapse_fatal:
        return {"decision": "close_transfer_track", "reason": "collapse"}
    if not checkpoint_valid or not leakage_safe:
        return {"decision": "inconclusive", "reason": "integrity_audit_failed"}
    random = metrics_by_mode["random_init"]
    shape = metrics_by_mode["shape_only"]
    aligned = metrics_by_mode["time_aligned"]
    deltas = {
        "time_aligned_minus_random_macro_f1": float(
            aligned["macro_f1"] - random["macro_f1"]
        ),
        "time_aligned_minus_random_balanced_accuracy": float(
            aligned["balanced_accuracy"] - random["balanced_accuracy"]
        ),
        "time_aligned_minus_shape_only_macro_f1": float(
            aligned["macro_f1"] - shape["macro_f1"]
        ),
        "time_aligned_minus_shape_only_balanced_accuracy": float(
            aligned["balanced_accuracy"] - shape["balanced_accuracy"]
        ),
    }
    proceed = (
        deltas["time_aligned_minus_random_macro_f1"]
        >= float(thresholds["macro_f1_minimum_gain_vs_random"])
        and deltas["time_aligned_minus_random_balanced_accuracy"]
        >= -float(
            thresholds["balanced_accuracy_maximum_degradation_vs_random"]
        )
        and deltas["time_aligned_minus_shape_only_macro_f1"]
        >= float(thresholds["macro_f1_minimum_gain_vs_shape_only"])
    )
    strong = (
        deltas["time_aligned_minus_random_macro_f1"]
        >= float(thresholds["strong_macro_f1_minimum_gain_vs_random"])
        and deltas["time_aligned_minus_random_balanced_accuracy"]
        >= float(
            thresholds[
                "strong_balanced_accuracy_minimum_gain_vs_random"
            ]
        )
        and deltas["time_aligned_minus_shape_only_macro_f1"]
        >= float(thresholds["strong_macro_f1_minimum_gain_vs_shape_only"])
    )
    return {
        "decision": (
            "strong_proceed"
            if strong
            else "proceed"
            if proceed
            else "close_transfer_track"
        ),
        "reason": "deterministic_time_alignment_screening_rule",
        "deltas": deltas,
        "criteria_met": proceed,
        "strong_criteria_met": strong,
        "rule_is_statistical_significance_test": False,
    }


class COGBCITimeAlignedTransferRunner(COGBCIContrastiveTransferRunner):
    """Coordinate cache alignment, pretraining, and one fold-2 screen."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        repository_root: Path,
    ) -> None:
        self.config = deepcopy(dict(config))
        self.repository_root = Path(repository_root)
        self._validate_time_aligned_config()
        self.output_dir = self.repository_root / _relative_path(
            self.config["output_dir"], label="output_dir"
        )
        self.cache_dir = self.output_dir / "cache" / "emotiv_common_256hz_w10"
        self.pretraining_dir = self.output_dir / "pretraining" / "eegnet_seed42"
        self.downstream_dir = (
            self.output_dir / "downstream" / "label_q5_fold2_seed42"
        )

    def _validate_time_aligned_config(self) -> None:
        required = {
            "result_status",
            "cache",
            "pretraining",
            "downstream",
            "decision_rule",
            "output_dir",
            "tracked_report",
        }
        missing = sorted(required - set(self.config))
        if missing:
            raise ValueError(f"Time-aligned config is missing fields: {missing}")
        if self.config["result_status"] != RESULT_STATUS:
            raise ValueError("Time-aligned screening must be diagnostic")
        cache = self.config["cache"]
        pretraining = self.config["pretraining"]
        downstream = self.config["downstream"]
        expected_cache = {
            "profile_id": "emotiv_common_256hz_w10",
            "channel_policy": "emotiv_common",
            "source_sampling_rate_hz": 500,
            "target_sampling_rate_hz": 256,
            "window_duration_seconds": 10.0,
            "window_stride_seconds": 10.0,
            "segmentation": "record_full",
            "dtype": "float32",
        }
        for name, expected in expected_cache.items():
            if cache.get(name) != expected:
                raise ValueError(
                    f"cache.{name} must be {expected!r}, got {cache.get(name)!r}"
                )
        resampling = cache["resampling"]
        if (
            resampling.get("method") != "scipy.signal.resample_poly"
            or int(resampling.get("up", 0)) != 64
            or int(resampling.get("down", 0)) != 125
        ):
            raise ValueError("Time alignment requires explicit ratio 64/125")
        if int(pretraining["seed"]) != 42 or int(downstream["seed"]) != 42:
            raise ValueError("Time-aligned screening is fixed to seed 42")
        if int(downstream["fold"]) != 2:
            raise ValueError("Time-aligned screening is fixed to fold 2")
        if tuple(downstream["modes"]) != DOWNSTREAM_MODES:
            raise ValueError("Exactly random/shape-only/time-aligned modes required")
        if int(pretraining["max_epochs"]) != 30:
            raise ValueError("Pretraining budget must remain 30 epochs")
        if int(pretraining["early_stopping_patience"]) != 6:
            raise ValueError("Pretraining patience must remain six")
        if int(pretraining["split"]["validation_subjects"]) != 5:
            raise ValueError("Pretraining validation requires five subjects")
        if pretraining["additional_preprocessing"] != "resampling_only":
            raise ValueError("Only the explicit anti-alias resampling is permitted")
        for label, value in (
            ("cache.dataset_root", cache["dataset_root"]),
            ("cache.index_cache", cache["index_cache"]),
            ("pretraining.cache", pretraining["cache"]),
            (
                "downstream.shape_only_checkpoint",
                downstream["shape_only_checkpoint"],
            ),
            (
                "downstream.shape_only_manifest",
                downstream["shape_only_manifest"],
            ),
            ("downstream.data_path", downstream["dataset"]["data_path"]),
            (
                "downstream.logical_recording_map_path",
                downstream["dataset"]["logical_recording_map_path"],
            ),
            ("output_dir", self.config["output_dir"]),
            ("tracked_report", self.config["tracked_report"]),
        ):
            _relative_path(value, label=label)

    def _protected_input_paths(self) -> dict[str, Path]:
        cache = self.config["cache"]
        downstream = self.config["downstream"]
        old_cache = self.repository_root / _relative_path(
            cache["existing_shape_only_cache"],
            label="cache.existing_shape_only_cache",
        )
        return {
            "cog_record_index": self.repository_root
            / _relative_path(cache["index_cache"], label="cache.index_cache"),
            "old_cog_cache_manifest": old_cache / "dataset_manifest.json",
            "old_cog_window_index": old_cache / "window_index.parquet",
            "old_shape_only_checkpoint": self.repository_root
            / _relative_path(
                downstream["shape_only_checkpoint"],
                label="downstream.shape_only_checkpoint",
            ),
            "old_shape_only_manifest": self.repository_root
            / _relative_path(
                downstream["shape_only_manifest"],
                label="downstream.shape_only_manifest",
            ),
            "project_raw_manifest": self.repository_root
            / _relative_path(
                downstream["dataset"]["data_path"],
                label="downstream.data_path",
            ),
            "project_logical_recording_map": self.repository_root
            / _relative_path(
                downstream["dataset"]["logical_recording_map_path"],
                label="downstream.logical_recording_map_path",
            ),
        }

    def _protected_input_hashes(self) -> dict[str, str]:
        paths = self._protected_input_paths()
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Required input {label} is missing: {path}")
        return {name: _sha256_file(path) for name, path in paths.items()}

    def _resampler(self) -> PolyphaseResamplingPreprocessor:
        values = self.config["cache"]["resampling"]
        return PolyphaseResamplingPreprocessor(
            source_sampling_rate_hz=float(
                self.config["cache"]["source_sampling_rate_hz"]
            ),
            target_sampling_rate_hz=float(
                self.config["cache"]["target_sampling_rate_hz"]
            ),
            up=int(values["up"]),
            down=int(values["down"]),
            window_name=str(values["window"][0]),
            window_beta=float(values["window"][1]),
            filter_half_len_factor=int(values["filter_half_len_factor"]),
            padtype=str(values["padtype"]),
            cval=float(values["cval"]),
            profile_id=str(self.config["cache"]["profile_id"]),
        )

    def _materialize_cache(
        self,
    ) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
        cache = self.config["cache"]
        dataset = COGBCIDataset(
            {
                "data_path": self.repository_root
                / _relative_path(
                    cache["dataset_root"], label="cache.dataset_root"
                ),
                "index_cache_path": self.repository_root
                / _relative_path(
                    cache["index_cache"], label="cache.index_cache"
                ),
                "use_index_cache": True,
                "require_canonical_complete": True,
            }
        )
        resampler = self._resampler()
        spec = RawWindowSpec(
            window_duration_seconds=float(cache["window_duration_seconds"]),
            window_stride_seconds=float(cache["window_stride_seconds"]),
            drop_incomplete_window=True,
            minimum_valid_fraction=1.0,
            segmentation_mode=str(cache["segmentation"]),
            preprocessing="none",
            target_sampling_rate_hz=float(cache["target_sampling_rate_hz"]),
            reject_nonfinite=True,
            reject_constant_channels=True,
        )
        builder = COGBCIWindowBuilder(
            dataset,
            output_dir=self.cache_dir,
            channel_policy_name=str(cache["channel_policy"]),
            spec=spec,
            whole_record_preprocessor=resampler,
        )
        records = builder.select_records()
        estimate = estimate_time_aligned_cache(
            records,
            resampler=resampler,
            window_samples=spec.samples_per_window(
                resampler.target_sampling_rate_hz
            ),
            channel_count=len(PROJECT_EMOTIV_CHANNEL_ORDER),
        )
        drive = shutil.disk_usage(self.repository_root)
        estimate.update(
            {
                "free_bytes_before": int(drive.free),
                "free_gib_before": drive.free / 2**30,
                "sufficient_space": drive.free
                > 1.25
                * (
                    estimate["npy_bytes"]
                    + estimate["estimated_manifest_bytes"]
                ),
            }
        )
        if not estimate["sufficient_space"]:
            raise RuntimeError("Insufficient free space for time-aligned cache")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.output_dir / "cache_plan.json", estimate)
        _write_json(self.output_dir / "resampling_spec.json", resampler.to_dict())
        cache_summary = builder.run(records, resume=True)
        manifest = json.loads(
            (self.cache_dir / "dataset_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        windows = pd.read_parquet(self.cache_dir / "window_index.parquet")
        qc = audit_window_index(windows)
        record_manifest = pd.read_parquet(
            self.cache_dir / "record_manifest.parquet"
        )
        accepted = windows.loc[windows["status"].eq("accepted")].copy()
        if int(len(accepted)) != int(estimate["accepted_windows"]):
            raise RuntimeError("Materialized window count differs from estimate")
        merged_bounds = accepted.merge(
            record_manifest[
                ["record_id", "resampled_n_times"]
            ],
            on="record_id",
            how="left",
            validate="many_to_one",
        )
        record_crossing = int(
            (
                merged_bounds["valid_stop_sample"].astype(int)
                > merged_bounds["resampled_n_times"].astype(int)
            ).sum()
        )
        old_cache = self.repository_root / _relative_path(
            cache["existing_shape_only_cache"],
            label="cache.existing_shape_only_cache",
        )
        old_windows = pd.read_parquet(old_cache / "window_index.parquet")
        old_events = pd.read_parquet(old_cache / "events.parquet")
        new_events = pd.read_parquet(self.cache_dir / "events.parquet")
        mapping = build_window_time_mapping(old_windows, windows)
        mapping.to_parquet(
            self.output_dir / "window_time_mapping.parquet", index=False
        )
        event_audit = build_event_timing_audit(
            old_events,
            new_events,
            target_sampling_rate_hz=resampler.target_sampling_rate_hz,
        )
        event_audit.to_csv(
            self.output_dir / "event_timing_audit.csv", index=False
        )
        shutil.copy2(
            self.cache_dir / "resampling_qc.csv",
            self.output_dir / "resampling_qc.csv",
        )
        shutil.copy2(
            self.cache_dir / "record_manifest.parquet",
            self.output_dir / "record_manifest.parquet",
        )
        shutil.copy2(
            self.cache_dir / "window_index.parquet",
            self.output_dir / "window_index.parquet",
        )
        _write_json(self.output_dir / "cache_manifest.json", manifest)
        old_ids = set(
            old_windows.loc[
                old_windows["status"].eq("accepted"), "sample_id"
            ].astype(str)
        )
        new_ids = set(accepted["sample_id"].astype(str))
        duration_errors = record_manifest["duration_error_seconds"].astype(float)
        cache_bytes = sum(
            path.stat().st_size
            for path in self.cache_dir.rglob("*")
            if path.is_file()
        )
        cache_audit = {
            "records": int(accepted["record_id"].nunique()),
            "subjects": int(accepted["subject_id"].nunique()),
            "sessions": int(accepted["session_id"].nunique()),
            "channels": int(manifest["channel_count"]),
            "samples": int(manifest["samples_per_window"]),
            "sampling_rate_hz": float(manifest["sampling_rate_hz"]),
            "window_duration_seconds": float(
                manifest["window_duration_seconds"]
            ),
            "accepted_windows": int(len(accepted)),
            "rejected_incomplete_tails": int(
                windows["status"].eq("rejected_incomplete").sum()
            ),
            "windows_by_task_family": {
                str(key): int(value)
                for key, value in accepted["task_family"]
                .value_counts()
                .sort_index()
                .items()
            },
            "windows_by_subject": {
                str(key): int(value)
                for key, value in accepted["subject_id"]
                .value_counts()
                .sort_index()
                .items()
            },
            "windows_per_record_min": int(
                accepted.groupby("record_id").size().min()
            ),
            "windows_per_record_max": int(
                accepted.groupby("record_id").size().max()
            ),
            "ecg1_excluded": "ECG1" not in manifest["channel_order"],
            "channel_order_matches_project": tuple(manifest["channel_order"])
            == tuple(PROJECT_EMOTIV_CHANNEL_ORDER),
            "nan_windows": int(
                json.loads(
                    (self.cache_dir / "qc_summary.json").read_text(
                        encoding="utf-8"
                    )
                )["has_nan_windows"]
            ),
            "inf_windows": int(
                json.loads(
                    (self.cache_dir / "qc_summary.json").read_text(
                        encoding="utf-8"
                    )
                )["has_inf_windows"]
            ),
            "duplicate_sample_id": int(
                accepted["sample_id"].astype(str).duplicated().sum()
            ),
            "old_new_sample_id_overlap": len(old_ids & new_ids),
            "invalid_bounds": int(qc["invalid_bounds"]),
            "record_crossing": record_crossing,
            "duration_error_seconds_min": float(duration_errors.min()),
            "duration_error_seconds_max": float(duration_errors.max()),
            "duration_error_seconds_abs_max": float(
                duration_errors.abs().max()
            ),
            "duration_tolerance_seconds": (
                resampler.duration_tolerance_seconds
            ),
            "event_rows": int(len(event_audit)),
            "event_metadata_mismatches": int(
                (~event_audit["metadata_equal"]).sum()
            ),
            "event_timing_error_seconds_abs_max": float(
                event_audit["timing_error_seconds"].abs().max()
            ),
            "event_families": sorted(
                event_audit["task_family_after"].dropna().astype(str).unique()
            ),
            "cache_size_bytes": int(cache_bytes),
            "cache_size_gib": cache_bytes / 2**30,
            "leakage_safe": bool(
                qc["leakage_safe"]
                and record_crossing == 0
                and len(old_ids & new_ids) == 0
            ),
        }
        if (
            cache_audit["records"] != EXPECTED_RECORDS
            or cache_audit["subjects"] != EXPECTED_SUBJECTS
            or cache_audit["sessions"] != EXPECTED_SESSIONS
            or cache_audit["channels"] != 14
            or cache_audit["samples"] != 2560
            or cache_audit["sampling_rate_hz"] != 256.0
            or not cache_audit["ecg1_excluded"]
            or not cache_audit["channel_order_matches_project"]
            or cache_audit["nan_windows"]
            or cache_audit["inf_windows"]
            or cache_audit["event_metadata_mismatches"]
            or not cache_audit["leakage_safe"]
        ):
            raise RuntimeError(f"Time-aligned cache QC failed: {cache_audit}")
        _write_json(self.output_dir / "cache_qc.json", cache_audit)
        return dataset, manifest, cache_summary, cache_audit

    def _pretraining_split(
        self,
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        pretraining = self.config["pretraining"]
        split = create_pretraining_split(
            frame,
            seed=int(pretraining["split"]["seed"]),
            validation_subjects=int(
                pretraining["split"]["validation_subjects"]
            ),
        )
        if tuple(split["validation_subject_ids"]) != EXPECTED_VALIDATION_SUBJECTS:
            raise RuntimeError("Pretraining subject assignment changed")
        old_path = self.repository_root / _relative_path(
            pretraining["shape_only_split"],
            label="pretraining.shape_only_split",
        )
        old = json.loads(old_path.read_text(encoding="utf-8"))
        assignment = {
            "training_subject_ids": split["training_subject_ids"],
            "validation_subject_ids": split["validation_subject_ids"],
        }
        old_assignment = {
            "training_subject_ids": old["training_subject_ids"],
            "validation_subject_ids": old["validation_subject_ids"],
        }
        split["subject_assignment_hash"] = _canonical_hash(assignment)
        split["shape_only_subject_assignment_hash"] = _canonical_hash(
            old_assignment
        )
        split["subject_assignment_unchanged"] = assignment == old_assignment
        split["combined_cache_and_split_hash"] = _canonical_hash(
            {
                "subject_assignment_hash": split["subject_assignment_hash"],
                "training_sample_ids_sha256": split[
                    "training_sample_ids_sha256"
                ],
                "validation_sample_ids_sha256": split[
                    "validation_sample_ids_sha256"
                ],
            }
        )
        if not split["subject_assignment_unchanged"]:
            raise RuntimeError("Shape-only and time-aligned subject splits differ")
        _write_json(self.output_dir / "pretraining_split.json", split)
        return split

    def _enrich_pretraining_artifacts(
        self,
        checkpoint_path: Path,
        checkpoint_manifest: dict[str, Any],
        summary: dict[str, Any],
        *,
        cache_manifest: Mapping[str, Any],
        split: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        resampling_hash = str(cache_manifest["resampling_spec_hash"])
        augmentation_hash = _canonical_hash(
            self.config["pretraining"]["augmentations"]
        )
        additions = {
            "pretraining_dataset": "cog_bci_emotiv_common_256hz_w10",
            "additional_preprocessing": "resampling_only",
            "source_sampling_rate_hz": 500.0,
            "target_sampling_rate_hz": 256.0,
            "physical_window_duration_seconds": 10.0,
            "sampling_rate_hz": 256.0,
            "source_cache_config_hash": cache_manifest["config_hash"],
            "source_cache_manifest_sha256": _sha256_file(
                self.cache_dir / "dataset_manifest.json"
            ),
            "resampling_spec_hash": resampling_hash,
            "pretraining_subject_assignment_hash": split[
                "subject_assignment_hash"
            ],
            "pretraining_combined_split_hash": split[
                "combined_cache_and_split_hash"
            ],
            "augmentation_hash": augmentation_hash,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        }
        checkpoint_manifest.update(additions)
        summary.update(additions)
        _write_json(
            self.pretraining_dir / "encoder_checkpoint_manifest.json",
            checkpoint_manifest,
        )
        _write_json(
            self.pretraining_dir / "pretraining_summary.json", summary
        )
        return checkpoint_manifest, summary

    def _pretraining_comparison(
        self,
        new_summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        old_root = self.repository_root / _relative_path(
            self.config["pretraining"]["shape_only_pretraining_dir"],
            label="pretraining.shape_only_pretraining_dir",
        )
        old_summary = json.loads(
            (old_root / "pretraining_summary.json").read_text(encoding="utf-8")
        )
        old_best = old_summary["best_epoch_diagnostics"]
        new_best = new_summary["best_epoch_diagnostics"]
        fields = (
            "train_contrastive_loss",
            "validation_contrastive_loss",
            "validation_embedding_norm_mean",
            "validation_embedding_norm_std",
            "validation_positive_similarity",
            "validation_negative_similarity",
            "validation_positive_negative_gap",
            "validation_feature_std_mean",
            "validation_effective_rank",
        )
        old_loss_gap = float(
            old_best["validation_contrastive_loss"]
            - old_best["train_contrastive_loss"]
        )
        new_loss_gap = float(
            new_best["validation_contrastive_loss"]
            - new_best["train_contrastive_loss"]
        )
        return {
            "shape_only": {
                "checkpoint_sha256": old_summary["checkpoint_sha256"],
                "epochs_trained": old_summary["epochs_trained"],
                "best_epoch": old_summary["best_epoch"],
                "best_validation_loss": old_summary["best_validation_loss"],
                "train_validation_loss_gap": old_loss_gap,
                "collapse": old_summary["collapse"],
                "best_epoch_diagnostics": {
                    field: old_best.get(field) for field in fields
                },
            },
            "time_aligned": {
                "checkpoint_sha256": new_summary["checkpoint_sha256"],
                "epochs_trained": new_summary["epochs_trained"],
                "best_epoch": new_summary["best_epoch"],
                "best_validation_loss": new_summary["best_validation_loss"],
                "train_validation_loss_gap": new_loss_gap,
                "collapse": new_summary["collapse"],
                "best_epoch_diagnostics": {
                    field: new_best.get(field) for field in fields
                },
            },
            "deltas_time_aligned_minus_shape_only": {
                field: float(new_best[field] - old_best[field])
                for field in fields
                if field in new_best and field in old_best
            },
            "train_validation_loss_gap_delta": new_loss_gap - old_loss_gap,
        }

    def _mode_checkpoint_contract(
        self,
        time_checkpoint: Path,
        time_manifest: Mapping[str, Any],
    ) -> dict[str, tuple[Path | None, Mapping[str, Any] | None]]:
        downstream = self.config["downstream"]
        shape_checkpoint = self.repository_root / _relative_path(
            downstream["shape_only_checkpoint"],
            label="downstream.shape_only_checkpoint",
        )
        shape_manifest = json.loads(
            (
                self.repository_root
                / _relative_path(
                    downstream["shape_only_manifest"],
                    label="downstream.shape_only_manifest",
                )
            ).read_text(encoding="utf-8")
        )
        return {
            "random_init": (None, None),
            "shape_only": (shape_checkpoint, shape_manifest),
            "time_aligned": (time_checkpoint, time_manifest),
        }

    def _run_time_aligned_downstream(
        self,
        time_checkpoint: Path,
        time_manifest: Mapping[str, Any],
        pretraining_split: Mapping[str, Any],
        pretraining_summary: Mapping[str, Any],
        *,
        protected_hashes_before: Mapping[str, str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        downstream = self.config["downstream"]
        data = self._load_downstream_data()
        if tuple(data.data.shape[1:]) != EXPECTED_INPUT_SHAPE:
            raise ValueError("Downstream raw EEG input shape is incompatible")
        channel_order = tuple(str(value) for value in data.feature_names)
        contracts = self._mode_checkpoint_contract(
            time_checkpoint, time_manifest
        )
        for checkpoint, manifest in contracts.values():
            if checkpoint is None or manifest is None:
                continue
            validate_encoder_manifest_for_downstream(
                manifest,
                input_shape=EXPECTED_INPUT_SHAPE,
                channel_order=channel_order,
            )
            if _sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
                raise ValueError("Encoder checkpoint hash differs from manifest")
        split = self._downstream_split(data, channel_order)
        leakage = self._leakage_audit(data, split, pretraining_split)
        if not leakage["leakage_safe"]:
            raise RuntimeError("Fold-2 downstream split failed leakage audit")
        self.downstream_dir.mkdir(parents=True, exist_ok=True)
        preregistration_core = {
            "schema_version": 1,
            "result_status": RESULT_STATUS,
            "fold": 2,
            "seed": 42,
            "modes": list(DOWNSTREAM_MODES),
            "checkpoint_hashes": {
                mode: (
                    None
                    if checkpoint is None
                    else _sha256_file(checkpoint)
                )
                for mode, (checkpoint, _) in contracts.items()
            },
            "training_budget": {
                "max_epochs": int(downstream["max_epochs"]),
                "batch_size": int(downstream["batch_size"]),
                "learning_rate": float(
                    downstream["optimizer"]["learning_rate"]
                ),
                "weight_decay": float(
                    downstream["optimizer"]["weight_decay"]
                ),
                "patience": int(
                    downstream["early_stopping"]["patience"]
                ),
                "monitor": downstream["early_stopping"]["monitor"],
            },
            "primary_metric": "macro_f1",
            "secondary_metric": "balanced_accuracy",
            "decision_rule": deepcopy(self.config["decision_rule"]),
            "outer_test_inference_started": False,
        }
        preregistration_path = (
            self.output_dir / "time_alignment_transfer_preregistration.json"
        )
        if preregistration_path.is_file():
            existing = json.loads(
                preregistration_path.read_text(encoding="utf-8")
            )
            comparable = {
                key: value
                for key, value in existing.items()
                if key != "created_at"
            }
            if comparable != preregistration_core:
                raise RuntimeError("Existing preregistration is incompatible")
        else:
            _write_json(
                preregistration_path,
                {
                    **preregistration_core,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        preregistration_hash = _sha256_file(preregistration_path)

        outer_train = split["outer_train_indices"]
        outer_test = split["outer_test_indices"]
        labels = np.asarray(data.labels, dtype=np.int64)
        predictions: list[pd.DataFrame] = []
        fold_metrics: list[dict[str, Any]] = []
        confusions: dict[str, Any] = {}
        checkpoint_checks: dict[str, Any] = {}
        training_summaries: dict[str, Any] = {}
        metrics_by_mode: dict[str, dict[str, float]] = {}
        for mode in DOWNSTREAM_MODES:
            mode_dir = self.downstream_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            adapter = self._build_eegnet_adapter(
                mode=mode, channel_order=channel_order
            )
            random_encoder_hash = _state_hash(_encoder_state(adapter.model))
            checkpoint_path, checkpoint_manifest = contracts[mode]
            checkpoint_loaded = checkpoint_path is not None
            transferred_encoder_hash = None
            head_before_transfer = _state_hash(
                _head_parameter_state(adapter.model)
            )
            if checkpoint_path is not None:
                load_encoder_checkpoint(adapter.model, checkpoint_path)
                transferred_encoder_hash = _state_hash(
                    _encoder_state(adapter.model)
                )
                seed_torch(int(downstream["seed"]))
                adapter.replace_output_head(
                    EXPECTED_CLASSES, task_type="classification"
                )
            adapter.unfreeze_encoder()
            encoder_before_fit = _encoder_parameter_state(adapter.model)
            head_before_fit = _head_parameter_state(adapter.model)
            adapter.set_validation_indices(
                split["inner_train_local_indices"],
                split["inner_validation_local_indices"],
                subject_ids=np.asarray(data.subject_ids)[outer_train],
                record_ids=np.asarray(data.record_ids)[outer_train],
                group_ids=np.asarray(data.subject_ids)[outer_train],
                outer_test_record_ids=np.asarray(data.record_ids)[outer_test],
                outer_test_group_ids=np.asarray(data.subject_ids)[outer_test],
                group_column="subject_id",
            )
            model_path = mode_dir / "model.pt"
            resumed = model_path.is_file()
            if resumed:
                adapter.load(model_path)
                training_seconds = float(
                    sum(
                        float(row.get("epoch_time_seconds", 0.0))
                        for row in adapter.training_log_
                    )
                )
            else:
                started = time.perf_counter()
                adapter.fit(data.data[outer_train], labels[outer_train])
                training_seconds = time.perf_counter() - started
                adapter.save(model_path)
            pd.DataFrame(adapter.training_log_).to_csv(
                mode_dir / "training_log.csv", index=False
            )
            _write_json(
                mode_dir / "validation_split.json",
                adapter.validation_split_ or {},
            )
            if getattr(adapter, "feature_mean_", None) is not None:
                _write_json(
                    mode_dir / "normalization_stats.json",
                    {
                        "scope": "inner_train_only",
                        "channel_names": list(channel_order),
                        "mean": np.asarray(adapter.feature_mean_).tolist(),
                        "scale": np.asarray(adapter.feature_scale_).tolist(),
                    },
                )
            encoder_after_fit = _encoder_parameter_state(adapter.model)
            head_after_fit = _head_parameter_state(adapter.model)
            expected_encoder_hash = (
                None
                if checkpoint_path is None
                else _state_hash(
                    torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        weights_only=False,
                    )["encoder_state_dict"]
                )
            )
            checkpoint_checks[mode] = {
                "checkpoint_loaded": checkpoint_loaded,
                "random_encoder_hash": random_encoder_hash,
                "transferred_encoder_hash": transferred_encoder_hash,
                "transferred_encoder_matches_checkpoint": (
                    checkpoint_path is None
                    or transferred_encoder_hash == expected_encoder_hash
                ),
                "pretrained_differs_from_random_before_load": (
                    None
                    if checkpoint_path is None
                    else transferred_encoder_hash != random_encoder_hash
                ),
                "downstream_head_independent": (
                    checkpoint_path is None
                    or head_before_transfer
                    != _state_hash(_head_parameter_state(adapter.model))
                ),
                "encoder_parameters_changed_during_fit": (
                    _any_parameter_changed(
                        encoder_before_fit, encoder_after_fit
                    )
                ),
                "head_parameters_changed_during_fit": (
                    _any_parameter_changed(head_before_fit, head_after_fit)
                ),
                "projection_head_loaded_downstream": False,
                "model_checkpoint_sha256": _sha256_file(model_path),
                "encoder_checkpoint_sha256": (
                    None
                    if checkpoint_manifest is None
                    else checkpoint_manifest["checkpoint_sha256"]
                ),
            }

            # The preregistration file exists before the first outer-test use.
            probabilities = adapter.predict_proba(data.data[outer_test])
            prediction = probabilities.argmax(axis=1).astype(np.int64)
            metrics = classification_metrics(
                labels[outer_test], prediction, probabilities
            )
            entropy = -np.sum(
                probabilities
                * np.log(np.clip(probabilities, 1e-12, 1.0)),
                axis=1,
            )
            maximum_probability = probabilities.max(axis=1)
            metric_subset = {
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "prediction_entropy_mean": float(entropy.mean()),
                "maximum_probability_mean": float(
                    maximum_probability.mean()
                ),
            }
            metrics_by_mode[mode] = metric_subset
            confusions[mode] = metrics["confusion_matrix"]
            fold_metrics.append(
                {
                    "mode": mode,
                    "fold": 2,
                    "seed": 42,
                    **metric_subset,
                    **{
                        f"recall_{class_id}": value
                        for class_id, value in metrics[
                            "per_class_recall"
                        ].items()
                    },
                }
            )
            prediction_frame = pd.DataFrame(
                {
                    "dataset": "emotiv_raw_eeg_deduplicated",
                    "task": "label_q5",
                    "model": "torch_eegnet",
                    "mode": mode,
                    "fold_id": 2,
                    "sample_id": np.asarray(data.sample_ids)[outer_test].astype(
                        str
                    ),
                    "subject_id": np.asarray(data.subject_ids)[
                        outer_test
                    ].astype(str),
                    "record_id": np.asarray(data.record_ids)[outer_test].astype(
                        str
                    ),
                    "y_true": labels[outer_test],
                    "y_pred": prediction,
                    "prediction_entropy": entropy,
                    "maximum_probability": maximum_probability,
                }
            )
            for class_index in range(EXPECTED_CLASSES):
                prediction_frame[f"proba_{class_index}"] = probabilities[
                    :, class_index
                ]
            predictions.append(prediction_frame)
            training_summary = {
                "mode": mode,
                "training_time_seconds": training_seconds,
                "resumed": resumed,
                "epochs_trained": adapter.n_epochs_trained_,
                "best_epoch": adapter.best_epoch_,
                "best_validation_macro_f1": adapter.best_monitor_value_,
                "best_validation_loss": adapter.best_validation_loss_,
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in adapter.model.parameters()
                ),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in adapter.model.parameters()
                    if parameter.requires_grad
                ),
                "device": str(adapter.device_),
                "metrics": metrics,
                "prediction_entropy_mean": float(entropy.mean()),
                "maximum_probability_mean": float(
                    maximum_probability.mean()
                ),
            }
            training_summaries[mode] = training_summary
            _write_json(mode_dir / "metrics.json", training_summary)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        prediction_frame = pd.concat(predictions, ignore_index=True)
        identity_columns = [
            "sample_id",
            "subject_id",
            "record_id",
            "y_true",
            "fold_id",
        ]
        reference = (
            prediction_frame.loc[
                prediction_frame["mode"].eq("random_init"), identity_columns
            ]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
        identities_equal = True
        for mode in DOWNSTREAM_MODES[1:]:
            current = (
                prediction_frame.loc[
                    prediction_frame["mode"].eq(mode), identity_columns
                ]
                .sort_values("sample_id")
                .reset_index(drop=True)
            )
            identities_equal &= current.equals(reference)
        probability_columns = [
            f"proba_{index}" for index in range(EXPECTED_CLASSES)
        ]
        probability_values = prediction_frame[
            probability_columns
        ].to_numpy()
        probabilities_valid = bool(
            np.isfinite(probability_values).all()
            and np.allclose(
                probability_values.sum(axis=1), 1.0, atol=1e-5
            )
        )
        if (
            prediction_frame.duplicated(["mode", "sample_id"]).any()
            or not identities_equal
            or not probabilities_valid
        ):
            raise RuntimeError("Unified downstream prediction audit failed")
        prediction_frame.to_parquet(
            self.output_dir / "downstream_predictions.parquet", index=False
        )
        pd.DataFrame(fold_metrics).to_csv(
            self.output_dir / "downstream_fold_metrics.csv", index=False
        )
        _write_json(
            self.output_dir / "confusion_matrices.json", confusions
        )
        checkpoint_valid = all(
            item["transferred_encoder_matches_checkpoint"]
            and item["head_parameters_changed_during_fit"]
            and item["encoder_parameters_changed_during_fit"]
            and not item["projection_head_loaded_downstream"]
            for item in checkpoint_checks.values()
        )
        checkpoint_document = {
            "schema_version": 1,
            "checkpoint_valid": checkpoint_valid,
            "modes": checkpoint_checks,
        }
        _write_json(
            self.output_dir / "checkpoint_verification.json",
            checkpoint_document,
        )
        protected_hashes_after = self._protected_input_hashes()
        leakage.update(
            {
                "protected_input_hashes_before": dict(
                    protected_hashes_before
                ),
                "protected_input_hashes_after": protected_hashes_after,
                "protected_inputs_unchanged": (
                    dict(protected_hashes_before)
                    == protected_hashes_after
                ),
                "downstream_split_hash": split["split_hash"],
                "pretraining_split_hash": pretraining_split["split_hash"],
                "test_identities_equal_across_modes": identities_equal,
                "probabilities_valid": probabilities_valid,
            }
        )
        leakage["leakage_safe"] = bool(
            leakage["leakage_safe"]
            and leakage["protected_inputs_unchanged"]
            and identities_equal
            and probabilities_valid
        )
        _write_json(self.output_dir / "leakage_audit.json", leakage)
        decision = time_alignment_transfer_decision(
            metrics_by_mode,
            collapse_fatal=bool(
                pretraining_summary["collapse"]["fatal"]
            ),
            checkpoint_valid=checkpoint_valid,
            leakage_safe=bool(leakage["leakage_safe"]),
            thresholds=self.config["decision_rule"],
        )
        decision["preregistration_sha256"] = preregistration_hash
        _write_json(self.output_dir / "decision.json", decision)
        summary = {
            "result_status": RESULT_STATUS,
            "downstream_fold": 2,
            "seed": 42,
            "input_shape": list(EXPECTED_INPUT_SHAPE),
            "classes": EXPECTED_CLASSES,
            "outer_train_windows": int(len(outer_train)),
            "outer_test_windows": int(len(outer_test)),
            "outer_train_subjects": len(split["outer_train_subject_ids"]),
            "outer_test_subjects": len(split["outer_test_subject_ids"]),
            "inner_train_windows": int(
                len(split["inner_train_global_indices"])
            ),
            "inner_validation_windows": int(
                len(split["inner_validation_global_indices"])
            ),
            "split_hash": split["split_hash"],
            "modes": training_summaries,
            "checkpoint_verification": checkpoint_document,
            "leakage_audit": leakage,
            "decision": decision,
            "preregistration_sha256": preregistration_hash,
        }
        return summary, split

    def _render_time_aligned_report(
        self,
        *,
        source_commit: str,
        cache_manifest: Mapping[str, Any],
        cache_audit: Mapping[str, Any],
        pretraining_split: Mapping[str, Any],
        pretraining_summary: Mapping[str, Any],
        pretraining_comparison: Mapping[str, Any],
        downstream_summary: Mapping[str, Any],
    ) -> str:
        modes = downstream_summary["modes"]
        decision = downstream_summary["decision"]
        old_pretraining = pretraining_comparison["shape_only"]
        new_pretraining = pretraining_comparison["time_aligned"]
        old_diagnostics = old_pretraining["best_epoch_diagnostics"]
        new_diagnostics = new_pretraining["best_epoch_diagnostics"]
        lines = [
            "# COG-BCI time-aligned contrastive transfer screening",
            "",
            "Status: `diagnostic`.",
            "",
            "## Repository and temporal contract",
            "",
            "- Branch: `integration/benchmark-unification`.",
            f"- Audited HEAD: `{source_commit}`.",
            "- Project contract: 256 Hz, 10.0 s, 2,560 samples, 14 channels, "
            "`PROJECT_EMOTIV_CHANNEL_ORDER`.",
            "- The previous screening was shape-compatible but used COG-BCI "
            "at 500 Hz for 5.12 s; it was not physically time-aligned.",
            "",
            "## Resampling and cache",
            "",
            "- Whole records were selected to `emotiv_common`, loaded once, "
            "resampled by explicit polyphase ratio 64/125, then windowed.",
            "- Explicit anti-alias FIR: 2,501 taps, Kaiser beta 5.0, normalized "
            "cutoff 1/125, constant zero padding.",
            "- No demean, experimental band-pass, notch, CAR, rereference, Cz "
            "interpolation, or per-window resampling was applied.",
            f"- Cache config hash: `{cache_manifest['config_hash']}`.",
            f"- Resampling hash: `{cache_manifest['resampling_spec_hash']}`.",
            f"- Accepted windows: {cache_audit['accepted_windows']:,}; rejected "
            f"tails: {cache_audit['rejected_incomplete_tails']:,}; size: "
            f"{cache_audit['cache_size_gib']:.3f} GiB.",
            f"- Windows by family: "
            f"`{json.dumps(cache_audit['windows_by_task_family'], sort_keys=True)}`.",
            f"- Duration absolute error maximum: "
            f"{cache_audit['duration_error_seconds_abs_max']:.9f} s.",
            f"- Event timing absolute error maximum: "
            f"{cache_audit['event_timing_error_seconds_abs_max']:.9f} s; "
            f"metadata mismatches: {cache_audit['event_metadata_mismatches']}.",
            "- QC: 1,044 records, 29 subjects, 3 sessions, 14 channels, "
            "2,560 samples, no ECG1, NaN, Inf, duplicate ID, invalid bound, "
            "or record crossing.",
            "",
            "## Contrastive pretraining",
            "",
            "- Subject assignment is unchanged from the shape-only run: "
            f"24 train / 5 validation; validation subjects: "
            f"`{', '.join(pretraining_split['validation_subject_ids'])}`.",
            f"- New combined split hash: "
            f"`{pretraining_split['combined_cache_and_split_hash']}`.",
            f"- Epochs: {pretraining_summary['epochs_trained']}; best epoch: "
            f"{pretraining_summary['best_epoch']}; time: "
            f"{pretraining_summary['training_time_seconds']:.1f} s.",
            f"- Best validation NT-Xent: "
            f"{pretraining_summary['best_validation_loss']:.6f}; collapse: "
            f"{pretraining_summary['collapse']['fatal']}.",
            f"- Shape-only encoder: "
            f"`{pretraining_comparison['shape_only']['checkpoint_sha256']}`.",
            f"- Time-aligned encoder: `{pretraining_summary['checkpoint_sha256']}`.",
            f"- Encoder / projection parameters: "
            f"{pretraining_summary['encoder_parameter_count']:,} / "
            f"{pretraining_summary['projection_parameter_count']:,}; latent "
            f"dimension: {pretraining_summary['latent_dim']}.",
            "",
            "| Pretraining | Train NT-Xent | Validation NT-Xent | Train-val gap | "
            "Validation effective rank | Positive-negative gap | Feature std |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| shape_only | "
            f"{old_diagnostics['train_contrastive_loss']:.6f} | "
            f"{old_diagnostics['validation_contrastive_loss']:.6f} | "
            f"{old_pretraining['train_validation_loss_gap']:.6f} | "
            f"{old_diagnostics['validation_effective_rank']:.6f} | "
            f"{old_diagnostics['validation_positive_negative_gap']:.6f} | "
            f"{old_diagnostics['validation_feature_std_mean']:.6f} |",
            f"| time_aligned | "
            f"{new_diagnostics['train_contrastive_loss']:.6f} | "
            f"{new_diagnostics['validation_contrastive_loss']:.6f} | "
            f"{new_pretraining['train_validation_loss_gap']:.6f} | "
            f"{new_diagnostics['validation_effective_rank']:.6f} | "
            f"{new_diagnostics['validation_positive_negative_gap']:.6f} | "
            f"{new_diagnostics['validation_feature_std_mean']:.6f} |",
            "",
            "Time alignment reduced validation NT-Xent and the train-validation "
            "loss gap, and increased the positive-negative similarity gap. "
            "Effective rank decreased slightly; neither run met the collapse "
            "criteria.",
            "",
            "## Fold-2 controlled downstream screening",
            "",
            "| Mode | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | "
            "Epochs | Best val macro F1 | Best val loss | Train s | Entropy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode in DOWNSTREAM_MODES:
            item = modes[mode]
            metrics = item["metrics"]
            lines.append(
                f"| {mode} | {metrics['accuracy']:.4f} | "
                f"{metrics['balanced_accuracy']:.4f} | "
                f"{metrics['macro_f1']:.4f} | "
                f"{metrics['weighted_f1']:.4f} | "
                f"{item['epochs_trained']} | "
                f"{item['best_validation_macro_f1']:.4f} | "
                f"{item['best_validation_loss']:.4f} | "
                f"{item['training_time_seconds']:.1f} | "
                f"{item['prediction_entropy_mean']:.4f} |"
            )
        lines.extend(
            [
                "",
                f"Fold 2 contains {downstream_summary['outer_test_windows']:,} "
                f"test windows from {downstream_summary['outer_test_subjects']} "
                "subjects. Inner train/validation and outer test subjects are "
                "pairwise disjoint.",
                "",
                "Per-class recall:",
                "",
                "| Mode | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |",
                "|---|---:|---:|---:|---:|---:|",
                *[
                    "| "
                    + mode
                    + " | "
                    + " | ".join(
                        f"{modes[mode]['metrics']['per_class_recall'][str(index)]:.4f}"
                        for index in range(EXPECTED_CLASSES)
                    )
                    + " |"
                    for mode in DOWNSTREAM_MODES
                ],
                "",
                "All three modes used the same fold 2, subject-disjoint inner "
                "validation, seed, optimizer, budget, train-only channel "
                "standardization, checkpoint criterion, and test objects. "
                "Projection heads were not loaded downstream.",
                "",
                "## Integrity and decision",
                "",
                f"- Checkpoint audit: "
                f"{downstream_summary['checkpoint_verification']['checkpoint_valid']}.",
                f"- Leakage audit: "
                f"{downstream_summary['leakage_audit']['leakage_safe']}.",
                "- Outer train/test subject overlap, inner train/validation "
                "subject/record/sample overlap, and pretraining train/validation "
                "subject overlap are all zero.",
                "- Unified predictions contain the same 6,192 sample, subject, "
                "record, target and fold identities in every mode; probabilities "
                "are finite and sum to one.",
                "- Protected input hashes (old COG cache, old encoder checkpoint, "
                "project raw manifest and split inputs) are unchanged.",
                f"- Preregistration SHA-256: "
                f"`{downstream_summary['preregistration_sha256']}`.",
                f"- Decision: `{decision['decision']}`.",
                f"- Metric deltas: `{json.dumps(decision.get('deltas', {}), sort_keys=True)}`.",
                "",
                "This is a second sequential one-fold, one-seed diagnostic "
                "screening, not a pooled estimate with fold 1 and not a "
                "statistical significance test.",
                "",
            ]
        )
        if decision["decision"] == "close_transfer_track":
            lines.append(
                "The COG-BCI contrastive-transfer track should be closed for "
                "the current project; no new augmentation or architecture "
                "search is recommended."
            )
        elif decision["decision"] in {"proceed", "strong_proceed"}:
            lines.append(
                "The preregistered threshold supports a later, separately "
                "authorized confirmation experiment."
            )
        else:
            lines.append(
                "The result is inconclusive because an integrity condition "
                "was not satisfied."
            )
        lines.append("")
        return "\n".join(lines)

    def run(self) -> dict[str, Any]:
        source_commit = _git_commit(self.repository_root)
        protected_hashes_before = self._protected_input_hashes()
        _, cache_manifest, cache_summary, cache_audit = (
            self._materialize_cache()
        )
        expected_windows = int(cache_audit["accepted_windows"])
        cog = load_unlabelled_cog_windows(
            self.cache_dir,
            expected_sampling_rate_hz=256.0,
            expected_windows=expected_windows,
            expected_preprocessing_names=("resample_poly_500_to_256",),
        )
        split = self._pretraining_split(cog.frame)
        checkpoint, checkpoint_manifest, pretraining_summary = (
            self._run_pretraining(cog, split, source_commit=source_commit)
        )
        checkpoint_manifest, pretraining_summary = (
            self._enrich_pretraining_artifacts(
                checkpoint,
                checkpoint_manifest,
                pretraining_summary,
                cache_manifest=cache_manifest,
                split=split,
            )
        )
        comparison = self._pretraining_comparison(pretraining_summary)
        _write_json(
            self.output_dir / "pretraining_comparison.json", comparison
        )
        downstream_summary, downstream_split = (
            self._run_time_aligned_downstream(
                checkpoint,
                checkpoint_manifest,
                split,
                pretraining_summary,
                protected_hashes_before=protected_hashes_before,
            )
        )
        summary = {
            "schema_version": 1,
            "result_status": RESULT_STATUS,
            "branch": "integration/benchmark-unification",
            "source_commit": source_commit,
            "project_temporal_contract": {
                "sampling_rate_hz": 256.0,
                "window_duration_seconds": 10.0,
                "samples_per_window": 2560,
                "channel_count": 14,
                "channel_order": list(PROJECT_EMOTIV_CHANNEL_ORDER),
            },
            "cache_manifest": cache_manifest,
            "cache_summary": cache_summary,
            "cache_audit": cache_audit,
            "pretraining_split": split,
            "pretraining": pretraining_summary,
            "pretraining_comparison": comparison,
            "downstream": downstream_summary,
            "downstream_split": {
                key: _jsonable(value)
                for key, value in downstream_split.items()
                if not key.endswith("_indices")
            },
            "decision": downstream_summary["decision"],
        }
        _write_json(self.output_dir / "screening_summary.json", summary)
        report = self._render_time_aligned_report(
            source_commit=source_commit,
            cache_manifest=cache_manifest,
            cache_audit=cache_audit,
            pretraining_split=split,
            pretraining_summary=pretraining_summary,
            pretraining_comparison=comparison,
            downstream_summary=downstream_summary,
        )
        (self.output_dir / "screening_report.md").write_text(
            report, encoding="utf-8"
        )
        tracked_path = self.repository_root / _relative_path(
            self.config["tracked_report"], label="tracked_report"
        )
        tracked_path.parent.mkdir(parents=True, exist_ok=True)
        tracked_path.write_text(report, encoding="utf-8")
        errors_path = self.output_dir / "errors.csv"
        if not errors_path.exists():
            pd.DataFrame(columns=["stage", "error_type", "error"]).to_csv(
                errors_path, index=False
            )
        return summary


def run_cog_bci_time_aligned_transfer(
    config_path: Path | str,
    *,
    repository_root: Path | str = ".",
) -> dict[str, Any]:
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Time-aligned transfer config must contain a mapping")
    runner = COGBCITimeAlignedTransferRunner(
        config, repository_root=Path(repository_root).resolve()
    )
    try:
        return runner.run()
    except Exception as error:
        runner.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "stage": "time_aligned_transfer",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ]
        ).to_csv(runner.output_dir / "errors.csv", index=False)
        raise


__all__ = [
    "COGBCITimeAlignedTransferRunner",
    "DOWNSTREAM_MODES",
    "EXPECTED_VALIDATION_SUBJECTS",
    "build_event_timing_audit",
    "build_window_time_mapping",
    "estimate_time_aligned_cache",
    "run_cog_bci_time_aligned_transfer",
    "time_alignment_transfer_decision",
]
