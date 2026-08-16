"""Leakage-safe CLARE/CL-Drive multimodal inventory and execution protocol."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

from bench.datasets.clare_cldrive_dataset import (
    EEG_CHANNELS,
    CognitiveLoadMultimodalRecord,
)
from bench.datasets.datasets_registry import get_dataset
from bench.features.cog_bci_spectral_features import (
    SpectralFeatureSpec,
    extract_spectral_feature_bundle,
)
from bench.validation.metrics import MetricsCalculator
from model_zoo.factory import build_model


SCHEMA_VERSION = "external-cognitive-load-multimodal-v1"
SUMMARY_STATISTICS = ("mean", "std", "min", "max", "median", "q25", "q75")
ECG_COLUMNS = ("ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL")
EDA_COLUMN = "GSR Conductance CAL"
PRIMARY_MODES = ("eeg_only", "peripheral_only", "eeg_peripheral")
EXPECTED_SHA256 = {
    "clare": "4146f8afcd475b9bdcfaacf0e82e286bd3d1c005102783d4cde0eab254235cba",
    "cl_drive": "83862102235105dfb600e36706b36406094c6c7887e9d133a90042f54736aac0",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") == "external-mefar-xgboost-v1":
        from bench.experiments.mefar_external_xgboost import load_config as load_mefar_config

        return load_mefar_config(path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    dataset = str(config.get("dataset", {}).get("name", ""))
    if dataset not in EXPECTED_SHA256:
        raise ValueError("dataset.name must be 'clare' or 'cl_drive'")
    expected = str(config["dataset"].get("expected_sha256", "")).lower()
    if expected != EXPECTED_SHA256[dataset]:
        raise ValueError(f"Unexpected validated archive SHA-256 for {dataset}")
    target = config.get("target", {})
    if target.get("target_id") != "subjective_cognitive_load_3class_fixed":
        raise ValueError("Primary target_id must be subjective_cognitive_load_3class_fixed")
    if target.get("fixed_bins") != {"0": [1, 3], "1": [4, 6], "2": [7, 9]}:
        raise ValueError("The fixed 1-3 / 4-6 / 7-9 target mapping changed")
    evaluation = config.get("evaluation", {})
    if evaluation.get("protocol") != "stratified_group_kfold_participant":
        raise ValueError("Outer protocol must remain participant-disjoint StratifiedGroupKFold")
    if int(evaluation.get("n_splits", 0)) != 5:
        raise ValueError("The protocol requires five outer folds")
    if bool(evaluation.get("oversampling", True)):
        raise ValueError("Oversampling before the outer split is forbidden")
    return config


def _safe_member_path(root: Path, member: str) -> Path:
    normalized = member.replace("\\", "/")
    candidate = (root / normalized).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe archive member: {member!r}") from exc
    return candidate


def _find_unrar() -> Path:
    discovered = shutil.which("unrar")
    candidates = [
        Path(discovered) if discovered else None,
        Path(r"C:\Program Files\WinRAR\UnRAR.exe"),
        Path(r"C:\Program Files\WinRAR\WinRAR.exe"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError("UnRAR is required to verify/extract nested RAR archives")


def verify_archive(config: Mapping[str, Any]) -> dict[str, Any]:
    archive = Path(config["dataset"]["archive"])
    digest = file_sha256(archive)
    if digest != str(config["dataset"]["expected_sha256"]).lower():
        raise ValueError(f"Source archive SHA-256 mismatch: {archive}")
    with zipfile.ZipFile(archive) as source:
        bad_member = source.testzip()
        infos = source.infolist()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failure: {bad_member}")
        for info in infos:
            _safe_member_path(Path(config["dataset"]["extracted_root"]), info.filename)
    nested = []
    unrar = _find_unrar()
    extracted = Path(config["dataset"]["extracted_root"])
    for path in sorted(extracted.glob("*.rar")):
        names = subprocess.run(
            [str(unrar), "lb", str(path)], check=True, capture_output=True, text=True
        ).stdout.splitlines()
        for name in names:
            _safe_member_path(extracted, name)
        result = subprocess.run(
            [str(unrar), "t", "-idq", str(path)], capture_output=True, text=True
        )
        if result.returncode:
            raise ValueError(f"RAR integrity failure: {path.name}")
        nested.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "entries": len(names),
                "unsafe_paths": 0,
                "crc_valid": True,
            }
        )
    return {
        "logical_path": config["dataset"]["archive"],
        "size_bytes": archive.stat().st_size,
        "sha256": digest,
        "zip_crc_valid": True,
        "outer_entries": len(infos),
        "nested_archives": nested,
    }


def extract_archives(config: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically extract a validated ZIP and its path-safe RAR members."""
    if config.get("dataset", {}).get("name") == "mefar":
        raise ValueError("MEFAR extraction is outside this adapter; reuse the existing MEFAR dataset")
    archive = Path(config["dataset"]["archive"])
    output = Path(config["dataset"]["extracted_root"])
    before = file_sha256(archive)
    if before != str(config["dataset"]["expected_sha256"]).lower():
        raise ValueError("Refusing to extract an archive with an unexpected SHA-256")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None:
            raise ValueError("Refusing to extract a ZIP with a failed CRC")
        for info in source.infolist():
            _safe_member_path(output, info.filename)
        source.extractall(output)
    unrar = _find_unrar()
    for rar in sorted(output.glob("*.rar")):
        names = subprocess.run(
            [str(unrar), "lb", str(rar)], check=True, capture_output=True, text=True
        ).stdout.splitlines()
        for name in names:
            _safe_member_path(output, name)
        subprocess.run([str(unrar), "x", "-o+", "-idq", str(rar), str(output) + "\\"], check=True)
    after = file_sha256(archive)
    if after != before:
        raise RuntimeError("Source archive changed during extraction")
    return {"archive_sha256_before": before, "archive_sha256_after": after, "immutable": True}


def _task_column(config: Mapping[str, Any], task_number: int) -> str:
    return f"level_{task_number}" if config["dataset"]["name"] == "clare" else f"lvl_{task_number}"


def _labels_for_record(
    config: Mapping[str, Any], root: Path, record: CognitiveLoadMultimodalRecord
) -> list[dict[str, Any]]:
    frame = pd.read_csv(root / record.label_path)
    column = _task_column(config, record.task_number)
    if column not in frame:
        raise ValueError(f"Missing label column {column} in {record.label_path}")
    scores = pd.to_numeric(frame[column], errors="coerce")
    valid = scores.notna()
    if config["dataset"]["name"] == "clare":
        ends = (np.arange(int(valid.sum()), dtype=float) + 1.0) * 10.0
        time_source = "row_order_inferred_10_second_intervals"
    else:
        ends = pd.to_numeric(frame.loc[valid, "time"], errors="raise").to_numpy(float)
        time_source = "explicit_time_column_seconds"
    values = scores.loc[valid].to_numpy(float)
    if not np.equal(values, np.floor(values)).all() or np.any((values < 1) | (values > 9)):
        raise ValueError(f"Labels outside the documented integer 1-9 scale: {record.label_path}")
    output = []
    for ordinal, (end, score_value) in enumerate(zip(ends, values)):
        score = int(score_value)
        target = 0 if score <= 3 else 1 if score <= 6 else 2
        output.append(
            {
                "sample_id": f"{record.record_id}__window-{ordinal:03d}",
                "record_id": record.record_id,
                "participant_id": record.participant_id,
                "source_participant_id": record.source_participant_id,
                "task_id": record.task_id,
                "task_number": record.task_number,
                "window_ordinal": ordinal,
                "window_start_seconds": float(end - 10.0),
                "window_end_seconds": float(end),
                "raw_subjective_score": score,
                "target": target,
                "class_name": ("low", "medium", "high")[target],
                "target_id": "subjective_cognitive_load_3class_fixed",
                "label_time_source": time_source,
                "label_source": "participant_self_report",
            }
        )
    return output


def _signal_data(path: Path, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["Timestamp", *columns])
    timestamps = pd.to_numeric(frame["Timestamp"], errors="coerce").to_numpy(float)
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return timestamps, values


def _signal_summary(
    path: Path, modality: str, columns: Sequence[str], *, require_all: bool
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    timestamps, values = _signal_data(path, columns)
    valid = np.isfinite(values).all(axis=1) if require_all else np.isfinite(values).any(axis=1)
    valid &= np.isfinite(timestamps)
    signal_times = timestamps[valid]
    relative = signal_times - signal_times[0] if len(signal_times) else np.asarray([], dtype=float)
    diffs = np.diff(signal_times)
    positive = diffs[diffs > 0]
    summary = {
        "modality": modality,
        "relative_path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "rows": int(len(timestamps)),
        "selected_columns": "|".join(columns),
        "valid_rows": int(valid.sum()),
        "missing_values": int(np.isnan(values).sum()),
        "infinite_values": int(np.isinf(values).sum()),
        "timestamp_start": None if not len(signal_times) else float(signal_times[0]),
        "timestamp_end": None if not len(signal_times) else float(signal_times[-1]),
        "duration_seconds": None if not len(signal_times) else float(signal_times[-1] - signal_times[0]),
        "median_positive_dt": None if not len(positive) else float(np.median(positive)),
        "nonpositive_timestamp_differences": int(np.sum(diffs <= 0)),
    }
    return summary, relative, values[valid]


def _file_inventory(root: Path) -> pd.DataFrame:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    def inspect(path: Path) -> dict[str, Any]:
        relative = path.relative_to(root).as_posix()
        return {
            "relative_path": relative,
            "top_level": relative.split("/")[0],
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "empty": path.stat().st_size == 0,
            "sha256": file_sha256(path),
        }
    with ThreadPoolExecutor(max_workers=4) as executor:
        return pd.DataFrame(executor.map(inspect, paths)).sort_values("relative_path").reset_index(drop=True)


def _window_count(relative_times: np.ndarray, start: float, end: float) -> int:
    return int(np.sum((relative_times >= start - 1e-9) & (relative_times < end - 1e-9)))


def build_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    archive_before = file_sha256(config["dataset"]["archive"])
    if archive_before != config["dataset"]["expected_sha256"]:
        raise ValueError("Source archive SHA-256 mismatch")
    root = Path(config["dataset"]["extracted_root"])
    dataset = get_dataset(config["dataset"]["name"], {"data_path": str(root)})
    records = list(dataset.iter_records())
    files = _file_inventory(root)
    modality_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    sync_rows: list[dict[str, Any]] = []
    for record in records:
        labels = _labels_for_record(config, root, record)
        validation = dataset.validate_record_files(record)
        summaries: dict[str, dict[str, Any]] = {}
        relative_times: dict[str, np.ndarray] = {}
        eeg_values: np.ndarray | None = None
        selections = {
            "EEG": (record.eeg_path, EEG_CHANNELS, True),
            "ECG": (record.ecg_path, ECG_COLUMNS, True),
            "EDA": (record.eda_path, (EDA_COLUMN,), True),
            "Gaze": (record.gaze_path, ("ET_PupilLeft", "ET_PupilRight", "Gaze X", "Gaze Y"), False),
        }
        for modality, (relative_path, columns, require_all) in selections.items():
            if relative_path is None:
                continue
            path = root / relative_path
            summary, relative, values = _signal_summary(path, modality, columns, require_all=require_all)
            summary.update({
                "record_id": record.record_id,
                "participant_id": record.participant_id,
                "task_id": record.task_id,
            })
            summary["relative_path"] = relative_path
            summaries[modality] = summary
            relative_times[modality] = relative
            modality_rows.append(summary)
            if modality == "EEG":
                eeg_timestamps, eeg_values = _signal_data(path, EEG_CHANNELS)
                if not np.isfinite(eeg_timestamps).all():
                    raise ValueError(f"EEG timestamps contain NaN/Inf: {relative_path}")
                relative_times[modality] = eeg_timestamps - eeg_timestamps[0]
        accepted = 0
        rejection_counts: dict[str, int] = {}
        for label in labels:
            start = label["window_start_seconds"]
            end = label["window_end_seconds"]
            reason = "accepted"
            eeg_count = _window_count(relative_times.get("EEG", np.asarray([])), start, end)
            ecg_count = _window_count(relative_times.get("ECG", np.asarray([])), start, end)
            eda_count = _window_count(relative_times.get("EDA", np.asarray([])), start, end)
            if not validation["complete_primary_modalities"]:
                reason = "missing_primary_modality"
            elif eeg_count != 2560:
                reason = "eeg_wrong_sample_count"
            elif eeg_values is None:
                reason = "missing_eeg"
            else:
                eeg_relative = relative_times["EEG"]
                mask = (eeg_relative >= start - 1e-9) & (eeg_relative < end - 1e-9)
                if not np.isfinite(eeg_values[mask]).all():
                    reason = "eeg_nonfinite"
                elif ecg_count < 4096:
                    reason = "ecg_insufficient_coverage"
                elif eda_count < 1024:
                    reason = "eda_insufficient_coverage"
            accepted_flag = reason == "accepted"
            accepted += int(accepted_flag)
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            label_rows.append(
                label
                | {
                    "eeg_sample_count": eeg_count,
                    "ecg_sample_count": ecg_count,
                    "eda_sample_count": eda_count,
                    "accepted_common_cohort": accepted_flag,
                    "rejection_reason": reason,
                }
            )
        starts = {name: value.get("timestamp_start") for name, value in summaries.items()}
        sync_rows.append(
            {
                "record_id": record.record_id,
                "participant_id": record.participant_id,
                "task_id": record.task_id,
                "eeg_timestamp_start": starts.get("EEG"),
                "ecg_timestamp_start": starts.get("ECG"),
                "eda_timestamp_start": starts.get("EDA"),
                "gaze_timestamp_start": starts.get("Gaze"),
                "fusion_alignment": "task_segment_relative_label_interval",
                "nearest_neighbour_merge": False,
                "common_absolute_clock_claimed": False,
                "safe_fusion_level": "10_second_label_interval_within_paired_task_files",
            }
        )
        task_rows.append(
            {
                **record.to_dict(),
                **validation,
                "label_count": len(labels),
                "accepted_common_windows": accepted,
                "rejection_counts": canonical_json(rejection_counts),
            }
        )
    labels_frame = pd.DataFrame(label_rows).sort_values("sample_id").reset_index(drop=True)
    cohort = labels_frame.loc[labels_frame["accepted_common_cohort"]].copy()
    if cohort["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate common-cohort sample IDs")
    archive_after = file_sha256(config["dataset"]["archive"])
    if archive_after != archive_before:
        raise RuntimeError("Source archive changed during inventory")
    tasks = pd.DataFrame(task_rows).sort_values("record_id").reset_index(drop=True)
    participants = (
        tasks.groupby(["participant_id", "source_participant_id"], sort=True)
        .agg(
            label_addressable_tasks=("record_id", "size"),
            complete_primary_tasks=("complete_primary_modalities", "sum"),
            accepted_common_windows=("accepted_common_windows", "sum"),
        )
        .reset_index()
    )
    duplicates = files.groupby("sha256").filter(lambda group: len(group) > 1)
    return {
        "archive_before": archive_before,
        "archive_after": archive_after,
        "files": files,
        "participants": participants,
        "tasks": tasks,
        "modalities": pd.DataFrame(modality_rows),
        "labels": labels_frame,
        "cohort": cohort,
        "synchronization": pd.DataFrame(sync_rows),
        "duplicates": duplicates,
    }


def eeg_feature_names() -> list[str]:
    spec = SpectralFeatureSpec(sampling_rate_hz=256.0, nperseg=512, noverlap=256)
    dummy = np.zeros((1, len(EEG_CHANNELS), 2560), dtype=np.float32)
    return list(extract_spectral_feature_bundle(dummy, channel_names=EEG_CHANNELS, spec=spec).channel_wise_spectral_columns)


def peripheral_feature_names() -> list[str]:
    columns = [f"ecg__{column}__{stat}" for column in ECG_COLUMNS for stat in SUMMARY_STATISTICS]
    columns.extend(f"eda__conductance__{stat}" for stat in SUMMARY_STATISTICS)
    return columns


def feature_names(mode: str) -> list[str]:
    eeg = eeg_feature_names()
    peripheral = peripheral_feature_names()
    if mode == "eeg_only":
        return eeg
    if mode == "peripheral_only":
        return peripheral
    if mode == "eeg_peripheral":
        return eeg + peripheral
    raise ValueError(f"Unknown modality mode {mode!r}")


def build_folds(config: Mapping[str, Any], cohort: pd.DataFrame) -> dict[str, Any]:
    groups = cohort["participant_id"].astype(str).to_numpy()
    labels = cohort["target"].to_numpy(dtype=int)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=int(config["evaluation"]["random_state"])
    )
    folds = []
    seen: set[str] = set()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros((len(labels), 1)), labels, groups), 1):
        train_groups = sorted(set(groups[train_idx]))
        test_groups = sorted(set(groups[test_idx]))
        overlap = sorted(set(train_groups) & set(test_groups))
        if overlap:
            raise RuntimeError(f"Participant leakage in fold {fold}: {overlap}")
        test_ids = sorted(cohort.iloc[test_idx]["sample_id"].astype(str))
        if seen & set(test_ids):
            raise RuntimeError("Evaluation sample IDs repeat across folds")
        seen.update(test_ids)
        folds.append(
            {
                "fold": fold,
                "train_participants": train_groups,
                "test_participants": test_groups,
                "train_sample_ids": sorted(cohort.iloc[train_idx]["sample_id"].astype(str)),
                "test_sample_ids": test_ids,
                "train_class_counts": {str(k): int(v) for k, v in cohort.iloc[train_idx]["target"].value_counts().sort_index().items()},
                "test_class_counts": {str(k): int(v) for k, v in cohort.iloc[test_idx]["target"].value_counts().sort_index().items()},
                "participant_overlap": 0,
            }
        )
    if seen != set(cohort["sample_id"].astype(str)):
        raise RuntimeError("Outer folds do not cover the common cohort exactly once")
    return {"protocol": "5-fold StratifiedGroupKFold by participant_id", "folds": folds}


def compatibility_matrix() -> pd.DataFrame:
    rows = [
        ("xgboost", "eeg_only", True, "52 spectral EEG features"),
        ("xgboost", "peripheral_only", True, "28 ECG/EDA summary features"),
        ("xgboost", "eeg_peripheral", True, "early fusion of 80 features"),
        ("torch_shallow_convnet", "eeg_only", True, "raw EEG [B,1,4,2560]"),
        ("torch_shallow_fusion", "eeg_peripheral", True, "separate shallow EEG and peripheral MLP branches"),
        ("torch_shallow_convnet", "peripheral_only", False, "tabular features cannot be passed to a raw EEG CNN"),
        ("torch_shallow_convnet", "eeg_peripheral", False, "use torch_shallow_fusion instead"),
        ("xgboost", "gaze_only", False, "eye tracking is a separate diagnostic modality"),
    ]
    return pd.DataFrame(rows, columns=["model", "mode", "supported", "reason"])


def build_run_matrix(config: Mapping[str, Any], folds: Mapping[str, Any]) -> pd.DataFrame:
    combinations = [
        ("xgboost", mode) for mode in PRIMARY_MODES
    ] + [
        ("torch_shallow_convnet", "eeg_only"),
        ("torch_shallow_fusion", "eeg_peripheral"),
    ]
    rows = []
    for fold in folds["folds"]:
        for model, mode in combinations:
            payload = {
                "experiment_id": config["experiment_id"],
                "fold": fold["fold"],
                "model": model,
                "mode": mode,
                "target_id": config["target"]["target_id"],
                "test_sample_ids": fold["test_sample_ids"],
            }
            rows.append(
                {
                    "run_id": f"{model}__{mode}__fold{fold['fold']:02d}__{stable_hash(payload)[:10]}",
                    "fold": fold["fold"],
                    "model": model,
                    "mode": mode,
                    "feature_count": len(feature_names(mode)) if model == "xgboost" else (0 if mode == "eeg_only" else 28),
                    "raw_eeg_shape": "[1,4,2560]" if model.startswith("torch_") else "",
                    "n_train_samples": len(fold["train_sample_ids"]),
                    "n_test_samples": len(fold["test_sample_ids"]),
                    "evaluation_sample_ids_hash": stable_hash(fold["test_sample_ids"]),
                    "specification_hash": stable_hash(payload),
                }
            )
    return pd.DataFrame(rows)


def build_protocol(config: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    cohort = inventory["cohort"]
    folds = build_folds(config, cohort)
    matrix = build_run_matrix(config, folds)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "diagnostic",
        "dataset": config["dataset"]["name"],
        "archive_sha256": inventory["archive_before"],
        "target": config["target"],
        "participants": int(cohort["participant_id"].nunique()),
        "common_cohort_windows": int(len(cohort)),
        "raw_eeg_shape": [1, 4, 2560],
        "sampling_rate_hz": 256.0,
        "channel_order": list(EEG_CHANNELS),
        "fusion_level": "10_second_label_interval_within_paired_task_files",
        "nearest_neighbour_merge": False,
        "feature_counts": {mode: len(feature_names(mode)) for mode in PRIMARY_MODES},
        "outer_protocol": folds["protocol"],
        "fold_manifest_hash": stable_hash(folds),
        "run_matrix_hash": stable_hash(matrix.to_dict(orient="records")),
        "run_count": int(len(matrix)),
        "future_training_units": int(len(matrix)),
        "primary_metrics": ["macro_f1", "balanced_accuracy"],
        "secondary_metrics": ["accuracy", "weighted_f1", "per_class_precision", "per_class_recall", "confusion_matrix"],
        "effects": {
            "xgboost": "macro_f1(eeg_peripheral)-macro_f1(eeg_only)",
            "shallow": "macro_f1(torch_shallow_fusion)-macro_f1(torch_shallow_convnet)",
        },
        "leakage_guards": {
            "participant_disjoint_outer_folds": True,
            "same_evaluation_sample_ids": True,
            "train_only_imputer_scaler": True,
            "oversampling_before_split": False,
            "target_columns_in_features": False,
            "dataset_mixing": False,
        },
        "runtime_measurement_contract": [
            "training_time", "model_inference_latency", "preprocessing_latency",
            "end_to_end_latency", "peak_ram", "peak_vram", "feature_extraction_latency",
        ],
    }
    protocol["protocol_hash"] = stable_hash(protocol)
    return {"protocol": protocol, "folds": folds, "matrix": matrix}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def plan_summary(inventory: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    cohort = inventory["cohort"]
    tasks = inventory["tasks"]
    modalities = inventory["modalities"]
    protocol = plan["protocol"]
    return {
        **protocol,
        "inventoried_participants": int(inventory["participants"]["participant_id"].nunique()),
        "usable_common_cohort_participants": int(cohort["participant_id"].nunique()),
        "label_addressable_tasks": int(len(tasks)),
        "complete_primary_tasks": int(tasks["complete_primary_modalities"].sum()),
        "modality_record_counts": {
            str(key): int(value)
            for key, value in modalities["modality"].value_counts().sort_index().items()
        },
        "raw_label_windows": int(len(inventory["labels"])),
        "class_distribution": {str(k): int(v) for k, v in cohort["target"].value_counts().sort_index().items()},
        "raw_score_distribution": {str(k): int(v) for k, v in cohort["raw_subjective_score"].value_counts().sort_index().items()},
        "folds": [
            {
                "fold": fold["fold"],
                "train_participants": len(fold["train_participants"]),
                "test_participants": len(fold["test_participants"]),
                "train_class_counts": fold["train_class_counts"],
                "test_class_counts": fold["test_class_counts"],
                "participant_overlap": fold["participant_overlap"],
            }
            for fold in plan["folds"]["folds"]
        ],
        "supported_combinations": int(compatibility_matrix()["supported"].sum()),
        "unsupported_combinations": int((~compatibility_matrix()["supported"]).sum()),
        "evaluation_units": int(len(plan["matrix"])),
        "expected_training_artifacts": [
            "runs/<run_id>/metrics.json",
            "runs/<run_id>/predictions.parquet",
            "runs/<run_id>/normalization_stats.json",
            "runs/<run_id>/model.pt (torch only)",
            "runs/<run_id>/training_log.csv (torch only)",
            "runs/<run_id>/validation_split.json (torch only)",
            "summary_xgboost.csv",
            "summary_shallow.csv",
        ],
        "models_trained": 0,
        "writes_performed": False,
    }


def write_inventory_artifacts(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    if config["dataset"]["name"] == "mefar":
        from bench.experiments.mefar_external_xgboost import write_plan_artifacts

        return write_plan_artifacts(config_path)
    archive_manifest = verify_archive(config)
    inventory = build_inventory(config)
    plan = build_protocol(config, inventory)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    archive_manifest.update({
        "sha256_after_inventory": inventory["archive_after"],
        "immutable": inventory["archive_before"] == inventory["archive_after"],
    })
    _write_json(output / "archive_manifest.json", archive_manifest)
    _write_csv(output / "file_inventory.csv", inventory["files"])
    _write_csv(output / "participant_inventory.csv", inventory["participants"])
    _write_csv(output / "task_inventory.csv", inventory["tasks"])
    _write_csv(output / "modality_inventory.csv", inventory["modalities"])
    _write_csv(output / "label_audit.csv", inventory["labels"])
    _write_csv(output / "synchronization_audit.csv", inventory["synchronization"])
    _write_csv(output / "cohort_inventory.csv", inventory["cohort"])
    _write_csv(output / "duplicate_file_audit.csv", inventory["duplicates"])
    _write_json(output / "fold_manifest.json", plan["folds"])
    _write_csv(output / "run_matrix.csv", plan["matrix"])
    _write_csv(output / "compatibility_matrix.csv", compatibility_matrix())
    _write_json(output / "protocol_manifest.json", plan["protocol"])
    summary = plan_summary(inventory, plan)
    summary["writes_performed"] = True
    _write_json(output / "plan_summary.json", summary)
    return summary


def plan_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    if config["dataset"]["name"] == "mefar":
        from bench.experiments.mefar_external_xgboost import plan_experiment as plan_mefar

        return plan_mefar(config_path)
    output = Path(config["output_dir"])
    required = [
        "protocol_manifest.json", "plan_summary.json", "fold_manifest.json",
        "run_matrix.csv", "cohort_inventory.csv",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run --inventory before --plan-only; missing {missing}")
    if file_sha256(config["dataset"]["archive"]) != config["dataset"]["expected_sha256"]:
        raise ValueError("Source archive changed after inventory")
    summary = json.loads((output / "plan_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "protocol_manifest.json").read_text(encoding="utf-8"))
    if summary["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Plan summary and protocol manifest hashes disagree")
    summary["writes_performed"] = False
    summary["models_trained"] = 0
    return summary


def _summary_features(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan")] * len(SUMMARY_STATISTICS)
    return [
        float(np.mean(finite)), float(np.std(finite)), float(np.min(finite)),
        float(np.max(finite)), float(np.median(finite)),
        float(np.quantile(finite, 0.25)), float(np.quantile(finite, 0.75)),
    ]


def materialize_model_inputs(config_path: str | Path) -> dict[str, Any]:
    """Materialize the audited common cohort without fitting any estimator."""
    config = load_config(config_path)
    if config["dataset"]["name"] == "mefar":
        raise ValueError("MEFAR uses session-level feature materialization, not raw EEG windows")
    output = Path(config["output_dir"])
    cohort = pd.read_csv(output / "cohort_inventory.csv")
    accepted = set(cohort["sample_id"].astype(str))
    root = Path(config["dataset"]["extracted_root"])
    dataset = get_dataset(config["dataset"]["name"], {"data_path": str(root)})
    metadata: list[dict[str, Any]] = []
    raw_windows: list[np.ndarray] = []
    peripheral_rows: list[list[float]] = []
    for record in dataset.iter_records():
        labels = _labels_for_record(config, root, record)
        selected_labels = [row for row in labels if row["sample_id"] in accepted]
        if not selected_labels:
            continue
        if record.eeg_path is None or record.ecg_path is None or record.eda_path is None:
            raise RuntimeError(f"Accepted record lacks a primary modality: {record.record_id}")
        eeg_t, eeg_x = _signal_data(root / record.eeg_path, EEG_CHANNELS)
        eeg_rel = eeg_t - eeg_t[0]
        ecg_t, ecg_x = _signal_data(root / record.ecg_path, ECG_COLUMNS)
        ecg_valid = np.isfinite(ecg_t) & np.isfinite(ecg_x).all(axis=1)
        ecg_t = ecg_t[ecg_valid]
        ecg_x = ecg_x[ecg_valid]
        ecg_rel = ecg_t - ecg_t[0]
        eda_t, eda_x = _signal_data(root / record.eda_path, (EDA_COLUMN,))
        eda_valid = np.isfinite(eda_t) & np.isfinite(eda_x[:, 0])
        eda_t = eda_t[eda_valid]
        eda_x = eda_x[eda_valid]
        eda_rel = eda_t - eda_t[0]
        for label in selected_labels:
            start, end = label["window_start_seconds"], label["window_end_seconds"]
            eeg_mask = (eeg_rel >= start - 1e-9) & (eeg_rel < end - 1e-9)
            ecg_mask = (ecg_rel >= start - 1e-9) & (ecg_rel < end - 1e-9)
            eda_mask = (eda_rel >= start - 1e-9) & (eda_rel < end - 1e-9)
            eeg_window = eeg_x[eeg_mask].T.astype(np.float32)
            if eeg_window.shape != (4, 2560) or not np.isfinite(eeg_window).all():
                raise RuntimeError(f"Cohort audit drift for {label['sample_id']}")
            peripheral = [
                value
                for column_index in range(len(ECG_COLUMNS))
                for value in _summary_features(ecg_x[ecg_mask, column_index])
            ]
            peripheral.extend(_summary_features(eda_x[eda_mask, 0]))
            metadata.append({key: label[key] for key in (
                "sample_id", "record_id", "participant_id", "task_id",
                "target", "raw_subjective_score",
            )})
            raw_windows.append(eeg_window)
            peripheral_rows.append(peripheral)
    metadata_frame = pd.DataFrame(metadata)
    order = {sample_id: index for index, sample_id in enumerate(cohort["sample_id"].astype(str))}
    indices = np.argsort([order[value] for value in metadata_frame["sample_id"].astype(str)])
    metadata_frame = metadata_frame.iloc[indices].reset_index(drop=True)
    raw = np.stack(raw_windows, axis=0)[indices]
    peripheral = np.asarray(peripheral_rows, dtype=np.float32)[indices]
    if set(metadata_frame["sample_id"]) != accepted or peripheral.shape[1] != 28:
        raise RuntimeError("Materialized common cohort no longer matches its manifest")
    spectral = extract_spectral_feature_bundle(
        raw,
        channel_names=EEG_CHANNELS,
        spec=SpectralFeatureSpec(sampling_rate_hz=256.0, nperseg=512, noverlap=256),
    )
    eeg_features = spectral.channel_wise[:, [
        spectral.channel_wise_columns.index(name)
        for name in spectral.channel_wise_spectral_columns
    ]]
    return {
        "metadata": metadata_frame,
        "raw_eeg": raw[:, None, :, :],
        "eeg_features": eeg_features,
        "peripheral_features": peripheral,
        "feature_names": {
            "eeg_only": list(spectral.channel_wise_spectral_columns),
            "peripheral_only": peripheral_feature_names(),
            "eeg_peripheral": list(spectral.channel_wise_spectral_columns) + peripheral_feature_names(),
        },
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([0, 1, 2])
    metrics = MetricsCalculator.calculate_all_metrics(
        y_true, y_pred, y_proba=proba, labels=labels
    )
    metrics["class_metrics"] = MetricsCalculator.calculate_class_metrics(
        y_true, y_pred, labels=labels
    )
    return metrics


def run_model_family(config_path: str | Path, model_family: str) -> dict[str, Any]:
    """Execute a preregistered family; intentionally separate from plan-only."""
    config = load_config(config_path)
    family = model_family.strip().lower()
    if config["dataset"]["name"] == "mefar":
        if family != "xgboost":
            raise ValueError(
                "MEFAR does not support ShallowConvNet: no suitable raw multichannel EEG is available"
            )
        from bench.experiments.mefar_external_xgboost import run_xgboost

        return run_xgboost(config_path)
    if family not in {"xgboost", "shallow"}:
        raise ValueError("model_family must be 'xgboost' or 'shallow'")
    plan_experiment(config_path)
    output = Path(config["output_dir"])
    folds = json.loads((output / "fold_manifest.json").read_text(encoding="utf-8"))["folds"]
    matrix = pd.read_csv(output / "run_matrix.csv")
    materialized = materialize_model_inputs(config_path)
    metadata = materialized["metadata"]
    y = metadata["target"].to_numpy(dtype=int)
    summary_rows = []
    rows = matrix.loc[
        matrix["model"].eq("xgboost")
        if family == "xgboost"
        else matrix["model"].isin(["torch_shallow_convnet", "torch_shallow_fusion"])
    ]
    for run in rows.to_dict(orient="records"):
        fold = folds[int(run["fold"]) - 1]
        train = metadata["sample_id"].isin(fold["train_sample_ids"]).to_numpy()
        test = metadata["sample_id"].isin(fold["test_sample_ids"]).to_numpy()
        run_dir = output / "runs" / str(run["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if family == "xgboost":
            mode = str(run["mode"])
            eeg = materialized["eeg_features"]
            peripheral = materialized["peripheral_features"]
            values = eeg if mode == "eeg_only" else peripheral if mode == "peripheral_only" else np.column_stack((eeg, peripheral))
            preprocessing_started = time.perf_counter()
            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(values[train])
            x_test = imputer.transform(values[test])
            preprocessing_time = time.perf_counter() - preprocessing_started
            model = build_model("xgboost", "classification", (values.shape[1],), 3, config["models"]["xgboost"])
            model.fit(x_train, y[train])
            inference_started = time.perf_counter()
            prediction = np.asarray(model.predict(x_test), dtype=int)
            probability = np.asarray(model.predict_proba(x_test), dtype=float)
            inference_time = time.perf_counter() - inference_started
            _write_json(run_dir / "normalization_stats.json", {
                "imputation": "outer_train_median", "statistics": imputer.statistics_.tolist(),
                "fit_sample_ids": sorted(metadata.loc[train, "sample_id"].astype(str)),
            })
        else:
            raw = materialized["raw_eeg"]
            peripheral = materialized["peripheral_features"]
            shallow = dict(config["models"]["torch_shallow"])
            shallow.update({"sampling_rate": 256.0, "channel_names": list(EEG_CHANNELS)})
            if run["model"] == "torch_shallow_convnet":
                values = raw
                model = build_model("torch_shallow_convnet", "classification", raw.shape[1:], 3, shallow)
            else:
                fusion = dict(config["models"]["torch_shallow_fusion"])
                fusion.update(shallow)
                fusion["eeg_dropout"] = fusion.pop("dropout")
                fusion["eeg_input_shape"] = [1, 4, 2560]
                values = np.column_stack((raw.reshape(len(raw), -1), peripheral)).astype(np.float32)
                model = build_model("torch_shallow_fusion", "classification", (values.shape[1],), 3, fusion)
            model.set_validation_groups(
                metadata.loc[train, "participant_id"].to_numpy(),
                subject_ids=metadata.loc[train, "participant_id"].to_numpy(),
                record_ids=metadata.loc[train, "record_id"].to_numpy(),
                outer_test_record_ids=metadata.loc[test, "record_id"].to_numpy(),
                outer_test_group_ids=metadata.loc[test, "participant_id"].to_numpy(),
                strategy="group_holdout",
                group_column="participant_id",
            )
            preprocessing_time = 0.0
            model.fit(values[train], y[train])
            inference_started = time.perf_counter()
            probability = np.asarray(model.predict_proba(values[test]), dtype=float)
            prediction = np.asarray(model.predict(values[test]), dtype=int)
            inference_time = time.perf_counter() - inference_started
            model.save(run_dir / "model.pt")
            pd.DataFrame(model.training_log_).to_csv(run_dir / "training_log.csv", index=False)
            _write_json(run_dir / "validation_split.json", model.validation_split_)
        metrics = _classification_metrics(y[test], prediction, probability)
        metrics.update({
            "training_and_inference_time_seconds": time.perf_counter() - started,
            "preprocessing_time_seconds": preprocessing_time,
            "model_inference_time_seconds": inference_time,
        })
        predictions = metadata.loc[test].copy()
        predictions["y_true"] = y[test]
        predictions["y_pred"] = prediction
        for class_index in range(3):
            predictions[f"proba_{class_index}"] = probability[:, class_index]
        predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        _write_json(run_dir / "metrics.json", metrics)
        summary_rows.append({**run, **{key: value for key, value in metrics.items() if np.isscalar(value)}})
    summary_path = output / f"summary_{family}.csv"
    _write_csv(summary_path, pd.DataFrame(summary_rows))
    return {
        "dataset": config["dataset"]["name"],
        "model_family": family,
        "models_trained": len(summary_rows),
        "summary_path": summary_path.as_posix(),
    }


__all__ = [
    "SCHEMA_VERSION", "EXPECTED_SHA256", "build_folds", "build_inventory",
    "build_protocol", "build_run_matrix", "compatibility_matrix",
    "eeg_feature_names", "extract_archives", "feature_names", "file_sha256",
    "load_config", "peripheral_feature_names", "plan_experiment", "stable_hash",
    "materialize_model_inputs", "run_model_family", "verify_archive",
    "write_inventory_artifacts",
]
