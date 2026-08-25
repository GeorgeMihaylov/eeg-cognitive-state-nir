"""Leakage-safe CL-Drive <-> CLARE raw-EEG DANN protocol.

The planning and cache-building paths never fit a model.  Execution is kept
behind an explicit confirmation flag and reuses the repository ShallowConvNet
encoder and DANN infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from bench.datasets.clare_cldrive_dataset import EEG_CHANNELS
from bench.datasets.datasets_registry import get_dataset
from bench.experiments.external_multimodal_protocol import (
    EXPECTED_SHA256,
    _labels_for_record,
    _signal_data,
    file_sha256,
)
from model_zoo.DL.adapter import TorchClassificationAdapter, seed_torch
from model_zoo.DL.dann import DANNFoldData, DANNModule, DANNObjective, DANNPartition
from model_zoo.factory import build_model


SCHEMA_VERSION = "cross-dataset-clare-cldrive-dann-v1"
EXPERIMENT_ID = "cross_dataset_clare_cldrive_dann_v1"
DATASETS = ("cl_drive", "clare")
DIRECTIONS = ("cl_drive_to_clare", "clare_to_cl_drive")
MODES = ("target_only", "source_only", "dann")
INPUT_SHAPE = (1, 4, 2560)
TARGET_ID = "subjective_cognitive_load_3class_fixed"
EXPECTED_COHORTS = {
    "cl_drive": {
        "windows": 3086,
        "participants": 21,
        "records": 182,
        "class_counts": {"0": 825, "1": 1612, "2": 649},
        "rejections": {"missing_eeg": 36, "nonfinite_eeg": 52, "wrong_sample_count": 138},
    },
    "clare": {
        "windows": 3829,
        "participants": 19,
        "records": 73,
        "class_counts": {"0": 602, "1": 2098, "2": 1129},
        "rejections": {"missing_eeg": 324, "nonfinite_eeg": 87, "wrong_sample_count": 14},
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def fixed_target(score: int | float) -> int:
    value = float(score)
    if not math.isfinite(value) or value != math.floor(value) or not 1 <= value <= 9:
        raise ValueError("Subjective cognitive-load score must be an integer in [1, 9]")
    return 0 if value <= 3 else 1 if value <= 6 else 2


def eeg_only_window(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    start_seconds: float,
    end_seconds: float,
) -> tuple[np.ndarray | None, str]:
    """Apply only the preregistered EEG inclusion checks to one interval."""
    times = np.asarray(timestamps, dtype=np.float64)
    signal = np.asarray(values)
    if times.ndim != 1 or signal.ndim != 2 or signal.shape[0] != len(times) or signal.shape[1] != 4:
        raise ValueError("EEG source arrays must have shapes [samples] and [samples, 4]")
    mask = (times >= float(start_seconds) - 1e-9) & (times < float(end_seconds) - 1e-9)
    window = signal[mask]
    if window.shape != (2560, 4):
        return None, "wrong_sample_count"
    if not np.isfinite(window).all():
        return None, "nonfinite_eeg"
    return window.T.astype(np.float32, copy=False)[None, :, :], "accepted"


def _repository_root(config_path: Path) -> Path:
    for parent in (config_path.resolve().parent, *config_path.resolve().parents):
        if (parent / "bench").is_dir() and (parent / "model_zoo").is_dir():
            return parent
    raise FileNotFoundError(f"Cannot locate repository root from {config_path}")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Expected experiment_id={EXPERIMENT_ID!r}")
    datasets = config.get("datasets", {})
    if tuple(sorted(datasets)) != tuple(sorted(DATASETS)):
        raise ValueError("datasets must contain exactly cl_drive and clare")
    for name in DATASETS:
        if datasets[name].get("expected_sha256") != EXPECTED_SHA256[name]:
            raise ValueError(f"Unexpected validated archive SHA-256 for {name}")
    target = config.get("target", {})
    if target.get("target_id") != TARGET_ID:
        raise ValueError(f"target.target_id must be {TARGET_ID}")
    if target.get("fixed_bins") != {"0": [1, 3], "1": [4, 6], "2": [7, 9]}:
        raise ValueError("Fixed target mapping must remain 1-3 / 4-6 / 7-9")
    windowing = config.get("windowing", {})
    expected_window = {
        "channels": list(EEG_CHANNELS),
        "sampling_rate_hz": 256,
        "window_seconds": 10.0,
        "samples": 2560,
    }
    if any(windowing.get(key) != value for key, value in expected_window.items()):
        raise ValueError(f"Raw EEG window contract changed: expected {expected_window}")
    evaluation = config.get("evaluation", {})
    if int(evaluation.get("target_folds", 0)) != 5 or int(evaluation.get("random_state", -1)) != 42:
        raise ValueError("The protocol requires five target folds and random_state=42")
    if tuple(config.get("directions", ())) != DIRECTIONS or tuple(config.get("modes", ())) != MODES:
        raise ValueError("directions or modes differ from the preregistered 2 x 3 matrix")
    if config.get("model", {}).get("name") != "torch_shallow_convnet":
        raise ValueError("Only torch_shallow_convnet is supported")
    if set(config.get("dann", {})) < {
        "n_domains", "domain_hidden_dims", "domain_dropout",
        "domain_loss_lambda", "gradient_reversal_schedule",
    }:
        raise ValueError("DANN parameters must be explicit")
    return config


def _dataset_config(config: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    return {
        "schema_version": "external-cognitive-load-multimodal-v1",
        "dataset": {"name": dataset},
        "target": config["target"],
    }


def _source_participants(config: Mapping[str, Any], root: Path, dataset: str) -> set[str]:
    data_root = _resolve(root, config["datasets"][dataset]["extracted_root"])
    loader = get_dataset(dataset, {"data_path": str(data_root)})
    return {str(record.source_participant_id) for record in loader.iter_records()}


def participant_identity_audit(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    participant_sets = {
        dataset: _source_participants(config, root, dataset) for dataset in DATASETS
    }
    overlapping = sorted(participant_sets["cl_drive"] & participant_sets["clare"])
    evidence = []
    for dataset in DATASETS:
        extracted = _resolve(root, config["datasets"][dataset]["extracted_root"])
        for name in ("README.txt", "MANIFEST.TXT"):
            path = extracted / name
            evidence.append({
                "dataset": dataset,
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
                "finding": (
                    "README calls directory and label identifiers participant IDs; "
                    "it does not state whether IDs are shared across studies"
                    if name.lower().startswith("readme")
                    else "archive member inventory only; no cross-study identity mapping"
                ),
            })
        for source_id in overlapping:
            label_path = extracted / "Labels" / f"{source_id}.csv"
            if not label_path.is_file():
                raise RuntimeError(f"Overlapping participant label file is missing: {label_path}")
            columns = pd.read_csv(label_path, nrows=0).columns.astype(str).tolist()
            evidence.append({
                "dataset": dataset,
                "path": label_path.relative_to(root).as_posix(),
                "sha256": file_sha256(label_path),
                "finding": (
                    "label schema exists for the same source participant ID; schemas differ "
                    "by study task design but contain no cross-study identity declaration; "
                    f"columns={columns}"
                ),
            })
    return {
        "overlapping_source_ids": [f"sub-{value}" for value in overlapping],
        "count": len(overlapping),
        "evidence_inspected": evidence,
        "conclusion": (
            "The local documentation does not prove that equal source IDs identify "
            "different people; independence cannot be established."
        ),
        "policy_used_for_protocol": (
            "conservative_shared_person_key_for_equal_source_id; dataset-qualified "
            "keys for all other participants; target-test keys excluded from source training"
        ),
    }


def _person_key(dataset: str, source_id: str, overlapping: set[str]) -> str:
    participant = f"sub-{source_id}"
    return participant if source_id in overlapping else f"{dataset}::{participant}"


def _cache_paths(config: Mapping[str, Any], root: Path, dataset: str) -> dict[str, Path]:
    directory = _resolve(root, config["output_dir"]) / "cache" / dataset
    return {
        "directory": directory,
        "raw": directory / "raw_eeg.npy",
        "index": directory / "index.parquet",
        "manifest": directory / "cache_manifest.json",
    }


def _validate_cache(config: Mapping[str, Any], root: Path, dataset: str) -> dict[str, Any] | None:
    paths = _cache_paths(config, root, dataset)
    if not all(paths[key].is_file() for key in ("raw", "index", "manifest")):
        return None
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("dataset") != dataset:
        return None
    raw = np.load(paths["raw"], mmap_mode="r")
    index = pd.read_parquet(paths["index"])
    checks = {
        "shape": list(raw.shape) == manifest.get("shape"),
        "dtype": str(raw.dtype) == manifest.get("dtype") == "float32",
        "rows": len(index) == len(raw) == int(manifest.get("rows", -1)),
        "sample_ids_unique": not index["sample_id"].duplicated().any(),
        "raw_sha256": file_sha256(paths["raw"]) == manifest.get("raw_eeg_sha256"),
        "index_sha256": file_sha256(paths["index"]) == manifest.get("index_sha256"),
    }
    identity_payload = dict(manifest.get("identity_payload", {}))
    checks["cache_identity"] = stable_hash(identity_payload) == manifest.get("cache_identity_hash")
    if not all(checks.values()):
        raise RuntimeError(f"Invalid existing {dataset} EEG-only cache: {checks}")
    manifest["reused"] = True
    return manifest


def _materialize_dataset(
    config: Mapping[str, Any], root: Path, dataset: str, overlapping: set[str]
) -> dict[str, Any]:
    cached = _validate_cache(config, root, dataset)
    if cached is not None:
        return cached
    data_cfg = config["datasets"][dataset]
    archive = _resolve(root, data_cfg["archive"])
    archive_hash = file_sha256(archive)
    if archive_hash != data_cfg["expected_sha256"]:
        raise RuntimeError(f"Validated source archive changed for {dataset}")
    data_root = _resolve(root, data_cfg["extracted_root"])
    loader = get_dataset(dataset, {"data_path": str(data_root)})
    label_cfg = _dataset_config(config, dataset)
    windows: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    rejections = {"missing_eeg": 0, "nonfinite_eeg": 0, "wrong_sample_count": 0}
    for record in loader.iter_records():
        labels = _labels_for_record(label_cfg, data_root, record)
        if record.eeg_path is None:
            rejections["missing_eeg"] += len(labels)
            continue
        timestamps, values = _signal_data(data_root / record.eeg_path, EEG_CHANNELS)
        if not len(timestamps) or not np.isfinite(timestamps[0]):
            rejections["wrong_sample_count"] += len(labels)
            continue
        relative = timestamps - timestamps[0]
        for label in labels:
            eeg_window, reason = eeg_only_window(
                relative,
                values,
                start_seconds=float(label["window_start_seconds"]),
                end_seconds=float(label["window_end_seconds"]),
            )
            if reason != "accepted" or eeg_window is None:
                rejections[reason] += 1
                continue
            target = fixed_target(label["raw_subjective_score"])
            if target != int(label["target"]):
                raise RuntimeError("Shared fixed target mapping drifted")
            rows.append({
                "sample_id": str(label["sample_id"]),
                "dataset": dataset,
                "participant_id": str(label["participant_id"]),
                "dataset_participant_id": f"{dataset}::{label['participant_id']}",
                "source_participant_id": str(label["source_participant_id"]),
                "cross_dataset_person_key": _person_key(
                    dataset, str(label["source_participant_id"]), overlapping
                ),
                "record_id": str(label["record_id"]),
                "task_id": str(label["task_id"]),
                "task_number": int(label["task_number"]),
                "window_ordinal": int(label["window_ordinal"]),
                "window_start_seconds": float(label["window_start_seconds"]),
                "window_end_seconds": float(label["window_end_seconds"]),
                "raw_subjective_score": int(label["raw_subjective_score"]),
                "target": target,
                "class_name": ("low", "medium", "high")[target],
                "target_id": TARGET_ID,
            })
            windows.append(eeg_window)
    index = pd.DataFrame(rows)
    if index.empty:
        raise RuntimeError(f"No accepted EEG-only windows for {dataset}")
    order = np.argsort(index["sample_id"].astype(str).to_numpy())
    index = index.iloc[order].reset_index(drop=True)
    raw = np.stack(windows, axis=0)[order]
    if raw.shape[1:] != INPUT_SHAPE or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise RuntimeError(f"Invalid materialized tensor for {dataset}: {raw.shape}, {raw.dtype}")
    if index["sample_id"].duplicated().any():
        raise RuntimeError(f"Duplicate sample IDs in {dataset} EEG-only cohort")
    observed = {
        "windows": len(index),
        "participants": int(index["participant_id"].nunique()),
        "records": int(index["record_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in index["target"].value_counts().sort_index().items()},
        "rejections": rejections,
    }
    if observed != EXPECTED_COHORTS[dataset]:
        raise RuntimeError(
            f"{dataset} EEG-only cohort differs from the audited contract: "
            f"observed={observed}, expected={EXPECTED_COHORTS[dataset]}"
        )
    paths = _cache_paths(config, root, dataset)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    raw_tmp = paths["raw"].with_suffix(".npy.tmp")
    with raw_tmp.open("wb") as stream:
        np.save(stream, raw, allow_pickle=False)
    os.replace(raw_tmp, paths["raw"])
    index_tmp = paths["index"].with_suffix(".parquet.tmp")
    index.to_parquet(index_tmp, index=False)
    os.replace(index_tmp, paths["index"])
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source_archive_sha256": archive_hash,
        "target_id": TARGET_ID,
        "fixed_bins": config["target"]["fixed_bins"],
        "channel_order": list(EEG_CHANNELS),
        "sampling_rate_hz": 256,
        "window_seconds": 10.0,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "sample_ids_hash": stable_hash(index["sample_id"].astype(str).tolist()),
        "cross_dataset_identity_policy": "conservative_shared_person_key_for_equal_source_id",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "rows": len(index),
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "channel_order": list(EEG_CHANNELS),
        "sampling_rate_hz": 256,
        "window_seconds": 10.0,
        "class_counts": observed["class_counts"],
        "participant_count": observed["participants"],
        "record_count": observed["records"],
        "rejection_counts": rejections,
        "source_archive_sha256": archive_hash,
        "raw_eeg_sha256": file_sha256(paths["raw"]),
        "index_sha256": file_sha256(paths["index"]),
        "identity_payload": identity_payload,
        "cache_identity_hash": stable_hash(identity_payload),
        "target_free_features": True,
        "labels_stored_in_index_only": True,
        "reused": False,
    }
    _write_json(paths["manifest"], manifest)
    return manifest


def materialize_eeg_only_cache(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = load_config(path)
    root = _repository_root(path)
    audit = participant_identity_audit(config, root)
    overlapping = {value.removeprefix("sub-") for value in audit["overlapping_source_ids"]}
    manifests = {
        dataset: _materialize_dataset(config, root, dataset, overlapping)
        for dataset in DATASETS
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "models_trained": 0,
        "participant_identity_audit": audit,
        "caches": manifests,
    }


def build_eeg_only_inventory(config_path: str | Path) -> dict[str, Any]:
    """Build/reuse caches and return their audited EEG-only cohort inventory."""
    result = materialize_eeg_only_cache(config_path)
    return {
        "experiment_id": result["experiment_id"],
        "models_trained": 0,
        "participant_identity_audit": result["participant_identity_audit"],
        "datasets": {
            name: {
                "windows": manifest["rows"],
                "participants": manifest["participant_count"],
                "records": manifest["record_count"],
                "class_counts": manifest["class_counts"],
                "rejection_counts": manifest["rejection_counts"],
                "cache_identity_hash": manifest["cache_identity_hash"],
            }
            for name, manifest in result["caches"].items()
        },
    }


def _load_cache(config: Mapping[str, Any], root: Path, dataset: str) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    manifest = _validate_cache(config, root, dataset)
    if manifest is None:
        raise FileNotFoundError(f"Build the {dataset} EEG-only cache before planning")
    paths = _cache_paths(config, root, dataset)
    return np.load(paths["raw"], mmap_mode="r"), pd.read_parquet(paths["index"]), manifest


def _inner_participant_split(frame: pd.DataFrame, *, seed: int) -> tuple[list[str], list[str]]:
    groups = frame["dataset_participant_id"].astype(str).to_numpy()
    labels = frame["target"].to_numpy(dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
    train_idx, validation_idx = next(splitter.split(np.zeros((len(frame), 1)), labels, groups))
    train = sorted(set(groups[train_idx]))
    validation = sorted(set(groups[validation_idx]))
    if set(train) & set(validation):
        raise RuntimeError("Inner participant split leaked")
    return train, validation


def build_cross_dataset_folds(
    config: Mapping[str, Any], indices: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    seed = int(config["evaluation"]["random_state"])
    for direction in DIRECTIONS:
        source_name, target_name = direction.split("_to_")
        source = indices[source_name]
        target = indices[target_name]
        groups = target["dataset_participant_id"].astype(str).to_numpy()
        labels = target["target"].to_numpy(dtype=int)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        covered: set[str] = set()
        for fold, (adapt_idx, test_idx) in enumerate(
            splitter.split(np.zeros((len(target), 1)), labels, groups), 1
        ):
            adaptation = target.iloc[adapt_idx]
            test = target.iloc[test_idx]
            test_person_keys = set(test["cross_dataset_person_key"].astype(str))
            source_train = source.loc[
                ~source["cross_dataset_person_key"].astype(str).isin(test_person_keys)
            ].copy()
            source_inner_train, source_validation = _inner_participant_split(
                source_train, seed=seed + fold
            )
            target_inner_train, target_validation = _inner_participant_split(
                adaptation, seed=seed + 100 + fold
            )
            test_ids = sorted(test["sample_id"].astype(str))
            if covered & set(test_ids):
                raise RuntimeError(f"Repeated target-test samples in {direction}")
            covered.update(test_ids)
            payload = {
                "direction": direction,
                "fold": fold,
                "source_dataset": source_name,
                "target_dataset": target_name,
                "source_train_participants": sorted(source_train["dataset_participant_id"].unique()),
                "source_train_sample_ids": sorted(source_train["sample_id"].astype(str)),
                "source_inner_train_participants": source_inner_train,
                "source_validation_participants": source_validation,
                "target_adaptation_participants": sorted(adaptation["dataset_participant_id"].unique()),
                "target_adaptation_sample_ids": sorted(adaptation["sample_id"].astype(str)),
                "target_inner_train_participants": target_inner_train,
                "target_validation_participants": target_validation,
                "target_test_participants": sorted(test["dataset_participant_id"].unique()),
                "target_test_cross_dataset_person_keys": sorted(test_person_keys),
                "target_test_sample_ids": test_ids,
                "target_test_class_counts": {
                    str(k): int(v) for k, v in test["target"].value_counts().sort_index().items()
                },
                "excluded_source_participants_due_identity_policy": sorted(
                    set(source["dataset_participant_id"]) - set(source_train["dataset_participant_id"])
                ),
            }
            payload["test_sample_ids_hash"] = stable_hash(test_ids)
            folds.append(payload)
        if covered != set(target["sample_id"].astype(str)):
            raise RuntimeError(f"Target folds do not cover {target_name} exactly once")
    return {
        "protocol": "five target-dataset StratifiedGroupKFold splits by participant",
        "random_state": seed,
        "folds": folds,
    }


def _fold_leakage_checks(fold: Mapping[str, Any], indices: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    source = indices[str(fold["source_dataset"])]
    target = indices[str(fold["target_dataset"])]
    source_rows = source[source["sample_id"].isin(fold["source_train_sample_ids"])]
    adapt_rows = target[target["sample_id"].isin(fold["target_adaptation_sample_ids"])]
    test_rows = target[target["sample_id"].isin(fold["target_test_sample_ids"])]
    checks = {
        "target_participant_overlap": bool(
            set(adapt_rows["dataset_participant_id"]) & set(test_rows["dataset_participant_id"])
        ),
        "target_sample_overlap": bool(set(adapt_rows["sample_id"]) & set(test_rows["sample_id"])),
        "source_test_sample_overlap": bool(set(source_rows["sample_id"]) & set(test_rows["sample_id"])),
        "cross_dataset_person_overlap": bool(
            set(source_rows["cross_dataset_person_key"]) & set(test_rows["cross_dataset_person_key"])
        ),
        "source_validation_test_person_overlap": bool(
            set(fold["source_validation_participants"]) & set(fold["target_test_participants"])
        ),
    }
    checks["clean"] = not any(value for key, value in checks.items() if key != "clean")
    return checks


def validate_protocol(
    config: Mapping[str, Any], folds: Mapping[str, Any], indices: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    fold_checks = []
    for fold in folds["folds"]:
        checks = _fold_leakage_checks(fold, indices)
        if not checks["clean"]:
            raise RuntimeError(f"Leakage in {fold['direction']} fold {fold['fold']}: {checks}")
        fold_checks.append({"direction": fold["direction"], "fold": fold["fold"], **checks})
    for direction in DIRECTIONS:
        direction_folds = [row for row in folds["folds"] if row["direction"] == direction]
        union = [sample for row in direction_folds for sample in row["target_test_sample_ids"]]
        target = indices[direction.split("_to_")[1]]
        if len(union) != len(set(union)) or set(union) != set(target["sample_id"].astype(str)):
            raise RuntimeError(f"Invalid test coverage for {direction}")
    return {
        "leakage_status": "clean",
        "fold_checks": fold_checks,
        "dann_task_loss_labels": "source_train_only",
        "target_adaptation_labels_exposed_to_training_step": False,
        "target_test_exposed_to_fit_or_adaptation": False,
        "paired_test_contract": True,
    }


def _protocol_specification(
    config: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]], folds: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "result_status": "diagnostic",
        "target": config["target"],
        "directions": list(DIRECTIONS),
        "modes": list(MODES),
        "raw_input": {
            "shape": list(INPUT_SHAPE), "dtype": "float32", "channels": list(EEG_CHANNELS),
            "sampling_rate_hz": 256, "window_seconds": 10.0,
        },
        "identity_policy": "conservative_shared_person_key_for_equal_source_id",
        "cache_identity_hashes": {
            dataset: manifests[dataset]["cache_identity_hash"] for dataset in DATASETS
        },
        "fold_manifest_hash": stable_hash(folds),
        "evaluation": config["evaluation"],
        "model": config["model"],
        "training": config["training"],
        "dann": config["dann"],
        "normalization": config["normalization"],
        "primary_metrics": ["participant_macro_macro_f1", "participant_macro_balanced_accuracy"],
        "secondary_metrics": [
            "participant_macro_accuracy", "pooled_window_macro_f1",
            "pooled_window_balanced_accuracy", "pooled_window_accuracy",
            "weighted_f1", "confusion_matrix", "per_class_precision", "per_class_recall",
        ],
        "participant_balanced_accuracy_definition": (
            "sklearn balanced_accuracy_score over classes present in each participant y_true; "
            "fixed labels [0,1,2] are used for precision/recall/F1 with zero_division=0"
        ),
        "effects": {
            "transfer_gap": "target_only - source_only",
            "adaptation_effect": "dann - source_only",
        },
        "planned_runs": 30,
    }


def build_protocol(
    config: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]], folds: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the deterministic protocol manifest before any training run."""
    specification = _protocol_specification(config, manifests, folds)
    protocol_hash = stable_hash(specification)
    matrix = build_run_matrix(config, folds, protocol_hash)
    return {
        **specification,
        "protocol_hash": protocol_hash,
        "run_matrix_hash": stable_hash(matrix.to_dict(orient="records")),
        "unsupported_runs": 0,
        "execution_ready": True,
        "training_status": "training_not_started",
    }


def build_run_matrix(
    config: Mapping[str, Any], folds: Mapping[str, Any], protocol_hash: str
) -> pd.DataFrame:
    rows = []
    for fold in folds["folds"]:
        for mode in MODES:
            specification = {
                "experiment_id": EXPERIMENT_ID,
                "direction": fold["direction"],
                "fold": fold["fold"],
                "mode": mode,
                "protocol_hash": protocol_hash,
                "test_sample_ids_hash": fold["test_sample_ids_hash"],
            }
            spec_hash = stable_hash(specification)
            rows.append({
                "run_id": f"{fold['direction']}__fold{fold['fold']:02d}__{mode}__{spec_hash[:10]}",
                "direction": fold["direction"],
                "fold": int(fold["fold"]),
                "mode": mode,
                "source_dataset": fold["source_dataset"],
                "target_dataset": fold["target_dataset"],
                "source_train_participants": len(fold["source_train_participants"]),
                "source_train_samples": len(fold["source_train_sample_ids"]),
                "target_adaptation_participants": len(fold["target_adaptation_participants"]),
                "target_adaptation_samples": len(fold["target_adaptation_sample_ids"]),
                "target_test_participants": len(fold["target_test_participants"]),
                "target_test_samples": len(fold["target_test_sample_ids"]),
                "test_sample_ids_hash": fold["test_sample_ids_hash"],
                "protocol_hash": protocol_hash,
                "specification_hash": spec_hash,
            })
    matrix = pd.DataFrame(rows)
    if len(matrix) != 30 or matrix["run_id"].duplicated().any():
        raise RuntimeError(f"Expected exactly 30 unique runs, got {len(matrix)}")
    return matrix


def compatibility_matrix() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model": "torch_shallow_convnet", "mode": mode, "supported": True,
            "input": "raw EEG [B,1,4,2560]", "task": "3-class classification",
        }
        for mode in MODES
    ])


def _fold_summary(folds: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        "direction": row["direction"],
        "fold": row["fold"],
        "source_train_participants": len(row["source_train_participants"]),
        "source_train_samples": len(row["source_train_sample_ids"]),
        "source_identity_exclusions": len(row["excluded_source_participants_due_identity_policy"]),
        "target_adaptation_participants": len(row["target_adaptation_participants"]),
        "target_adaptation_samples": len(row["target_adaptation_sample_ids"]),
        "target_test_participants": len(row["target_test_participants"]),
        "target_test_samples": len(row["target_test_sample_ids"]),
        "target_test_class_0": row["target_test_class_counts"].get("0", 0),
        "target_test_class_1": row["target_test_class_counts"].get("1", 0),
        "target_test_class_2": row["target_test_class_counts"].get("2", 0),
        "test_sample_ids_hash": row["test_sample_ids_hash"],
    } for row in folds["folds"]])


def plan_experiment(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = load_config(path)
    root = _repository_root(path)
    identity = participant_identity_audit(config, root)
    caches = {}
    indices = {}
    for dataset in DATASETS:
        _, indices[dataset], caches[dataset] = _load_cache(config, root, dataset)
    folds = build_cross_dataset_folds(config, indices)
    leakage = validate_protocol(config, folds, indices)
    protocol = build_protocol(config, caches, folds)
    protocol_hash = protocol["protocol_hash"]
    matrix = build_run_matrix(config, folds, protocol_hash)
    output = _resolve(root, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "participant_identity_audit.json", identity)
    dataset_rows = []
    class_rows = []
    for dataset in DATASETS:
        frame = indices[dataset]
        dataset_rows.append({
            "dataset": dataset, "windows": len(frame),
            "participants": frame["participant_id"].nunique(),
            "records": frame["record_id"].nunique(),
            "cache_identity_hash": caches[dataset]["cache_identity_hash"],
        })
        class_rows.extend({
            "dataset": dataset, "class_id": class_id, "windows": count,
            "proportion": count / len(frame),
        } for class_id, count in frame["target"].value_counts().sort_index().items())
    _write_csv(output / "dataset_summary.csv", pd.DataFrame(dataset_rows))
    _write_csv(output / "class_distribution.csv", pd.DataFrame(class_rows))
    _write_json(output / "fold_manifest.json", folds)
    _write_csv(output / "fold_summary.csv", _fold_summary(folds))
    _write_csv(output / "run_matrix.csv", matrix)
    _write_csv(output / "compatibility_matrix.csv", compatibility_matrix())
    _write_json(output / "leakage_audit.json", leakage)
    _write_json(output / "protocol_manifest.json", protocol)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": protocol_hash,
        "planned_runs": len(matrix),
        "unsupported_runs": 0,
        "models_trained": 0,
        "execution_ready": True,
        "training_status": "training_not_started",
        "leakage_status": leakage["leakage_status"],
        "cache_identity_hashes": protocol["cache_identity_hashes"],
    }
    _write_json(output / "plan_summary.json", summary)
    return summary


def _adapter(config: Mapping[str, Any], *, seed: int) -> TorchClassificationAdapter:
    params = dict(config["model"]["params"])
    params.update({
        "device": config["training"]["device"],
        "random_state": seed,
        "batch_size": config["training"]["batch_size"],
        "max_epochs": config["training"]["max_epochs"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "early_stopping_patience": config["training"]["early_stopping_patience"],
        "standardize": True,
        "feature_scaling": {"strategy": "standard"},
        "sampling_rate": 256.0,
        "channel_names": list(EEG_CHANNELS),
    })
    adapter = build_model("torch_shallow_convnet", "classification", INPUT_SHAPE, 3, params)
    if not isinstance(adapter, TorchClassificationAdapter):
        raise TypeError("Factory did not return TorchClassificationAdapter")
    return adapter


def smoke_forward_backward(config_path: str | Path) -> dict[str, Any]:
    """Run one finite CPU/CUDA DANN batch without an epoch or checkpoint."""
    path = Path(config_path)
    config = load_config(path)
    root = _repository_root(path)
    plan = plan_experiment(path)
    output = _resolve(root, config["output_dir"])
    folds = json.loads((output / "fold_manifest.json").read_text(encoding="utf-8"))
    fold = folds["folds"][0]
    source_raw, source_index, _ = _load_cache(config, root, fold["source_dataset"])
    target_raw, target_index, _ = _load_cache(config, root, fold["target_dataset"])
    source_lookup = pd.Series(np.arange(len(source_index)), index=source_index["sample_id"])
    target_lookup = pd.Series(np.arange(len(target_index)), index=target_index["sample_id"])
    source_ids = fold["source_train_sample_ids"][:2]
    target_ids = fold["target_adaptation_sample_ids"][:2]
    source_positions = source_lookup.loc[source_ids].to_numpy(int)
    target_positions = target_lookup.loc[target_ids].to_numpy(int)
    source_x = np.asarray(source_raw[source_positions], dtype=np.float32)
    target_x = np.asarray(target_raw[target_positions], dtype=np.float32)
    source_rows = source_index.iloc[source_positions]
    target_rows = target_index.iloc[target_positions]
    adapter = _adapter(config, seed=int(config["evaluation"]["random_state"]))
    module = DANNModule(
        adapter.model,
        n_domains=int(config["dann"]["n_domains"]),
        domain_hidden_dims=config["dann"]["domain_hidden_dims"],
        domain_dropout=float(config["dann"]["domain_dropout"]),
    ).to(adapter.device_)
    partition_source = DANNPartition(
        "source_train", source_x, np.zeros(2, dtype=np.int64), source_ids,
        source_rows["record_id"].astype(str).tolist(),
        source_rows["dataset_participant_id"].astype(str).tolist(),
        source_rows["target"].to_numpy(np.int64),
    )
    partition_target = DANNPartition(
        "target_adaptation", target_x, np.ones(2, dtype=np.int64), target_ids,
        target_rows["record_id"].astype(str).tolist(),
        target_rows["dataset_participant_id"].astype(str).tolist(), None,
    )
    # DANNFoldData additionally proves the adaptation loader has no target label field.
    fold_data = DANNFoldData(
        partition_source,
        partition_target,
        DANNPartition(
            "source_validation", source_x.copy(), np.zeros(2, dtype=np.int64),
            [f"smoke-val-{i}" for i in range(2)], [f"smoke-val-record-{i}" for i in range(2)],
            [f"smoke-val-subject-{i}" for i in range(2)], np.asarray([0, 1], dtype=np.int64),
        ),
        DANNPartition(
            "outer_test", target_x.copy(), np.ones(2, dtype=np.int64),
            [f"smoke-test-{i}" for i in range(2)], [f"smoke-test-record-{i}" for i in range(2)],
            [f"smoke-test-subject-{i}" for i in range(2)], np.asarray([0, 1], dtype=np.int64),
        ),
    )
    batch = next(iter(fold_data.training_loader(batch_size=2, shuffle=False))).to(adapter.device_)
    outputs = module(batch.source_inputs, batch.target_inputs, gradient_reversal_alpha=0.5)
    losses = DANNObjective(
        task_type="classification",
        lambda_domain=float(config["dann"]["domain_loss_lambda"]),
    )(outputs, batch.source_task_labels, batch.domain_ids)
    losses.total_loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in module.parameters()
    )
    result = {
        **plan,
        "device": str(adapter.device_),
        "source_shape": list(batch.source_inputs.shape),
        "target_shape": list(batch.target_inputs.shape),
        "source_task_output_shape": list(outputs.source_task_outputs.shape),
        "domain_output_shape": list(outputs.domain_outputs.shape),
        "losses": losses.detached_metrics(),
        "finite_gradients": bool(finite_gradients),
        "target_task_label_field_present": hasattr(batch, "target_task_labels"),
        "epochs_trained": 0,
        "training_runs_started": 0,
        "smoke_status": "finite_forward_backward_passed",
    }
    if not finite_gradients or result["target_task_label_field_present"]:
        raise RuntimeError(f"DANN smoke firewall failed: {result}")
    _write_json(output / "smoke_forward_backward.json", result)
    return result


def _positions(index: pd.DataFrame, sample_ids: Sequence[str]) -> np.ndarray:
    lookup = pd.Series(np.arange(len(index), dtype=np.int64), index=index["sample_id"].astype(str))
    requested = pd.Index([str(value) for value in sample_ids])
    missing = requested.difference(lookup.index)
    if len(missing):
        raise RuntimeError(f"Cache misses protocol samples: {missing[:5].tolist()}")
    return lookup.loc[requested].to_numpy(dtype=np.int64)


def _partition_positions(
    index: pd.DataFrame, sample_ids: Sequence[str], participants: Sequence[str]
) -> np.ndarray:
    selected = _positions(index, sample_ids)
    frame = index.iloc[selected]
    mask = frame["dataset_participant_id"].astype(str).isin([str(value) for value in participants])
    result = selected[mask.to_numpy()]
    if not len(result):
        raise RuntimeError("Protocol partition is empty")
    return result


def _channel_statistics(raw: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(4, dtype=np.float64)
    channel_square_sum = np.zeros(4, dtype=np.float64)
    count = 0
    for start in range(0, len(positions), 128):
        batch = np.asarray(raw[positions[start : start + 128]], dtype=np.float64)
        channel_sum += batch.sum(axis=(0, 1, 3))
        channel_square_sum += np.square(batch).sum(axis=(0, 1, 3))
        count += batch.shape[0] * batch.shape[1] * batch.shape[3]
    mean = channel_sum / count
    variance = np.maximum(channel_square_sum / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale < 1e-8, 1.0, scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise RuntimeError("Channel normalization statistics are not finite")
    return mean.astype(np.float32), scale.astype(np.float32)


def _normalized(raw: np.ndarray, positions: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(raw[positions], dtype=np.float32)
    values = (values - mean[None, None, :, None]) / scale[None, None, :, None]
    if not np.isfinite(values).all():
        raise RuntimeError("Channel normalization produced non-finite values")
    return np.ascontiguousarray(values, dtype=np.float32)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([0, 1, 2], dtype=int)
    present = np.asarray(sorted(set(np.asarray(y_true, dtype=int).tolist())), dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall_score(
            y_true, y_pred, labels=present, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_precision": precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).astype(float).tolist(),
        "per_class_recall": recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).astype(float).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def _prediction_artifacts(index: pd.DataFrame, positions: np.ndarray, probabilities: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = index.iloc[positions].reset_index(drop=True)
    prediction = probabilities.argmax(axis=1).astype(int)
    rows = frame[[
        "sample_id", "dataset", "dataset_participant_id", "cross_dataset_person_key",
        "record_id", "target",
    ]].rename(columns={"target": "y_true"}).copy()
    rows["y_pred"] = prediction
    for class_id in range(3):
        rows[f"proba_{class_id}"] = probabilities[:, class_id]
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Invalid prediction probabilities")
    participant_rows = []
    for participant, group in rows.groupby("dataset_participant_id", sort=True):
        values = _metrics(group["y_true"].to_numpy(int), group["y_pred"].to_numpy(int))
        participant_rows.append({
            "dataset_participant_id": participant,
            "windows": len(group),
            **{key: value for key, value in values.items() if not isinstance(value, list)},
        })
    participant = pd.DataFrame(participant_rows)
    pooled = _metrics(rows["y_true"].to_numpy(int), rows["y_pred"].to_numpy(int))
    aggregate = {
        "participant_macro": {
            metric: float(participant[metric].mean())
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        },
        "pooled_window": pooled,
        "participants": len(participant),
        "windows": len(rows),
    }
    return rows, aggregate


def _fit_supervised_mode(
    config: Mapping[str, Any], fold: Mapping[str, Any], mode: str,
    raw: np.ndarray, index: pd.DataFrame, test_source_raw: np.ndarray,
    test_positions: np.ndarray, test_index: pd.DataFrame,
    run_dir: Path, seed: int,
) -> dict[str, Any]:
    if mode == "target_only":
        sample_ids = fold["target_adaptation_sample_ids"]
        train_participants = fold["target_inner_train_participants"]
        validation_participants = fold["target_validation_participants"]
    else:
        sample_ids = fold["source_train_sample_ids"]
        train_participants = fold["source_inner_train_participants"]
        validation_participants = fold["source_validation_participants"]
    all_positions = _positions(index, sample_ids)
    selected = index.iloc[all_positions].reset_index(drop=True)
    train_mask = selected["dataset_participant_id"].astype(str).isin(train_participants).to_numpy()
    validation_mask = selected["dataset_participant_id"].astype(str).isin(validation_participants).to_numpy()
    if np.any(train_mask & validation_mask) or not train_mask.any() or not validation_mask.any():
        raise RuntimeError(f"Invalid precomputed validation split for {mode}")
    adapter = _adapter(config, seed=seed)
    adapter.set_validation_indices(
        np.flatnonzero(train_mask), np.flatnonzero(validation_mask),
        subject_ids=selected["dataset_participant_id"].astype(str).to_numpy(),
        record_ids=selected["record_id"].astype(str).to_numpy(),
        group_ids=selected["dataset_participant_id"].astype(str).to_numpy(),
        outer_test_group_ids=test_index["dataset_participant_id"].astype(str).to_numpy(),
        outer_test_record_ids=test_index["record_id"].astype(str).to_numpy(),
        group_column="dataset_participant_id",
    )
    started = time.perf_counter()
    adapter.fit(np.asarray(raw[all_positions], dtype=np.float32), selected["target"].to_numpy(int))
    training_time = time.perf_counter() - started
    # The outer-test tensor is first read after fit/checkpoint selection finishes.
    test_raw = np.asarray(test_source_raw[test_positions], dtype=np.float32)
    probabilities = np.asarray(adapter.predict_proba(np.asarray(test_raw, dtype=np.float32)), dtype=float)
    adapter.save(run_dir / "model.pt")
    pd.DataFrame(adapter.training_log_).to_csv(run_dir / "training_log.csv", index=False)
    _write_json(run_dir / "validation_split.json", adapter.validation_split_)
    _write_json(run_dir / "normalization_stats.json", {
        "fit_partition": "target_inner_train" if mode == "target_only" else "source_inner_train",
        "mean": None if adapter.feature_mean_ is None else adapter.feature_mean_.tolist(),
        "scale": None if adapter.feature_scale_ is None else adapter.feature_scale_.tolist(),
        "target_test_statistics_used": False,
    })
    return {
        "probabilities": probabilities,
        "training_time_seconds": training_time,
        "epochs_trained": len(adapter.training_log_),
        "best_epoch": adapter.best_epoch_,
        "best_validation_loss": adapter.best_validation_loss_,
        "device": str(adapter.device_),
    }


def _source_validation_metric(model: torch.nn.Module, values: np.ndarray, labels: np.ndarray, device: torch.device) -> tuple[float, float]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(values), 128):
            logits = model(torch.from_numpy(values[start : start + 128]).to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    y_pred = np.asarray(predictions, dtype=int)
    return (
        float(f1_score(labels, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        float(balanced_accuracy_score(labels, y_pred)),
    )


def _fit_dann_mode(
    config: Mapping[str, Any], fold: Mapping[str, Any],
    source_raw: np.ndarray, source_index: pd.DataFrame,
    target_raw: np.ndarray, target_index: pd.DataFrame,
    test_positions: np.ndarray, run_dir: Path, seed: int,
) -> dict[str, Any]:
    source_train_positions = _partition_positions(
        source_index, fold["source_train_sample_ids"], fold["source_inner_train_participants"]
    )
    source_validation_positions = _partition_positions(
        source_index, fold["source_train_sample_ids"], fold["source_validation_participants"]
    )
    target_positions = _positions(target_index, fold["target_adaptation_sample_ids"])
    mean, scale = _channel_statistics(source_raw, source_train_positions)
    source_train_x = _normalized(source_raw, source_train_positions, mean, scale)
    source_validation_x = _normalized(source_raw, source_validation_positions, mean, scale)
    target_x = _normalized(target_raw, target_positions, mean, scale)
    source_train_rows = source_index.iloc[source_train_positions]
    source_validation_rows = source_index.iloc[source_validation_positions]
    target_rows = target_index.iloc[target_positions]
    source_partition = DANNPartition(
        "source_inner_train", source_train_x, np.zeros(len(source_train_x), np.int64),
        source_train_rows["sample_id"].astype(str).tolist(),
        source_train_rows["record_id"].astype(str).tolist(),
        source_train_rows["dataset_participant_id"].astype(str).tolist(),
        source_train_rows["target"].to_numpy(np.int64),
    )
    target_partition = DANNPartition(
        "target_adaptation", target_x, np.ones(len(target_x), np.int64),
        target_rows["sample_id"].astype(str).tolist(), target_rows["record_id"].astype(str).tolist(),
        target_rows["dataset_participant_id"].astype(str).tolist(), None,
    )
    adapter = _adapter(config, seed=seed)
    adapter.standardize = False
    adapter.feature_scaling_config_ = {"strategy": "none"}
    device = adapter.device_
    module = DANNModule(
        adapter.model,
        n_domains=int(config["dann"]["n_domains"]),
        domain_hidden_dims=config["dann"]["domain_hidden_dims"],
        domain_dropout=float(config["dann"]["domain_dropout"]),
    ).to(device)
    # Validation/test partitions are explicit provenance guards; neither enters the loader.
    guard = DANNFoldData(
        source_partition,
        target_partition,
        DANNPartition(
            "source_validation", source_validation_x,
            np.zeros(len(source_validation_x), np.int64),
            source_validation_rows["sample_id"].astype(str).tolist(),
            source_validation_rows["record_id"].astype(str).tolist(),
            source_validation_rows["dataset_participant_id"].astype(str).tolist(),
            source_validation_rows["target"].to_numpy(np.int64),
        ),
        DANNPartition(
            "outer_test_guard", np.zeros((1, *INPUT_SHAPE), np.float32), np.ones(1, np.int64),
            [str(fold["target_test_sample_ids"][0])], ["locked-target-test-record"],
            [str(fold["target_test_participants"][0])], np.asarray([0], np.int64),
        ),
    )
    loader = guard.training_loader(
        batch_size=int(config["training"]["batch_size"]), shuffle=True, random_state=seed
    )
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    objective = DANNObjective(
        task_type="classification", lambda_domain=float(config["dann"]["domain_loss_lambda"])
    )
    maximum_epochs = int(config["training"]["max_epochs"])
    total_steps = max(len(loader) * maximum_epochs, 1)
    patience = int(config["training"]["early_stopping_patience"])
    best_key = (-math.inf, -math.inf)
    best_state = None
    best_epoch = None
    stale = 0
    global_step = 0
    log_rows = []
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        module.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            progress = global_step / max(total_steps - 1, 1)
            alpha = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
            optimizer.zero_grad(set_to_none=True)
            output = module(batch.source_inputs, batch.target_inputs, gradient_reversal_alpha=alpha)
            result = objective(output, batch.source_task_labels, batch.domain_ids)
            if not torch.isfinite(result.total_loss):
                raise RuntimeError("DANN training loss became non-finite")
            result.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            losses.append(result.detached_metrics())
            global_step += 1
        macro_f1, balanced = _source_validation_metric(
            module.task_model, source_validation_x,
            source_validation_rows["target"].to_numpy(int), device,
        )
        row = {
            "epoch": epoch,
            "train_task_loss": float(np.mean([value["task_loss"] for value in losses])),
            "train_domain_loss": float(np.mean([value["domain_loss"] for value in losses])),
            "train_domain_accuracy": float(np.mean([value["domain_accuracy"] for value in losses])),
            "source_validation_macro_f1": macro_f1,
            "source_validation_balanced_accuracy": balanced,
        }
        log_rows.append(row)
        key = (macro_f1, balanced)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    training_time = time.perf_counter() - started
    if best_state is None or best_epoch is None:
        raise RuntimeError("DANN did not produce a checkpoint")
    module.load_state_dict(best_state, strict=True)
    module.save(run_dir / "model.pt", metadata={"best_epoch": best_epoch})
    pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)
    _write_json(run_dir / "normalization_stats.json", {
        "fit_partition": "source_inner_train", "mean": mean.tolist(), "scale": scale.tolist(),
        "target_adaptation_statistics_used": False, "target_test_statistics_used": False,
    })
    # The outer-test tensor is first read after the source-validation checkpoint is fixed.
    test_raw = np.asarray(target_raw[test_positions], dtype=np.float32)
    normalized_test = (
        np.asarray(test_raw, dtype=np.float32) - mean[None, None, :, None]
    ) / scale[None, None, :, None]
    module.task_model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(normalized_test), 128):
            logits = module.task_model(torch.from_numpy(normalized_test[start : start + 128]).to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return {
        "probabilities": np.concatenate(probabilities),
        "training_time_seconds": training_time,
        "epochs_trained": len(log_rows),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_key[0],
        "best_validation_balanced_accuracy": best_key[1],
        "device": str(device),
        "target_adaptation_task_labels_accessible": False,
    }


def _resumable_summary(path: Path, specification_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    if payload.get("specification_hash") != str(specification_hash):
        return None
    return payload


def execute(config_path: str | Path, *, resume: bool = True, confirm_training: bool = False) -> dict[str, Any]:
    """Execute the confirmation-gated 30-run matrix with hash-safe resume."""
    if not confirm_training:
        raise PermissionError("Full training requires confirm_training=True")
    path = Path(config_path)
    config = load_config(path)
    root = _repository_root(path)
    plan = plan_experiment(path)
    output = _resolve(root, config["output_dir"])
    protocol_manifest = json.loads((output / "protocol_manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((output / "fold_manifest.json").read_text(encoding="utf-8"))["folds"]
    matrix = pd.read_csv(output / "run_matrix.csv")
    if protocol_manifest["protocol_hash"] != plan["protocol_hash"] or len(matrix) != 30:
        raise RuntimeError("Execution plan changed after validation")
    cache = {}
    for dataset in DATASETS:
        cache[dataset] = _load_cache(config, root, dataset)
    summaries = []
    for row in matrix.to_dict(orient="records"):
        run_dir = output / "runs" / str(row["run_id"])
        summary_path = run_dir / "run_summary.json"
        previous = _resumable_summary(summary_path, row["specification_hash"]) if resume else None
        if previous is not None:
            summaries.append(previous)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        fold = next(
            value for value in folds
            if value["direction"] == row["direction"] and int(value["fold"]) == int(row["fold"])
        )
        source_raw, source_index, _ = cache[fold["source_dataset"]]
        target_raw, target_index, _ = cache[fold["target_dataset"]]
        test_positions = _positions(target_index, fold["target_test_sample_ids"])
        test_index = target_index.iloc[test_positions].reset_index(drop=True)
        seed = int(config["evaluation"]["random_state"]) + int(row["fold"])
        if row["mode"] == "target_only":
            trained = _fit_supervised_mode(
                config, fold, "target_only", target_raw, target_index,
                target_raw, test_positions, test_index, run_dir, seed,
            )
        elif row["mode"] == "source_only":
            trained = _fit_supervised_mode(
                config, fold, "source_only", source_raw, source_index,
                target_raw, test_positions, test_index, run_dir, seed,
            )
        else:
            trained = _fit_dann_mode(
                config, fold, source_raw, source_index, target_raw, target_index,
                test_positions, run_dir, seed,
            )
        predictions, metrics = _prediction_artifacts(
            test_index, np.arange(len(test_index), dtype=np.int64), trained.pop("probabilities")
        )
        predictions.insert(0, "mode", row["mode"])
        predictions.insert(0, "fold", int(row["fold"]))
        predictions.insert(0, "direction", row["direction"])
        predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        summary = {
            "experiment_id": EXPERIMENT_ID,
            "status": "complete",
            "result_status": "diagnostic",
            "protocol_hash": plan["protocol_hash"],
            "specification_hash": row["specification_hash"],
            "run_id": row["run_id"],
            "direction": row["direction"],
            "fold": int(row["fold"]),
            "mode": row["mode"],
            "test_sample_ids_hash": row["test_sample_ids_hash"],
            "metrics": metrics,
            **trained,
        }
        _write_json(summary_path, summary)
        summaries.append(summary)
    final = {
        **plan,
        "training_status": "complete",
        "completed_runs": len(summaries),
        "result_status": "diagnostic",
    }
    _write_json(output / "execution_summary.json", final)
    return final


__all__ = [
    "DATASETS", "DIRECTIONS", "EXPERIMENT_ID", "INPUT_SHAPE", "MODES",
    "TARGET_ID", "build_cross_dataset_folds", "build_eeg_only_inventory", "build_protocol", "build_run_matrix",
    "eeg_only_window", "execute", "fixed_target", "load_config", "materialize_eeg_only_cache",
    "participant_identity_audit", "plan_experiment", "smoke_forward_backward",
    "stable_hash", "validate_protocol",
]
