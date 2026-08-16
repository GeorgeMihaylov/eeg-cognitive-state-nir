"""Leakage-safe orchestration for the selected-model seven-PM confirmation.

The module extends :class:`BenchmarkRunner`; it does not own a training loop.
Plan construction reads only manifests, indexes and target columns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from bench.bench_runner import BenchmarkRunner
from bench.experiments.preliminary_model_zoo_comparison import (
    ResourceSampler,
    measure_model_only_latency,
    measure_prediction_latency,
)
from bench.tasks.target_registry import get_target_spec
from bench.tasks.target_transforms import (
    FoldLocalQuantileTargetTransform,
    build_target_transform_manifest,
)
from bench.validation.metrics import MetricsCalculator
from model_zoo import build_model


SCHEMA_VERSION = "pm-confirmatory-selected-models-v1"
MODELS = (
    "random_forest",
    "xgboost",
    "torch_shallow_convnet",
    "torch_lstm",
)
TASK_TYPES = ("classification", "regression")
COHORT_TYPES = ("native", "common_sequence_eligible")
FEATURE_MODELS = frozenset({"random_forest", "xgboost"})
RAW_MODELS = frozenset({"torch_shallow_convnet"})
SEQUENCE_MODELS = frozenset({"torch_lstm"})


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_id(pm_name: str, task_type: str) -> str:
    suffix = "q3_fold_local" if task_type == "classification" else "regression"
    return f"pm_{pm_name}_{suffix}"


def target_column(pm_name: str) -> str:
    return f"target_{pm_name}"


def model_family(model: str) -> str:
    if model in FEATURE_MODELS:
        return "features"
    if model in RAW_MODELS:
        return "raw"
    if model in SEQUENCE_MODELS:
        return "sequence"
    raise ValueError(f"Unknown confirmatory model: {model!r}")


def is_supported(model: str, task_type: str) -> bool:
    if model not in MODELS or task_type not in TASK_TYPES:
        return False
    return not (model == "torch_lstm" and task_type == "regression")


def unit_id(fold: int, pm_name: str, task_type: str, model: str) -> str:
    return f"fold_{fold:02d}__{pm_name}__{task_type}__{model}"


def build_training_matrix(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in config["folds"]:
        for pm_name in config["pm_names"]:
            for task_type in TASK_TYPES:
                for model in MODELS:
                    supported = is_supported(model, task_type)
                    rows.append({
                        "unit_id": unit_id(int(fold), pm_name, task_type, model),
                        "fold": int(fold),
                        "pm": pm_name,
                        "target_id": target_id(pm_name, task_type),
                        "task_type": task_type,
                        "model": model,
                        "input_family": model_family(model),
                        "seed": int(config["seed"]),
                        "supported": supported,
                        "status": "planned" if supported else "unsupported",
                        "unsupported_reason": (
                            "torch_lstm factory exposes classification only"
                            if not supported else ""
                        ),
                    })
    return pd.DataFrame(rows)


def sequence_endpoint_ids(
    metadata: pd.DataFrame,
    eligible_sample_ids: Iterable[Any],
    *,
    length: int,
    stride: int,
    max_gap_seconds: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return deterministic last-window IDs using the preliminary contract."""
    required = {
        "source", "subject_id", "record_id", "record_group_id",
        "sample_id", "t_start",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Sequence metadata is missing columns: {missing}")
    if length <= 0 or stride <= 0:
        raise ValueError("Sequence length and stride must be positive")
    eligible = set(eligible_sample_ids)
    endpoints: list[Any] = []
    candidate_without_history: set[Any] = set()
    gap_rejected: set[Any] = set()
    groups = metadata.groupby(
        ["source", "subject_id", "record_group_id"],
        sort=True,
        dropna=False,
    )
    for _, group in groups:
        if group["record_id"].nunique(dropna=False) != 1:
            raise ValueError(
                "A sequence group spans multiple record_id values; refusing "
                "to cross a record boundary"
            )
        ordered = group.sort_values(["t_start", "sample_id"], kind="mergesort")
        ids = ordered["sample_id"].to_numpy()
        times = ordered["t_start"].to_numpy(dtype=float)
        for start in range(0, max(0, len(ids) - length + 1), stride):
            candidate_without_history.add(ids[start + length - 1])
        deltas = np.diff(times)
        breaks = np.flatnonzero((deltas <= 0) | (deltas > max_gap_seconds)) + 1
        for segment in np.split(np.arange(len(ids)), breaks):
            for start in range(0, max(0, len(segment) - length + 1), stride):
                endpoint = ids[segment[start + length - 1]]
                if endpoint in eligible:
                    endpoints.append(endpoint)
        for endpoint in eligible & candidate_without_history:
            if endpoint not in endpoints:
                gap_rejected.add(endpoint)
    result = np.asarray(sorted(set(endpoints)))
    stats = {
        "full_target_count": int(len(eligible)),
        "sequence_endpoint_count": int(len(result)),
        "dropped_no_history": int(len(eligible - candidate_without_history)),
        "dropped_gap": int(len(gap_rejected)),
        "dropped_other": int(
            len(eligible - set(result) - (eligible - candidate_without_history) - gap_rejected)
        ),
    }
    return result, stats


def validate_feature_cache_identity(
    cache_dir: str | Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(cache_dir)
    manifest_path = root / "feature_materialization_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Feature cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = dict(manifest.get("identity", {}))
    checked = (
        "cache_schema_version", "cache_identity_hash", "feature_hash",
        "sample_id_universe_hash", "raw_preprocessing_hash", "rows",
        "n_features", "dtype",
    )
    mismatches = {
        key: {"expected": expected.get(key), "actual": identity.get(key)}
        for key in checked
        if identity.get(key) != expected.get(key)
    }
    required_files = (
        root / "features.npy",
        root / "feature_index.parquet",
        root / "feature_names.json",
    )
    if mismatches or any(not path.is_file() for path in required_files):
        raise ValueError(
            "Feature cache identity gate failed: "
            f"mismatches={mismatches}, missing="
            f"{[str(p) for p in required_files if not p.is_file()]}"
        )
    return identity


def _target_table(path: Path, pm_names: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["subject_id", "record_id", *(target_column(pm) for pm in pm_names)],
    )
    if "sample_id" not in frame.columns:
        frame = frame.copy()
        frame.insert(0, "sample_id", frame.index.to_numpy())
    frame = frame.reset_index(drop=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Processed target table contains duplicate sample_id")
    return frame


def build_metadata_plan(
    config: Mapping[str, Any],
    feature_index: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    """Build folds, target transforms and sequence cohorts without tensors."""
    required = {
        "sample_id", "source", "subject_id", "record_id",
        "record_group_id", "t_start", "outer_fold",
    }
    missing = sorted(required - set(feature_index.columns))
    if missing:
        raise ValueError(f"Feature index lacks canonical metadata: {missing}")
    index = feature_index.copy()
    if index["sample_id"].duplicated().any():
        raise ValueError("Feature index contains duplicate sample_id")
    if sorted(index["outer_fold"].astype(int).unique().tolist()) != list(config["folds"]):
        raise ValueError("Fixed outer folds differ from configured folds")
    joined = index.merge(targets, on="sample_id", how="left", suffixes=("", "_target"))
    # Both canonical dataset loaders expose continuous single-output targets as
    # float32 before the fold-local transform.  The dry plan must hash those
    # exact runtime values, not the parquet's wider storage dtype.
    for pm_name in config["pm_names"]:
        column = target_column(pm_name)
        joined[column] = pd.to_numeric(joined[column], errors="coerce").astype(
            np.float32
        )
    for identity in ("subject_id", "record_id"):
        other = f"{identity}_target"
        if other in joined and not joined[identity].astype(str).eq(joined[other].astype(str)).all():
            raise ValueError(f"Target join changed {identity}")

    fold_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    transforms: dict[str, dict[str, Any]] = {}
    sequence = config["sequence"]
    for fold in config["folds"]:
        fold = int(fold)
        train_subjects = set(joined.loc[joined.outer_fold.ne(fold), "subject_id"].astype(str))
        test_subjects = set(joined.loc[joined.outer_fold.eq(fold), "subject_id"].astype(str))
        overlap = sorted(train_subjects & test_subjects)
        if overlap:
            raise RuntimeError(f"Outer subject leakage in fold {fold}: {overlap}")
        train_groups = set(joined.loc[joined.outer_fold.ne(fold), "record_group_id"].astype(str))
        test_groups = set(joined.loc[joined.outer_fold.eq(fold), "record_group_id"].astype(str))
        group_overlap = sorted(train_groups & test_groups)
        if group_overlap:
            raise RuntimeError(f"Outer record-group leakage in fold {fold}: {group_overlap[:10]}")
        fold_rows.append({
            "fold": fold,
            "train_subjects": len(train_subjects),
            "test_subjects": len(test_subjects),
            "subject_overlap": 0,
            "record_group_overlap": 0,
        })
        for pm in config["pm_names"]:
            column = target_column(pm)
            available = joined[column].notna()
            train = joined.loc[joined.outer_fold.ne(fold) & available]
            test = joined.loc[joined.outer_fold.eq(fold) & available]
            spec = get_target_spec(target_id(pm, "classification"))
            transform = FoldLocalQuantileTargetTransform(3, duplicates="drop").fit(
                train[column].to_numpy(dtype=np.float32)
            )
            manifest = build_target_transform_manifest(
                spec,
                transform,
                outer_fold=fold,
                outer_train_sample_ids=train["sample_id"].to_numpy(),
                outer_train_targets=train[column].to_numpy(dtype=np.float32),
            )
            key = f"fold_{fold:02d}__{pm}"
            transforms[key] = manifest
            test_context = joined.loc[joined.outer_fold.eq(fold), list(required)].copy()
            common_ids, stats = sequence_endpoint_ids(
                test_context,
                test["sample_id"].tolist(),
                length=int(sequence["length"]),
                stride=int(sequence["stride"]),
                max_gap_seconds=float(sequence["max_gap_seconds"]),
            )
            common_hash = stable_hash(common_ids.tolist())
            cohort_rows.append({
                "fold": fold,
                "pm": pm,
                "target_column": column,
                "native_train_count": int(len(train)),
                "native_test_count": int(len(test)),
                "common_test_count": int(len(common_ids)),
                "sequence_exclusions": int(len(test) - len(common_ids)),
                "dropped_no_history": stats["dropped_no_history"],
                "dropped_gap": stats["dropped_gap"],
                "dropped_other": stats["dropped_other"],
                "common_sample_id_hash": common_hash,
                "q3_transform_hash": manifest["transform_hash"],
            })
            values = test.set_index("sample_id")[column]
            for sample_id in common_ids:
                common_rows.append({
                    "fold": fold,
                    "pm": pm,
                    "sample_id": sample_id,
                    "continuous_target": float(values.loc[sample_id]),
                    "q3_target": int(transform.transform(np.asarray([values.loc[sample_id]]))[0]),
                    "common_sample_id_hash": common_hash,
                })
    folds = pd.DataFrame(fold_rows)
    cohorts = pd.DataFrame(cohort_rows)
    common = pd.DataFrame(common_rows)
    return folds, cohorts, transforms, common


def _torch_checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return {
        "input_shape": list(payload.get("input_shape", ())),
        "num_outputs": int(payload.get("num_outputs", payload.get("num_classes", -1))),
        "task_type": str(payload.get("task_type", "classification")),
        "model_metadata": dict(payload.get("model_metadata", {})),
        "training_config": dict(payload.get("training_config", {})),
    }


def _expected_input_shape(config: Mapping[str, Any], model: str) -> list[int]:
    if model in FEATURE_MODELS:
        return [int(config["feature_cache_identity"]["n_features"])]
    if model in RAW_MODELS:
        return list(config["raw_input"]["input_shape"])
    return list(config["sequence"]["input_shape"])


def _find_lstm_checkpoint(root: Path, target: str) -> Path | None:
    slug = target.removeprefix("pm_").removesuffix("_fold_local")
    candidates = sorted((root / "runs" / "torch_lstm" / slug).rglob("model.pt"))
    return candidates[-1] if candidates else None


def _shallow_checkpoint_map(preliminary_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    source = preliminary_root.parent / "preliminary_streaming_handoff_shallow_fold1"
    manifest_path = source / "manifest.json"
    summary_path = source / "summary.csv"
    if not manifest_path.is_file() or not summary_path.is_file():
        return {}, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, Path] = {}
    for row in pd.read_csv(summary_path).to_dict("records"):
        checkpoint = source / str(row.get("checkpoint", ""))
        if checkpoint.is_file():
            result[str(row["target_id"])] = checkpoint
    return result, manifest


def audit_preliminary_checkpoints(
    config: Mapping[str, Any],
    training: pd.DataFrame,
    preliminary_root: str | Path,
) -> pd.DataFrame:
    """Apply a strict metadata gate to preliminary fold-1 checkpoints."""
    root = Path(preliminary_root)
    shallow, shallow_manifest = _shallow_checkpoint_map(root)
    rows: list[dict[str, Any]] = []
    generic_training_keys = (
        "batch_size", "max_epochs", "learning_rate", "weight_decay",
        "validation_size", "early_stopping_patience", "random_state", "standardize",
    )
    for unit in training.loc[training.fold.eq(1)].to_dict("records"):
        model = str(unit["model"])
        task = str(unit["task_type"])
        target = str(unit["target_id"])
        reasons: list[str] = []
        checkpoint: Path | None = None
        metadata: dict[str, Any] = {}
        if not bool(unit["supported"]):
            reasons.append(str(unit["unsupported_reason"]))
        elif model in FEATURE_MODELS:
            slug = target.removeprefix("pm_").removesuffix("_fold_local")
            candidates = sorted((root / "runs" / model / slug).rglob("model.joblib"))
            checkpoint = candidates[-1] if candidates else None
            if checkpoint is None:
                reasons.append(
                    "preliminary run has metrics but no serialized sklearn checkpoint"
                )
        elif model == "torch_lstm":
            checkpoint = _find_lstm_checkpoint(root, target)
            if checkpoint is None:
                reasons.append("preliminary LSTM checkpoint is missing")
        elif model == "torch_shallow_convnet":
            checkpoint = shallow.get(target)
            if checkpoint is None:
                reasons.append("preliminary ShallowConvNet checkpoint is missing")
            if shallow_manifest:
                if shallow_manifest.get("evaluation", {}).get("folds") != [1]:
                    reasons.append("outer fold mismatch")
                if shallow_manifest.get("evaluation", {}).get("random_state") != int(config["seed"]):
                    reasons.append("seed mismatch")
                actual_raw_hash = shallow_manifest.get("composite_audit", {}).get("preprocessing_hash")
                expected_raw_hash = config["feature_cache_identity"]["raw_preprocessing_hash"]
                if actual_raw_hash != expected_raw_hash:
                    reasons.append("raw preprocessing hash mismatch")
        if checkpoint is not None and checkpoint.suffix == ".pt":
            metadata = _torch_checkpoint_metadata(checkpoint)
            expected_shape = _expected_input_shape(config, model)
            expected_outputs = 3 if task == "classification" else 1
            if metadata["input_shape"] != expected_shape:
                reasons.append("input shape mismatch")
            if metadata["num_outputs"] != expected_outputs:
                reasons.append("output shape mismatch")
            if metadata["task_type"] != task:
                reasons.append("task type mismatch")
            if metadata["model_metadata"].get("model_type") != model:
                reasons.append("model type mismatch")
            expected_params = config["models"][model][task]
            stored_training = metadata["training_config"]
            for key in generic_training_keys:
                if key in expected_params and stored_training.get(key) != expected_params[key]:
                    reasons.append(f"model config mismatch: {key}")
            stored_architecture = metadata["model_metadata"]
            architecture_keys = (
                ("n_filters", "temporal_kernel_samples", "pool_size", "pool_stride", "dropout")
                if model == "torch_shallow_convnet"
                else ("hidden_size", "num_layers", "bidirectional", "dropout", "classifier_hidden")
            )
            for key in architecture_keys:
                if stored_architecture.get(key) != expected_params.get(key):
                    reasons.append(f"model architecture mismatch: {key}")
            normalization = checkpoint.parent / "normalization_stats.json"
            if not normalization.is_file():
                reasons.append("normalization artifact is missing")
            transform = checkpoint.parent / "target_transform.json"
            expected_transform = str(unit.get("q3_transform_hash", ""))
            if task == "classification":
                if not transform.is_file():
                    reasons.append("Q3 transform artifact is missing")
                else:
                    stored = json.loads(transform.read_text(encoding="utf-8"))
                    if stored.get("transform_hash") != expected_transform:
                        reasons.append("Q3 transform hash mismatch")
            if model == "torch_lstm":
                sequence_stats = checkpoint.parent / "sequence_stats.json"
                if not sequence_stats.is_file():
                    reasons.append("sequence contract artifact is missing")
                feature_manifest = checkpoint.parent / "feature_manifest.json"
                if not feature_manifest.is_file():
                    reasons.append("feature schema artifact is missing")
                else:
                    feature = json.loads(feature_manifest.read_text(encoding="utf-8"))
                    if int(feature.get("feature_count", -1)) != int(
                        config["feature_cache_identity"]["n_features"]
                    ):
                        reasons.append("feature dimension mismatch")
        reusable = bool(unit["supported"]) and checkpoint is not None and not reasons
        rows.append({
            "unit_id": unit["unit_id"],
            "model": model,
            "target_id": target,
            "task_type": task,
            "fold": 1,
            "checkpoint": "" if checkpoint is None else str(checkpoint),
            "checkpoint_sha256": "" if checkpoint is None else file_sha256(checkpoint),
            "reusable": reusable,
            "reason": "compatible" if reusable else "; ".join(dict.fromkeys(reasons)),
        })
    return pd.DataFrame(rows)


def build_evaluation_matrix(
    training: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    lookup = cohorts.set_index(["fold", "pm"])
    rows: list[dict[str, Any]] = []
    for unit in training.loc[training.supported].to_dict("records"):
        cohort = lookup.loc[(unit["fold"], unit["pm"])]
        for cohort_type in COHORT_TYPES:
            is_sequence = unit["model"] == "torch_lstm"
            count = (
                int(cohort["common_test_count"])
                if cohort_type == "common_sequence_eligible" or is_sequence
                else int(cohort["native_test_count"])
            )
            rows.append({
                "evaluation_id": f"{unit['unit_id']}__{cohort_type}",
                "unit_id": unit["unit_id"],
                "fold": unit["fold"],
                "pm": unit["pm"],
                "target_id": unit["target_id"],
                "task_type": unit["task_type"],
                "model": unit["model"],
                "cohort_type": cohort_type,
                "n_samples": count,
                "sample_id_hash": (
                    cohort["common_sample_id_hash"]
                    if cohort_type == "common_sequence_eligible" or is_sequence
                    else "native_target_complete_case"
                ),
                "requires_new_checkpoint": False,
            })
    return pd.DataFrame(rows)


def write_plan(
    config: Mapping[str, Any],
    *,
    data_root: str | Path,
    feature_cache_dir: str | Path,
    preliminary_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic, metadata-only execution plan."""
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported confirmatory config schema")
    if tuple(config.get("models", {})) != MODELS:
        raise ValueError("The selected model set or order changed")
    if int(config.get("seed", -1)) != 42:
        raise ValueError("The confirmatory protocol is frozen at seed 42")
    root = Path(data_root)
    cache = Path(feature_cache_dir)
    output = Path(output_dir or config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    identity = validate_feature_cache_identity(cache, config["feature_cache_identity"])
    feature_index = pd.read_parquet(cache / "feature_index.parquet")
    targets = _target_table(root / config["data"]["processed_targets"], config["pm_names"])
    folds, cohorts, transforms, common = build_metadata_plan(
        config, feature_index, targets
    )
    training = build_training_matrix(config)
    q3_lookup = {
        (int(key[5:7]), key.split("__", 1)[1]): value["transform_hash"]
        for key, value in transforms.items()
    }
    training["q3_transform_hash"] = [
        q3_lookup[(row.fold, row.pm)] if row.task_type == "classification" else ""
        for row in training.itertuples()
    ]
    if training.loc[training.task_type.eq("classification")].groupby(
        ["fold", "pm"]
    )["q3_transform_hash"].nunique().ne(1).any():
        raise RuntimeError("Q3 transform hash is not invariant between models")
    reuse = audit_preliminary_checkpoints(config, training, preliminary_root)
    reuse_lookup = reuse.set_index("unit_id")
    training["reusable_checkpoint"] = [
        bool(reuse_lookup.at[item, "reusable"])
        if item in reuse_lookup.index else False
        for item in training["unit_id"]
    ]
    training["requires_training"] = training["supported"] & ~training["reusable_checkpoint"]
    evaluations = build_evaluation_matrix(training, cohorts)
    fixed_fold_hash = stable_hash(
        feature_index.loc[:, ["sample_id", "subject_id", "outer_fold"]]
        .sort_values("sample_id", kind="mergesort")
        .astype(str).to_dict("records")
    )
    scientific_config = {
        key: value for key, value in config.items() if key != "output_dir"
    }
    protocol_hash = stable_hash({
        "schema_version": SCHEMA_VERSION,
        "config": scientific_config,
        "feature_cache_identity": identity,
        "fixed_fold_hash": fixed_fold_hash,
        "q3_hashes": {key: value["transform_hash"] for key, value in transforms.items()},
    })
    run_matrix_hash = stable_hash({
        "protocol_hash": protocol_hash,
        "training": training.loc[:, [
            "unit_id", "supported", "q3_transform_hash", "reusable_checkpoint"
        ]].to_dict("records"),
        "evaluations": evaluations.loc[:, [
            "evaluation_id", "n_samples", "sample_id_hash"
        ]].to_dict("records"),
    })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "confirmatory_plan",
        "training_executed": False,
        "protocol_hash": protocol_hash,
        "run_matrix_hash": run_matrix_hash,
        "fixed_fold_hash": fixed_fold_hash,
        "feature_cache_identity": identity,
        "matrix_cells": int(len(training)),
        "supported_training_units": int(training.supported.sum()),
        "unsupported_training_units": int((~training.supported).sum()),
        "reusable_fold1_units": int(training.reusable_checkpoint.sum()),
        "new_trainings_required": int(training.requires_training.sum()),
        "native_evaluations": int(evaluations.cohort_type.eq("native").sum()),
        "common_cohort_evaluations": int(
            evaluations.cohort_type.eq("common_sequence_eligible").sum()
        ),
        "runtime_paths": {
            "data_root": str(root),
            "feature_cache_dir": str(cache),
            "preliminary_root": str(preliminary_root),
        },
    }
    training.to_csv(output / "training_units.csv", index=False)
    evaluations.to_csv(output / "evaluation_units.csv", index=False)
    folds.to_csv(output / "fixed_fold_audit.csv", index=False)
    cohorts.to_csv(output / "cohort_inventory.csv", index=False)
    common.to_parquet(output / "common_sequence_eligible_samples.parquet", index=False)
    reuse.to_csv(output / "checkpoint_reuse_audit.csv", index=False)
    (output / "q3_target_transforms.json").write_text(
        json.dumps(transforms, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status_path = output / "execution_status.csv"
    if not status_path.exists():
        status = training.loc[:, ["unit_id", "supported"]].copy()
        status["training_status"] = np.where(status.supported, "pending", "unsupported")
        status["native_status"] = np.where(status.supported, "pending", "unsupported")
        status["common_status"] = np.where(status.supported, "pending", "unsupported")
        status["error_type"] = ""
        status["error_message"] = ""
        status.to_csv(status_path, index=False)
    return manifest


def benchmark_run_config(
    config: Mapping[str, Any],
    unit: Mapping[str, Any],
    *,
    data_root: Path,
    feature_cache_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    model = str(unit["model"])
    task = str(unit["task_type"])
    target = str(unit["target_id"])
    family = model_family(model)
    if family == "raw":
        dataset_name = "emotiv_raw_eeg"
        dataset = {
            "data_path": str(data_root / config["data"]["raw_manifest"]),
            "cache_path_root": str(data_root),
            "target_data_path": str(data_root / config["data"]["processed_targets"]),
            "target_id": target,
            "dataset_mode": "raw_deduplicated_logical_records",
            "logical_recording_map_path": str(
                data_root / config["data"]["logical_recording_map"]
            ),
            "raw_preprocessing": dict(config["raw_preprocessing"]),
        }
    else:
        dataset_name = "cogstate_features"
        dataset = {
            "data_path": str(feature_cache_dir),
            "target_data_path": str(data_root / config["data"]["processed_targets"]),
            "target_id": target,
            "sampling_rate": 256,
        }
    result = {
        "output_dir": str(output_dir / "units" / str(unit["unit_id"]) / "benchmark"),
        "result_status": "confirmatory",
        "datasets": {dataset_name: dataset},
        "tasks": [target],
        "task_config": {"target_id": target, "random_state": int(config["seed"])},
        "models": {
            model: {
                "type": model,
                "task_type": task,
                "params": dict(config["models"][model][task]),
            }
        },
        "evaluation": {
            "protocol": "group_kfold_subject",
            "n_splits": 5,
            "group_column": "subject_id",
            "precomputed_fold_column": "outer_fold",
            "folds": [int(unit["fold"])],
            "random_state": int(config["seed"]),
        },
        "validation": dict(config["validation"]),
        "run_within_subject": False,
        "run_loso": False,
    }
    if family == "sequence":
        result["sequence"] = {
            key: config["sequence"][key]
            for key in (
                "length", "stride", "target_position",
                "expected_step_seconds", "max_gap_seconds",
            )
        }
    return result


def checkpoint_identity(
    config: Mapping[str, Any],
    unit: Mapping[str, Any],
    *,
    protocol_hash: str,
    checkpoint: Path,
) -> dict[str, Any]:
    model = str(unit["model"])
    task = str(unit["task_type"])
    q3_value = unit.get("q3_transform_hash")
    q3_hash = (
        None
        if q3_value is None or pd.isna(q3_value) or str(q3_value) == ""
        else str(q3_value)
    )
    payload = {
        "schema_version": "pm-confirmatory-checkpoint-v1",
        "protocol_hash": protocol_hash,
        "unit_id": unit["unit_id"],
        "model": model,
        "model_config_hash": stable_hash(config["models"][model][task]),
        "target_id": unit["target_id"],
        "task_type": task,
        "fold": int(unit["fold"]),
        "seed": int(unit["seed"]),
        "input_shape": _expected_input_shape(config, model),
        "num_outputs": 3 if task == "classification" else 1,
        "raw_preprocessing_hash": config["feature_cache_identity"]["raw_preprocessing_hash"],
        "feature_cache_identity_hash": (
            config["feature_cache_identity"]["cache_identity_hash"]
            if model_family(model) != "raw" else None
        ),
        "feature_hash": (
            config["feature_cache_identity"]["feature_hash"]
            if model_family(model) != "raw" else None
        ),
        "q3_transform_hash": q3_hash,
        "sequence_contract_hash": (
            stable_hash(config["sequence"]) if model_family(model) == "sequence" else None
        ),
        "normalization_scope": (
            "inner_train_only" if model.startswith("torch_") else "estimator_native"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    payload["identity_hash"] = stable_hash(payload)
    return payload


def validate_checkpoint_identity(
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    checked = (
        "protocol_hash", "unit_id", "model_config_hash", "target_id",
        "task_type", "fold", "seed", "input_shape", "num_outputs",
        "raw_preprocessing_hash", "feature_cache_identity_hash", "feature_hash",
        "q3_transform_hash", "sequence_contract_hash", "normalization_scope",
    )
    mismatches = {
        key: {"expected": expected.get(key), "actual": manifest.get(key)}
        for key in checked if manifest.get(key) != expected.get(key)
    }
    checkpoint = Path(str(manifest.get("checkpoint", "")))
    if not checkpoint.is_file():
        mismatches["checkpoint"] = {"expected": "existing file", "actual": str(checkpoint)}
    elif file_sha256(checkpoint) != manifest.get("checkpoint_sha256"):
        mismatches["checkpoint_sha256"] = {
            "expected": manifest.get("checkpoint_sha256"),
            "actual": file_sha256(checkpoint),
        }
    if mismatches:
        raise ValueError(f"Incompatible checkpoint: {mismatches}")


def _predict(model: Any, X: np.ndarray, task_type: str) -> tuple[np.ndarray, np.ndarray | None]:
    y_pred = np.asarray(model.predict(X))
    probabilities = None
    if task_type == "classification" and hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(model.predict_proba(X))
        except (AttributeError, NotImplementedError):
            probabilities = None
    return y_pred, probabilities


def evaluate_view(
    model: Any,
    split: Any,
    *,
    unit: Mapping[str, Any],
    cohort_type: str,
    common_sample_ids: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    sample_ids = np.asarray(split.sample_id_test)
    if cohort_type == "common_sequence_eligible":
        expected = np.asarray(sorted(common_sample_ids.tolist()))
        if model_family(str(unit["model"])) == "sequence":
            if not np.array_equal(np.sort(sample_ids), expected):
                raise RuntimeError("LSTM endpoint IDs differ from the common cohort")
            positions = np.argsort(sample_ids)
        else:
            lookup = {value: index for index, value in enumerate(sample_ids.tolist())}
            missing = [value for value in expected if value not in lookup]
            if missing:
                raise RuntimeError(f"Single-window split lacks common sample IDs: {missing[:10]}")
            positions = np.asarray([lookup[value] for value in expected], dtype=np.int64)
    else:
        positions = np.arange(len(sample_ids), dtype=np.int64)
    X = np.asarray(split.X_test)[positions]
    y_true = np.asarray(split.y_test)[positions]
    selected_ids = sample_ids[positions]
    y_pred, probabilities = _predict(model, X, str(unit["task_type"]))
    metrics = MetricsCalculator.calculate_all_metrics(
        y_true,
        y_pred,
        probabilities,
        task_type=str(unit["task_type"]),
        labels=(np.arange(3) if unit["task_type"] == "classification" else None),
    )
    view_dir = output_dir / "units" / str(unit["unit_id"]) / "evaluations" / cohort_type
    view_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "fold": int(unit["fold"]),
        "pm": unit["pm"],
        "target_id": unit["target_id"],
        "task_type": unit["task_type"],
        "model": unit["model"],
        "cohort_type": cohort_type,
        "sample_id": selected_ids,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    if probabilities is not None:
        for class_index in range(probabilities.shape[1]):
            predictions[f"proba_{class_index}"] = probabilities[:, class_index]
    predictions_path = view_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    payload = {
        "unit_id": unit["unit_id"],
        "cohort_type": cohort_type,
        "n_samples": int(len(predictions)),
        "sample_id_hash": stable_hash(sorted(selected_ids.tolist())),
        "metrics": metrics,
        "predictions": str(predictions_path),
    }
    (view_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


class ConfirmatoryExecutor:
    """Failure-isolating execution over the standard benchmark runner."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        data_root: Path,
        feature_cache_dir: Path,
        preliminary_root: Path,
        output_dir: Path,
        resume: bool,
    ) -> None:
        self.config = dict(config)
        self.data_root = Path(data_root)
        self.feature_cache_dir = Path(feature_cache_dir)
        self.preliminary_root = Path(preliminary_root)
        self.output = Path(output_dir)
        self.resume = bool(resume)
        manifest_path = self.output / "plan_manifest.json"
        if not manifest_path.is_file():
            write_plan(
                config,
                data_root=self.data_root,
                feature_cache_dir=self.feature_cache_dir,
                preliminary_root=self.preliminary_root,
                output_dir=self.output,
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.training = pd.read_csv(self.output / "training_units.csv")
        self.reuse = pd.read_csv(self.output / "checkpoint_reuse_audit.csv").set_index("unit_id")
        self.common = pd.read_parquet(self.output / "common_sequence_eligible_samples.parquet")
        self.status_path = self.output / "execution_status.csv"
        self.status = pd.read_csv(self.status_path).fillna("")

    def _save_status(self) -> None:
        self.status.to_csv(self.status_path, index=False)

    def _load_model(self, checkpoint: Path, unit: Mapping[str, Any], split: Any) -> Any:
        if checkpoint.suffix == ".joblib":
            import joblib

            return joblib.load(checkpoint)
        model = build_model(
            str(unit["model"]),
            str(unit["task_type"]),
            tuple(np.asarray(split.X_train).shape[1:]),
            3 if unit["task_type"] == "classification" else 1,
            self.config["models"][unit["model"]][unit["task_type"]],
        )
        model.load(checkpoint)
        return model

    def run_one(self, unit: Mapping[str, Any]) -> None:
        unit_key = str(unit["unit_id"])
        mask = self.status.unit_id.eq(unit_key)
        if not bool(unit["supported"]):
            return
        unit_dir = self.output / "units" / unit_key
        manifest_path = unit_dir / "checkpoint_manifest.json"
        run_config = benchmark_run_config(
            self.config,
            unit,
            data_root=self.data_root,
            feature_cache_dir=self.feature_cache_dir,
            output_dir=self.output,
        )
        runner = BenchmarkRunner(run_config)
        dataset_name = next(iter(run_config["datasets"]))
        checkpoint: Path
        model: Any
        split: Any
        result: Mapping[str, Any] | None = None
        resources: dict[str, Any] = {}
        if self.resume and manifest_path.is_file():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = checkpoint_identity(
                self.config,
                unit,
                protocol_hash=self.manifest["protocol_hash"],
                checkpoint=Path(saved["checkpoint"]),
            )
            validate_checkpoint_identity(saved, expected)
            split, _, _ = runner.prepare_group_fold(
                dataset_name, str(unit["target_id"]), int(unit["fold"]), str(unit["model"])
            )
            checkpoint = Path(saved["checkpoint"])
            model = self._load_model(checkpoint, unit, split)
        elif bool(unit.get("reusable_checkpoint", False)):
            row = self.reuse.loc[unit_key]
            checkpoint = Path(str(row["checkpoint"]))
            split, _, _ = runner.prepare_group_fold(
                dataset_name, str(unit["target_id"]), int(unit["fold"]), str(unit["model"])
            )
            model = self._load_model(checkpoint, unit, split)
        else:
            fold_name = f"fold_{int(unit['fold']):02d}"
            completed = (
                BenchmarkRunner.find_completed_run(run_config)
                if self.resume else None
            )
            if completed is not None:
                results = json.loads(completed.result_file.read_text(encoding="utf-8"))
                result = results[dataset_name]["models"][unit["target_id"]][unit["model"]][
                    "group_kfold_subject"
                ]["folds"][fold_name]
                split, _, _ = runner.prepare_group_fold(
                    dataset_name,
                    str(unit["target_id"]),
                    int(unit["fold"]),
                    str(unit["model"]),
                )
                checkpoint = Path(str(result["artifacts"].get("model", "")))
                if not checkpoint.is_file():
                    raise RuntimeError(
                        "Validated completed run has no serialized checkpoint"
                    )
                model = self._load_model(checkpoint, unit, split)
            else:
                with ResourceSampler() as sampler:
                    runner.run()
                resources = sampler.result()
                model = runner.last_fitted_model
                split = runner.last_evaluated_split
                if model is None or split is None:
                    raise RuntimeError(
                        "BenchmarkRunner did not expose the completed training unit"
                    )
                result = runner.results[dataset_name]["models"][unit["target_id"]][unit["model"]][
                    "group_kfold_subject"
                ]["folds"][fold_name]
            checkpoint = Path(str(result["artifacts"].get("model", "")))
            if not checkpoint.is_file():
                raise RuntimeError("Completed training unit has no serialized checkpoint")
        if split.metadata.get("subject_overlap") or split.metadata.get("record_group_overlap"):
            raise RuntimeError("Prepared split failed the outer leakage gate")
        expected_q3 = str(unit.get("q3_transform_hash", ""))
        if unit["task_type"] == "classification" and split.metadata.get(
            "target_transform_hash"
        ) != expected_q3:
            raise RuntimeError("Runtime Q3 transform differs from the shared fold/PM hash")
        validation = getattr(model, "validation_split_", None)
        if validation is not None and int(validation.get("inner_group_overlap", -1)) != 0:
            raise RuntimeError("Inner validation is not record-group-disjoint")
        unit_dir.mkdir(parents=True, exist_ok=True)
        identity = checkpoint_identity(
            self.config,
            unit,
            protocol_hash=self.manifest["protocol_hash"],
            checkpoint=checkpoint,
        )
        manifest_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        sample = np.asarray(split.X_test[0], dtype=np.float32)
        model_latency = measure_model_only_latency(model, sample)
        end_to_end_latency = measure_prediction_latency(model, sample)
        runtime = {
            "training_time_seconds": (
                None if result is None else result.get("training_time")
            ),
            "model_only_latency": model_latency,
            "preprocessing_feature_latency": {
                "measurement_scope": "cached_input",
                "value_ms": 0.0,
                "note": (
                    "Online raw-to-feature extraction is reported separately "
                    "by the preliminary feature-pipeline benchmark"
                    if model_family(str(unit["model"])) != "raw"
                    else "Canonical raw input requires no configured filtering"
                ),
            },
            "end_to_end_latency": end_to_end_latency,
            "peak_ram_mb": resources.get("peak_ram_mb"),
            "peak_vram_mb": resources.get(
                "peak_vram_mb", getattr(model, "peak_gpu_memory_bytes_", 0) / 2**20
            ),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_reused": bool(unit.get("reusable_checkpoint", False)),
        }
        (unit_dir / "runtime.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        common_ids = self.common.loc[
            self.common.fold.eq(int(unit["fold"])) & self.common.pm.eq(unit["pm"]),
            "sample_id",
        ].to_numpy()
        native_path = unit_dir / "evaluations" / "native" / "metrics.json"
        if not (self.resume and native_path.is_file()):
            evaluate_view(
                model, split, unit=unit, cohort_type="native",
                common_sample_ids=common_ids, output_dir=self.output,
            )
        common_path = unit_dir / "evaluations" / "common_sequence_eligible" / "metrics.json"
        if not (self.resume and common_path.is_file()):
            evaluate_view(
                model, split, unit=unit, cohort_type="common_sequence_eligible",
                common_sample_ids=common_ids, output_dir=self.output,
            )
        self.status.loc[mask, ["training_status", "native_status", "common_status"]] = [
            "reused" if bool(unit.get("reusable_checkpoint", False)) else "completed",
            "completed",
            "completed",
        ]
        self.status.loc[mask, ["error_type", "error_message"]] = ["", ""]
        self._save_status()

    def run(
        self,
        *,
        folds: Sequence[int] | None = None,
        models: Sequence[str] | None = None,
        targets: Sequence[str] | None = None,
    ) -> dict[str, int]:
        selected = self.training.copy()
        if folds:
            selected = selected.loc[selected.fold.isin([int(value) for value in folds])]
        if models:
            selected = selected.loc[selected.model.isin(models)]
        if targets:
            selected = selected.loc[selected.target_id.isin(targets)]
        for unit in selected.to_dict("records"):
            try:
                self.run_one(unit)
            except Exception as exc:  # one cell must not abort the matrix
                mask = self.status.unit_id.eq(unit["unit_id"])
                self.status.loc[mask, "training_status"] = "failed"
                self.status.loc[mask, "error_type"] = type(exc).__name__
                self.status.loc[mask, "error_message"] = str(exc)
                self._save_status()
        write_result_tables(self.output)
        return {
            str(key): int(value)
            for key, value in self.status.training_status.value_counts().items()
        }


def write_result_tables(output_dir: str | Path) -> None:
    output = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for path in output.glob("units/*/evaluations/*/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        unit = payload["unit_id"].split("__")
        metrics = payload["metrics"]
        rows.append({
            "unit_id": payload["unit_id"],
            "fold": int(unit[0].removeprefix("fold_")),
            "pm": unit[1],
            "task_type": unit[2],
            "model": unit[3],
            "cohort_type": payload["cohort_type"],
            "n_samples": payload["n_samples"],
            **{name: metrics.get(name) for name in (
                "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "mae", "rmse", "r2", "pearson", "spearman",
            )},
        })
    if not rows:
        return
    frame = pd.DataFrame(rows)
    classification = frame.loc[frame.task_type.eq("classification")].copy()
    regression = frame.loc[frame.task_type.eq("regression")].copy()
    classification.to_csv(output / "classification_by_fold.csv", index=False)
    regression.to_csv(output / "regression_by_fold.csv", index=False)

    def summarize(values: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
        records = []
        for keys, group in values.groupby(["model", "pm", "cohort_type"], sort=True):
            row = {"model": keys[0], "pm": keys[1], "cohort_type": keys[2], "n_folds": len(group)}
            for metric in metrics:
                series = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = float(series.mean())
                row[f"{metric}_std"] = float(series.std(ddof=1)) if len(series) > 1 else np.nan
                row[f"{metric}_median"] = float(series.median())
            records.append(row)
        return pd.DataFrame(records)

    class_summary = summarize(
        classification, ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    )
    reg_summary = summarize(regression, ("mae", "rmse", "r2", "pearson", "spearman"))
    class_summary.to_csv(output / "classification_summary.csv", index=False)
    reg_summary.to_csv(output / "regression_summary.csv", index=False)
    macro_rows = []
    for task, summary in (("classification", class_summary), ("regression", reg_summary)):
        for keys, group in summary.groupby(["model", "cohort_type"], sort=True):
            row = {"task_type": task, "model": keys[0], "cohort_type": keys[1], "pm_count": group.pm.nunique()}
            for column in summary.columns:
                if column.endswith(("_mean", "_std", "_median")):
                    row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
            macro_rows.append(row)
    pd.DataFrame(macro_rows).to_csv(output / "pm_macro_summary.csv", index=False)

    common = frame.loc[frame.cohort_type.eq("common_sequence_eligible")].copy()
    comparisons = []
    for task_type, reference, comparators, metrics in (
        (
            "classification",
            "torch_lstm",
            ("random_forest", "xgboost", "torch_shallow_convnet"),
            ("macro_f1", "balanced_accuracy"),
        ),
        (
            "regression",
            "torch_shallow_convnet",
            ("random_forest", "xgboost"),
            ("mae", "r2", "pearson"),
        ),
    ):
        subset = common.loc[common.task_type.eq(task_type)]
        reference_rows = subset.loc[subset.model.eq(reference)].set_index(["pm", "fold"])
        for comparator in comparators:
            other = subset.loc[subset.model.eq(comparator)].set_index(["pm", "fold"])
            for key in sorted(set(reference_rows.index) & set(other.index)):
                row = {"task_type": task_type, "pm": key[0], "fold": key[1], "reference": reference, "comparator": comparator}
                for metric in metrics:
                    row[f"delta_{metric}"] = float(
                        reference_rows.at[key, metric] - other.at[key, metric]
                    )
                comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(output / "common_cohort_comparison.csv", index=False)
