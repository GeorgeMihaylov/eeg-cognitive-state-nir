"""Final target-independent artifact-removal ablation for all seven PM."""

from __future__ import annotations

import json
import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from bench.datasets.datasets_registry import get_dataset
from bench.experiments.artifact_removal_ablation import (
    _balanced_subset,
    _q3_labels,
    _q3_thresholds,
)
from bench.preprocessing.artifact_removal_cache_v2 import (
    ARTIFACT_VARIANTS_V2,
    build_preprocessing_cache,
    load_cached_view,
    preprocessing_cache_hash,
    preprocessing_implementation_hashes,
    source_contract_hash,
)
from bench.preprocessing.fold_artifact_transform import stable_hash
from bench.tasks.target_registry import PM_METRICS
from bench.validation.metrics import MetricsCalculator
from cogstate.model_zoo import build_model


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_TYPES = ("classification", "regression")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_path(value: str | Path) -> str:
    path = _repo_path(value)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def load_config(path: str | Path) -> dict[str, Any]:
    document = yaml.safe_load(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Artifact-removal v2 config must be a mapping")
    required = {
        "experiment_id",
        "dataset",
        "targets",
        "variants",
        "task_types",
        "preprocessing",
        "model",
        "evaluation",
        "smoke",
        "output_dir",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Artifact-removal v2 config is missing sections: {missing}")
    if document["experiment_id"] != "artifact_removal_ablation_v2":
        raise ValueError("experiment_id must be artifact_removal_ablation_v2")
    if tuple(document["targets"]) != PM_METRICS:
        raise ValueError(f"targets must contain all seven PM in order: {PM_METRICS}")
    if tuple(document["variants"]) != ARTIFACT_VARIANTS_V2:
        raise ValueError(f"variants must equal {ARTIFACT_VARIANTS_V2}")
    if tuple(document["task_types"]) != TASK_TYPES:
        raise ValueError(f"task_types must equal {TASK_TYPES}")
    if tuple(map(int, document["evaluation"]["folds"])) != (1, 2, 3, 4, 5):
        raise ValueError("Artifact-removal v2 must use fixed folds [1, 2, 3, 4, 5]")
    if document["evaluation"]["inner_validation_group_column"] != "record_group_id":
        raise ValueError("Inner validation must be grouped by record_group_id")
    if document["model"]["name"] != "torch_shallow_convnet":
        raise ValueError("Artifact-removal v2 is locked to torch_shallow_convnet")
    preprocessing = document["preprocessing"]
    if float(preprocessing["z_threshold"]) != 3.0:
        raise ValueError("z_threshold is locked to 3.0")
    if preprocessing["interpolation_method"] != "mean":
        raise ValueError("interpolation_method is locked to mean")
    if bool(preprocessing["average_reference"]):
        raise ValueError("CAR/average reference is forbidden in artifact-removal v2")
    if not bool(preprocessing["full_faster"]["run_ica"]):
        raise ValueError("Full FASTER must include its internal ICA stage")
    smoke = document["smoke"]
    if tuple(smoke["targets"]) != PM_METRICS:
        raise ValueError("Smoke must include all seven PM")
    if tuple(smoke["variants"]) != ARTIFACT_VARIANTS_V2:
        raise ValueError("Smoke must include all four variants")
    if tuple(smoke["task_types"]) != TASK_TYPES:
        raise ValueError("Smoke must include classification and regression")
    if tuple(map(int, smoke["folds"])) != (1,):
        raise ValueError("Smoke must use exactly fold 1")
    return document


@dataclass(frozen=True)
class ArtifactRunSpecV2:
    metric: str
    variant: str
    fold: int
    task_type: str
    seed: int = 42

    @property
    def run_id(self) -> str:
        return (
            f"{self.metric}__{self.variant}__{self.task_type}__"
            f"fold{self.fold:02d}__seed{self.seed}"
        )


@dataclass
class SignalUniverse:
    data: Any
    manifest: pd.DataFrame
    targets: pd.DataFrame
    target_sample_ids: dict[str, set[str]]
    source_contract_hash: str


def build_run_matrix(config: Mapping[str, Any]) -> list[ArtifactRunSpecV2]:
    specs = [
        ArtifactRunSpecV2(
            metric=str(metric),
            variant=str(variant),
            fold=int(fold),
            task_type=str(task_type),
            seed=42,
        )
        for metric in config["targets"]
        for variant in config["variants"]
        for fold in config["evaluation"]["folds"]
        for task_type in config["task_types"]
    ]
    if len(specs) != 280:
        raise RuntimeError(f"Artifact-removal v2 matrix must contain 280 runs, got {len(specs)}")
    return specs


def run_specification_hash(
    spec: ArtifactRunSpecV2,
    *,
    protocol_hash: str,
    cache_hash: str,
) -> str:
    return stable_hash(
        {
            "run_spec": asdict(spec),
            "protocol_hash": protocol_hash,
            "preprocessing_cache_hash": cache_hash,
        }
    )


def _dataset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    return {
        "data_path": str(_repo_path(dataset["raw_window_index"])),
        "target_id": "pm_engagement_regression",
        "target_data_path": str(_repo_path(dataset["target_table"])),
        "dataset_mode": "raw_deduplicated_logical_records",
        "logical_recording_map_path": str(_repo_path(dataset["logical_recording_map"])),
        "raw_preprocessing": dict(dataset["raw_preprocessing"]),
    }


def _load_target_table(config: Mapping[str, Any]) -> pd.DataFrame:
    path = _repo_path(config["dataset"]["target_table"])
    frame = pd.read_parquet(path)
    if "sample_id" not in frame.columns:
        frame = frame.copy()
        frame.insert(0, "sample_id", frame.index.to_numpy(dtype=np.int64))
    required = {"sample_id", "subject_id", "record_id"} | {
        f"target_{metric}" for metric in PM_METRICS
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical target table is missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Canonical target table contains duplicate sample_id")
    return frame[
        ["sample_id", "subject_id", "record_id"]
        + [f"target_{metric}" for metric in PM_METRICS]
    ].copy()


def load_signal_universe(config: Mapping[str, Any]) -> SignalUniverse:
    """Load the 30,958-window union before applying any target mask."""
    dataset = get_dataset("emotiv_raw_eeg", _dataset_config(config))
    data = dataset.load()
    if tuple(data.data.shape[1:]) != (1, 14, 2560):
        raise ValueError(f"Canonical raw input shape changed: {data.data.shape}")
    manifest = data.data.manifest.reset_index(drop=True).copy()
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Signal universe contains duplicate sample_id")
    target_table = _load_target_table(config)
    targets = manifest[["sample_id", "subject_id", "record_id"]].merge(
        target_table,
        on="sample_id",
        how="left",
        suffixes=("", "_target"),
        validate="one_to_one",
        sort=False,
    )
    available = targets["subject_id_target"].notna()
    subject_mismatch = available & (
        targets["subject_id"].astype(str) != targets["subject_id_target"].astype(str)
    )
    record_mismatch = available & (
        targets["record_id"].astype(str) != targets["record_id_target"].astype(str)
    )
    if subject_mismatch.any() or record_mismatch.any():
        raise ValueError("Target table identifiers disagree with raw signal universe")
    targets = targets.drop(columns=["subject_id_target", "record_id_target"])
    target_sets: dict[str, set[str]] = {}
    for metric in PM_METRICS:
        column = f"target_{metric}"
        numeric = pd.to_numeric(targets[column], errors="coerce")
        targets[column] = numeric
        target_sets[metric] = set(
            targets.loc[np.isfinite(numeric), "sample_id"].astype(str)
        )
    universe_ids = set(manifest["sample_id"].astype(str))
    union_ids = set().union(*target_sets.values())
    if union_ids != universe_ids:
        raise RuntimeError(
            "Target-independent signal universe does not equal the seven-PM union: "
            f"universe={len(universe_ids)}, union={len(union_ids)}"
        )
    expected = config["dataset"].get("expected_counts", {})
    if expected:
        if len(universe_ids) != int(expected["signal_windows"]):
            raise RuntimeError("Canonical signal-universe window count changed")
        for metric in PM_METRICS:
            if len(target_sets[metric]) != int(expected[metric]):
                raise RuntimeError(f"Canonical complete-case count changed for {metric}")
    group_contract = manifest.groupby("record_group_id", sort=True).agg(
        subjects=("subject_id", "nunique"),
        records=("record_id", "nunique"),
        sources=("source", "nunique"),
    )
    if not (
        (group_contract["subjects"] == 1)
        & (group_contract["records"] == 1)
        & (group_contract["sources"] == 1)
    ).all():
        raise RuntimeError("record_group_id contract is not record-local")
    source_hash = source_contract_hash(
        manifest,
        raw_preprocessing=config["dataset"]["raw_preprocessing"],
        input_shape=data.data.shape[1:],
    )
    return SignalUniverse(
        data=data,
        manifest=manifest,
        targets=targets,
        target_sample_ids=target_sets,
        source_contract_hash=source_hash,
    )


def _fold_subject_audit(
    universe: SignalUniverse, config: Mapping[str, Any]
) -> dict[str, Any]:
    reference_path = _repo_path(config["dataset"]["reference_fold_manifest"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))["folds"]
    observed_by_subject = (
        universe.manifest.assign(subject_id=lambda frame: frame["subject_id"].astype(str))
        .groupby("subject_id")["outer_fold"]
        .agg(lambda values: sorted(set(map(int, values))))
    )
    if any(len(values) != 1 for values in observed_by_subject):
        raise RuntimeError("A subject appears in more than one outer fold")
    reference_by_subject = {
        str(subject): int(fold)
        for fold, payload in reference.items()
        for subject in payload["test_subject_ids"]
    }
    mismatches = {
        subject: {
            "observed": values[0],
            "reference": reference_by_subject.get(subject),
        }
        for subject, values in observed_by_subject.items()
        if reference_by_subject.get(subject) != values[0]
    }
    folds: dict[str, Any] = {}
    for fold in config["evaluation"]["folds"]:
        train = universe.manifest["outer_fold"].astype(int) != int(fold)
        test = ~train
        train_subjects = set(universe.manifest.loc[train, "subject_id"].astype(str))
        test_subjects = set(universe.manifest.loc[test, "subject_id"].astype(str))
        overlap = sorted(train_subjects & test_subjects)
        if overlap:
            raise RuntimeError(f"Outer participant leakage in fold {fold}: {overlap}")
        folds[str(fold)] = {
            "train_windows": int(train.sum()),
            "test_windows": int(test.sum()),
            "train_subject_ids": sorted(train_subjects),
            "test_subject_ids": sorted(test_subjects),
            "participant_overlap": overlap,
            "split_hash": stable_hash(
                {
                    "train": universe.manifest.loc[train, "sample_id"].astype(str).tolist(),
                    "test": universe.manifest.loc[test, "sample_id"].astype(str).tolist(),
                }
            ),
        }
    return {
        "reference_path": _relative_path(reference_path),
        "subject_fold_assignments_match": not mismatches,
        "mismatches": mismatches,
        "folds": folds,
    }


def _protocol_components(
    config_path: str | Path,
) -> tuple[dict[str, Any], SignalUniverse, list[ArtifactRunSpecV2], dict[str, Any]]:
    config = load_config(config_path)
    universe = load_signal_universe(config)
    specs = build_run_matrix(config)
    fold_audit = _fold_subject_audit(universe, config)
    cache_hash = preprocessing_cache_hash(
        config["preprocessing"], universe.source_contract_hash
    )
    run_matrix_hash = stable_hash([asdict(spec) for spec in specs])
    semantic = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "implementation_sha256": {
            "experiment_runner": _file_sha256(Path(__file__)),
            **preprocessing_implementation_hashes(),
        },
        "dataset": {
            key: config["dataset"][key]
            for key in (
                "raw_window_index",
                "target_table",
                "logical_recording_map",
                "reference_fold_manifest",
                "raw_preprocessing",
            )
        },
        "targets": list(PM_METRICS),
        "variants": list(ARTIFACT_VARIANTS_V2),
        "task_types": list(TASK_TYPES),
        "preprocessing": config["preprocessing"],
        "model": config["model"],
        "evaluation": config["evaluation"],
        "source_preprocessing_contract_hash": universe.source_contract_hash,
        "preprocessing_cache_hash": cache_hash,
        "run_matrix_hash": run_matrix_hash,
        "fold_split_hashes": {
            fold: payload["split_hash"]
            for fold, payload in fold_audit["folds"].items()
        },
        "target_sample_id_hashes": {
            metric: stable_hash(sorted(sample_ids))
            for metric, sample_ids in universe.target_sample_ids.items()
        },
    }
    return config, universe, specs, {
        "fold_audit": fold_audit,
        "cache_hash": cache_hash,
        "run_matrix_hash": run_matrix_hash,
        "protocol_semantic": semantic,
        "protocol_hash": stable_hash(semantic),
    }


def protocol_plan(config_path: str | Path) -> dict[str, Any]:
    config, universe, specs, components = _protocol_components(config_path)
    target_counts = {
        metric: {
            "sample_count": int(len(sample_ids)),
            "is_subset_of_signal_universe": sample_ids.issubset(
                set(universe.manifest["sample_id"].astype(str))
            ),
            "sample_id_hash": stable_hash(sorted(sample_ids)),
        }
        for metric, sample_ids in universe.target_sample_ids.items()
    }
    smoke_ids = smoke_sample_ids(universe, config)
    smoke_source_hash = stable_hash(
        {
            "canonical_source_preprocessing_contract_hash": universe.source_contract_hash,
            "profile": "smoke",
            "sample_ids": sorted(smoke_ids.tolist()),
        }
    )
    smoke_cache_hash = preprocessing_cache_hash(
        config["preprocessing"], smoke_source_hash
    )
    manifest = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "result_status": config.get("result_status", "baseline"),
        "analysis_role": "confirmatory_protocol_not_executed",
        "config_path": _relative_path(config_path),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "signal_universe": {
            "target_independent": True,
            "dataset_mode": "raw_deduplicated_logical_records",
            "windows": int(len(universe.manifest)),
            "subjects": int(universe.manifest["subject_id"].nunique()),
            "record_group_ids": int(universe.manifest["record_group_id"].nunique()),
            "input_shape": [1, 14, 2560],
            "sampling_rate_hz": 256,
            "source_preprocessing_contract_hash": universe.source_contract_hash,
        },
        "target_cohorts": target_counts,
        "all_target_cohorts_subset_of_signal_universe": all(
            payload["is_subset_of_signal_universe"]
            for payload in target_counts.values()
        ),
        "target_mask_application_order": "after_preprocessing_cache",
        "targets": list(PM_METRICS),
        "variants": list(ARTIFACT_VARIANTS_V2),
        "task_types": list(TASK_TYPES),
        "run_count": len(specs),
        "expected_run_count": 280,
        "smoke_run_count": 56,
        "preprocessing_cache_hash": components["cache_hash"],
        "smoke_preprocessing_cache_hash": smoke_cache_hash,
        "smoke_signal_window_count": int(len(smoke_ids)),
        "run_matrix_hash": components["run_matrix_hash"],
        "protocol_hash": components["protocol_hash"],
        "fixed_outer_folds": components["fold_audit"],
        "classification": {
            "target_transform": "outer_train_q33_q67_before_preprocessing_rejection",
            "classes": {"0": "low", "1": "medium", "2": "high"},
            "primary_metrics": [
                "participant_macro_macro_f1",
                "participant_macro_balanced_accuracy",
            ],
        },
        "regression": {
            "target_transform": "none",
            "primary_aggregation": "participant_macro",
            "metrics": ["mae", "rmse", "r2", "pearson", "spearman"],
        },
        "paired_comparison": {
            "reference": "raw predictions restricted to identical retained sample_id",
            "delta": "method_minus_paired_raw",
            "coverage_denominator": "target_valid_test_before_preprocessing_rejection",
        },
        "normalization": "torch_adapter_inner_train_only_after_preprocessing",
        "inner_validation": "record_group_id_disjoint_inside_outer_train",
        "protocol_semantic": components["protocol_semantic"],
    }
    return manifest


def _matrix_frame(
    specs: Sequence[ArtifactRunSpecV2],
    *,
    protocol_hash: str,
    cache_hash: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **asdict(spec),
                "run_id": spec.run_id,
                "specification_hash": run_specification_hash(
                    spec, protocol_hash=protocol_hash, cache_hash=cache_hash
                ),
            }
            for spec in specs
        ]
    )


def write_plan(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = protocol_plan(config_path)
    specs = build_run_matrix(config)
    output = _repo_path(config["output_dir"])
    _write_json(output / "protocol_manifest.json", manifest)
    matrix = _matrix_frame(
        specs,
        protocol_hash=manifest["protocol_hash"],
        cache_hash=manifest["preprocessing_cache_hash"],
    )
    _write_csv(output / "run_matrix.csv", matrix)
    smoke = config["smoke"]
    smoke_specs = [
        spec
        for spec in specs
        if spec.metric in smoke["targets"]
        and spec.variant in smoke["variants"]
        and spec.task_type in smoke["task_types"]
        and spec.fold in set(map(int, smoke["folds"]))
    ]
    smoke_matrix = _matrix_frame(
        smoke_specs,
        protocol_hash=manifest["protocol_hash"],
        cache_hash=manifest["smoke_preprocessing_cache_hash"],
    )
    if len(smoke_matrix) != 56:
        raise RuntimeError(f"Smoke matrix must contain 56 runs, got {len(smoke_matrix)}")
    _write_csv(output / "smoke_run_matrix.csv", smoke_matrix)
    _write_json(
        output / "plan_summary.json",
        {
            "experiment_id": config["experiment_id"],
            "protocol_hash": manifest["protocol_hash"],
            "preprocessing_cache_hash": manifest["preprocessing_cache_hash"],
            "smoke_preprocessing_cache_hash": manifest[
                "smoke_preprocessing_cache_hash"
            ],
            "run_matrix_hash": manifest["run_matrix_hash"],
            "run_count": len(matrix),
            "smoke_run_count": len(smoke_matrix),
        },
    )
    return manifest


def _thresholds_for_metric_fold(
    universe: SignalUniverse, metric: str, fold: int
) -> list[float]:
    column = f"target_{metric}"
    values = universe.targets[column].to_numpy(dtype=np.float64)
    train = (
        universe.manifest["outer_fold"].to_numpy(dtype=int) != int(fold)
    ) & np.isfinite(values)
    return _q3_thresholds(values[train])


def smoke_sample_ids(
    universe: SignalUniverse, config: Mapping[str, Any]
) -> np.ndarray:
    selected: set[str] = set()
    folds = universe.manifest["outer_fold"].to_numpy(dtype=int)
    subjects = universe.manifest["subject_id"].astype(str).to_numpy()
    sample_ids = universe.manifest["sample_id"].astype(str).to_numpy()
    for metric in PM_METRICS:
        values = universe.targets[f"target_{metric}"].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        thresholds = _thresholds_for_metric_fold(universe, metric, 1)
        labels = np.full(len(values), -1, dtype=np.int64)
        labels[valid] = _q3_labels(values[valid], thresholds)
        for fold_mask, limit in (
            (folds != 1, int(config["smoke"]["max_train_windows"])),
            (folds == 1, int(config["smoke"]["max_test_windows"])),
        ):
            positions = np.flatnonzero(valid & fold_mask)
            chosen = _balanced_subset(
                positions,
                subjects=subjects,
                labels=labels,
                sample_ids=sample_ids,
                limit=limit,
            )
            selected.update(sample_ids[chosen].tolist())
    return np.asarray(sorted(selected), dtype=str)


def _cache_root(config: Mapping[str, Any], profile: str) -> Path:
    base = _repo_path(config["output_dir"]) / "preprocessing_cache"
    return base if profile == "full" else base / "smoke"


def build_cache(
    config_path: str | Path,
    *,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    universe = load_signal_universe(config)
    profile = "smoke" if smoke else "full"
    base_view = universe.data.data
    source_hash = universe.source_contract_hash
    if smoke:
        selected_ids = set(smoke_sample_ids(universe, config))
        positions = np.flatnonzero(
            universe.manifest["sample_id"].astype(str).isin(selected_ids).to_numpy()
        )
        base_view = base_view[positions]
        source_hash = stable_hash(
            {
                "canonical_source_preprocessing_contract_hash": source_hash,
                "profile": "smoke",
                "sample_ids": sorted(selected_ids),
            }
        )
    result = build_preprocessing_cache(
        base_view,
        cache_root=_cache_root(config, profile),
        config=config["preprocessing"],
        source_hash=source_hash,
        resume=resume,
    )
    result["profile"] = profile
    result["canonical_source_preprocessing_contract_hash"] = (
        universe.source_contract_hash
    )
    result["cached_signal_windows"] = int(len(base_view))
    _write_json(_cache_root(config, profile) / "preprocessing_cache_manifest.json", result)
    return result


def _task_arrays(
    universe: SignalUniverse,
    manifest: pd.DataFrame,
    spec: ArtifactRunSpecV2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float] | None]:
    target_map = universe.targets.set_index(
        universe.targets["sample_id"].astype(str)
    )[f"target_{spec.metric}"]
    values = manifest["sample_id"].astype(str).map(target_map).to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    train = valid & (manifest["outer_fold"].to_numpy(dtype=int) != spec.fold)
    test = valid & (manifest["outer_fold"].to_numpy(dtype=int) == spec.fold)
    thresholds: list[float] | None = None
    if spec.task_type == "classification":
        thresholds = _thresholds_for_metric_fold(universe, spec.metric, spec.fold)
        targets = np.full(len(values), -1, dtype=np.int64)
        targets[valid] = _q3_labels(values[valid], thresholds)
    else:
        targets = values.astype(np.float32)
    return np.flatnonzero(train), np.flatnonzero(test), targets, thresholds


def calculate_coverage(
    retained_sample_ids: Iterable[str],
    original_sample_ids: Iterable[str],
    *,
    variant: str,
) -> tuple[float, list[str]]:
    retained = set(map(str, retained_sample_ids))
    original = set(map(str, original_sample_ids))
    if not original:
        raise ValueError("Coverage denominator cannot be empty")
    if not retained.issubset(original):
        raise RuntimeError("Retained sample_id are outside the original cohort")
    coverage = float(len(retained) / len(original))
    if variant == "raw" and coverage != 1.0:
        raise RuntimeError(f"Raw coverage must equal 1, got {coverage}")
    return coverage, sorted(original - retained)


def _classification_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    labels = [0, 1, 2]
    return {
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "per_class_f1": f1_score(
            truth, prediction, labels=labels, average=None, zero_division=0
        ).astype(float).tolist(),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).astype(int).tolist(),
        "n_samples": int(len(truth)),
    }


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result = MetricsCalculator.calculate_regression_metrics(truth, prediction)
    return {
        key: result[key]
        for key in ("mae", "rmse", "r2", "pearson", "spearman", "n_samples")
    }


def _metric_function(task_type: str) -> Any:
    return _classification_metrics if task_type == "classification" else _regression_metrics


def _participant_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    subjects: np.ndarray,
    *,
    task_type: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    calculate = _metric_function(task_type)
    for subject in sorted(set(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        metrics = calculate(truth[mask], prediction[mask])
        rows.append({"subject_id": subject, **metrics})
    frame = pd.DataFrame(rows)
    names = (
        ("macro_f1", "balanced_accuracy", "accuracy")
        if task_type == "classification"
        else ("mae", "rmse", "r2", "pearson", "spearman")
    )
    macro = {
        name: float(pd.to_numeric(frame[name], errors="coerce").mean(skipna=True))
        for name in names
    }
    return frame, macro


def _run_dir(config: Mapping[str, Any], profile: str, spec: ArtifactRunSpecV2) -> Path:
    return _repo_path(config["output_dir"]) / profile / spec.run_id


def _paired_raw_metrics(
    config: Mapping[str, Any],
    profile: str,
    spec: ArtifactRunSpecV2,
    current_predictions: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, float], pd.DataFrame]:
    if spec.variant == "raw":
        raw = current_predictions.copy()
    else:
        raw_spec = ArtifactRunSpecV2(
            metric=spec.metric,
            variant="raw",
            fold=spec.fold,
            task_type=spec.task_type,
            seed=spec.seed,
        )
        path = _run_dir(config, profile, raw_spec) / "predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"Paired raw predictions must exist before {spec.run_id}: {path}"
            )
        raw_all = pd.read_parquet(path)
        wanted = current_predictions["sample_id"].astype(str)
        raw = raw_all.loc[raw_all["sample_id"].astype(str).isin(set(wanted))].copy()
        raw = raw.set_index(raw["sample_id"].astype(str)).loc[wanted].reset_index(drop=True)
        if raw["sample_id"].astype(str).tolist() != wanted.tolist():
            raise RuntimeError("Paired raw sample_id order does not match method predictions")
        if not np.allclose(
            raw["y_true"].to_numpy(dtype=float),
            current_predictions["y_true"].to_numpy(dtype=float),
        ):
            raise RuntimeError("Paired raw y_true differs on identical sample_id")
    truth = raw["y_true"].to_numpy()
    prediction = raw["y_pred"].to_numpy()
    if spec.task_type == "classification":
        truth = truth.astype(np.int64)
        prediction = prediction.astype(np.int64)
    else:
        truth = truth.astype(np.float64)
        prediction = prediction.astype(np.float64)
    window = _metric_function(spec.task_type)(truth, prediction)
    participant, macro = _participant_metrics(
        truth,
        prediction,
        raw["subject_id"].astype(str).to_numpy(),
        task_type=spec.task_type,
    )
    return window, macro, participant


def execute_run(
    config_path: str | Path,
    spec: ArtifactRunSpecV2,
    *,
    smoke: bool,
    resume: bool,
    protocol: Mapping[str, Any] | None = None,
    universe: SignalUniverse | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    protocol = dict(protocol or protocol_plan(config_path))
    profile = "smoke" if smoke else "full"
    cache_root = _cache_root(config, profile)
    cache_manifest_path = cache_root / "preprocessing_cache_manifest.json"
    if not cache_manifest_path.is_file():
        raise FileNotFoundError(
            f"Build the {profile} preprocessing cache first: {cache_manifest_path}"
        )
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache_hash = str(cache_manifest["cache_hash"])
    specification_hash = run_specification_hash(
        spec,
        protocol_hash=str(protocol["protocol_hash"]),
        cache_hash=cache_hash,
    )
    run_dir = _run_dir(config, profile, spec)
    summary_path = run_dir / "run_summary.json"
    if resume and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("specification_hash") == specification_hash
        ):
            return existing
    universe = universe or load_signal_universe(config)
    view = load_cached_view(cache_root, spec.variant, repo_root=REPO_ROOT)
    train_idx, test_idx, targets, thresholds = _task_arrays(
        universe, view.manifest, spec
    )
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Empty target-specific split for {spec.run_id}")
    train_subjects = set(view.manifest.iloc[train_idx]["subject_id"].astype(str))
    test_subjects = set(view.manifest.iloc[test_idx]["subject_id"].astype(str))
    overlap = sorted(train_subjects & test_subjects)
    if overlap:
        raise RuntimeError(f"Outer participant leakage for {spec.run_id}: {overlap}")
    X_train = view[train_idx]
    X_test = view[test_idx]
    y_train = targets[train_idx]
    y_test = targets[test_idx]
    params = dict(config["model"]["params"])
    params["random_state"] = spec.seed
    if smoke:
        params["max_epochs"] = int(config["smoke"]["max_epochs"])
        params["batch_size"] = min(
            int(params["batch_size"]), int(config["smoke"].get("batch_size", 64))
        )
    num_outputs = 3 if spec.task_type == "classification" else 1
    model = build_model(
        "torch_shallow_convnet",
        spec.task_type,
        X_train.shape[1:],
        num_outputs,
        params,
    )
    train_manifest = X_train.manifest
    test_manifest = X_test.manifest
    model.set_validation_groups(
        train_manifest["record_group_id"].astype(str).to_numpy(),
        subject_ids=train_manifest["subject_id"].astype(str).to_numpy(),
        record_ids=train_manifest["record_id"].astype(str).to_numpy(),
        outer_test_record_ids=test_manifest["record_id"].astype(str).to_numpy(),
        outer_test_group_ids=test_manifest["record_group_id"].astype(str).to_numpy(),
        strategy="group_record",
        group_column="record_group_id",
        validation_size=float(config["model"]["params"]["validation_size"]),
        random_state=spec.seed,
    )
    training_started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - training_started
    inference_started = time.perf_counter()
    prediction = model.predict(X_test)
    probabilities = (
        model.predict_proba(X_test)
        if spec.task_type == "classification"
        else None
    )
    inference_seconds = time.perf_counter() - inference_started
    if spec.task_type == "classification":
        prediction = np.asarray(prediction, dtype=np.int64)
        y_test = np.asarray(y_test, dtype=np.int64)
    else:
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    window_metrics = _metric_function(spec.task_type)(y_test, prediction)
    participant, participant_macro = _participant_metrics(
        y_test,
        prediction,
        test_manifest["subject_id"].astype(str).to_numpy(),
        task_type=spec.task_type,
    )
    predictions = pd.DataFrame(
        {
            "sample_id": test_manifest["sample_id"].to_numpy(),
            "subject_id": test_manifest["subject_id"].astype(str).to_numpy(),
            "record_id": test_manifest["record_id"].astype(str).to_numpy(),
            "record_group_id": test_manifest["record_group_id"].astype(str).to_numpy(),
            "fold": spec.fold,
            "pm": spec.metric,
            "variant": spec.variant,
            "task_type": spec.task_type,
            "y_true": y_test,
            "y_pred": prediction,
        }
    )
    if probabilities is not None:
        for class_id in range(3):
            predictions[f"proba_{class_id}"] = probabilities[:, class_id]
    paired_window, paired_macro, paired_participant = _paired_raw_metrics(
        config, profile, spec, predictions
    )
    metric_names = (
        ("macro_f1", "balanced_accuracy", "accuracy")
        if spec.task_type == "classification"
        else ("mae", "rmse", "r2", "pearson", "spearman")
    )
    deltas = {
        name: float(participant_macro[name] - paired_macro[name])
        for name in metric_names
    }
    original_test_ids = universe.target_sample_ids[spec.metric] & set(
        universe.manifest.loc[
            universe.manifest["outer_fold"].astype(int).eq(spec.fold), "sample_id"
        ].astype(str)
    )
    if smoke:
        smoke_ids = set(smoke_sample_ids(universe, config))
        original_test_ids &= smoke_ids
    retained_test_ids = set(predictions["sample_id"].astype(str))
    coverage, rejected_ids = calculate_coverage(
        retained_test_ids, original_test_ids, variant=spec.variant
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    participant.insert(0, "aggregation", "method")
    paired_participant.insert(0, "aggregation", "paired_raw")
    _write_csv(
        run_dir / "participant_metrics.csv",
        pd.concat([participant, paired_participant], ignore_index=True),
    )
    pd.DataFrame({"sample_id": rejected_ids}).to_parquet(
        run_dir / "rejected_test_sample_ids.parquet", index=False
    )
    model.save(run_dir / "model.pt")
    _write_csv(run_dir / "training_log.csv", pd.DataFrame(model.training_log_))
    _write_json(run_dir / "validation_split.json", model.validation_split_)
    _write_json(
        run_dir / "normalization_stats.json",
        {
            "fit_scope": "inner_train_only_after_preprocessing",
            "mean": None if model.feature_mean_ is None else model.feature_mean_.tolist(),
            "scale": None if model.feature_scale_ is None else model.feature_scale_.tolist(),
        },
    )
    threshold_hash = stable_hash(thresholds) if thresholds is not None else None
    _write_json(
        run_dir / "preprocessing_manifest.json",
        {
            "cache_root": _relative_path(cache_root),
            "cache_hash": cache_hash,
            "variant_preprocessing_hash": cache_manifest["variants"][spec.variant][
                "preprocessing_hash"
            ],
            "target_independent_cache": True,
            "target_mask_application_order": "after_preprocessing_cache",
            "q3_fit_scope": (
                "all_target_valid_outer_train_before_preprocessing_rejection"
                if thresholds is not None
                else "not_applicable"
            ),
            "q3_thresholds": thresholds,
            "q3_thresholds_hash": threshold_hash,
            "outer_participant_overlap": overlap,
            "coverage": coverage,
            "rejected_test_sample_ids": rejected_ids,
        },
    )
    metrics = {
        "window": window_metrics,
        "participant_macro": participant_macro,
        "paired_raw_window": paired_window,
        "paired_raw_participant_macro": paired_macro,
        "delta_participant_macro_method_minus_paired_raw": deltas,
        "coverage": coverage,
    }
    _write_json(run_dir / "metrics.json", metrics)
    summary = {
        "status": "complete",
        "result_status": "smoke" if smoke else config.get("result_status", "baseline"),
        "run_id": spec.run_id,
        "specification_hash": specification_hash,
        "protocol_hash": protocol["protocol_hash"],
        "preprocessing_cache_hash": cache_hash,
        **asdict(spec),
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "original_target_valid_test_windows": int(len(original_test_ids)),
        "coverage": coverage,
        "rejected_test_windows": int(len(rejected_ids)),
        "q3_thresholds": thresholds,
        "q3_thresholds_hash": threshold_hash,
        "metrics": metrics,
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "epochs": int(model.n_epochs_trained_),
        "best_validation_loss": model.best_validation_loss_,
        "device": str(model.device_),
        "parameter_count": int(sum(parameter.numel() for parameter in model.model.parameters())),
        "outer_participant_overlap": overlap,
        "inner_validation_group_overlap": model.validation_split_.get(
            "inner_group_overlap", []
        ),
    }
    _write_json(summary_path, summary)
    return summary


def _flatten_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        key: summary[key]
        for key in (
            "run_id",
            "metric",
            "variant",
            "fold",
            "task_type",
            "seed",
            "train_windows",
            "test_windows",
            "coverage",
            "rejected_test_windows",
            "training_seconds",
            "inference_seconds",
            "epochs",
            "best_validation_loss",
            "device",
            "parameter_count",
        )
    }
    metrics = summary["metrics"]
    for prefix, values in (
        ("window", metrics["window"]),
        ("participant_macro", metrics["participant_macro"]),
        ("paired_raw_window", metrics["paired_raw_window"]),
        ("paired_raw_participant_macro", metrics["paired_raw_participant_macro"]),
        ("delta_participant_macro", metrics["delta_participant_macro_method_minus_paired_raw"]),
    ):
        for name, value in values.items():
            if np.isscalar(value) and not isinstance(value, str):
                row[f"{prefix}_{name}"] = value
    return row


def aggregate_results(
    config_path: str | Path,
    *,
    smoke: bool,
    summaries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    profile = "smoke" if smoke else "full"
    output = _repo_path(config["output_dir"])
    if summaries is None:
        summaries = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output / profile).glob("*/run_summary.json"))
        ]
    complete = [summary for summary in summaries if summary.get("status") == "complete"]
    frame = pd.DataFrame([_flatten_summary(summary) for summary in complete])
    if frame.empty:
        raise ValueError(f"No completed {profile} runs to aggregate")
    _write_csv(output / f"{profile}_run_metrics.csv", frame)
    metric_columns = [
        column
        for column in frame.columns
        if column.startswith("participant_macro_")
        or column.startswith("delta_participant_macro_")
    ]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["task_type", "metric", "variant"], sort=True, observed=True
    ):
        row = dict(zip(("task_type", "metric", "variant"), keys))
        row["fold_count"] = int(group["fold"].nunique())
        row["coverage_mean"] = float(group["coverage"].mean())
        row["preprocessing_runtime_seconds"] = float(
            json.loads(
                (_cache_root(config, profile) / keys[2] / "cache_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["elapsed_seconds"]
        )
        for column in metric_columns:
            numeric = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(numeric.mean(skipna=True))
            row[f"{column}_sample_sd"] = float(numeric.std(ddof=1, skipna=True))
            if column.startswith("delta_participant_macro_"):
                lower_is_better = any(
                    column.endswith(name) for name in ("mae", "rmse")
                )
                row[f"folds_better_paired_raw_{column.removeprefix('delta_participant_macro_')}"] = int(
                    ((numeric < 0) if lower_is_better else (numeric > 0)).sum()
                )
        summary_rows.append(row)
    final = pd.DataFrame(summary_rows)
    _write_csv(output / f"{profile}_final_summary_by_pm.csv", final)
    fold_macro = (
        frame.groupby(["task_type", "variant", "fold"], as_index=False, sort=True)[
            metric_columns + ["coverage", "training_seconds", "inference_seconds"]
        ]
        .mean(numeric_only=True)
    )
    _write_csv(output / f"{profile}_pm_macro_by_fold.csv", fold_macro)
    pm_macro_rows: list[dict[str, Any]] = []
    for keys, group in fold_macro.groupby(
        ["task_type", "variant"], sort=True, observed=True
    ):
        row = dict(zip(("task_type", "variant"), keys))
        row["fold_count"] = int(group["fold"].nunique())
        for column in metric_columns + ["coverage", "training_seconds", "inference_seconds"]:
            numeric = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(numeric.mean(skipna=True))
            row[f"{column}_sample_sd"] = float(numeric.std(ddof=1, skipna=True))
        pm_macro_rows.append(row)
    pm_macro = pd.DataFrame(pm_macro_rows)
    _write_csv(output / f"{profile}_pm_macro_summary.csv", pm_macro)
    aggregation = {
        "profile": profile,
        "completed_runs": int(len(frame)),
        "expected_runs": 56 if smoke else 280,
        "all_seven_pm_present": set(frame["metric"]) == set(PM_METRICS),
        "all_four_variants_present": set(frame["variant"]) == set(ARTIFACT_VARIANTS_V2),
        "both_task_types_present": set(frame["task_type"]) == set(TASK_TYPES),
        "run_metrics_path": _relative_path(output / f"{profile}_run_metrics.csv"),
        "final_summary_path": _relative_path(output / f"{profile}_final_summary_by_pm.csv"),
        "pm_macro_summary_path": _relative_path(output / f"{profile}_pm_macro_summary.csv"),
    }
    _write_json(output / f"{profile}_aggregation_summary.json", aggregation)
    return aggregation


def _selected_specs(
    specs: Iterable[ArtifactRunSpecV2],
    *,
    smoke: bool,
    config: Mapping[str, Any],
    metrics: set[str] | None,
    variants: set[str] | None,
    folds: set[int] | None,
    task_types: set[str] | None,
) -> list[ArtifactRunSpecV2]:
    selected = list(specs)
    if smoke:
        smoke_config = config["smoke"]
        selected = [
            spec
            for spec in selected
            if spec.metric in smoke_config["targets"]
            and spec.variant in smoke_config["variants"]
            and spec.fold in set(map(int, smoke_config["folds"]))
            and spec.task_type in smoke_config["task_types"]
        ]
    if metrics:
        selected = [spec for spec in selected if spec.metric in metrics]
    if variants:
        selected = [spec for spec in selected if spec.variant in variants]
    if folds:
        selected = [spec for spec in selected if spec.fold in folds]
    if task_types:
        selected = [spec for spec in selected if spec.task_type in task_types]
    return selected


def run_experiment(
    config_path: str | Path,
    *,
    smoke: bool,
    resume: bool = True,
    metrics: set[str] | None = None,
    variants: set[str] | None = None,
    folds: set[int] | None = None,
    task_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    protocol = write_plan(config_path)
    if smoke and not (_cache_root(config, "smoke") / "preprocessing_cache_manifest.json").is_file():
        build_cache(config_path, smoke=True, resume=resume)
    specs = _selected_specs(
        build_run_matrix(config),
        smoke=smoke,
        config=config,
        metrics=metrics,
        variants=variants,
        folds=folds,
        task_types=task_types,
    )
    universe = load_signal_universe(config)
    summaries = [
        execute_run(
            config_path,
            spec,
            smoke=smoke,
            resume=resume,
            protocol=protocol,
            universe=universe,
        )
        for spec in specs
    ]
    unfiltered = not any((metrics, variants, folds, task_types))
    if unfiltered:
        aggregate_results(config_path, smoke=smoke, summaries=summaries)
    return summaries


