"""Canonical seven-PM feature baseline with fixed subject folds."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from bench.datasets.base_eeg_data_loader import (
    feature_list_sha256,
    resolve_feature_columns,
)
from bench.datasets.logical_recordings import ensure_record_group_ids
from bench.tasks.target_registry import PM_METRICS, get_target_spec
from bench.validation.metrics import MetricsCalculator
from model_zoo import build_model
from model_zoo.DL.feature_preprocessing import FeaturePreprocessor


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_STATUSES = {
    "pending",
    "running",
    "complete",
    "failed_technical",
    "failed_numerical",
}
FEATURE_SET_ORDER = ("eeg", "pow", "eeg_pow")
METRIC_NAMES = ("mae", "rmse", "r2", "pearson", "spearman")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return _relative_path(value)
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, sample_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    digest.update("\n".join(map(str, sample_ids.tolist())).encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def load_baseline_config(path: str | Path) -> dict[str, Any]:
    config_path = _repo_path(path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("PM baseline config must be a mapping")
    required = {
        "experiment_id",
        "dataset",
        "targets",
        "feature_sets",
        "models",
        "evaluation",
        "metrics",
        "paired_single_vs_multioutput",
        "output_dir",
        "resume",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"PM baseline config is missing sections: {missing}")
    if document["experiment_id"] != "pm_all_targets_feature_baseline_v1":
        raise ValueError("Unexpected experiment_id")
    return document


@dataclass(frozen=True)
class RunSpec:
    analysis: str
    target_id: str
    target_name: str
    feature_set: str
    model: str
    seed: int
    fold: int
    cohort_target_id: str
    scaling: str
    params: dict[str, Any]

    @property
    def semantic_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def specification_hash(self) -> str:
        return stable_hash(self.semantic_payload)

    @property
    def run_id(self) -> str:
        target = self.target_name if self.analysis == "paired_single" else self.target_id
        return (
            f"{self.analysis}__{target}__{self.feature_set}__{self.model}__"
            f"seed{self.seed}__fold{self.fold:02d}__{self.specification_hash[:10]}"
        )


@dataclass
class ProtocolContext:
    config: dict[str, Any]
    config_path: Path
    output_dir: Path
    frame: pd.DataFrame
    feature_names: dict[str, tuple[str, ...]]
    features: dict[str, np.ndarray]
    target_values: dict[str, np.ndarray]
    target_masks: dict[str, np.ndarray]
    fold_by_subject: dict[str, int]
    folds: dict[int, dict[str, Any]]
    run_specs: list[RunSpec]
    preregistration: dict[str, Any]
    cohort_summary: pd.DataFrame


def _model_specs(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    models = config["models"][section]
    if not isinstance(models, Mapping) or not models:
        raise ValueError(f"models.{section} must be a non-empty mapping")
    return models


def build_run_matrix(config: Mapping[str, Any]) -> list[RunSpec]:
    folds = tuple(int(value) for value in config["evaluation"]["folds"])
    feature_sets = tuple(config["feature_sets"])
    if feature_sets != FEATURE_SET_ORDER:
        raise ValueError(f"Feature sets must be ordered as {FEATURE_SET_ORDER}")
    single_targets = tuple(config["targets"]["single"])
    multi_target = str(config["targets"]["multioutput"])
    output_order = tuple(config["targets"]["output_order"])
    if output_order != PM_METRICS:
        raise ValueError(f"PM output order must be {PM_METRICS}")
    expected_single = tuple(f"pm_{metric}_regression" for metric in PM_METRICS)
    if single_targets != expected_single:
        raise ValueError(f"Single PM targets must be {expected_single}")
    specs: list[RunSpec] = []
    for target_id in single_targets:
        target_name = get_target_spec(target_id).output_names[0]
        for feature_set in feature_sets:
            for model, definition in _model_specs(config, "single_target").items():
                for seed in definition["seeds"]:
                    for fold in folds:
                        specs.append(
                            RunSpec(
                                analysis="main_single",
                                target_id=target_id,
                                target_name=target_name,
                                feature_set=feature_set,
                                model=str(model),
                                seed=int(seed),
                                fold=fold,
                                cohort_target_id=target_id,
                                scaling=str(definition["scaling"]),
                                params=dict(definition.get("params", {})),
                            )
                        )
    for feature_set in feature_sets:
        for model, definition in _model_specs(config, "multioutput").items():
            for seed in definition["seeds"]:
                for fold in folds:
                    specs.append(
                        RunSpec(
                            analysis="main_multi",
                            target_id=multi_target,
                            target_name="all_pm",
                            feature_set=feature_set,
                            model=str(model),
                            seed=int(seed),
                            fold=fold,
                            cohort_target_id=multi_target,
                            scaling=str(definition["scaling"]),
                            params=dict(definition.get("params", {})),
                        )
                    )
    paired = config["paired_single_vs_multioutput"]
    if paired.get("enabled", False):
        paired_models = set(map(str, paired["models"]))
        for target_id in single_targets:
            target_name = get_target_spec(target_id).output_names[0]
            for feature_set in paired["feature_sets"]:
                for model, definition in _model_specs(config, "single_target").items():
                    if model not in paired_models:
                        continue
                    for seed in definition["seeds"]:
                        for fold in folds:
                            specs.append(
                                RunSpec(
                                    analysis="paired_single",
                                    target_id=target_id,
                                    target_name=target_name,
                                    feature_set=str(feature_set),
                                    model=str(model),
                                    seed=int(seed),
                                    fold=fold,
                                    cohort_target_id=str(paired["cohort_target_id"]),
                                    scaling=str(definition["scaling"]),
                                    params=dict(definition.get("params", {})),
                                )
                            )
    ids = [spec.run_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Run matrix contains duplicate run IDs")
    return specs


def _load_reference_folds(
    frame: pd.DataFrame, reference_path: Path, folds: Sequence[int]
) -> tuple[dict[str, int], dict[int, dict[str, Any]]]:
    reference = pd.read_parquet(reference_path, columns=["sample_id", "subject_id", "fold"])
    if reference["sample_id"].duplicated().any():
        raise ValueError("Reference predictions contain duplicate sample_id")
    subject_fold_counts = reference.groupby("subject_id")["fold"].nunique()
    if not subject_fold_counts.eq(1).all():
        raise ValueError("Reference predictions assign a subject to multiple folds")
    mapping = (
        reference.drop_duplicates("subject_id")
        .set_index("subject_id")["fold"]
        .astype(int)
        .to_dict()
    )
    mapping = {str(key): int(value) for key, value in mapping.items()}
    target_subjects = set(frame.loc[frame["target_focus"].notna(), "subject_id"].astype(str))
    if target_subjects != set(mapping):
        raise ValueError(
            "Reference subject universe differs from canonical supervised PM universe"
        )
    fold_documents: dict[int, dict[str, Any]] = {}
    all_subjects = sorted(mapping)
    for fold in folds:
        test = sorted(subject for subject, value in mapping.items() if value == fold)
        train = sorted(set(all_subjects) - set(test))
        document = {
            "fold": int(fold),
            "group_column": "subject_id",
            "train_subject_ids": train,
            "test_subject_ids": test,
            "subject_overlap": [],
        }
        document["split_hash"] = stable_hash(document)
        fold_documents[int(fold)] = document
    if set(fold_documents) != set(map(int, folds)):
        raise ValueError("Fixed fold IDs differ from configuration")
    return mapping, fold_documents


def _cohort_summary(
    frame: pd.DataFrame,
    target_masks: Mapping[str, np.ndarray],
    fold_by_subject: Mapping[str, int],
    folds: Sequence[int],
) -> pd.DataFrame:
    subject_folds = frame["subject_id"].astype(str).map(fold_by_subject).to_numpy()
    rows: list[dict[str, Any]] = []
    for target_id, mask in target_masks.items():
        for fold in folds:
            test_subject_mask = subject_folds == fold
            train_subject_mask = np.isfinite(subject_folds.astype(float)) & ~test_subject_mask
            train = mask & train_subject_mask
            test = mask & test_subject_mask
            rows.append(
                {
                    "target_id": target_id,
                    "fold": int(fold),
                    "train_windows": int(train.sum()),
                    "test_windows": int(test.sum()),
                    "train_participants": int(frame.loc[train, "subject_id"].nunique()),
                    "test_participants": int(frame.loc[test, "subject_id"].nunique()),
                    "train_source_records": int(frame.loc[train, "record_id"].nunique()),
                    "test_source_records": int(frame.loc[test, "record_id"].nunique()),
                    "train_logical_records": int(frame.loc[train, "record_group_id"].nunique()),
                    "test_logical_records": int(frame.loc[test, "record_group_id"].nunique()),
                    "gpn_data_windows": int((test & frame["source"].eq("gpn_data").to_numpy()).sum()),
                    "Old_EEG_windows": int((test & frame["source"].eq("Old_EEG").to_numpy()).sum()),
                    "missing_windows": int((test_subject_mask & ~mask).sum()),
                }
            )
    return pd.DataFrame(rows)


def prepare_protocol(config_path: str | Path) -> ProtocolContext:
    config_path = _repo_path(config_path)
    config = load_baseline_config(config_path)
    dataset_path = _repo_path(config["dataset"]["path"])
    reference_path = _repo_path(config["dataset"]["reference_predictions"])
    if not dataset_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("Canonical feature dataset or reference predictions are absent")
    frame = ensure_record_group_ids(pd.read_parquet(dataset_path))
    if "sample_id" not in frame:
        frame.insert(0, "sample_id", frame.index.to_numpy(dtype=np.int64))
    if frame["sample_id"].duplicated().any():
        raise ValueError("Canonical feature table has duplicate sample_id")
    feature_names: dict[str, tuple[str, ...]] = {}
    features: dict[str, np.ndarray] = {}
    for feature_set, definition in config["feature_sets"].items():
        names = tuple(resolve_feature_columns(frame.columns.tolist(), feature_set))
        actual_hash = feature_list_sha256(list(names))
        if len(names) != int(definition["expected_count"]):
            raise ValueError(f"Feature count mismatch for {feature_set}")
        if actual_hash != str(definition["expected_hash"]):
            raise ValueError(f"Feature hash mismatch for {feature_set}")
        forbidden = [
            name for name in names
            if name.startswith(("PM.", "target_", "label_"))
            or name in {"target_main", "subject_id", "record_id", "record_group_id", "sample_id", "source"}
        ]
        if forbidden:
            raise ValueError(f"Target leakage in {feature_set}: {forbidden}")
        values = frame.loc[:, names].to_numpy(dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError(f"Feature set {feature_set} contains infinite values")
        feature_names[str(feature_set)] = names
        features[str(feature_set)] = values
    target_ids = [*config["targets"]["single"], config["targets"]["multioutput"]]
    target_values: dict[str, np.ndarray] = {}
    target_masks: dict[str, np.ndarray] = {}
    for target_id in target_ids:
        spec = get_target_spec(str(target_id))
        values = frame.loc[:, list(spec.processed_columns)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
        mask = np.isfinite(values).all(axis=1)
        target_masks[spec.target_id] = mask
        target_values[spec.target_id] = values[:, 0] if spec.output_dim == 1 else values
    folds = tuple(int(value) for value in config["evaluation"]["folds"])
    fold_by_subject, fold_documents = _load_reference_folds(frame, reference_path, folds)
    for target_id, mask in target_masks.items():
        missing_subjects = sorted(
            set(frame.loc[mask, "subject_id"].astype(str)) - set(fold_by_subject)
        )
        if missing_subjects:
            raise ValueError(f"Target {target_id} has subjects absent from fixed folds: {missing_subjects}")
    cohort = _cohort_summary(frame, target_masks, fold_by_subject, folds)
    specs = build_run_matrix(config)
    run_rows = [
        {**spec.semantic_payload, "run_id": spec.run_id, "specification_hash": spec.specification_hash}
        for spec in specs
    ]
    preregistration = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_before_training": True,
        "config_path": _relative_path(config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_path": _relative_path(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "reference_predictions_path": _relative_path(reference_path),
        "reference_predictions_sha256": file_sha256(reference_path),
        "target_ids": target_ids,
        "target_order": list(PM_METRICS),
        "feature_sets": {
            name: {
                "n_features": len(names),
                "feature_list_sha256": feature_list_sha256(list(names)),
            }
            for name, names in feature_names.items()
        },
        "outer_folds": fold_documents,
        "run_count": len(specs),
        "run_matrix_hash": stable_hash(run_rows),
        "models": config["models"],
        "metrics": config["metrics"],
        "leakage_controls": {
            "target_columns_in_features": False,
            "fixed_subject_folds": True,
            "scaler_fit_scope": "outer_train_only",
            "imputation": "outer_train_median",
            "outer_test_used_for_selection": False,
        },
        "git_commit": _git_head(),
    }
    preregistration["protocol_hash"] = stable_hash(preregistration)
    return ProtocolContext(
        config=config,
        config_path=config_path,
        output_dir=_repo_path(config["output_dir"]),
        frame=frame,
        feature_names=feature_names,
        features=features,
        target_values=target_values,
        target_masks=target_masks,
        fold_by_subject=fold_by_subject,
        folds=fold_documents,
        run_specs=specs,
        preregistration=preregistration,
        cohort_summary=cohort,
    )


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _run_matrix_frame(context: ProtocolContext) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "run_id": spec.run_id,
            **spec.semantic_payload,
            "params": json.dumps(spec.params, sort_keys=True),
            "specification_hash": spec.specification_hash,
        }
        for spec in context.run_specs
    ])


def write_preregistration(context: ProtocolContext) -> None:
    root = context.output_dir
    (root / "preregistration").mkdir(parents=True, exist_ok=True)
    write_json(root / "preregistration" / "preregistration_manifest.json", context.preregistration)
    write_csv(root / "run_matrix.csv", _run_matrix_frame(context))
    write_csv(root / "target_cohort_summary.csv", context.cohort_summary)
    config_copy = root / "preregistration" / "config.yaml"
    config_copy.write_text(
        yaml.safe_dump(context.config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(
        root / "preregistration" / "fold_manifest.json",
        {"folds": context.folds, "reference_match": True},
    )


def _semantic_preregistration(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return protocol inputs without run-environment provenance fields."""
    return {
        str(key): value
        for key, value in document.items()
        if key not in {"git_commit", "protocol_hash"}
    }


def _ensure_preregistration(context: ProtocolContext) -> None:
    """Create preregistration once or reuse a semantically identical manifest."""
    path = context.output_dir / "preregistration" / "preregistration_manifest.json"
    if not path.exists():
        write_preregistration(context)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if stable_hash(_semantic_preregistration(existing)) != stable_hash(
        _semantic_preregistration(context.preregistration)
    ):
        raise ValueError(
            "Existing preregistration semantic inputs differ from current protocol"
        )
    # A commit made after a successful run changes provenance, not the scientific
    # protocol. Preserve the immutable original manifest and its protocol hash.
    context.preregistration = existing


def _initial_registry(context: ProtocolContext) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.preregistration["protocol_hash"],
        "runs": {
            spec.run_id: {
                "status": "pending",
                "specification_hash": spec.specification_hash,
                "attempts": [],
            }
            for spec in context.run_specs
        },
    }


def _load_registry(context: ProtocolContext, *, resume: bool) -> dict[str, Any]:
    path = context.output_dir / "run_registry" / "run_registry.json"
    if not path.exists():
        registry = _initial_registry(context)
        write_json(path, registry)
        return registry
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("protocol_hash") != context.preregistration["protocol_hash"]:
        raise ValueError("Existing run registry protocol hash differs from preregistration")
    if not resume:
        completed = [key for key, row in registry["runs"].items() if row["status"] == "complete"]
        if completed:
            raise ValueError("Completed runs exist; use --resume to preserve them")
    return registry


def _save_registry(context: ProtocolContext, registry: Mapping[str, Any]) -> None:
    write_json(context.output_dir / "run_registry" / "run_registry.json", registry)


def _selected_specs(context: ProtocolContext, *, smoke: bool) -> list[RunSpec]:
    if not smoke:
        return list(context.run_specs)
    smoke_config = context.config["smoke"]
    folds = set(map(int, smoke_config["folds"]))
    features = set(map(str, smoke_config["feature_sets"]))
    models = set(map(str, smoke_config["models"]))
    return [
        spec for spec in context.run_specs
        if spec.analysis in {"main_single", "main_multi"}
        and spec.fold in folds
        and spec.feature_set in features
        and spec.model in models
    ]


def _indices_for_run(context: ProtocolContext, spec: RunSpec) -> tuple[np.ndarray, np.ndarray]:
    mask = context.target_masks[spec.cohort_target_id]
    fold_values = context.frame["subject_id"].astype(str).map(context.fold_by_subject).to_numpy()
    test = mask & (fold_values == spec.fold)
    train = mask & np.isfinite(fold_values.astype(float)) & (fold_values != spec.fold)
    train_indices = np.flatnonzero(train)
    test_indices = np.flatnonzero(test)
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(f"Run {spec.run_id} has an empty train or test partition")
    overlap = np.intersect1d(
        context.frame.iloc[train_indices]["subject_id"].astype(str).unique(),
        context.frame.iloc[test_indices]["subject_id"].astype(str).unique(),
    )
    if len(overlap):
        raise RuntimeError(f"Subject leakage in {spec.run_id}: {overlap.tolist()}")
    return train_indices, test_indices


def _target_for_run(
    context: ProtocolContext, spec: RunSpec, indices: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    if spec.analysis == "main_multi":
        return context.target_values[spec.target_id][indices], list(PM_METRICS)
    values = context.target_values[spec.target_id][indices]
    return np.asarray(values, dtype=np.float32), [spec.target_name]


def _undefined_reason(metric: str, truth: np.ndarray, prediction: np.ndarray) -> str | None:
    if metric in {"r2", "pearson", "spearman"} and len(truth) < 2:
        return "fewer_than_two_windows"
    if metric in {"r2", "pearson", "spearman"} and np.ptp(truth) == 0:
        return "constant_target"
    if metric in {"pearson", "spearman"} and np.ptp(prediction) == 0:
        return "constant_prediction"
    return None


def participant_metric_rows(
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    subjects: np.ndarray,
    sources: np.ndarray,
    target_names: Sequence[str],
    train_target_std: np.ndarray,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    y_true = np.asarray(truth, dtype=float)
    y_pred = np.asarray(prediction, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    rows: list[dict[str, Any]] = []
    for subject in sorted(np.unique(subjects.astype(str))):
        subject_mask = subjects.astype(str) == subject
        subject_sources = sorted(np.unique(sources[subject_mask].astype(str)).tolist())
        for target_index, target_name in enumerate(target_names):
            current_truth = y_true[subject_mask, target_index]
            current_prediction = y_pred[subject_mask, target_index]
            metrics = MetricsCalculator.calculate_regression_metrics(
                current_truth, current_prediction
            )
            denominator = float(train_target_std[target_index])
            normalized_mae = (
                float(metrics["mae"] / denominator)
                if np.isfinite(denominator) and denominator > 0 else np.nan
            )
            row = {
                **metadata,
                "subject_id": subject,
                "source_membership": "+".join(subject_sources),
                "target_name": str(target_name),
                "n_windows": int(subject_mask.sum()),
                **{name: metrics[name] for name in METRIC_NAMES},
                "normalized_mae": normalized_mae,
                "train_target_std": denominator,
            }
            for metric in ("r2", "pearson", "spearman"):
                row[f"{metric}_undefined_reason"] = (
                    _undefined_reason(metric, current_truth, current_prediction)
                    if not np.isfinite(metrics[metric]) else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _window_metric_rows(
    truth: np.ndarray,
    prediction: np.ndarray,
    target_names: Sequence[str],
    metadata: Mapping[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    y_true = np.asarray(truth, dtype=float)
    y_pred = np.asarray(prediction, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    rows = []
    for index, target_name in enumerate(target_names):
        metrics = MetricsCalculator.calculate_regression_metrics(
            y_true[:, index], y_pred[:, index]
        )
        rows.append(
            {
                **metadata,
                "source": source,
                "target_name": target_name,
                "n_windows": len(y_true),
                **{name: metrics[name] for name in METRIC_NAMES},
            }
        )
    return rows


def _run_directory(context: ProtocolContext, spec: RunSpec) -> Path:
    section = {
        "main_single": "single_target",
        "main_multi": "multioutput",
        "paired_single": "paired_single_vs_multioutput",
    }[spec.analysis]
    return context.output_dir / section / spec.run_id


def execute_run(context: ProtocolContext, spec: RunSpec) -> dict[str, Any]:
    run_dir = _run_directory(context, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_idx, test_idx = _indices_for_run(context, spec)
    frame = context.frame
    names = context.feature_names[spec.feature_set]
    X_train = context.features[spec.feature_set][train_idx]
    X_test = context.features[spec.feature_set][test_idx]
    y_train, target_names = _target_for_run(context, spec, train_idx)
    y_test, _ = _target_for_run(context, spec, test_idx)
    sample_train = frame.iloc[train_idx]["sample_id"].to_numpy()
    sample_test = frame.iloc[test_idx]["sample_id"].to_numpy()
    split_manifest = {
        **context.folds[spec.fold],
        "target_id": spec.target_id,
        "cohort_target_id": spec.cohort_target_id,
        "train_sample_count": len(train_idx),
        "test_sample_count": len(test_idx),
        "train_sample_hash": stable_hash(sample_train.tolist()),
        "test_sample_hash": stable_hash(sample_test.tolist()),
        "reference_predictions_sha256": context.preregistration["reference_predictions_sha256"],
    }
    split_manifest["target_specific_split_hash"] = stable_hash(split_manifest)
    write_json(run_dir / "run_specification.json", {
        **spec.semantic_payload,
        "run_id": spec.run_id,
        "specification_hash": spec.specification_hash,
        "protocol_hash": context.preregistration["protocol_hash"],
    })
    write_json(run_dir / "split_manifest.json", split_manifest)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imputed = np.asarray(imputer.fit_transform(X_train), dtype=np.float32)
    X_test_imputed = np.asarray(imputer.transform(X_test), dtype=np.float32)
    if not np.isfinite(X_train_imputed).all() or not np.isfinite(X_test_imputed).all():
        raise ValueError("Train-only median imputation produced non-finite values")
    imputer_state = {
        "used": True,
        "strategy": "median",
        "scope": "outer_train_only",
        "n_fit_samples": len(X_train),
        "statistics": np.asarray(imputer.statistics_, dtype=float).tolist(),
        "train_missing_values": int(np.isnan(X_train).sum()),
        "test_missing_values": int(np.isnan(X_test).sum()),
    }
    if spec.scaling == "standard":
        preprocessor = FeaturePreprocessor(
            {"strategy": "standard"}, feature_names=names
        ).fit(X_train_imputed)
        X_train_model = preprocessor.transform(X_train_imputed)
        X_test_model = preprocessor.transform(X_test_imputed)
        scaler_state: dict[str, Any] = preprocessor.to_state()
        # FeaturePreprocessor is shared with adapters whose fit partition is
        # called inner-train.  In this classical baseline there is no inner
        # validation: the corresponding fit partition is the outer-train set.
        scaler_state["scope"] = "outer_train_only"
    elif spec.scaling == "none":
        X_train_model = X_train_imputed
        X_test_model = X_test_imputed
        scaler_state = {
            "strategy": "none",
            "scope": "not_applicable",
            "train_only": True,
            "n_fit_samples": 0,
        }
    else:
        raise ValueError(f"Unsupported scaling strategy {spec.scaling}")
    feature_manifest = {
        "feature_set": spec.feature_set,
        "n_features": len(names),
        "feature_names": list(names),
        "feature_list_sha256": feature_list_sha256(list(names)),
        "train_feature_hash": array_sha256(X_train_model, sample_train),
        "test_feature_hash": array_sha256(X_test_model, sample_test),
        "scaler": scaler_state,
        "imputer": imputer_state,
    }
    write_json(run_dir / "feature_manifest.json", feature_manifest)
    train_2d = y_train[:, None] if np.asarray(y_train).ndim == 1 else np.asarray(y_train)
    train_std = np.std(train_2d.astype(float), axis=0, ddof=0)
    target_manifest = {
        "target_id": spec.target_id,
        "cohort_target_id": spec.cohort_target_id,
        "target_names": target_names,
        "output_dim": len(target_names),
        "train_target_std": train_std.tolist(),
        "train_target_hash": array_sha256(np.asarray(y_train), sample_train),
        "test_target_hash": array_sha256(np.asarray(y_test), sample_test),
        "missing_value_policy": "target_complete_cases_inside_fixed_fold",
    }
    write_json(run_dir / "target_manifest.json", target_manifest)
    params = dict(spec.params)
    if spec.model in {"random_forest", "hist_gradient_boosting"}:
        params["random_state"] = spec.seed
    model = build_model(
        spec.model,
        "regression",
        input_shape=(X_train_model.shape[1],),
        num_outputs=len(target_names),
        params=params,
    )
    started = time.perf_counter()
    model.fit(X_train_model, y_train)
    prediction = np.asarray(model.predict(X_test_model), dtype=np.float32)
    training_seconds = float(time.perf_counter() - started)
    if prediction.shape != np.asarray(y_test).shape:
        raise ValueError(
            f"Prediction shape mismatch: {prediction.shape} != {np.asarray(y_test).shape}"
        )
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Model produced non-finite predictions")
    metadata = {
        "run_id": spec.run_id,
        "analysis": spec.analysis,
        "target_id": spec.target_id,
        "feature_set": spec.feature_set,
        "model": spec.model,
        "seed": spec.seed,
        "fold": spec.fold,
    }
    predictions = pd.DataFrame(
        {
            **metadata,
            "sample_id": sample_test,
            "subject_id": frame.iloc[test_idx]["subject_id"].astype(str).to_numpy(),
            "record_id": frame.iloc[test_idx]["record_id"].astype(str).to_numpy(),
            "record_group_id": frame.iloc[test_idx]["record_group_id"].astype(str).to_numpy(),
            "source": frame.iloc[test_idx]["source"].astype(str).to_numpy(),
        }
    )
    if len(target_names) == 1:
        predictions["y_true"] = np.asarray(y_test)
        predictions["y_pred"] = prediction
    else:
        for index, target_name in enumerate(target_names):
            predictions[f"y_true_{target_name}"] = np.asarray(y_test)[:, index]
            predictions[f"y_pred_{target_name}"] = prediction[:, index]
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    participant = participant_metric_rows(
        truth=np.asarray(y_test),
        prediction=prediction,
        subjects=predictions["subject_id"].to_numpy(),
        sources=predictions["source"].to_numpy(),
        target_names=target_names,
        train_target_std=train_std,
        metadata=metadata,
    )
    write_csv(run_dir / "participant_metrics.csv", participant)
    window_rows = _window_metric_rows(
        np.asarray(y_test), prediction, target_names, metadata, "overall"
    )
    source_rows = [
        {**row, "aggregation_level": "window", "subject_id": None}
        for row in window_rows
    ]
    overall_participant = participant.copy()
    overall_participant["source"] = "overall"
    overall_participant["aggregation_level"] = "participant"
    source_frames = [overall_participant]
    for source in ("gpn_data", "Old_EEG"):
        source_mask = predictions["source"].eq(source).to_numpy()
        if source_mask.any():
            source_rows.extend(
                {
                    **row,
                    "aggregation_level": "window",
                    "subject_id": None,
                }
                for row in _window_metric_rows(
                    np.asarray(y_test)[source_mask],
                    prediction[source_mask],
                    target_names,
                    metadata,
                    source,
                )
            )
            source_participant = participant_metric_rows(
                truth=np.asarray(y_test)[source_mask],
                prediction=prediction[source_mask],
                subjects=predictions.loc[source_mask, "subject_id"].to_numpy(),
                sources=predictions.loc[source_mask, "source"].to_numpy(),
                target_names=target_names,
                train_target_std=train_std,
                metadata=metadata,
            )
            source_participant["source"] = source
            source_participant["aggregation_level"] = "participant"
            source_frames.append(source_participant)
    source_frame = pd.concat(
        [pd.DataFrame(source_rows), *source_frames], ignore_index=True, sort=False
    )
    write_csv(run_dir / "source_metrics.csv", source_frame)
    write_json(run_dir / "window_metrics.json", {"rows": window_rows})
    error_rows = []
    for row in participant.to_dict("records"):
        for metric in ("r2", "pearson", "spearman"):
            if not np.isfinite(float(row[metric])):
                error_rows.append(
                    {
                        **metadata,
                        "subject_id": row["subject_id"],
                        "target_name": row["target_name"],
                        "metric": metric,
                        "reason": row[f"{metric}_undefined_reason"],
                    }
                )
    write_csv(run_dir / "errors.csv", pd.DataFrame(error_rows, columns=[
        *metadata.keys(), "subject_id", "target_name", "metric", "reason"
    ]))
    training_manifest = {
        "model_factory": "model_zoo.build_model",
        "model": spec.model,
        "params": params,
        "seed": spec.seed,
        "training_seconds": training_seconds,
        "outer_test_used_for_fit_or_selection": False,
        "inner_validation_required": False,
        "optimizer": None,
        "device": "cpu_sklearn",
    }
    write_json(run_dir / "training_manifest.json", training_manifest)
    macro = {
        metric: float(participant[metric].mean(skipna=True))
        for metric in (*METRIC_NAMES, "normalized_mae")
    }
    summary = {
        **metadata,
        "status": "complete",
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_participants": int(participant["subject_id"].nunique()),
        "participant_macro": macro,
        "undefined": {
            metric: int(participant[metric].isna().sum())
            for metric in ("r2", "pearson", "spearman")
        },
        "training_seconds": training_seconds,
        "artifacts": {
            name: name for name in (
                "run_specification.json", "split_manifest.json", "feature_manifest.json",
                "target_manifest.json", "training_manifest.json", "predictions.parquet",
                "participant_metrics.csv", "window_metrics.json", "source_metrics.csv",
                "errors.csv",
            )
        },
    }
    write_json(run_dir / "run_summary.json", summary)
    return summary


def _read_complete_frames(
    context: ProtocolContext, registry: Mapping[str, Any], filename: str
) -> list[pd.DataFrame]:
    specs = {spec.run_id: spec for spec in context.run_specs}
    frames = []
    for run_id, row in registry["runs"].items():
        if row["status"] != "complete" or run_id not in specs:
            continue
        path = _run_directory(context, specs[run_id]) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Complete run lacks {filename}: {run_id}")
        frames.append(pd.read_csv(path))
    return frames


def _metric_aggregation(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, partition in frame.groupby(list(groups), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["rows"] = len(partition)
        row["participants"] = int(partition["subject_id"].nunique()) if "subject_id" in partition else None
        for metric in (*METRIC_NAMES, "normalized_mae"):
            values = pd.to_numeric(partition[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
            row[f"{metric}_valid"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _audit_completed_runs(
    context: ProtocolContext, registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate persisted split, feature, prediction, and pairing contracts."""
    specs = {spec.run_id: spec for spec in context.run_specs}
    required = {
        "run_specification.json",
        "split_manifest.json",
        "feature_manifest.json",
        "target_manifest.json",
        "training_manifest.json",
        "predictions.parquet",
        "participant_metrics.csv",
        "window_metrics.json",
        "source_metrics.csv",
        "run_summary.json",
        "errors.csv",
    }
    violations: dict[str, list[str]] = {
        "missing_artifacts": [],
        "subject_overlap": [],
        "scaler_scope": [],
        "imputer_scope": [],
        "feature_contract": [],
        "prediction_rows": [],
        "duplicate_prediction_sample_id": [],
        "paired_sample_or_feature_identity": [],
    }
    multi_identity: dict[tuple[str, str, int, int], tuple[str, str, str]] = {}
    paired_identity: list[tuple[RunSpec, tuple[str, str, str]]] = []
    complete = 0
    for run_id, row in registry["runs"].items():
        if row["status"] != "complete" or run_id not in specs:
            continue
        complete += 1
        spec = specs[run_id]
        run_dir = _run_directory(context, spec)
        missing = sorted(name for name in required if not (run_dir / name).is_file())
        if missing:
            violations["missing_artifacts"].append(f"{run_id}: {missing}")
            continue
        split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
        feature = json.loads((run_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        if split.get("subject_overlap"):
            violations["subject_overlap"].append(run_id)
        scaler_scope = feature.get("scaler", {}).get("scope")
        if scaler_scope not in {"outer_train_only", "not_applicable"}:
            violations["scaler_scope"].append(f"{run_id}: {scaler_scope}")
        if feature.get("imputer", {}).get("scope") != "outer_train_only":
            violations["imputer_scope"].append(run_id)
        names = feature.get("feature_names", [])
        if (
            len(names) != int(feature.get("n_features", -1))
            or feature_list_sha256(names) != feature.get("feature_list_sha256")
            or any(str(name).startswith(("PM.", "target_", "label_")) for name in names)
        ):
            violations["feature_contract"].append(run_id)
        predictions = pd.read_parquet(run_dir / "predictions.parquet", columns=["sample_id"])
        if len(predictions) != int(split.get("test_sample_count", -1)):
            violations["prediction_rows"].append(run_id)
        if predictions["sample_id"].duplicated().any():
            violations["duplicate_prediction_sample_id"].append(run_id)
        identity = (
            str(split.get("test_sample_hash")),
            str(feature.get("train_feature_hash")),
            str(feature.get("test_feature_hash")),
        )
        pairing_key = (spec.feature_set, spec.model, spec.seed, spec.fold)
        if spec.analysis == "main_multi" and spec.model in {"ridge", "random_forest"}:
            multi_identity[pairing_key] = identity
        elif spec.analysis == "paired_single":
            paired_identity.append((spec, identity))
    for spec, identity in paired_identity:
        reference = multi_identity.get((spec.feature_set, spec.model, spec.seed, spec.fold))
        if reference is None or reference != identity:
            violations["paired_sample_or_feature_identity"].append(spec.run_id)
    failed = sum(str(row["status"]).startswith("failed") for row in registry["runs"].values())
    counts = {name: len(items) for name, items in violations.items()}
    return {
        "complete_run_count": complete,
        "failed_run_count": failed,
        "fold_assignments_match_reference": counts["subject_overlap"] == 0,
        "target_columns_in_features": counts["feature_contract"] != 0,
        "scaler_fit_scope": "outer_train_only",
        "violation_counts": counts,
        "violation_examples": {
            name: items[:10] for name, items in violations.items() if items
        },
        "clean": failed == 0 and all(count == 0 for count in counts.values()),
    }


def aggregate_results(context: ProtocolContext, registry: Mapping[str, Any]) -> dict[str, Any]:
    aggregation = context.output_dir / "aggregation"
    aggregation.mkdir(parents=True, exist_ok=True)
    participant_frames = _read_complete_frames(context, registry, "participant_metrics.csv")
    source_frames = _read_complete_frames(context, registry, "source_metrics.csv")
    participant = pd.concat(participant_frames, ignore_index=True) if participant_frames else pd.DataFrame()
    source_details = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    write_csv(context.output_dir / "participant_metrics.csv", participant)
    write_csv(aggregation / "participant_metrics.csv", participant)
    if participant.empty:
        summary = {"status": "no_completed_runs", "complete_runs": 0}
        write_json(context.output_dir / "benchmark_summary.json", summary)
        return summary
    fold = _metric_aggregation(
        participant,
        ["analysis", "target_id", "target_name", "feature_set", "model", "seed", "fold"],
    )
    seed = _metric_aggregation(
        participant,
        ["analysis", "target_id", "target_name", "feature_set", "model", "seed"],
    )
    single = _metric_aggregation(
        participant.loc[participant["analysis"].eq("main_single")],
        ["target_id", "target_name", "feature_set", "model"],
    )
    multi = _metric_aggregation(
        participant.loc[participant["analysis"].eq("main_multi")],
        ["target_id", "target_name", "feature_set", "model"],
    )
    source_participant = source_details.loc[
        source_details.get("aggregation_level", pd.Series(index=source_details.index, dtype=str))
        .eq("participant")
    ].copy()
    source_summary = (
        _metric_aggregation(
            source_participant,
            [
                "analysis", "target_id", "target_name", "feature_set",
                "model", "seed", "fold", "source",
            ],
        )
        if not source_participant.empty else pd.DataFrame()
    )
    summary_frames = {
        "fold_metrics.csv": fold,
        "seed_metrics.csv": seed,
        "single_target_results.csv": single,
        "multioutput_results.csv": multi,
        "source_metrics.csv": source_summary,
    }
    for filename, frame in summary_frames.items():
        write_csv(context.output_dir / filename, frame)
        write_csv(aggregation / filename, frame)
    feature_comparison = _feature_comparisons(participant)
    single_multi = _single_multi_comparisons(participant)
    dummy = _dummy_improvements(participant)
    undefined = _undefined_audit(participant)
    write_csv(context.output_dir / "feature_set_comparison.csv", feature_comparison)
    write_csv(context.output_dir / "single_vs_multioutput_comparison.csv", single_multi)
    write_csv(context.output_dir / "dummy_improvement.csv", dummy)
    write_csv(context.output_dir / "undefined_metrics_audit.csv", undefined)
    write_csv(aggregation / "feature_set_comparison.csv", feature_comparison)
    write_csv(aggregation / "single_vs_multioutput_comparison.csv", single_multi)
    write_csv(aggregation / "dummy_improvement.csv", dummy)
    write_csv(aggregation / "undefined_metrics_audit.csv", undefined)
    error_frames = _read_complete_frames(context, registry, "errors.csv")
    write_csv(
        context.output_dir / "errors.csv",
        pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(),
    )
    complete = sum(row["status"] == "complete" for row in registry["runs"].values())
    failed = sum(str(row["status"]).startswith("failed") for row in registry["runs"].values())
    leakage = _audit_completed_runs(context, registry)
    write_json(context.output_dir / "global_leakage_audit.json", leakage)
    total = len(context.run_specs)
    required_models = {"dummy_mean", "ridge", "random_forest"}
    complete_models = {
        spec.model for spec in context.run_specs
        if registry["runs"][spec.run_id]["status"] == "complete"
    }
    if complete == total and failed == 0:
        status = "pm_feature_baseline_complete"
    elif required_models.issubset(complete_models) and failed == 0:
        status = "pm_feature_baseline_partially_complete"
    else:
        status = "running_or_incomplete"
    summary = {
        "experiment_id": context.config["experiment_id"],
        "status": status,
        "protocol_hash": context.preregistration["protocol_hash"],
        "run_matrix_hash": context.preregistration["run_matrix_hash"],
        "planned_runs": total,
        "complete_runs": complete,
        "failed_runs": failed,
        "pending_runs": total - complete - failed,
        "targets": context.preregistration["target_ids"],
        "feature_sets": list(FEATURE_SET_ORDER),
        "leakage_audit_clean": leakage["clean"],
    }
    write_json(context.output_dir / "benchmark_summary.json", summary)
    write_csv(context.output_dir / "run_registry.csv", pd.DataFrame([
        {"run_id": run_id, **row} for run_id, row in registry["runs"].items()
    ]))
    write_json(context.output_dir / "run_registry.json", registry)
    report = _render_full_report(
        context,
        summary,
        single,
        multi,
        feature_comparison,
        single_multi,
        source_summary,
        undefined,
    )
    runtime_report = context.output_dir / "reports" / "benchmark_report.md"
    runtime_report.parent.mkdir(parents=True, exist_ok=True)
    runtime_report.write_text(report, encoding="utf-8")
    canonical_output = _repo_path(
        "benchmark_results/pm_all_targets_feature_baseline_v1"
    ).resolve()
    if context.output_dir.resolve() == canonical_output:
        tracked_report = (
            REPO_ROOT / "reports" / "integration" /
            "pm_all_targets_feature_baseline.md"
        )
        tracked_report.write_text(report, encoding="utf-8")
    return summary


def _feature_comparisons(participant: pd.DataFrame) -> pd.DataFrame:
    base = participant.loc[participant["analysis"].eq("main_single")].copy()
    keys = ["target_id", "target_name", "model", "seed", "fold", "subject_id"]
    rows = []
    for left, right in (("eeg", "pow"), ("eeg", "eeg_pow"), ("pow", "eeg_pow")):
        a = base.loc[base["feature_set"].eq(left), [*keys, *METRIC_NAMES, "normalized_mae"]]
        b = base.loc[base["feature_set"].eq(right), [*keys, *METRIC_NAMES, "normalized_mae"]]
        merged = a.merge(b, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
        for metric in (*METRIC_NAMES, "normalized_mae"):
            merged[f"{metric}_difference"] = merged[f"{metric}_right"] - merged[f"{metric}_left"]
        merged["comparison"] = f"{right}_minus_{left}"
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _single_multi_comparisons(participant: pd.DataFrame) -> pd.DataFrame:
    single = participant.loc[participant["analysis"].eq("paired_single")].copy()
    multi = participant.loc[participant["analysis"].eq("main_multi")].copy()
    keys = ["target_name", "feature_set", "model", "seed", "fold", "subject_id"]
    columns = [*keys, *METRIC_NAMES, "normalized_mae"]
    if single.empty or multi.empty:
        return pd.DataFrame()
    merged = single[columns].merge(
        multi[columns], on=keys, suffixes=("_single", "_multi"), validate="one_to_one"
    )
    for metric in (*METRIC_NAMES, "normalized_mae"):
        merged[f"{metric}_multi_minus_single"] = (
            merged[f"{metric}_multi"] - merged[f"{metric}_single"]
        )
    return merged


def _dummy_improvements(participant: pd.DataFrame) -> pd.DataFrame:
    base = participant.loc[participant["analysis"].eq("main_single")].copy()
    dummy = base.loc[base["model"].eq("dummy_mean")]
    models = base.loc[~base["model"].eq("dummy_mean")]
    keys = ["target_id", "target_name", "feature_set", "fold", "subject_id"]
    merged = models.merge(
        dummy[[*keys, *METRIC_NAMES, "normalized_mae"]],
        on=keys, suffixes=("", "_dummy"), validate="many_to_one",
    )
    for metric in (*METRIC_NAMES, "normalized_mae"):
        merged[f"{metric}_difference_vs_dummy"] = merged[metric] - merged[f"{metric}_dummy"]
    return merged


def _undefined_audit(participant: pd.DataFrame) -> pd.DataFrame:
    groups = ["analysis", "target_id", "target_name", "feature_set", "model", "seed", "fold"]
    rows = []
    for keys, part in participant.groupby(groups, sort=True):
        row = dict(zip(groups, keys))
        row["participants"] = int(len(part))
        for metric in ("r2", "pearson", "spearman"):
            row[f"undefined_{metric}"] = int(part[metric].isna().sum())
            row[f"undefined_{metric}_fraction"] = float(part[metric].isna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(
    frame: pd.DataFrame, columns: Sequence[str], *, max_rows: int = 100
) -> str:
    if frame.empty:
        return "_No completed rows._"
    available = [column for column in columns if column in frame.columns]
    view = frame.loc[:, available].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
        )
    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join("---" for _ in available) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _render_report(
    context: ProtocolContext,
    summary: Mapping[str, Any],
    single: pd.DataFrame,
    multi: pd.DataFrame,
    feature_comparison: pd.DataFrame,
    single_multi: pd.DataFrame,
    undefined: pd.DataFrame,
) -> str:
    cohort = context.cohort_summary.groupby("target_id")["test_windows"].sum().to_dict()
    return f"""# Canonical seven-PM feature baseline

## Status

`{summary['status']}` on branch `integration/benchmark-unification`, commit `{context.preregistration['git_commit']}`.

## Protocol and preregistration

- Experiment: `{context.config['experiment_id']}`
- Protocol hash: `{context.preregistration['protocol_hash']}`
- Run-matrix hash: `{context.preregistration['run_matrix_hash']}`
- Runs: {summary['complete_runs']} complete / {summary['planned_runs']} planned; {summary['failed_runs']} failed.
- Five immutable outer folds by `subject_id`; reference assignments match the existing label-Q5 benchmark.

## Targets and cohorts

Seven continuous PM targets and the fixed-order seven-output target are included. Cohort window counts: `{json.dumps(cohort, sort_keys=True)}`. Target-specific complete cases are applied inside fixed folds.

## Features

EEG=168, POW=280, EEG+POW=448. Device POW columns are stored engineered power features, not spectra recomputed from raw EEG. PM, target, label and identity columns are excluded.

## Models and seeds

Dummy mean, Ridge, Random Forest and single-output HistGradientBoosting are fixed baselines. Random Forest uses seeds 42, 123 and 2026; deterministic models run once. LightGBM was not installed and was not added.

## Leakage audit

Subject overlap is zero. Median imputation and Ridge scaling are fitted only on outer-train. Outer-test is not used for fitting, early stopping, selection or target statistics.

## Results

Single-target aggregate rows: {len(single)}. Multi-output aggregate rows: {len(multi)}. Feature-set paired rows: {len(feature_comparison)}. Paired single-versus-multioutput rows: {len(single_multi)}.

Detailed participant-, fold-, seed-, source-, feature-comparison-, dummy-improvement- and undefined-metric tables are stored under `benchmark_results/pm_all_targets_feature_baseline_v1/`.

## Participant and source interpretation

Participant metrics use equal subject weights. Undefined R²/Pearson/Spearman values remain missing with explicit reasons; they are never replaced by zero. `gpn_data` and `Old_EEG` slices are descriptive and are not treated as independent confirmation datasets.

## Limitations

This is a classical engineered-feature baseline, not a raw-EEG, personalization, FOMAML or DANN experiment. Different PM targets have different complete-case cohorts; direct single-versus-multioutput comparisons therefore use only the identical seven-output cohort. Negative R² values are retained.
"""


def _render_full_report(
    context: ProtocolContext,
    summary: Mapping[str, Any],
    single: pd.DataFrame,
    multi: pd.DataFrame,
    feature_comparison: pd.DataFrame,
    single_multi: pd.DataFrame,
    source_summary: pd.DataFrame,
    undefined: pd.DataFrame,
) -> str:
    cohort = context.cohort_summary.groupby("target_id")["test_windows"].sum().to_dict()
    metric_columns = [
        "target_name", "feature_set", "model", "mae_mean", "rmse_mean",
        "r2_mean", "pearson_mean", "spearman_mean", "normalized_mae_mean",
        "participants",
    ]
    dummy_table = (
        single.loc[
            single["model"].eq("dummy_mean") & single["feature_set"].eq("eeg_pow")
        ] if not single.empty else pd.DataFrame()
    )
    single_table = (
        single.loc[
            single["feature_set"].eq("eeg_pow") & ~single["model"].eq("dummy_mean")
        ] if not single.empty else pd.DataFrame()
    )
    multi_table = (
        multi.loc[multi["feature_set"].eq("eeg_pow")]
        if not multi.empty else pd.DataFrame()
    )
    feature_summary = pd.DataFrame()
    if not feature_comparison.empty:
        feature_summary = feature_comparison.groupby(
            ["model", "comparison"], as_index=False, sort=True
        )[[
            "mae_difference", "r2_difference", "pearson_difference",
            "spearman_difference", "normalized_mae_difference",
        ]].mean()
    single_multi_summary = pd.DataFrame()
    if not single_multi.empty:
        single_multi_summary = single_multi.groupby(
            ["target_name", "feature_set", "model"], as_index=False, sort=True
        )[[
            "mae_multi_minus_single", "r2_multi_minus_single",
            "pearson_multi_minus_single", "spearman_multi_minus_single",
            "normalized_mae_multi_minus_single",
        ]].mean()
    source_table = pd.DataFrame()
    if not source_summary.empty:
        source_table = source_summary.loc[
            source_summary["analysis"].eq("main_single")
            & source_summary["feature_set"].eq("eeg_pow")
        ].groupby(["model", "source"], as_index=False, sort=True)[
            ["mae_mean", "r2_mean", "pearson_mean", "spearman_mean"]
        ].mean()
    undefined_totals = pd.DataFrame()
    if not undefined.empty:
        undefined_totals = undefined.groupby("model", as_index=False, sort=True)[
            ["participants", "undefined_r2", "undefined_pearson", "undefined_spearman"]
        ].sum()
    return f"""# Canonical seven-PM feature baseline

## Status

`{summary['status']}` on branch `integration/benchmark-unification`, commit `{context.preregistration['git_commit']}`.

## Protocol and preregistration

- Experiment: `{context.config['experiment_id']}`
- Protocol hash: `{context.preregistration['protocol_hash']}`
- Run-matrix hash: `{context.preregistration['run_matrix_hash']}`
- Runs: {summary['complete_runs']} complete / {summary['planned_runs']} planned; {summary['failed_runs']} failed.
- Five immutable outer folds by `subject_id`; reference assignments match the existing label-Q5 benchmark.

## Targets and cohorts

Seven continuous PM targets and the fixed-order seven-output target are included. Cohort window counts: `{json.dumps(cohort, sort_keys=True)}`. Target-specific complete cases are applied inside fixed folds.

## Features, models and seeds

EEG=168, POW=280, EEG+POW=448. Device POW columns are stored engineered power features, not spectra recomputed from raw EEG. PM, target, label and identity columns are excluded. Dummy mean, Ridge, Random Forest and single-output HistGradientBoosting are fixed baselines. Random Forest uses seeds 42, 123 and 2026; deterministic models run once. LightGBM was not installed and was not added.

## Leakage audit

Subject overlap is zero. Median imputation and Ridge scaling are fitted only on outer-train. Outer-test is not used for fitting, early stopping, selection or target statistics. The machine-readable audit is `global_leakage_audit.json`.

## Dummy mean, EEG+POW

{_markdown_table(dummy_table, metric_columns)}

## Single-output PM regressions, EEG+POW

Random Forest rows combine all three preregistered seeds.

{_markdown_table(single_table, metric_columns)}

## Seven-output regression, EEG+POW

{_markdown_table(multi_table, metric_columns)}

## Paired feature-view differences

Differences are right minus left on identical participant/fold/seed units. Negative MAE and positive correlations indicate improvement.

{_markdown_table(feature_summary, ["model", "comparison", "mae_difference", "r2_difference", "pearson_difference", "spearman_difference", "normalized_mae_difference"])}

## Paired multi-output minus single-output

These rows use only the identical seven-target complete-case cohort.

{_markdown_table(single_multi_summary, ["target_name", "feature_set", "model", "mae_multi_minus_single", "r2_multi_minus_single", "pearson_multi_minus_single", "spearman_multi_minus_single", "normalized_mae_multi_minus_single"])}

## Descriptive source slices

{_markdown_table(source_table, ["model", "source", "mae_mean", "r2_mean", "pearson_mean", "spearman_mean"])}

Source slices are descriptive and are not treated as independent confirmation datasets.

## Participant variability and undefined metrics

Participant metrics use equal subject weights. The standard deviations in the result tables capture participant variability. Undefined R-squared/Pearson/Spearman values remain missing with explicit reasons; they are never replaced by zero.

{_markdown_table(undefined_totals, ["model", "participants", "undefined_r2", "undefined_pearson", "undefined_spearman"])}

Detailed participant-, fold-, seed-, source-, comparison-, dummy-improvement- and undefined-metric tables are stored under `benchmark_results/pm_all_targets_feature_baseline_v1/`.

## Limitations and final status

This is a classical engineered-feature baseline, not a raw-EEG, personalization, FOMAML or DANN experiment. Different PM targets have different complete-case cohorts; direct single-versus-multioutput comparisons therefore use only the identical seven-output cohort. Negative R-squared values are retained. No target is declared solved from relative ranking alone.

Participant-level R-squared is unstable for subjects whose within-subject target variance is near zero; it must be interpreted together with MAE, normalized MAE and correlations. Ridge emitted ill-conditioned-matrix warnings and produced poor finite predictions despite train-only scaling, so those negative results are retained rather than hidden or used to retune the preregistered model.
"""


def run_baseline(
    config_path: str | Path,
    *,
    plan_only: bool = False,
    smoke: bool = False,
    resume: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    context = prepare_protocol(config_path)
    _ensure_preregistration(context)
    registry = _load_registry(context, resume=resume or plan_only)
    if plan_only:
        return {
            "status": "protocol_audit_complete",
            "protocol_hash": context.preregistration["protocol_hash"],
            "run_matrix_hash": context.preregistration["run_matrix_hash"],
            "planned_runs": len(context.run_specs),
            "cohorts": context.cohort_summary.groupby("target_id")["test_windows"].sum().to_dict(),
        }
    selected = _selected_specs(context, smoke=smoke)
    if max_runs is not None:
        selected = selected[: int(max_runs)]
    for spec in selected:
        row = registry["runs"][spec.run_id]
        if row["specification_hash"] != spec.specification_hash:
            raise ValueError(f"Run specification changed for {spec.run_id}")
        if row["status"] == "complete":
            if resume:
                continue
            raise ValueError(f"Run already complete: {spec.run_id}")
        if row["status"] not in {"pending", "failed_technical", "running"}:
            continue
        attempt = {
            "attempt": len(row["attempts"]) + 1,
            "started_unix": time.time(),
            "previous_status": row["status"],
        }
        row["status"] = "running"
        row["attempts"].append(attempt)
        _save_registry(context, registry)
        try:
            result = execute_run(context, spec)
        except (FloatingPointError, ArithmeticError) as exc:
            row["status"] = "failed_numerical"
            attempt.update({"finished_unix": time.time(), "error": repr(exc)})
            _save_registry(context, registry)
            continue
        except Exception as exc:
            row["status"] = "failed_technical"
            attempt.update({"finished_unix": time.time(), "error": repr(exc)})
            _save_registry(context, registry)
            continue
        row["status"] = "complete"
        row["run_summary"] = _relative_path(_run_directory(context, spec) / "run_summary.json")
        attempt.update({"finished_unix": time.time(), "result": "complete"})
        _save_registry(context, registry)
    return aggregate_results(context, registry)
