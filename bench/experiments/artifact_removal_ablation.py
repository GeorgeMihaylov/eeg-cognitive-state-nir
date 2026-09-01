"""Leakage-safe raw EEG artifact-removal ablation orchestration."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from bench.datasets.datasets_registry import get_dataset
from bench.preprocessing.fold_artifact_transform import (
    ARTIFACT_VARIANTS,
    ArtifactTransformedRawView,
    FoldArtifactConfig,
    FoldArtifactTransform,
    stable_hash,
)
from bench.tasks.target_registry import PM_METRICS
from cogstate.preprocessing.artifact_removal import FasterConfig, IcaConfig
from cogstate.model_zoo import build_model


REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    document = yaml.safe_load(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Artifact-removal config must be a mapping")
    required = {"experiment_id", "dataset", "targets", "variants", "ica", "model", "evaluation", "smoke", "output_dir"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Artifact-removal config is missing sections: {missing}")
    if tuple(document["targets"]) != PM_METRICS:
        raise ValueError(f"targets must contain all seven PM in order: {PM_METRICS}")
    if tuple(document["variants"]) != ARTIFACT_VARIANTS:
        raise ValueError(f"variants must equal {ARTIFACT_VARIANTS}")
    folds = tuple(map(int, document["evaluation"]["folds"]))
    if folds != (1, 2, 3, 4, 5):
        raise ValueError("Artifact ablation must use fixed folds [1, 2, 3, 4, 5]")
    if document["model"]["name"] != "torch_shallow_convnet":
        raise ValueError("Artifact ablation is locked to torch_shallow_convnet")
    return document


@dataclass(frozen=True)
class ArtifactRunSpec:
    metric: str
    variant: str
    fold: int
    seed: int = 42

    @property
    def run_id(self) -> str:
        return f"{self.metric}__{self.variant}__fold{self.fold:02d}__seed{self.seed}"

    @property
    def specification_hash(self) -> str:
        return stable_hash(asdict(self))


def build_run_matrix(config: Mapping[str, Any]) -> list[ArtifactRunSpec]:
    return [
        ArtifactRunSpec(metric=metric, variant=variant, fold=int(fold), seed=42)
        for metric in config["targets"]
        for variant in config["variants"]
        for fold in config["evaluation"]["folds"]
    ]


def _dataset_config(config: Mapping[str, Any], metric: str) -> dict[str, Any]:
    dataset = config["dataset"]
    return {
        "data_path": str(_repo_path(dataset["raw_window_index"])),
        "target_id": f"pm_{metric}_regression",
        "target_data_path": str(_repo_path(dataset["target_table"])),
        "dataset_mode": "raw_deduplicated_logical_records",
        "logical_recording_map_path": str(_repo_path(dataset["logical_recording_map"])),
        "raw_preprocessing": dict(dataset["raw_preprocessing"]),
    }


def _load_metric_data(config: Mapping[str, Any], metric: str) -> Any:
    dataset = get_dataset("emotiv_raw_eeg", _dataset_config(config, metric))
    data = dataset.load()
    if tuple(data.data.shape[1:]) != (1, 14, 2560):
        raise ValueError(f"Canonical raw input shape changed: {data.data.shape}")
    if "outer_fold" not in data.row_metadata or "record_group_id" not in data.row_metadata:
        raise ValueError("Raw data lack fixed outer_fold or record_group_id metadata")
    return data


def _q3_thresholds(values: np.ndarray) -> list[float]:
    thresholds = np.quantile(np.asarray(values, dtype=float), [1 / 3, 2 / 3]).tolist()
    if len(set(map(float, thresholds))) != 2:
        raise ValueError("Outer-train Q3 thresholds are not unique")
    return [float(value) for value in thresholds]


def _q3_labels(values: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    labels = np.searchsorted(np.asarray(thresholds, dtype=float), values, side="right")
    if np.any((labels < 0) | (labels > 2)):
        raise ValueError("Q3 transform produced labels outside [0, 2]")
    return labels.astype(np.int64)


def _balanced_subset(
    indices: np.ndarray,
    *,
    subjects: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    limit: int | None,
) -> np.ndarray:
    if limit is None or len(indices) <= limit:
        return indices
    frame = pd.DataFrame({
        "position": indices,
        "subject_id": subjects[indices].astype(str),
        "label": labels[indices].astype(int),
        "sample_id": sample_ids[indices].astype(str),
    }).sort_values(["subject_id", "label", "sample_id"], kind="mergesort")
    groups = max(1, frame.groupby(["subject_id", "label"], observed=True).ngroups)
    per_group = max(1, int(np.ceil(limit / groups)))
    chosen = frame.groupby(["subject_id", "label"], sort=True, observed=True).head(per_group)
    chosen = chosen.head(limit)
    if len(chosen) < limit:
        remaining = frame.loc[~frame["position"].isin(chosen["position"])]
        chosen = pd.concat([chosen, remaining.head(limit - len(chosen))])
    result = chosen.sort_values("sample_id", kind="mergesort")["position"].to_numpy(np.int64)
    if len(np.unique(labels[result])) != 3:
        raise ValueError("Smoke subset does not preserve all Q3 classes")
    return result


def _fold_indices(data: Any, fold: int) -> tuple[np.ndarray, np.ndarray]:
    fold_values = np.asarray(data.row_metadata["outer_fold"], dtype=int)
    train = np.flatnonzero(fold_values != fold)
    test = np.flatnonzero(fold_values == fold)
    train_subjects = set(np.asarray(data.subject_ids)[train].astype(str))
    test_subjects = set(np.asarray(data.subject_ids)[test].astype(str))
    overlap = sorted(train_subjects & test_subjects)
    if overlap:
        raise RuntimeError(f"Outer participant leakage in fold {fold}: {overlap}")
    return train, test


def protocol_plan(config_path: str | Path) -> dict[str, Any]:
    path = _repo_path(config_path)
    config = load_config(path)
    specs = build_run_matrix(config)
    data = _load_metric_data(config, PM_METRICS[0])
    reference_path = _repo_path(config["dataset"]["reference_fold_manifest"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_folds = reference["folds"]
    actual_subjects = set(np.asarray(data.subject_ids).astype(str))
    reference_subject_to_fold: dict[str, int] = {}
    for fold_key, fold_payload in reference_folds.items():
        for subject in fold_payload["test_subject_ids"]:
            if subject in reference_subject_to_fold:
                raise RuntimeError(f"Duplicate subject in reference folds: {subject}")
            reference_subject_to_fold[str(subject)] = int(fold_key)
    actual_subject_to_fold = {
        str(subject): int(fold)
        for subject, fold in zip(data.subject_ids, data.row_metadata["outer_fold"])
    }
    fold_mismatches = {
        subject: {
            "reference_fold": reference_subject_to_fold.get(subject),
            "raw_target_fold": actual_subject_to_fold.get(subject),
        }
        for subject in sorted(actual_subjects)
        if reference_subject_to_fold.get(subject) != actual_subject_to_fold.get(subject)
    }
    folds: dict[str, Any] = {}
    for fold in config["evaluation"]["folds"]:
        train, test = _fold_indices(data, int(fold))
        folds[str(fold)] = {
            "train_windows": int(len(train)),
            "test_windows": int(len(test)),
            "train_participants": sorted(np.unique(np.asarray(data.subject_ids)[train].astype(str)).tolist()),
            "test_participants": sorted(np.unique(np.asarray(data.subject_ids)[test].astype(str)).tolist()),
            "participant_overlap": [],
            "split_hash": stable_hash({
                "train_sample_ids": np.asarray(data.sample_ids)[train].astype(str).tolist(),
                "test_sample_ids": np.asarray(data.sample_ids)[test].astype(str).tolist(),
            }),
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "result_status": config.get("result_status", "baseline"),
        "analysis_role": "confirmatory_protocol_not_executed",
        "config_path": path.relative_to(REPO_ROOT).as_posix(),
        "targets": list(PM_METRICS),
        "target_contract": "outer_train_q3_low_medium_high",
        "variants": list(ARTIFACT_VARIANTS),
        "model": config["model"],
        "outer_folds": folds,
        "fixed_fold_reference": {
            "path": reference_path.relative_to(REPO_ROOT).as_posix(),
            "reference_subject_count": len(reference_subject_to_fold),
            "target_cohort_subject_count": len(actual_subjects),
            "target_cohort_missing_reference_subjects": sorted(
                set(reference_subject_to_fold) - actual_subjects
            ),
            "unexpected_target_cohort_subjects": sorted(
                actual_subjects - set(reference_subject_to_fold)
            ),
            "subject_fold_mismatches": fold_mismatches,
            "subject_fold_assignments_match": not fold_mismatches,
        },
        "run_count": len(specs),
        "expected_run_count": 140,
        "raw_cache_roots": data.metadata.get("cache_roots", []),
        "accepted_deduplicated_windows": int(data.n_samples),
        "input_shape": list(data.data.shape[1:]),
        "preprocessing_contract": {
            "base": "canonical_raw_cache_no_bandpass_no_notch_no_car",
            "faster": "apply_faster_per_window_mean_channel_interpolation_no_epoch_drop",
            "ica": "one_outer_train_calibrated_state_per_fold_and_variant",
            "normalization": "torch_adapter_inner_train_only_after_artifact_transform",
        },
        "calibration_strategy": {
            "selection": "deterministic_participant_balanced_outer_train",
            "max_windows": int(config["ica"]["calibration_max_windows"]),
            "estimated_peak_bytes_per_fold": int(
                config["ica"]["calibration_max_windows"] * 14 * 2560 * 8 * 3
            ),
        },
        "artifact_runtime": {
            **dict(config.get("artifact_runtime", {})),
            "cache_scope": "single_run_memory_only_not_persisted",
            "estimated_full_cohort_cache_bytes": int(
                data.n_samples * np.prod(data.data.shape[1:]) * 4
            ),
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    manifest["run_matrix_hash"] = stable_hash([asdict(spec) for spec in specs])
    manifest["protocol_hash"] = stable_hash(manifest)
    return manifest


def write_plan(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output = _repo_path(config["output_dir"])
    manifest = protocol_plan(config_path)
    specs = build_run_matrix(config)
    _write_json(output / "protocol_manifest.json", manifest)
    pd.DataFrame([
        {**asdict(spec), "run_id": spec.run_id, "specification_hash": spec.specification_hash}
        for spec in specs
    ]).to_csv(output / "run_matrix.csv", index=False, lineterminator="\n")
    smoke = config["smoke"]
    smoke_specs = [
        spec for spec in specs
        if spec.metric in set(smoke["targets"])
        and spec.fold in set(map(int, smoke["folds"]))
    ]
    pd.DataFrame([
        {**asdict(spec), "run_id": spec.run_id, "specification_hash": spec.specification_hash}
        for spec in smoke_specs
    ]).to_csv(output / "smoke_run_matrix.csv", index=False, lineterminator="\n")
    return manifest


def _transform_config(config: Mapping[str, Any], variant: str, *, smoke: bool) -> FoldArtifactConfig:
    faster = FasterConfig(**dict(config.get("faster", {})))
    ica_values = dict(config["ica"])
    maximum = int(
        config["smoke"].get("ica_calibration_max_windows", ica_values["calibration_max_windows"])
        if smoke else ica_values["calibration_max_windows"]
    )
    ica = IcaConfig(
        n_components=ica_values.get("n_components"),
        max_iter=int(ica_values["max_iter"]),
        random_state=int(ica_values["random_state"]),
        faster_config=faster,
    )
    return FoldArtifactConfig(
        variant=variant,
        sample_rate=256.0,
        calibration_max_windows=maximum,
        faster=faster,
        ica=ica,
    )


def execute_run(
    config_path: str | Path,
    spec: ArtifactRunSpec,
    *,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    profile = "smoke" if smoke else "full"
    run_dir = _repo_path(config["output_dir"]) / profile / spec.run_id
    summary_path = run_dir / "run_summary.json"
    if resume and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("specification_hash") == spec.specification_hash and existing.get("status") == "complete":
            return existing
    data = _load_metric_data(config, spec.metric)
    train_idx, test_idx = _fold_indices(data, spec.fold)
    continuous = np.asarray(data.labels, dtype=np.float64)
    thresholds = _q3_thresholds(continuous[train_idx])
    labels = _q3_labels(continuous, thresholds)
    if smoke:
        smoke_config = config["smoke"]
        train_idx = _balanced_subset(
            train_idx, subjects=np.asarray(data.subject_ids), labels=labels,
            sample_ids=np.asarray(data.sample_ids), limit=int(smoke_config["max_train_windows"]),
        )
        test_idx = _balanced_subset(
            test_idx, subjects=np.asarray(data.subject_ids), labels=labels,
            sample_ids=np.asarray(data.sample_ids), limit=int(smoke_config["max_test_windows"]),
        )
    base_train = data.data[train_idx]
    base_test = data.data[test_idx]
    transform = FoldArtifactTransform(
        _transform_config(config, spec.variant, smoke=smoke)
    )
    preprocessing_started = time.perf_counter()
    transform.fit(base_train, fold=spec.fold)
    calibration_fit_seconds = time.perf_counter() - preprocessing_started
    cache_transformed = bool(
        config.get("artifact_runtime", {}).get("cache_transformed_windows", True)
    ) and spec.variant != "raw"
    X_train = ArtifactTransformedRawView(
        base_train, transform, cache_transformed_windows=cache_transformed
    )
    X_test = ArtifactTransformedRawView(
        base_test, transform, cache_transformed_windows=cache_transformed
    )
    params = dict(config["model"]["params"])
    params["random_state"] = spec.seed
    if smoke:
        params["max_epochs"] = int(config["smoke"]["max_epochs"])
    model = build_model(
        "torch_shallow_convnet", "classification", X_train.shape[1:], 3, params
    )
    train_manifest = X_train.manifest
    test_manifest = X_test.manifest
    model.set_validation_groups(
        train_manifest["record_group_id"].astype(str).to_numpy(),
        subject_ids=train_manifest["subject_id"].astype(str).to_numpy(),
        record_ids=train_manifest["record_id"].astype(str).to_numpy(),
        outer_test_record_ids=test_manifest["record_id"].astype(str).to_numpy(),
        outer_test_group_ids=test_manifest["record_group_id"].astype(str).to_numpy(),
        strategy="group_record", group_column="record_group_id",
        validation_size=float(config["model"]["params"]["validation_size"]),
        random_state=spec.seed,
    )
    training_started = time.perf_counter()
    transform_seconds_before_training = transform.transform_seconds_
    model.fit(X_train, labels[train_idx])
    training_total_seconds = time.perf_counter() - training_started
    training_transform_seconds = (
        transform.transform_seconds_ - transform_seconds_before_training
    )
    inference_started = time.perf_counter()
    transform_seconds_before_inference = transform.transform_seconds_
    prediction = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    inference_total_seconds = time.perf_counter() - inference_started
    inference_transform_seconds = (
        transform.transform_seconds_ - transform_seconds_before_inference
    )
    preprocessing_seconds = calibration_fit_seconds + transform.transform_seconds_
    truth = labels[test_idx]
    metrics = {
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.pt")
    pd.DataFrame(model.training_log_).to_csv(
        run_dir / "training_log.csv", index=False, lineterminator="\n"
    )
    _write_json(run_dir / "validation_split.json", model.validation_split_)
    _write_json(run_dir / "normalization_stats.json", {
        "scope": "inner_train_only_after_artifact_transform",
        "mean": None if model.feature_mean_ is None else model.feature_mean_.tolist(),
        "scale": None if model.feature_scale_ is None else model.feature_scale_.tolist(),
    })
    preprocessing_manifest = {
        **(transform.manifest_ or {}),
        "runtime_diagnostics": transform.runtime_diagnostics(),
        "transformed_window_cache": {
            "train": X_train.cache_diagnostics(),
            "test": X_test.cache_diagnostics(),
            "scope": "single_run_memory_only_not_persisted",
        },
        "outer_test_participants": sorted(test_manifest["subject_id"].astype(str).unique().tolist()),
        "outer_participant_overlap": sorted(
            set(train_manifest["subject_id"].astype(str))
            & set(test_manifest["subject_id"].astype(str))
        ),
        "q3_fit_scope": "outer_train_full_before_smoke_subsampling",
        "q3_thresholds": thresholds,
        "q3_thresholds_hash": stable_hash(thresholds),
    }
    _write_json(run_dir / "preprocessing_manifest.json", preprocessing_manifest)
    predictions = pd.DataFrame({
        "sample_id": np.asarray(data.sample_ids)[test_idx],
        "subject_id": np.asarray(data.subject_ids)[test_idx],
        "record_id": np.asarray(data.record_ids)[test_idx],
        "fold": spec.fold,
        "pm": spec.metric,
        "variant": spec.variant,
        "y_true": truth,
        "y_pred": prediction,
    })
    for class_id in range(3):
        predictions[f"proba_{class_id}"] = probabilities[:, class_id]
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    _write_json(run_dir / "metrics.json", metrics)
    summary = {
        "status": "complete",
        "result_status": "smoke" if smoke else config.get("result_status", "baseline"),
        "run_id": spec.run_id,
        "specification_hash": spec.specification_hash,
        **asdict(spec),
        "metrics": metrics,
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "q3_thresholds": thresholds,
        "preprocessing_seconds": preprocessing_seconds,
        "calibration_fit_seconds": calibration_fit_seconds,
        "artifact_transform_seconds": transform.transform_seconds_,
        "artifact_transform_calls": transform.transform_calls_,
        "artifact_transform_changed_windows": transform.changed_windows_,
        "transformed_window_cache": {
            "train": X_train.cache_diagnostics(),
            "test": X_test.cache_diagnostics(),
        },
        "training_seconds": max(0.0, training_total_seconds - training_transform_seconds),
        "training_total_seconds": training_total_seconds,
        "training_artifact_transform_seconds": training_transform_seconds,
        "inference_seconds": max(0.0, inference_total_seconds - inference_transform_seconds),
        "inference_total_seconds": inference_total_seconds,
        "inference_artifact_transform_seconds": inference_transform_seconds,
        "epochs": int(model.n_epochs_trained_),
        "best_validation_loss": model.best_validation_loss_,
        "device": str(model.device_),
        "parameter_count": int(sum(parameter.numel() for parameter in model.model.parameters())),
        "artifact_transform_fit_count": transform.fit_count_,
        "ica_state_hash": preprocessing_manifest["ica_state_hash"],
    }
    _write_json(summary_path, summary)
    return summary


def _aggregate(output: Path, profile: str, summaries: Sequence[Mapping[str, Any]]) -> None:
    rows = [{**{key: row[key] for key in ("metric", "variant", "fold", "seed", "train_windows", "test_windows", "preprocessing_seconds", "training_seconds", "inference_seconds", "epochs")}, **row["metrics"]} for row in summaries]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{profile}_metrics.csv", index=False, lineterminator="\n")
    raw = frame.loc[frame["variant"].eq("raw")].set_index(["metric", "fold"])
    comparisons = []
    for _, row in frame.iterrows():
        baseline = raw.loc[(row["metric"], row["fold"])]
        comparisons.append({
            "metric": row["metric"], "fold": row["fold"], "variant": row["variant"],
            **{f"delta_{name}": float(row[name] - baseline[name]) for name in ("macro_f1", "balanced_accuracy", "accuracy")},
        })
    pd.DataFrame(comparisons).to_csv(
        output / f"{profile}_aggregate_comparison.csv", index=False, lineterminator="\n"
    )
    metric_columns = ["macro_f1", "balanced_accuracy", "accuracy"]
    fold_macro = (
        frame.groupby(["variant", "fold"], as_index=False, sort=True)[
            metric_columns + ["preprocessing_seconds", "training_seconds", "inference_seconds"]
        ]
        .mean()
    )
    raw_fold = fold_macro.loc[fold_macro["variant"].eq("raw")].set_index("fold")
    for metric_name in metric_columns:
        fold_macro[f"delta_{metric_name}_vs_raw"] = [
            float(row[metric_name] - raw_fold.loc[row["fold"], metric_name])
            for _, row in fold_macro.iterrows()
        ]
    fold_macro.to_csv(
        output / f"{profile}_pm_macro_by_fold.csv", index=False, lineterminator="\n"
    )
    summary_rows = []
    for variant, group in fold_macro.groupby("variant", sort=True):
        row: dict[str, Any] = {"variant": variant, "fold_count": int(len(group))}
        for metric_name in metric_columns:
            delta_name = f"delta_{metric_name}_vs_raw"
            row[f"{metric_name}_mean"] = float(group[metric_name].mean())
            row[f"{metric_name}_sample_sd"] = float(group[metric_name].std(ddof=1))
            row[f"{delta_name}_mean"] = float(group[delta_name].mean())
            row[f"{delta_name}_sample_sd"] = float(group[delta_name].std(ddof=1))
            row[f"folds_better_raw_{metric_name}"] = int((group[delta_name] > 0).sum())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        output / f"{profile}_variant_summary.csv", index=False, lineterminator="\n"
    )
    lines = [
        "# Artifact-removal ablation",
        "",
        f"Profile: `{profile}`. Result status: `{'smoke' if profile == 'smoke' else 'baseline'}`.",
        "",
        "FASTER variants use the project apply_faster per-window implementation with mean-channel interpolation; this is FASTER-like, not the complete classical FASTER algorithm.",
        "",
        frame.to_markdown(index=False),
        "",
        "## PM-macro by fold",
        "",
        fold_macro.to_markdown(index=False),
    ]
    (output / f"{profile}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    config_path: str | Path,
    *,
    smoke: bool,
    resume: bool = True,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    write_plan(config_path)
    specs = build_run_matrix(config)
    if smoke:
        smoke_config = config["smoke"]
        specs = [
            spec for spec in specs
            if spec.metric in set(smoke_config["targets"])
            and spec.fold in set(map(int, smoke_config["folds"]))
        ]
    summaries = [execute_run(config_path, spec, smoke=smoke, resume=resume) for spec in specs]
    _aggregate(_repo_path(config["output_dir"]), "smoke" if smoke else "full", summaries)
    return summaries
