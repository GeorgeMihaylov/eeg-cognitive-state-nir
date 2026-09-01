"""Read-only diagnostics for XGBoost participant feature alignment.

This module intentionally does not add an alignment mode to the scientific
personalization protocol.  It reuses one completed XGBoost outer-fold base,
the frozen participant calibration/evaluation identities, and performs only
post-hoc, label-free feature transformations and inference.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from bench.bench_runner import BenchmarkRunner
from bench.experiments.personalization_calibration import (
    PersonalizationCalibrationPlanner,
    _participant_partition,
    _sample_hash,
    stable_hash,
    validate_temporal_partition,
)
from bench.experiments.personalization_calibration_execution import (
    BenchmarkPersonalizationBackend,
    XGBOOST_CHECKPOINT_NAME,
    base_run_directory,
    base_unit_id,
)
from bench.validation.metrics import MetricsCalculator
from cogstate.adaptation import (
    FeatureAligner,
    FeatureAlignmentConfig,
    apply_alignment_shrinkage,
)
from cogstate.model_zoo.ML.xgboost_personalization import xgboost_state_sha256


SCHEMA_VERSION = "xgboost-feature-alignment-diagnostics-v1"
PM = "focus"
OUTER_FOLD = 1
BUDGET_FRACTION = 0.20
CLASS_LABELS = np.arange(3, dtype=np.int64)
FULL_METHODS = (
    "standard_location_scale",
    "robust_location_scale",
)
LOCATION_ONLY_VARIANTS = {
    "standard_location_only": "standard_location_scale",
    "robust_location_only": "robust_location_scale",
}
FULL_VARIANTS = {
    "standard_location_scale": "standard_location_scale",
    "robust_location_scale": "robust_location_scale",
}
ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
)


def sha256_file(path: str | Path) -> str:
    """Return SHA-256 without loading a potentially large file at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_features(
    X: np.ndarray,
    *,
    name: str,
    expected_features: int | None = None,
) -> np.ndarray:
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if expected_features is not None and array.shape[1] != expected_features:
        raise ValueError(
            f"{name} has {array.shape[1]} features; expected {expected_features}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(array)


def estimate_location_scale(
    X: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the same statistics as FeatureAligner for diagnostics."""

    features = _validate_features(X, name="X")
    if method == "standard_location_scale":
        return np.mean(features, axis=0), np.std(features, axis=0, ddof=0)
    if method == "robust_location_scale":
        center = np.median(features, axis=0)
        scale = (
            np.quantile(features, 0.75, axis=0)
            - np.quantile(features, 0.25, axis=0)
        )
        return center, scale
    raise ValueError(f"Unknown alignment method {method!r}")


def apply_location_only(
    X: np.ndarray,
    *,
    reference_center: np.ndarray,
    calibration_center: np.ndarray,
) -> np.ndarray:
    """Translate participant values without modifying their feature scale."""

    features = _validate_features(X, name="X")
    reference = np.asarray(reference_center, dtype=np.float64).reshape(-1)
    calibration = np.asarray(calibration_center, dtype=np.float64).reshape(-1)
    if reference.shape != calibration.shape or len(reference) != features.shape[1]:
        raise ValueError("Location vectors must match the feature width")
    transformed = features + reference[None, :] - calibration[None, :]
    if not np.isfinite(transformed).all():
        raise RuntimeError("Location-only alignment produced NaN or Inf")
    return np.ascontiguousarray(transformed)


def apply_shrinkage(
    original: np.ndarray,
    fully_aligned: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate between identity and a fully aligned feature matrix."""

    return apply_alignment_shrinkage(original, fully_aligned, alpha)


def displacement_table(
    original: np.ndarray,
    transformed: np.ndarray,
    *,
    reference_scale: np.ndarray,
    feature_names: Sequence[str],
    scale_epsilon: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Summarize feature-wise movement caused by one full transform."""

    source = _validate_features(original, name="original")
    shifted = _validate_features(
        transformed,
        name="transformed",
        expected_features=source.shape[1],
    )
    if source.shape != shifted.shape:
        raise ValueError("original and transformed must have identical shapes")
    names = [str(name) for name in feature_names]
    if len(names) != source.shape[1]:
        raise ValueError("feature_names must match the feature width")
    scale = np.asarray(reference_scale, dtype=np.float64).reshape(-1)
    if len(scale) != source.shape[1] or not np.isfinite(scale).all():
        raise ValueError("reference_scale must be finite and match feature width")
    denominator = np.maximum(scale, float(scale_epsilon))
    absolute = np.abs(shifted - source)
    median_absolute = np.median(absolute, axis=0)
    normalized = median_absolute / denominator
    table = pd.DataFrame(
        {
            "feature_index": np.arange(source.shape[1], dtype=np.int64),
            "feature_name": names,
            "median_absolute_displacement": median_absolute,
            "reference_scale": scale,
            "normalized_displacement": normalized,
        }
    )
    quantiles = np.quantile(normalized, [0.50, 0.90, 0.95, 0.99])
    summary = {
        "median_absolute_feature_displacement": float(np.median(median_absolute)),
        "normalized_displacement_p50": float(quantiles[0]),
        "normalized_displacement_p90": float(quantiles[1]),
        "normalized_displacement_p95": float(quantiles[2]),
        "normalized_displacement_p99": float(quantiles[3]),
        "normalized_displacement_max": float(np.max(normalized)),
    }
    for threshold in (0.25, 0.5, 1.0, 2.0):
        suffix = str(threshold).replace(".", "_")
        summary[f"fraction_features_normalized_gt_{suffix}"] = float(
            np.mean(normalized > threshold)
        )
    return table, summary


def temporal_drift_summary(
    calibration: np.ndarray,
    evaluation: np.ndarray,
    *,
    reference_scale: np.ndarray,
    method: str,
    scale_epsilon: float,
) -> dict[str, float]:
    """Compare calibration and later evaluation distributions post hoc."""

    cal_center, cal_scale = estimate_location_scale(calibration, method)
    eval_center, eval_scale = estimate_location_scale(evaluation, method)
    reference = np.maximum(
        np.asarray(reference_scale, dtype=np.float64),
        float(scale_epsilon),
    )
    center_drift = np.abs(eval_center - cal_center) / reference
    scale_drift = np.abs(eval_scale - cal_scale) / reference
    output: dict[str, float] = {}
    for name, values in (
        ("center_drift", center_drift),
        ("scale_drift", scale_drift),
    ):
        q50, q90, q95, q99 = np.quantile(values, [0.50, 0.90, 0.95, 0.99])
        output.update(
            {
                f"{name}_p50": float(q50),
                f"{name}_p90": float(q90),
                f"{name}_p95": float(q95),
                f"{name}_p99": float(q99),
                f"{name}_max": float(np.max(values)),
            }
        )
    output["temporal_drift_combined_p50"] = float(
        np.median(np.concatenate([center_drift, scale_drift]))
    )
    return output


def _predict(
    estimator: Any,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    predictions = np.asarray(estimator.predict(X), dtype=np.int64)
    if probabilities.shape != (len(predictions), len(CLASS_LABELS)):
        raise RuntimeError(
            "Expected three-class probabilities, got "
            f"{probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise RuntimeError("Invalid XGBoost probabilities")
    return predictions, probabilities


def prediction_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    baseline_pred: np.ndarray,
    baseline_proba: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return distribution/flip diagnostics and per-class metrics."""

    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(y_proba, dtype=np.float64)
    reference_prediction = np.asarray(baseline_pred, dtype=np.int64).reshape(-1)
    reference_proba = np.asarray(baseline_proba, dtype=np.float64)
    if not (
        truth.shape == prediction.shape == reference_prediction.shape
        and probabilities.shape == reference_proba.shape
        and probabilities.shape == (len(truth), len(CLASS_LABELS))
    ):
        raise ValueError("Prediction diagnostics received incompatible shapes")

    counts = np.bincount(prediction, minlength=len(CLASS_LABELS))
    probability_change = np.abs(probabilities - reference_proba)
    normalized_confusion = confusion_matrix(
        truth,
        prediction,
        labels=CLASS_LABELS,
        normalize="true",
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    summary: dict[str, Any] = {
        "prediction_flip_rate": float(np.mean(prediction != reference_prediction)),
        "mean_absolute_probability_change": float(np.mean(probability_change)),
        "p95_absolute_probability_change": float(
            np.quantile(probability_change, 0.95)
        ),
        "predicted_class_count": int(np.sum(counts > 0)),
        "class_collapse": bool(np.sum(counts > 0) == 1),
        "missing_predicted_classes": "|".join(
            str(class_id) for class_id in CLASS_LABELS[counts == 0]
        ),
    }
    for class_id, count in enumerate(counts):
        summary[f"predicted_fraction_class_{class_id}"] = float(count / len(truth))
    class_rows: list[dict[str, Any]] = []
    for class_id in CLASS_LABELS:
        row = {
            "class_id": int(class_id),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
            "predicted_fraction": float(counts[class_id] / len(truth)),
        }
        for predicted_class in CLASS_LABELS:
            row[f"confusion_pred_{predicted_class}"] = float(
                normalized_confusion[class_id, predicted_class]
            )
        class_rows.append(row)
    return summary, class_rows


def safe_correlations(
    x: Sequence[float],
    y: Sequence[float],
) -> dict[str, float | int | None]:
    """Compute exploratory correlations without constant-input warnings."""

    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if len(first) < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return {"n": int(len(first)), "pearson": None, "spearman": None}
    return {
        "n": int(len(first)),
        "pearson": float(scipy_stats.pearsonr(first, second).statistic),
        "spearman": float(scipy_stats.spearmanr(first, second).statistic),
    }


def subject_macro_reference_diagnostics(
    X_outer_train: np.ndarray,
    subject_ids: Sequence[Any] | None,
    feature_names: Sequence[str],
    *,
    scale_epsilon: float = 1e-12,
    top_n: int = 15,
) -> dict[str, Any]:
    """Quantify pooled-versus-equal-subject reference center differences."""

    if subject_ids is None:
        return {"available": False, "reason": "subject_id metadata unavailable"}
    features = _validate_features(X_outer_train, name="X_outer_train")
    subjects = np.asarray(subject_ids).astype(str)
    if len(subjects) != len(features):
        raise ValueError("subject_ids must match outer-train rows")
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        return {"available": False, "reason": "fewer than two train subjects"}
    names = [str(name) for name in feature_names]
    result: dict[str, Any] = {
        "available": True,
        "n_subjects": int(len(unique_subjects)),
        "participant_weighting": "equal_subject_weight",
        "methods": {},
    }
    for method in FULL_METHODS:
        pooled_center, pooled_scale = estimate_location_scale(features, method)
        subject_centers = np.vstack(
            [
                estimate_location_scale(features[subjects == subject], method)[0]
                for subject in unique_subjects
            ]
        )
        macro_center = np.mean(subject_centers, axis=0)
        normalized_difference = np.abs(macro_center - pooled_center) / np.maximum(
            pooled_scale, float(scale_epsilon)
        )
        order = np.argsort(normalized_difference)[::-1][:top_n]
        result["methods"][method] = {
            "normalized_center_difference_p50": float(
                np.quantile(normalized_difference, 0.50)
            ),
            "normalized_center_difference_p90": float(
                np.quantile(normalized_difference, 0.90)
            ),
            "normalized_center_difference_p95": float(
                np.quantile(normalized_difference, 0.95)
            ),
            "normalized_center_difference_max": float(
                np.max(normalized_difference)
            ),
            "top_features": [
                {
                    "feature_name": names[index],
                    "normalized_center_difference": float(
                        normalized_difference[index]
                    ),
                }
                for index in order
            ],
        }
    return result


def _metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    metrics = MetricsCalculator.calculate_all_metrics(
        y_true,
        y_pred,
        y_proba,
        task_type="classification",
        labels=CLASS_LABELS,
    )
    return {name: float(metrics[name]) for name in METRIC_NAMES}


def _aggregate_metrics(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_participants"] = int(group["subject_id"].nunique())
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            if f"delta_{metric}" in group:
                row[f"delta_{metric}_mean"] = float(group[f"delta_{metric}"].mean())
                row[f"positive_fraction_{metric}"] = float(
                    np.mean(group[f"delta_{metric}"] > 0)
                )
        rows.append(row)
    return rows


def _top_displaced_features(
    feature_displacement: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    grouped = (
        feature_displacement.groupby(
            ["method", "feature_index", "feature_name"],
            sort=True,
            as_index=False,
        )
        .agg(
            normalized_displacement_mean=("normalized_displacement", "mean"),
            normalized_displacement_median=("normalized_displacement", "median"),
            normalized_displacement_max=("normalized_displacement", "max"),
            median_absolute_displacement_mean=(
                "median_absolute_displacement", "mean"
            ),
            n_participants=("subject_id", "nunique"),
        )
    )
    grouped["rank_within_method"] = grouped.groupby("method")[
        "normalized_displacement_mean"
    ].rank(method="first", ascending=False).astype(int)
    selected = grouped.loc[grouped["rank_within_method"] <= int(top_n)].copy()
    overall = (
        feature_displacement.groupby(
            ["feature_index", "feature_name"], sort=True, as_index=False
        )
        .agg(
            normalized_displacement_mean=("normalized_displacement", "mean"),
            normalized_displacement_median=("normalized_displacement", "median"),
            normalized_displacement_max=("normalized_displacement", "max"),
            median_absolute_displacement_mean=(
                "median_absolute_displacement", "mean"
            ),
            n_participants=("subject_id", "nunique"),
        )
        .sort_values("normalized_displacement_mean", ascending=False)
        .head(int(top_n))
    )
    overall.insert(0, "method", "all_full_methods")
    overall["rank_within_method"] = np.arange(1, len(overall) + 1)
    return pd.concat([selected, overall], ignore_index=True).sort_values(
        ["method", "rank_within_method"], kind="stable"
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _load_source_tables(
    plan_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = _read_json(plan_dir / "protocol_manifest.json", label="plan manifest")
    matrix_path = plan_dir / "run_matrix.csv"
    participant_path = plan_dir / "participant_calibration_plan.csv"
    if not matrix_path.is_file() or not participant_path.is_file():
        raise FileNotFoundError("Source plan tables are incomplete")
    return manifest, pd.read_csv(matrix_path), pd.read_csv(participant_path)


def _resolve_base_row(matrix: pd.DataFrame) -> dict[str, Any]:
    selected = matrix.loc[
        matrix["pm"].eq(PM)
        & matrix["task_type"].eq("classification")
        & matrix["model"].eq("xgboost")
        & matrix["outer_fold"].eq(OUTER_FOLD)
        & matrix["mode"].eq("zero_shot")
        & matrix["budget_fraction"].eq(0.0)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one exact source base row, found {len(selected)}")
    row = selected.iloc[0].to_dict()
    row["outer_fold"] = int(row["outer_fold"])
    row["seed"] = int(row["seed"])
    row["budget_fraction"] = float(row["budget_fraction"])
    return row


def _prepare_read_only_base(
    planner: PersonalizationCalibrationPlanner,
    *,
    plan_hash: str,
    base: Mapping[str, Any],
) -> tuple[BenchmarkPersonalizationBackend, Any, dict[str, Any]]:
    """Load an exact completed base while making missing state a hard failure."""

    backend = BenchmarkPersonalizationBackend(planner, plan_hash=plan_hash)
    config = backend._base_config(base)
    base_dir = base_run_directory(planner.output_dir, base_unit_id(base))
    checkpoint = base_dir / XGBOOST_CHECKPOINT_NAME
    manifest_path = base_dir / "base_checkpoint_manifest.json"
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Diagnostics require the completed native XGBoost checkpoint and manifest"
        )
    completed = BenchmarkRunner.find_completed_run(
        config,
        search_directories=[config["output_dir"]],
    )
    if completed is None:
        raise RuntimeError(
            "No exact completed BenchmarkRunner base exists; diagnostics refuse "
            "to train a replacement"
        )
    checkpoint_hash_before = sha256_file(checkpoint)
    manifest_bytes_before = manifest_path.read_bytes()
    handle = backend.ensure_base(base, resume=True)
    if not handle.resumed:
        raise RuntimeError("Read-only diagnostic base was not resumed")
    if Path(handle.checkpoint_path).resolve() != checkpoint.resolve():
        raise RuntimeError("Loaded a different XGBoost checkpoint")
    if sha256_file(checkpoint) != checkpoint_hash_before:
        raise RuntimeError("Base checkpoint changed while loading diagnostics")
    if manifest_path.read_bytes() != manifest_bytes_before:
        raise RuntimeError("Base checkpoint manifest changed while loading diagnostics")
    manifest = _read_json(manifest_path, label="base checkpoint manifest")
    return backend, handle, {
        "completed_run_directory": str(completed.run_directory),
        "base_directory": str(base_dir),
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": checkpoint_hash_before,
        "base_checkpoint_identity_hash": manifest["checkpoint_identity_hash"],
        "base_manifest_sha256": hashlib.sha256(manifest_bytes_before).hexdigest(),
        "base_resumed": True,
        "base_artifacts_unchanged_during_load": True,
    }


def run_diagnostics(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    source_plan_dir: str | Path,
    source_run_dir: str | Path,
    output_dir: str | Path,
    smoke_audit_path: str | Path | None = None,
    top_n: int = 15,
) -> dict[str, Any]:
    """Execute the isolated post-hoc diagnostic and write compact artifacts."""

    root = Path(repo_root).resolve()

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    config = resolve(config_path)
    plan_dir = resolve(source_plan_dir)
    run_dir = resolve(source_run_dir)
    destination = resolve(output_dir)
    smoke_audit = None if smoke_audit_path is None else resolve(smoke_audit_path)
    if destination == run_dir or run_dir in destination.parents:
        raise ValueError("Diagnostic output must be isolated from source results")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Diagnostic output already exists: {destination}")
    if top_n < 1:
        raise ValueError("top_n must be positive")

    plan_manifest, run_matrix, participant_plan = _load_source_tables(plan_dir)
    execution_manifest = _read_json(
        run_dir / "execution_manifest.json", label="execution manifest"
    )
    planner = PersonalizationCalibrationPlanner(
        config,
        data_root=root,
        output_dir=run_dir,
    )
    if planner.protocol_hash != plan_manifest.get("protocol_hash"):
        raise RuntimeError("Current config does not match source protocol hash")
    if execution_manifest.get("protocol_hash") != planner.protocol_hash:
        raise RuntimeError("Source execution protocol hash mismatch")
    plan_hash = str(plan_manifest["plan_hash"])
    if execution_manifest.get("plan_hash") != plan_hash:
        raise RuntimeError("Source execution plan hash mismatch")
    if bool(plan_manifest.get("training_executed")):
        raise RuntimeError("Source plan manifest unexpectedly reports training")

    base = _resolve_base_row(run_matrix)
    backend, handle, base_audit = _prepare_read_only_base(
        planner,
        plan_hash=plan_hash,
        base=base,
    )
    if base_audit["base_checkpoint_sha256"] != handle.checkpoint_sha256:
        raise RuntimeError("Base handle checkpoint SHA mismatch")
    if base_audit["base_checkpoint_identity_hash"] != handle.checkpoint_identity_hash:
        raise RuntimeError("Base handle identity mismatch")
    if not hasattr(handle.adapter, "global_model"):
        raise TypeError("Expected XGBoostMarginHeadAdapter")
    estimator = handle.adapter.global_model
    booster_hash_before = xgboost_state_sha256(estimator)

    if smoke_audit is not None:
        previous_smoke = _read_json(smoke_audit, label="feature-alignment smoke audit")
        if previous_smoke.get("base_checkpoint_sha256") != handle.checkpoint_sha256:
            raise RuntimeError("Diagnostics do not use the checkpoint from smoke")
        if previous_smoke.get("booster_hash_before") != booster_hash_before:
            raise RuntimeError("Diagnostics booster differs from smoke booster")
    else:
        previous_smoke = None

    split = handle.split
    X_outer_train = _validate_features(split.X_train, name="X_outer_train")
    feature_names = [str(name) for name in (split.feature_names or [])]
    if len(feature_names) != X_outer_train.shape[1] or len(set(feature_names)) != len(
        feature_names
    ):
        raise RuntimeError("Split feature_names are missing, duplicated, or misaligned")
    reference_aligners: dict[str, FeatureAligner] = {}
    for method in FULL_METHODS:
        reference_aligners[method] = FeatureAligner(
            FeatureAlignmentConfig(method=method)
        ).fit_reference(X_outer_train)

    eligible = participant_plan.loc[
        participant_plan["pm"].eq(PM)
        & participant_plan["outer_fold"].eq(OUTER_FOLD)
        & np.isclose(participant_plan["budget_fraction"], BUDGET_FRACTION)
        & participant_plan["status"].eq("planned")
    ].copy()
    eligible = eligible.sort_values("subject_id", kind="stable").reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("No eligible participants in the requested source plan")
    target_frame = planner._load_target_frame(PM)

    participant_rows: list[dict[str, Any]] = []
    feature_rows: list[pd.DataFrame] = []
    temporal_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    calibration_hash_matches: list[bool] = []
    evaluation_hash_matches: list[bool] = []
    q3_hash_matches: list[bool] = []
    fit_evaluation_overlaps: list[int] = []

    for participant in eligible.to_dict("records"):
        subject_id = str(participant["subject_id"])
        subject = target_frame.loc[
            target_frame["outer_fold"].eq(OUTER_FOLD)
            & target_frame["subject_id"].astype(str).eq(subject_id)
        ]
        partition, _ = _participant_partition(
            subject,
            budget=BUDGET_FRACTION,
            reference_budget=BUDGET_FRACTION,
            protocol=planner.config["protocol"],
        )
        temporal_audit = validate_temporal_partition(partition)
        calibration_ids = partition.calibration_metadata["sample_id"].astype(str).to_numpy()
        evaluation_ids = partition.evaluation_metadata["sample_id"].astype(str).to_numpy()
        actual_calibration_hash = _sample_hash(calibration_ids)
        actual_evaluation_hash = _sample_hash(evaluation_ids)
        calibration_match = actual_calibration_hash == str(
            participant["calibration_sample_hash"]
        )
        evaluation_match = actual_evaluation_hash == str(
            participant["evaluation_sample_hash"]
        )
        q3_match = str(participant["q3_transform_hash"]) == handle.target_transform_hash
        overlap = len(set(calibration_ids) & set(evaluation_ids))
        calibration_hash_matches.append(calibration_match)
        evaluation_hash_matches.append(evaluation_match)
        q3_hash_matches.append(q3_match)
        fit_evaluation_overlaps.append(overlap)
        if not calibration_match or not evaluation_match or not q3_match or overlap:
            raise RuntimeError(f"Participant identity mismatch for {subject_id}")

        X_calibration, _ = backend._subset_by_ids(split, calibration_ids)
        X_evaluation, y_evaluation = backend._subset_by_ids(split, evaluation_ids)
        X_calibration = _validate_features(
            X_calibration,
            name="X_calibration",
            expected_features=len(feature_names),
        )
        X_evaluation = _validate_features(
            X_evaluation,
            name="X_evaluation",
            expected_features=len(feature_names),
        )
        y_evaluation = np.asarray(y_evaluation, dtype=np.int64)
        zero_pred, zero_proba = _predict(estimator, X_evaluation)
        zero_metrics = _metric_row(y_evaluation, zero_pred, zero_proba)

        variant_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "zero_shot": (zero_pred, zero_proba)
        }
        full_aligned: dict[str, np.ndarray] = {}
        fitted_aligners: dict[str, FeatureAligner] = {}
        for method, reference_aligner in reference_aligners.items():
            aligner = deepcopy(reference_aligner).fit_calibration(X_calibration)
            fitted_aligners[method] = aligner
            full = aligner.transform(X_evaluation)
            full_aligned[method] = full
            reference_stats = aligner.reference_stats_
            calibration_stats = aligner.calibration_stats_
            if reference_stats is None or calibration_stats is None:
                raise RuntimeError("FeatureAligner did not expose fitted statistics")
            location_name = (
                "standard_location_only"
                if method == "standard_location_scale"
                else "robust_location_only"
            )
            location_only = apply_location_only(
                X_evaluation,
                reference_center=reference_stats.center,
                calibration_center=calibration_stats.center,
            )
            variant_predictions[location_name] = _predict(estimator, location_only)
            variant_predictions[method] = _predict(estimator, full)

            feature_table, displacement = displacement_table(
                X_evaluation,
                full,
                reference_scale=reference_stats.scale,
                feature_names=feature_names,
                scale_epsilon=aligner.config.scale_epsilon,
            )
            feature_table.insert(0, "subject_id", subject_id)
            feature_table.insert(1, "method", method)
            feature_rows.append(feature_table)
            drift = temporal_drift_summary(
                X_calibration,
                X_evaluation,
                reference_scale=reference_stats.scale,
                method=method,
                scale_epsilon=aligner.config.scale_epsilon,
            )
            aligned_metrics = _metric_row(
                y_evaluation,
                *variant_predictions[method],
            )
            delta_macro_f1 = aligned_metrics["macro_f1"] - zero_metrics["macro_f1"]
            temporal_rows.append(
                {
                    "subject_id": subject_id,
                    "method": method,
                    **drift,
                    "delta_macro_f1": delta_macro_f1,
                }
            )
            participant_rows.append(
                {
                    "pm": PM,
                    "outer_fold": OUTER_FOLD,
                    "budget_fraction": BUDGET_FRACTION,
                    "subject_id": subject_id,
                    "method": method,
                    "calibration_windows": int(len(calibration_ids)),
                    "evaluation_windows": int(len(evaluation_ids)),
                    "calibration_sample_hash": actual_calibration_hash,
                    "evaluation_sample_hash": actual_evaluation_hash,
                    "q3_transform_hash": handle.target_transform_hash,
                    "calibration_hash_match": calibration_match,
                    "evaluation_hash_match": evaluation_match,
                    "q3_hash_match": q3_match,
                    "fit_evaluation_sample_overlap": overlap,
                    "calibration_before_evaluation": temporal_audit[
                        "calibration_before_evaluation"
                    ],
                    "calibration_degenerate_features": int(
                        np.sum(
                            calibration_stats.scale
                            <= aligner.config.scale_epsilon
                        )
                    ),
                    **displacement,
                    **drift,
                    "zero_shot_macro_f1": zero_metrics["macro_f1"],
                    "aligned_macro_f1": aligned_metrics["macro_f1"],
                    "delta_macro_f1": delta_macro_f1,
                }
            )

        for variant, (prediction, probabilities) in variant_predictions.items():
            metrics = _metric_row(y_evaluation, prediction, probabilities)
            row = {
                "pm": PM,
                "outer_fold": OUTER_FOLD,
                "budget_fraction": BUDGET_FRACTION,
                "subject_id": subject_id,
                "variant": variant,
                **metrics,
            }
            for metric in METRIC_NAMES:
                row[f"delta_{metric}"] = metrics[metric] - zero_metrics[metric]
            variant_rows.append(row)
            prediction_summary, per_class = prediction_diagnostics(
                y_evaluation,
                prediction,
                probabilities,
                baseline_pred=zero_pred,
                baseline_proba=zero_proba,
            )
            prediction_rows.append(
                {
                    "subject_id": subject_id,
                    "variant": variant,
                    "evaluation_windows": int(len(y_evaluation)),
                    **prediction_summary,
                }
            )
            class_rows.extend(
                {
                    "subject_id": subject_id,
                    "variant": variant,
                    **class_row,
                }
                for class_row in per_class
            )

        for method in FULL_METHODS:
            for alpha in ALPHAS:
                if alpha == 0.0:
                    prediction, probabilities = zero_pred, zero_proba
                elif alpha == 1.0:
                    prediction, probabilities = variant_predictions[method]
                else:
                    partial = apply_shrinkage(
                        X_evaluation,
                        full_aligned[method],
                        alpha,
                    )
                    prediction, probabilities = _predict(estimator, partial)
                metrics = _metric_row(y_evaluation, prediction, probabilities)
                row = {
                    "pm": PM,
                    "outer_fold": OUTER_FOLD,
                    "budget_fraction": BUDGET_FRACTION,
                    "subject_id": subject_id,
                    "method": method,
                    "alpha": float(alpha),
                    **metrics,
                }
                for metric in METRIC_NAMES:
                    row[f"delta_{metric}"] = metrics[metric] - zero_metrics[metric]
                partial_rows.append(row)

    booster_hash_after = xgboost_state_sha256(estimator)
    if booster_hash_before != booster_hash_after:
        raise RuntimeError("Frozen XGBoost booster changed during diagnostics")

    participant_diagnostics = pd.DataFrame(participant_rows)
    feature_displacement = pd.concat(feature_rows, ignore_index=True)
    temporal_drift = pd.DataFrame(temporal_rows)
    variant_results = pd.DataFrame(variant_rows)
    partial_curve = pd.DataFrame(partial_rows)
    prediction_shift = pd.DataFrame(prediction_rows)
    class_metrics = pd.DataFrame(class_rows)
    top_features = _top_displaced_features(feature_displacement, top_n=top_n)

    variant_summary = _aggregate_metrics(variant_results, ["variant"])
    partial_summary = _aggregate_metrics(partial_curve, ["method", "alpha"])
    drift_correlations: dict[str, Any] = {}
    for method, group in temporal_drift.groupby("method", sort=True):
        drift_correlations[str(method)] = {
            metric: safe_correlations(group[metric], group["delta_macro_f1"])
            for metric in (
                "center_drift_p50",
                "scale_drift_p50",
                "temporal_drift_combined_p50",
            )
        }
    subject_ids_train = split.row_metadata_train.get("subject_id")
    subject_id_source = "row_metadata_train.subject_id"
    if subject_ids_train is None and split.subject_train is not None:
        # TaskSplit carries canonical group identity separately even when a
        # dataset does not duplicate it into row_metadata_train.
        subject_ids_train = split.subject_train
        subject_id_source = "TaskSplit.subject_train"
    subject_macro_bias = subject_macro_reference_diagnostics(
        X_outer_train,
        subject_ids_train,
        feature_names,
        top_n=top_n,
    )
    subject_macro_bias["subject_id_source"] = subject_id_source

    invariants = {
        "source_protocol_hash_matches_config": True,
        "source_execution_plan_hash_matches_plan": True,
        "exact_completed_base_reused": bool(handle.resumed),
        "base_checkpoint_sha_matches_manifest": (
            handle.checkpoint_sha256 == base_audit["base_checkpoint_sha256"]
        ),
        "base_checkpoint_identity_matches_manifest": (
            handle.checkpoint_identity_hash
            == base_audit["base_checkpoint_identity_hash"]
        ),
        "booster_unchanged": booster_hash_before == booster_hash_after,
        "all_calibration_sample_hashes_match": all(calibration_hash_matches),
        "all_evaluation_sample_hashes_match": all(evaluation_hash_matches),
        "all_q3_transform_hashes_match": all(q3_hash_matches),
        "evaluation_samples_used_for_fit": False,
        "labels_used_for_alignment_fit": False,
        "fit_evaluation_sample_overlap_max": max(fit_evaluation_overlaps),
        "feature_names_from_split": True,
        "participant_count_matches_source_plan": (
            participant_diagnostics["subject_id"].nunique() == len(eligible)
        ),
        "exploratory_alpha_not_selected": True,
    }
    if not all(
        value is True
        for key, value in invariants.items()
        if key not in {"evaluation_samples_used_for_fit", "labels_used_for_alignment_fit",
                       "fit_evaluation_sample_overlap_max"}
    ):
        raise RuntimeError(f"One or more diagnostic invariants failed: {invariants}")
    if (
        invariants["evaluation_samples_used_for_fit"]
        or invariants["labels_used_for_alignment_fit"]
        or invariants["fit_evaluation_sample_overlap_max"] != 0
    ):
        raise RuntimeError(f"Leakage invariant failed: {invariants}")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "exploratory_diagnostic_not_model_selection",
        "pm": PM,
        "outer_fold": OUTER_FOLD,
        "budget_fraction": BUDGET_FRACTION,
        "n_participants": int(len(eligible)),
        "n_features": int(len(feature_names)),
        "variant_summary": variant_summary,
        "partial_alignment_summary": partial_summary,
        "temporal_drift_correlations": drift_correlations,
        "subject_macro_reference_bias": subject_macro_bias,
        "class_collapse_counts": (
            prediction_shift.groupby("variant")["class_collapse"].sum().astype(int).to_dict()
        ),
        "top_displaced_features": top_features.loc[
            top_features["method"].eq("all_full_methods")
        ].to_dict("records"),
        "interpretation_guard": (
            "Evaluation labels are used only for post-hoc diagnostics. Alpha and "
            "alignment variants must not be selected from these results as final "
            "hyperparameters."
        ),
    }
    audit_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_experiment_id": plan_manifest["experiment_id"],
        "source_protocol_hash": planner.protocol_hash,
        "source_plan_hash": plan_hash,
        "source_plan_dir": str(plan_dir),
        "source_run_dir": str(run_dir),
        "source_plan_manifest_sha256": sha256_file(
            plan_dir / "protocol_manifest.json"
        ),
        "source_execution_manifest_sha256": sha256_file(
            run_dir / "execution_manifest.json"
        ),
        "feature_names_hash": stable_hash(feature_names),
        "q3_transform_hash": handle.target_transform_hash,
        "booster_hash_before": booster_hash_before,
        "booster_hash_after": booster_hash_after,
        "previous_smoke_checked": previous_smoke is not None,
        **base_audit,
        "invariants": invariants,
    }

    destination.mkdir(parents=True, exist_ok=True)
    participant_diagnostics.to_csv(
        destination / "participant_diagnostics.csv", index=False
    )
    feature_displacement.to_csv(
        destination / "feature_displacement.csv", index=False
    )
    top_features.to_csv(destination / "feature_displacement_top.csv", index=False)
    temporal_drift.to_csv(destination / "temporal_drift.csv", index=False)
    prediction_shift.to_csv(destination / "prediction_shift.csv", index=False)
    class_metrics.to_csv(destination / "class_metrics.csv", index=False)
    variant_results.to_csv(
        destination / "alignment_variant_results.csv", index=False
    )
    partial_curve.to_csv(destination / "partial_alignment_curve.csv", index=False)
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "audit_manifest.json", audit_manifest)
    readme = f"""# XGBoost feature-alignment diagnostics v1

This is an isolated exploratory analysis, not a new personalization mode.

- Source protocol: `{planner.protocol_hash}`
- Source plan: `{plan_hash}`
- Scope: `{PM}`, outer fold `{OUTER_FOLD}`, budget `{BUDGET_FRACTION:.2f}`
- Eligible participants: `{len(eligible)}`
- Features: `{len(feature_names)}` from the resolved split
- Base checkpoint reused: `{handle.resumed}`
- Booster unchanged: `{booster_hash_before == booster_hash_after}`
- Evaluation used for fitting: `false`
- Labels used for alignment fitting: `false`

The partial-alignment alpha curve and all correlations are exploratory.  The
outer-test/evaluation labels must not be used to select alpha, a reference
weighting policy, or an alignment variant for a future confirmatory run.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    return {"output_dir": str(destination), "summary": summary, "audit": audit_manifest}


def _summary_lines(result: Mapping[str, Any]) -> list[str]:
    summary = result["summary"]
    audit = result["audit"]
    variants = {
        row["variant"]: row for row in summary["variant_summary"]
    }
    lines = [
        "XGBoost feature-alignment diagnostics",
        f"A. baseline macro-F1: {variants['zero_shot']['macro_f1_mean']:.6f}",
    ]
    for name in (
        "standard_location_scale",
        "robust_location_scale",
        "standard_location_only",
        "robust_location_only",
    ):
        row = variants[name]
        lines.append(
            f"B/C. {name}: macro-F1={row['macro_f1_mean']:.6f}, "
            f"delta={row['delta_macro_f1_mean']:+.6f}"
        )
    lines.append("D. partial alignment (macro-F1 mean):")
    for row in summary["partial_alignment_summary"]:
        lines.append(
            f"   {row['method']} alpha={row['alpha']:.2f}: "
            f"{row['macro_f1_mean']:.6f} "
            f"({row['delta_macro_f1_mean']:+.6f})"
        )
    flip = pd.read_csv(Path(result["output_dir"]) / "prediction_shift.csv")
    lines.append("E. prediction flip rate mean:")
    for variant, value in flip.groupby("variant")["prediction_flip_rate"].mean().items():
        lines.append(f"   {variant}: {value:.6f}")
    lines.append("F. temporal drift correlations are exploratory:")
    for method, metrics in summary["temporal_drift_correlations"].items():
        combined = metrics["temporal_drift_combined_p50"]
        lines.append(
            f"   {method}: Pearson={combined['pearson']}, "
            f"Spearman={combined['spearman']}"
        )
    lines.append("G. top 15 displaced features:")
    for row in summary["top_displaced_features"][:15]:
        lines.append(
            f"   {row['rank_within_method']:02d}. {row['feature_name']}: "
            f"{row['normalized_displacement_mean']:.6f}"
        )
    lines.append("H. invariants:")
    for key, value in audit["invariants"].items():
        lines.append(f"   {key}: {value}")
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--config",
        default="experiments/calibration/personalization_calibration_xgboost_v1.json",
    )
    parser.add_argument(
        "--source-plan-dir",
        default="benchmark_results/_plan_personalization_xgboost_v1",
    )
    parser.add_argument(
        "--source-run-dir",
        default="benchmark_results/personalization_calibration_xgboost_v1",
    )
    parser.add_argument(
        "--smoke-audit",
        default="benchmark_results/xgboost_feature_alignment_smoke_v1/audit_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/xgboost_feature_alignment_diagnostics_v1",
    )
    parser.add_argument("--top-n", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_diagnostics(
        repo_root=args.repo_root,
        config_path=args.config,
        source_plan_dir=args.source_plan_dir,
        source_run_dir=args.source_run_dir,
        output_dir=args.output_dir,
        smoke_audit_path=args.smoke_audit,
        top_n=args.top_n,
    )
    print("\n".join(_summary_lines(result)))
    print(f"Artifacts: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALPHAS",
    "FULL_METHODS",
    "apply_location_only",
    "apply_shrinkage",
    "displacement_table",
    "estimate_location_scale",
    "prediction_diagnostics",
    "run_diagnostics",
    "safe_correlations",
    "subject_macro_reference_diagnostics",
    "temporal_drift_summary",
]
