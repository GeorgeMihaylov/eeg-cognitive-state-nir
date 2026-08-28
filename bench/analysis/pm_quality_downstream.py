"""Deferred EEG-feature predictability check for PM temporal variants.

The runner is deliberately separate from BenchmarkRunner because its labels are
experimental derived targets.  It still reuses the repository model factory,
metrics, fixed outer folds, and fold-local Q3 implementation.  No hyperparameter
selection is performed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from bench.analysis.pm_temporal_quality import (
    PM_METRICS,
    TARGET_COLUMNS,
    VARIANT_ORDER,
    _jsonable,
    _repo_path,
    _write_csv,
    _write_json,
    build_variants,
    load_config,
    prepare_pm_frame,
    stable_hash,
)
from bench.tasks.target_registry import get_target_spec
from bench.tasks.target_transforms import (
    build_fold_local_target_transform,
    build_target_transform_manifest,
)
from bench.validation.metrics import MetricsCalculator
from cogstate.model_zoo import build_model


def build_downstream_plan(
    config: Mapping[str, Any],
    *,
    outer_folds: Sequence[int] | None = None,
) -> pd.DataFrame:
    definition = config["downstream"]
    folds = tuple(int(value) for value in (outer_folds or config["folds"]["fold_ids"]))
    if not folds or not set(folds).issubset({1, 2, 3, 4, 5}):
        raise ValueError("outer_folds must be a non-empty subset of 1..5")
    rows: list[dict[str, Any]] = []
    for metric in PM_METRICS:
        for variant in VARIANT_ORDER:
            for fold in folds:
                for task_type in ("classification", "regression"):
                    for model_name, model_definition in definition["models"][task_type].items():
                        payload = {
                            "pm": metric,
                            "variant": variant,
                            "outer_fold": fold,
                            "task_type": task_type,
                            "model": model_name,
                            "params": model_definition["params"],
                            "scaling": model_definition["scaling"],
                            "seed": int(definition["random_state"]),
                        }
                        rows.append(
                            {
                                **payload,
                                "run_id": (
                                    f"{metric}__{variant}__{task_type}__{model_name}__"
                                    f"fold{fold:02d}__{stable_hash(payload)[:10]}"
                                ),
                                "specification_hash": stable_hash(payload),
                            }
                        )
    return pd.DataFrame(rows)


def plan_downstream(
    config_path: str | Path,
    *,
    outer_folds: Sequence[int] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    matrix = build_downstream_plan(config, outer_folds=outer_folds)
    return {
        "experiment_id": config["downstream"]["experiment_id"],
        "feature_count": int(config["downstream"]["expected_feature_count"]),
        "pm_targets": list(PM_METRICS),
        "variants": list(VARIANT_ORDER),
        "outer_folds": sorted(matrix["outer_fold"].unique().tolist()),
        "models": config["downstream"]["models"],
        "run_count": int(len(matrix)),
        "run_matrix_hash": stable_hash(matrix.to_dict(orient="records")),
        "hyperparameter_search": False,
        "models_trained": 0,
        "writes_performed": False,
    }


def build_downstream_manifest(
    config: Mapping[str, Any],
    matrix: pd.DataFrame,
    *,
    completed_run_count: int,
) -> dict[str, Any]:
    """Build runtime provenance without changing the execution protocol."""
    manifest = {
        "experiment_id": config["downstream"]["experiment_id"],
        "result_status": config["result_status"],
        "feature_cache_mode": "read_only",
        "feature_count": int(config["downstream"]["expected_feature_count"]),
        "run_count": int(completed_run_count),
        "run_matrix_hash": stable_hash(matrix.to_dict(orient="records")),
        "fixed_outer_folds": sorted(matrix["outer_fold"].unique().tolist()),
        "sample_universe_identical_across_variants": True,
        "q3_fit_scope": "outer_train_only",
        "hyperparameter_search": False,
    }
    manifest["protocol_hash"] = stable_hash(manifest)
    return manifest


def _load_feature_cache(cache_dir: Path, expected_count: int) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    required = {
        "features.npy",
        "feature_index.parquet",
        "feature_names.json",
        "feature_materialization_manifest.json",
    }
    missing = sorted(name for name in required if not (cache_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Feature cache is incomplete: {missing}")
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    index = pd.read_parquet(cache_dir / "feature_index.parquet")
    names_payload = json.loads((cache_dir / "feature_names.json").read_text(encoding="utf-8"))
    names = list(names_payload.get("feature_names", names_payload) if isinstance(names_payload, dict) else names_payload)
    if features.ndim != 2 or features.shape[1] != expected_count:
        raise ValueError(f"Expected feature shape [n, {expected_count}], got {features.shape}")
    if len(index) != len(features) or len(names) != expected_count:
        raise ValueError("Feature tensor, index, and names disagree")
    required_columns = {"sample_id", "subject_id", "record_id", "record_group_id", "outer_fold"}
    if not required_columns.issubset(index.columns):
        raise ValueError(f"Feature index is missing {sorted(required_columns - set(index.columns))}")
    if index["sample_id"].duplicated().any():
        raise ValueError("Feature cache contains duplicate sample_id")
    return features, index.reset_index(drop=True), names


def _prepare_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    scaling: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    train = imputer.fit_transform(x_train)
    test = imputer.transform(x_test)
    metadata: dict[str, Any] = {
        "imputation": "outer_train_median",
        "scaling": scaling,
        "fit_scope": "outer_train_only",
    }
    if scaling == "standard":
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
        metadata["mean"] = scaler.mean_.tolist()
        metadata["scale"] = scaler.scale_.tolist()
    elif scaling != "none":
        raise ValueError(f"Unsupported downstream scaling {scaling!r}")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Fold-local feature preprocessing produced NaN/Inf")
    return train, test, metadata


def run_downstream(
    config_path: str | Path,
    *,
    feature_cache_dir: str | Path,
    data_path: str | Path,
    output_dir: str | Path | None = None,
    outer_folds: Sequence[int] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    definition = config["downstream"]
    cache_dir = Path(feature_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = _repo_path(cache_dir)
    effective_output = _repo_path(output_dir or definition["output_dir"])
    features, feature_index, feature_names = _load_feature_cache(
        cache_dir, int(definition["expected_feature_count"])
    )
    columns = ["source", "subject_id", "record_id", "t_start", *TARGET_COLUMNS]
    pm_frame = prepare_pm_frame(pd.read_parquet(Path(data_path), columns=columns), config)
    variants = build_variants(pm_frame, config)
    position_by_id = pd.Series(pm_frame.index.to_numpy(), index=pm_frame["sample_id"]).to_dict()
    if not set(feature_index["sample_id"]).issubset(position_by_id):
        raise ValueError("Feature cache contains sample IDs absent from canonical PM data")
    pm_positions = np.asarray([position_by_id[value] for value in feature_index["sample_id"]], dtype=np.int64)
    matrix = build_downstream_plan(config, outer_folds=outer_folds)
    cohort_reference: dict[tuple[str, int], np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    effective_output.mkdir(parents=True, exist_ok=True)
    _write_csv(effective_output / "run_matrix.csv", matrix)
    for run in matrix.to_dict(orient="records"):
        metric = str(run["pm"])
        variant = str(run["variant"])
        fold = int(run["outer_fold"])
        task_type = str(run["task_type"])
        values = variants.values[metric][variant][pm_positions]
        available = np.isfinite(values)
        key = (metric, fold)
        sample_universe = feature_index.loc[available, "sample_id"].to_numpy()
        if key not in cohort_reference:
            cohort_reference[key] = sample_universe
        elif not np.array_equal(cohort_reference[key], sample_universe):
            raise RuntimeError("Target availability differs between PM variants")
        train_mask = available & feature_index["outer_fold"].ne(fold).to_numpy()
        test_mask = available & feature_index["outer_fold"].eq(fold).to_numpy()
        if not train_mask.any() or not test_mask.any():
            raise ValueError(f"Empty train/test partition for {metric} fold {fold}")
        y_train_continuous = values[train_mask]
        y_test_continuous = values[test_mask]
        target_manifest: dict[str, Any] | None = None
        if task_type == "classification":
            target_spec = get_target_spec(f"pm_{metric}_q3_fold_local")
            transform = build_fold_local_target_transform(target_spec).fit(y_train_continuous)
            y_train = transform.transform(y_train_continuous).astype(int)
            y_test = transform.transform(y_test_continuous).astype(int)
            target_manifest = build_target_transform_manifest(
                target_spec,
                transform,
                outer_fold=fold,
                outer_train_sample_ids=feature_index.loc[train_mask, "sample_id"].to_numpy(),
                outer_train_targets=y_train_continuous,
            )
            num_outputs = 3
        else:
            y_train, y_test = y_train_continuous, y_test_continuous
            num_outputs = 1
        x_train, x_test, preprocessing = _prepare_features(
            np.asarray(features[train_mask]),
            np.asarray(features[test_mask]),
            scaling=str(run["scaling"]),
        )
        params = dict(run["params"])
        model = build_model(
            model_name=str(run["model"]),
            task_type=task_type,
            input_shape=(x_train.shape[1],),
            num_outputs=num_outputs,
            params=params,
        )
        started = time.perf_counter()
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_test))
        elapsed = time.perf_counter() - started
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Downstream model produced NaN/Inf")
        probabilities = None
        if task_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
        metrics = (
            MetricsCalculator.calculate_all_metrics(
                y_test,
                prediction,
                y_proba=probabilities,
                labels=np.arange(3),
            )
            if task_type == "classification"
            else MetricsCalculator.calculate_regression_metrics(y_test, prediction)
        )
        run_dir = effective_output / str(run["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions = feature_index.loc[
            test_mask,
            ["sample_id", "subject_id", "record_id", "record_group_id", "outer_fold"],
        ].copy()
        predictions.insert(0, "pm", metric)
        predictions.insert(1, "variant", variant)
        predictions.insert(2, "task_type", task_type)
        predictions.insert(3, "model", str(run["model"]))
        predictions["y_true"] = y_test
        predictions["y_pred"] = prediction
        if probabilities is not None:
            for class_id in range(probabilities.shape[1]):
                predictions[f"proba_{class_id}"] = probabilities[:, class_id]
        predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "normalization_stats.json", preprocessing)
        _write_json(
            run_dir / "split.json",
            {
                "outer_fold": fold,
                "group_column": "subject_id",
                "train_subjects": sorted(feature_index.loc[train_mask, "subject_id"].astype(str).unique().tolist()),
                "test_subjects": sorted(feature_index.loc[test_mask, "subject_id"].astype(str).unique().tolist()),
                "sample_universe_hash": stable_hash(sample_universe.tolist()),
            },
        )
        if target_manifest is not None:
            _write_json(run_dir / "target_transform.json", target_manifest)
        scalar_metrics = {
            key: value for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        summary_rows.append({**run, "training_seconds": elapsed, **scalar_metrics})
    summary = pd.DataFrame(summary_rows)
    _write_csv(effective_output / "summary.csv", summary)
    manifest = build_downstream_manifest(
        config,
        matrix,
        completed_run_count=len(summary),
    )
    _write_json(effective_output / "manifest.json", manifest)
    return _jsonable(manifest)


__all__ = [
    "build_downstream_manifest",
    "build_downstream_plan",
    "plan_downstream",
    "run_downstream",
]
