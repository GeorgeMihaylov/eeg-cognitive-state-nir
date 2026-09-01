"""Train-relative EEG/POW outlier audit and scaling-trial summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.core.abstract_task import TaskSplit


DEFAULT_EXTREME_FEATURES = (
    "POW.T8.BetaL__min",
    "POW.T8.Alpha__min",
)


def _feature_statistics(
    values: np.ndarray,
    feature_names: Sequence[str],
    *,
    scope: str,
    near_constant_threshold: float,
) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.percentile(values, [25, 50, 75], axis=0)
    standard_deviation = values.std(axis=0, dtype=np.float64)
    rows = []
    for index, feature_name in enumerate(feature_names):
        column = values[:, index]
        iqr = float(q3[index] - q1[index])
        rows.append({
            "scope": scope,
            "feature_index": index,
            "feature_name": feature_name,
            "feature_group": (
                "POW"
                if str(feature_name).upper().startswith("POW.")
                else "EEG"
            ),
            "mean": float(np.mean(column)),
            "std": float(standard_deviation[index]),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
            "median": float(median[index]),
            "q1": float(q1[index]),
            "q3": float(q3[index]),
            "iqr": iqr,
            "zero_fraction": float(np.mean(column == 0)),
            "unique_fraction": float(len(np.unique(column)) / len(column)),
            "near_constant": bool(
                standard_deviation[index] < near_constant_threshold
                or iqr < near_constant_threshold
            ),
        })
    return pd.DataFrame(rows)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        json.dump(
            payload,
            output,
            indent=2,
            default=lambda value: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else (
                    value.item()
                    if isinstance(value, np.generic)
                    else str(value)
                )
            ),
        )


def run_feature_outlier_audit(
    split: TaskSplit,
    inner_train_indices: Sequence[int],
    inner_validation_indices: Sequence[int],
    output_dir: Path,
    *,
    audited_subject: str = "8191f1d9",
    extreme_features: Sequence[str] = DEFAULT_EXTREME_FEATURES,
    near_constant_threshold: float = 1e-8,
    extreme_window_count: int = 20,
) -> Dict[str, Any]:
    """Audit one outer fold using only inner-train distribution estimates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_indices = np.asarray(inner_train_indices, dtype=np.int64)
    validation_indices = np.asarray(
        inner_validation_indices, dtype=np.int64
    )
    X_outer_train = np.asarray(split.X_train, dtype=np.float64)
    X_inner_train = X_outer_train[train_indices]
    X_inner_validation = X_outer_train[validation_indices]
    feature_names = tuple(str(name) for name in split.feature_names or ())
    if X_outer_train.ndim != 2 or X_outer_train.shape[1] != len(feature_names):
        raise ValueError(
            "Feature audit requires a two-dimensional feature matrix with "
            "ordered feature names"
        )
    if not np.isfinite(X_outer_train).all():
        raise ValueError("Feature audit input contains NaN or infinite values")
    if near_constant_threshold <= 0:
        raise ValueError("near_constant_threshold must be positive")

    distribution = _feature_statistics(
        X_inner_train,
        feature_names,
        scope="inner_train",
        near_constant_threshold=near_constant_threshold,
    )
    source_values = np.asarray(
        split.row_metadata_train.get(
            "source", np.full(len(X_outer_train), "unknown", dtype=object)
        )
    ).astype(str)
    feature_lookup = {name: index for index, name in enumerate(feature_names)}
    missing_extreme = [
        name for name in extreme_features if name not in feature_lookup
    ]
    if missing_extreme:
        raise ValueError(
            f"Extreme audit features are missing: {missing_extreme}"
        )
    source_rows = []
    inner_sources = source_values[train_indices]
    for source in sorted(np.unique(source_values)):
        source_matrix = X_outer_train[source_values == source]
        for feature_name in extreme_features:
            index = feature_lookup[feature_name]
            train_column = X_inner_train[:, index]
            source_column = source_matrix[:, index]
            q1, q3 = np.percentile(source_column, [25, 75])
            source_rows.append({
                "scope": f"source:{source}",
                "feature_index": index,
                "feature_name": feature_name,
                "feature_group": "POW",
                "mean": float(np.mean(source_column)),
                "std": float(np.std(source_column)),
                "min": float(np.min(source_column)),
                "max": float(np.max(source_column)),
                "median": float(np.median(source_column)),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "zero_fraction": float(np.mean(source_column == 0)),
                "unique_fraction": float(
                    len(np.unique(source_column)) / len(source_column)
                ),
                "near_constant": bool(
                    np.std(source_column) < near_constant_threshold
                    or q3 - q1 < near_constant_threshold
                ),
                "train_q99_9_exceedance_fraction": float(
                    np.mean(source_column > np.percentile(train_column, 99.9))
                ),
            })
    distribution = pd.concat(
        [distribution, pd.DataFrame(source_rows)],
        ignore_index=True,
        sort=False,
    )
    distribution_path = output_dir / "feature_distribution_audit.csv"
    distribution.to_csv(distribution_path, index=False)

    near_constant = distribution.loc[
        (distribution["scope"] == "inner_train")
        & distribution["near_constant"].astype(bool)
    ].copy()
    near_constant_path = output_dir / "near_constant_features.csv"
    near_constant.to_csv(near_constant_path, index=False)

    train_mean = X_inner_train.mean(axis=0, dtype=np.float64)
    train_std = X_inner_train.std(axis=0, dtype=np.float64)
    train_q1, train_median, train_q3 = np.percentile(
        X_inner_train, [25, 50, 75], axis=0
    )
    train_iqr = train_q3 - train_q1
    safe_std = np.where(train_std < near_constant_threshold, 1.0, train_std)
    safe_iqr = np.where(train_iqr < near_constant_threshold, 1.0, train_iqr)
    train_min = X_inner_train.min(axis=0)
    train_max = X_inner_train.max(axis=0)

    validation_subjects = np.asarray(
        split.subject_train
    ).astype(str)[validation_indices]
    validation_sources = source_values[validation_indices]
    shift_rows = []
    for subject_id in np.unique(validation_subjects):
        mask = validation_subjects == subject_id
        subject_values = X_inner_validation[mask]
        z_scores = np.abs((subject_values - train_mean) / safe_std)
        robust_z = np.abs((subject_values - train_median) / safe_iqr)
        feature_maximums = np.max(z_scores, axis=0)
        max_flat_index = int(np.argmax(z_scores))
        _, max_feature_index = np.unravel_index(
            max_flat_index, z_scores.shape
        )
        shift_rows.append({
            "subject_id": subject_id,
            "source": "|".join(
                sorted(set(validation_sources[mask].astype(str).tolist()))
            ),
            "windows": int(np.sum(mask)),
            "max_abs_z": float(np.max(z_scores)),
            "p95_abs_z": float(np.percentile(z_scores, 95)),
            "p99_abs_z": float(np.percentile(z_scores, 99)),
            "features_abs_z_gt_5": int(np.sum(feature_maximums > 5)),
            "features_abs_z_gt_10": int(np.sum(feature_maximums > 10)),
            "features_abs_z_gt_100": int(np.sum(feature_maximums > 100)),
            "features_abs_z_gt_1000": int(np.sum(feature_maximums > 1000)),
            "max_robust_z": float(np.max(robust_z)),
            "values_outside_train_minmax": int(np.sum(
                (subject_values < train_min) | (subject_values > train_max)
            )),
            "max_z_feature": feature_names[max_feature_index],
        })
    subject_shift = pd.DataFrame(shift_rows).sort_values(
        "max_abs_z", ascending=False
    )
    subject_shift_path = output_dir / "subject_shift_audit.csv"
    subject_shift.to_csv(subject_shift_path, index=False)

    subject_ids = np.asarray(split.subject_train).astype(str)
    subject_mask = subject_ids == str(audited_subject)
    if not np.any(subject_mask):
        raise ValueError(
            f"Audited subject {audited_subject!r} is absent from outer train"
        )
    subject_indices = np.flatnonzero(subject_mask)
    sample_ids = np.asarray(split.sample_id_train)
    record_ids = np.asarray(split.record_id_train).astype(str)
    metadata = split.row_metadata_train
    extreme_rows = []
    for feature_name in extreme_features:
        feature_index = feature_lookup[feature_name]
        order = subject_indices[
            np.argsort(
                -X_outer_train[subject_indices, feature_index],
                kind="mergesort",
            )[:extreme_window_count]
        ]
        for index in order:
            raw_value = X_outer_train[index, feature_index]
            extreme_rows.append({
                "feature_name": feature_name,
                "sample_id": sample_ids[index],
                "subject_id": subject_ids[index],
                "source": source_values[index],
                "record_id": record_ids[index],
                "record_group_id": np.asarray(
                    metadata.get("record_group_id", record_ids)
                )[index],
                "t_start": np.asarray(
                    metadata.get("t_start", np.full(len(subject_ids), np.nan))
                )[index],
                "t_end": np.asarray(
                    metadata.get("t_end", np.full(len(subject_ids), np.nan))
                )[index],
                "raw_value": float(raw_value),
                "train_median": float(train_median[feature_index]),
                "train_mean": float(train_mean[feature_index]),
                "train_std": float(train_std[feature_index]),
                "train_iqr": float(train_iqr[feature_index]),
                "standard_z": float(
                    (raw_value - train_mean[feature_index])
                    / safe_std[feature_index]
                ),
                "robust_z": float(
                    (raw_value - train_median[feature_index])
                    / safe_iqr[feature_index]
                ),
                "partition": (
                    "inner_validation"
                    if index in set(validation_indices.tolist())
                    else "inner_train"
                ),
            })
    extreme_windows = pd.DataFrame(extreme_rows)
    extreme_path = output_dir / "extreme_windows.csv"
    extreme_windows.to_csv(extreme_path, index=False)

    subject_frame = pd.DataFrame({
        "sample_id": sample_ids[subject_mask],
        "record_id": record_ids[subject_mask],
        "source": source_values[subject_mask],
    })
    for key in ("record_group_id", "t_start", "t_end"):
        if key in metadata:
            subject_frame[key] = np.asarray(metadata[key])[subject_mask]
    targets = np.asarray(split.y_train, dtype=np.float64)[subject_mask]
    target_names = list(split.metadata.get("target_names") or [])
    target_summary = {
        name: {
            "mean": float(np.mean(targets[:, index])),
            "std": float(np.std(targets[:, index])),
            "min": float(np.min(targets[:, index])),
            "max": float(np.max(targets[:, index])),
        }
        for index, name in enumerate(target_names)
    }
    subject_features = X_outer_train[subject_mask]
    eeg_mask = np.asarray(
        [not name.upper().startswith("POW.") for name in feature_names]
    )
    pow_mask = ~eeg_mask
    audited_standard_z = np.abs(
        (subject_features - train_mean) / safe_std
    )
    audited_robust_z = np.abs(
        (subject_features - train_median) / safe_iqr
    )
    inner_scope = distribution["scope"] == "inner_train"
    distribution.loc[
        inner_scope, "audited_subject_max_abs_z"
    ] = np.max(audited_standard_z, axis=0)
    distribution.loc[
        inner_scope, "audited_subject_p99_abs_z"
    ] = np.percentile(audited_standard_z, 99, axis=0)
    distribution.loc[
        inner_scope, "audited_subject_max_abs_robust_z"
    ] = np.max(audited_robust_z, axis=0)
    distribution.to_csv(distribution_path, index=False)
    record_extremes = []
    for record_id in np.unique(record_ids[subject_mask]):
        record_mask = subject_mask & (record_ids == record_id)
        row = {
            "record_id": record_id,
            "windows": int(np.sum(record_mask)),
        }
        for feature_name in extreme_features:
            index = feature_lookup[feature_name]
            row[f"max_{feature_name}"] = float(
                np.max(X_outer_train[record_mask, index])
            )
        record_extremes.append(row)
    summary = {
        "scope": "outer_fold_01_with_inner_train_relative_statistics",
        "audited_subject": audited_subject,
        "outer_train_rows": int(len(X_outer_train)),
        "inner_train_rows": int(len(train_indices)),
        "inner_validation_rows": int(len(validation_indices)),
        "inner_train_subjects": int(
            len(np.unique(subject_ids[train_indices]))
        ),
        "inner_validation_subjects": int(
            len(np.unique(subject_ids[validation_indices]))
        ),
        "fit_validation_subject_overlap": np.intersect1d(
            subject_ids[train_indices], subject_ids[validation_indices]
        ).tolist(),
        "audited_subject_windows": int(np.sum(subject_mask)),
        "audited_subject_sources": subject_frame["source"]
        .value_counts()
        .to_dict(),
        "audited_subject_records": int(subject_frame["record_id"].nunique()),
        "audited_subject_record_ids": sorted(
            subject_frame["record_id"].unique().tolist()
        ),
        "audited_subject_record_group_ids": sorted(
            subject_frame.get("record_group_id", subject_frame["record_id"])
            .astype(str)
            .unique()
            .tolist()
        ),
        "audited_subject_time_range": {
            key: (
                None
                if key not in subject_frame
                else [
                    float(subject_frame[key].min()),
                    float(subject_frame[key].max()),
                ]
            )
            for key in ("t_start", "t_end")
        },
        "audited_subject_duplicate_sample_ids": int(
            subject_frame["sample_id"].duplicated().sum()
        ),
        "nan_values": int(np.isnan(subject_features).sum()),
        "infinite_values": int(np.isinf(subject_features).sum()),
        "target_distributions": target_summary,
        "EEG_distribution": {
            "min": float(np.min(subject_features[:, eeg_mask])),
            "median": float(np.median(subject_features[:, eeg_mask])),
            "max": float(np.max(subject_features[:, eeg_mask])),
        },
        "POW_distribution": {
            "min": float(np.min(subject_features[:, pow_mask])),
            "median": float(np.median(subject_features[:, pow_mask])),
            "max": float(np.max(subject_features[:, pow_mask])),
        },
        "record_extremes": record_extremes,
        "near_constant_threshold": near_constant_threshold,
        "near_constant_feature_count": int(len(near_constant)),
        "maximum_standard_z": float(subject_shift["max_abs_z"].max()),
        "maximum_standard_z_subject": str(
            subject_shift.iloc[0]["subject_id"]
        ),
        "maximum_standard_z_feature": str(
            subject_shift.iloc[0]["max_z_feature"]
        ),
        "source_membership_inner_train": pd.Series(inner_sources)
        .value_counts()
        .to_dict(),
    }
    summary_path = output_dir / "feature_outlier_audit_summary.json"
    _json_dump(summary_path, summary)
    return {
        "feature_distribution_audit": str(distribution_path),
        "subject_shift_audit": str(subject_shift_path),
        "extreme_windows": str(extreme_path),
        "near_constant_features": str(near_constant_path),
        "summary": str(summary_path),
        "summary_payload": summary,
    }


def summarize_scaling_results(
    benchmark_results_path: Path,
    output_dir: Path,
) -> Dict[str, str]:
    """Combine one-fold A--F artifacts into comparison CSV files."""
    benchmark_results_path = Path(benchmark_results_path)
    output_dir = Path(output_dir)
    with open(benchmark_results_path, encoding="utf-8") as source:
        payload = json.load(source)
    trial_rows = []
    subject_frames = []
    for dataset_name, dataset_result in payload.items():
        task_models = dataset_result["models"][
            "performance_metrics_regression"
        ]
        for model_name, model_result in task_models.items():
            fold = model_result["group_kfold_subject"]["folds"]["fold_01"]
            metrics = fold["metrics"]
            training = fold.get("training", {})
            artifacts = fold["artifacts"]
            with open(artifacts["feature_transform"], encoding="utf-8") as source:
                transform = json.load(source)
            subject_frame = pd.read_csv(
                artifacts["robust_scaling_subject_metrics"]
            )
            subject_frame["trial_id"] = model_name
            subject_frames.append(subject_frame)
            predictions = pd.read_parquet(artifacts["predictions"])
            prediction_columns = [
                column
                for column in predictions
                if column.startswith("y_pred_")
            ]
            prediction_values = predictions[prediction_columns].to_numpy(
                dtype=float
            )
            validation_diag = transform["diagnostics"]["inner_validation"]
            validation_prediction_diag = transform.get(
                "validation_prediction_diagnostics", {}
            )
            subject_819 = subject_frame.loc[
                subject_frame["subject_id"].astype(str) == "8191f1d9"
            ]
            worst = subject_frame.sort_values("mse", ascending=False).iloc[0]
            trial_rows.append({
                "trial_id": model_name,
                "strategy": transform["strategy"],
                "mae_macro": metrics["mae_macro"],
                "rmse_macro": metrics["rmse_macro"],
                "r2_macro": metrics["r2_macro"],
                "pearson_macro": metrics["pearson_macro"],
                "spearman_macro": metrics["spearman_macro"],
                "best_epoch": training.get("best_epoch"),
                "best_validation_loss": training.get(
                    "best_validation_loss"
                ),
                "training_time_seconds": fold["training_time"],
                "validation_max_abs_transformed_feature": validation_diag[
                    "max_abs"
                ],
                "validation_p99_abs_transformed_feature": validation_diag[
                    "p99_abs"
                ],
                "validation_values_abs_gt_5": validation_diag[
                    "values_abs_gt_5"
                ],
                "validation_values_abs_gt_10": validation_diag[
                    "values_abs_gt_10"
                ],
                "validation_values_abs_gt_100": validation_diag[
                    "values_abs_gt_100"
                ],
                "validation_max_abs_prediction": (
                    validation_prediction_diag.get("max_abs")
                ),
                "validation_p99_abs_prediction": (
                    validation_prediction_diag.get("p99_abs")
                ),
                "outer_max_abs_prediction": float(
                    np.max(np.abs(prediction_values))
                ),
                "outer_p99_abs_prediction": float(
                    np.percentile(np.abs(prediction_values), 99)
                ),
                "nonfinite_predictions": int(
                    np.sum(~np.isfinite(prediction_values))
                    + int(validation_prediction_diag.get(
                        "nonfinite_values", 0
                    ))
                ),
                "subject_8191f1d9_validation_mse": (
                    None
                    if subject_819.empty
                    else float(subject_819.iloc[0]["mse"])
                ),
                "worst_validation_subject": str(worst["subject_id"]),
                "worst_validation_subject_mse": float(worst["mse"]),
                "fit_validation_overlap": transform["leakage_audit"][
                    "inner_group_overlap"
                ],
                "fit_test_overlap": transform["leakage_audit"][
                    "train_outer_test_overlap_count"
                ],
            })
    trials = pd.DataFrame(trial_rows)
    subjects = pd.concat(subject_frames, ignore_index=True)
    baseline_name = (
        "A_standard"
        if "A_standard" in set(subjects["trial_id"])
        else subjects["trial_id"].iloc[0]
    )
    baseline = subjects.loc[
        subjects["trial_id"] == baseline_name,
        ["subject_id", "mse", "mae"],
    ].rename(columns={"mse": "baseline_mse", "mae": "baseline_mae"})
    subjects = subjects.merge(baseline, on="subject_id", how="left")
    subjects["mse_change_vs_standard"] = (
        subjects["mse"] - subjects["baseline_mse"]
    )
    subjects["mae_change_vs_standard"] = (
        subjects["mae"] - subjects["baseline_mae"]
    )
    subjects["improved_vs_standard"] = (
        subjects["mse_change_vs_standard"] < 0
    )
    improvement = (
        subjects.groupby("trial_id")["improved_vs_standard"]
        .mean()
        .rename("validation_subject_improvement_fraction")
    )
    trials = trials.merge(improvement, on="trial_id", how="left")
    trials_path = output_dir / "robust_scaling_trials.csv"
    subjects_path = output_dir / "robust_scaling_subject_metrics.csv"
    trials.to_csv(trials_path, index=False)
    subjects.to_csv(subjects_path, index=False)
    return {
        "robust_scaling_trials": str(trials_path),
        "robust_scaling_subject_metrics": str(subjects_path),
    }
