"""Simple non-EEG controls for temporal target structure.

This is intentionally a thin analytical wrapper.  It reuses the sklearn model
factory and benchmark metrics, while GroupKFold, preprocessing, and estimator
fit are kept strictly inside each outer-train partition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bench.analysis.temporal_target_structure import (
    DIAGNOSTIC_METRICS,
    _selected_metrics,
    metrics_by_group,
)
from model_zoo.ML.sklearn_models import build_sklearn_model


POSITION_COLUMNS = [
    "normalized_record_progress",
    "absolute_window_index",
    "record_duration",
]
DIAGNOSTIC_FEATURES = {
    "D0": [],
    "D1": ["source"],
    "D2": POSITION_COLUMNS,
    "D3": ["source", *POSITION_COLUMNS],
}
FORBIDDEN_FEATURE_COLUMNS = {
    "subject_id",
    "record_id",
    "target_focus",
    "target_main",
    "label_q5",
    "sample_id",
    "t_start",
    "t_end",
}


def assign_subject_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign deterministic sklearn GroupKFold folds by subject."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    groups = frame["subject_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough subjects for requested GroupKFold")
    labels = frame["label_q5"].to_numpy(dtype=int)
    assignments = np.zeros(len(frame), dtype=np.int64)
    folds: dict[str, Any] = {}
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros((len(frame), 1)), labels, groups), start=1
    ):
        train_subjects = sorted(np.unique(groups[train_indices]).tolist())
        test_subjects = sorted(np.unique(groups[test_indices]).tolist())
        overlap = sorted(set(train_subjects) & set(test_subjects))
        if overlap:
            raise RuntimeError(f"Subject overlap in fold {fold}: {overlap}")
        assignments[test_indices] = fold
        folds[f"fold_{fold:02d}"] = {
            "fold": fold,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "subject_overlap": overlap,
        }
    if np.any(assignments == 0):
        raise RuntimeError("Every supervised row must be assigned to one test fold")
    output = frame.copy()
    output["outer_fold"] = assignments
    return output, {
        "protocol": "group_kfold_subject",
        "n_splits": n_splits,
        "shuffle": False,
        "group_column": "subject_id",
        "folds": folds,
    }


def align_with_canonical_predictions(
    frame: pd.DataFrame,
    reference_path: str | Path,
    *,
    label_col: str = "label_q5",
) -> dict[str, Any]:
    """Require exact identity/fold/target alignment with the canonical RF run."""

    path = Path(reference_path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical prediction artifact not found: {path}")
    reference = pd.read_parquet(
        path, columns=["sample_id", "fold", "subject_id", "y_true"]
    ).copy()
    current = frame[
        ["sample_id", "outer_fold", "subject_id", label_col]
    ].rename(columns={"outer_fold": "fold", label_col: "y_true"})
    for table in (reference, current):
        table["subject_id"] = table["subject_id"].astype(str)
        table["fold"] = table["fold"].astype(int)
        table["y_true"] = table["y_true"].astype(int)
    reference = reference.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    current = current.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if len(reference) != len(current):
        raise ValueError(
            f"Canonical alignment row mismatch: {len(reference)} != {len(current)}"
        )
    columns = ["sample_id", "fold", "subject_id", "y_true"]
    mismatches = {
        column: int((reference[column].to_numpy() != current[column].to_numpy()).sum())
        for column in columns
    }
    if any(mismatches.values()):
        raise ValueError(f"Canonical alignment failed: {mismatches}")
    return {
        "reference_path": path,
        "rows": int(len(current)),
        "columns": columns,
        "mismatches": mismatches,
        "exact_match": True,
    }


def _preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in feature_columns if column == "source"]
    numeric = [column for column in feature_columns if column != "source"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _probabilities_for_all_classes(
    estimator: Any,
    features: pd.DataFrame,
    *,
    n_classes: int,
) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = np.asarray(estimator.named_steps["model"].classes_, dtype=int)
    probabilities = np.zeros((len(features), n_classes), dtype=float)
    probabilities[:, classes] = raw
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Diagnostic probabilities contain NaN or Inf")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
        raise RuntimeError("Diagnostic probability rows do not sum to one")
    return probabilities


def _prediction_frame(
    test: pd.DataFrame,
    *,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    diagnostic_set: str,
    model_name: str,
    label_col: str,
    fold: int,
) -> pd.DataFrame:
    output = test[
        [
            "sample_id",
            "source",
            "subject_id",
            "record_id",
            "t_start",
            "absolute_window_index",
            "normalized_record_progress",
            "record_duration",
        ]
    ].copy()
    output["y_true"] = test[label_col].to_numpy(dtype=int)
    output["y_pred"] = np.asarray(y_pred, dtype=int)
    output["fold"] = int(fold)
    output["protocol"] = "group_kfold_subject"
    output["diagnostic_set"] = diagnostic_set
    output["model"] = model_name
    output["previous_sample_id"] = pd.NA
    output["prediction_id"] = [
        f"group_kfold_subject:{diagnostic_set}:{model_name}:{fold}:{sample_id}"
        for sample_id in output["sample_id"]
    ]
    for class_id in range(y_proba.shape[1]):
        output[f"proba_{class_id}"] = y_proba[:, class_id]
    return output


def _fold_metric_row(
    predictions: pd.DataFrame,
    *,
    diagnostic_set: str,
    model_name: str,
    fold: int,
) -> dict[str, Any]:
    return {
        "diagnostic_set": diagnostic_set,
        "model": model_name,
        "fold": int(fold),
        **_selected_metrics(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["y_pred"].to_numpy(dtype=int),
        ),
    }


def _summarize_folds(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(fold_rows)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["diagnostic_set", "model"], sort=True, observed=True
    ):
        diagnostic_set, model_name = keys
        row: dict[str, Any] = {
            "diagnostic_set": str(diagnostic_set),
            "model": str(model_name),
            "folds": int(len(group)),
        }
        for metric in DIAGNOSTIC_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return rows


def run_diagnostic_baselines(
    frame: pd.DataFrame,
    *,
    label_col: str = "label_q5",
    n_classes: int = 5,
    spec: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit D0-D3 controls with all learned state restricted to outer-train."""

    settings = dict(spec or {})
    expected_features = set().union(*DIAGNOSTIC_FEATURES.values())
    missing = sorted(expected_features - set(frame.columns))
    if missing:
        raise ValueError(f"Diagnostic covariates are missing: {missing}")
    forbidden_used = sorted(
        set().union(*DIAGNOSTIC_FEATURES.values()) & FORBIDDEN_FEATURE_COLUMNS
    )
    if forbidden_used:
        raise RuntimeError(f"Forbidden diagnostic inputs configured: {forbidden_used}")
    folds = sorted(frame["outer_fold"].astype(int).unique().tolist())
    expected_folds = list(range(1, int(settings.get("n_splits", 5)) + 1))
    if folds != expected_folds:
        raise ValueError(f"Expected folds {expected_folds}, observed {folds}")

    logistic_params = {
        "max_iter": 500,
        "solver": "lbfgs",
        "random_state": int(settings.get("random_state", 42)),
        **dict(settings.get("logistic_regression", {})),
    }
    forest_params = {
        "n_estimators": 50,
        "max_depth": 6,
        "min_samples_leaf": 10,
        "random_state": int(settings.get("random_state", 42)),
        "n_jobs": -1,
        **dict(settings.get("random_forest", {})),
    }
    model_settings = {
        "logistic_regression": logistic_params,
        "random_forest": forest_params,
    }
    prediction_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    fit_audit: list[dict[str, Any]] = []

    for fold in folds:
        train = frame.loc[frame["outer_fold"] != fold].copy()
        test = frame.loc[frame["outer_fold"] == fold].copy()
        train_subjects = sorted(train["subject_id"].astype(str).unique().tolist())
        test_subjects = sorted(test["subject_id"].astype(str).unique().tolist())
        overlap = sorted(set(train_subjects) & set(test_subjects))
        if overlap:
            raise RuntimeError(f"Outer-test subjects entered train in fold {fold}: {overlap}")
        y_train = train[label_col].to_numpy(dtype=int)

        majority_class = int(np.argmax(np.bincount(y_train, minlength=n_classes)))
        majority_pred = np.full(len(test), majority_class, dtype=int)
        majority_proba = np.zeros((len(test), n_classes), dtype=float)
        majority_proba[:, majority_class] = 1.0
        majority_frame = _prediction_frame(
            test,
            y_pred=majority_pred,
            y_proba=majority_proba,
            diagnostic_set="D0",
            model_name="majority_outer_train",
            label_col=label_col,
            fold=fold,
        )
        prediction_parts.append(majority_frame)
        fold_metrics.append(
            _fold_metric_row(
                majority_frame,
                diagnostic_set="D0",
                model_name="majority_outer_train",
                fold=fold,
            )
        )
        fit_audit.append(
            {
                "fold": fold,
                "diagnostic_set": "D0",
                "model": "majority_outer_train",
                "feature_columns": [],
                "train_rows_fit": int(len(train)),
                "test_rows": int(len(test)),
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
                "subject_overlap": overlap,
                "majority_class": majority_class,
                "majority_from_outer_train_only": True,
            }
        )

        for diagnostic_set in ("D1", "D2", "D3"):
            feature_columns = DIAGNOSTIC_FEATURES[diagnostic_set]
            categorical = [column for column in feature_columns if column == "source"]
            numeric = [column for column in feature_columns if column != "source"]
            for model_name, params in model_settings.items():
                transformer = _preprocessor(feature_columns)
                estimator = build_sklearn_model(
                    model_name=model_name,
                    task_type="classification",
                    params=params,
                )
                pipeline = Pipeline(
                    [("preprocess", transformer), ("model", estimator)]
                )
                pipeline.fit(train[feature_columns], y_train)
                y_pred = pipeline.predict(test[feature_columns]).astype(int)
                y_proba = _probabilities_for_all_classes(
                    pipeline, test[feature_columns], n_classes=n_classes
                )
                prediction_frame = _prediction_frame(
                    test,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    diagnostic_set=diagnostic_set,
                    model_name=model_name,
                    label_col=label_col,
                    fold=fold,
                )
                prediction_parts.append(prediction_frame)
                fold_metrics.append(
                    _fold_metric_row(
                        prediction_frame,
                        diagnostic_set=diagnostic_set,
                        model_name=model_name,
                        fold=fold,
                    )
                )
                audit_row: dict[str, Any] = {
                    "fold": fold,
                    "diagnostic_set": diagnostic_set,
                    "model": model_name,
                    "feature_columns": feature_columns,
                    "categorical_columns": categorical,
                    "numeric_columns": numeric,
                    "forbidden_feature_overlap": sorted(
                        set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
                    ),
                    "train_rows_fit": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_subjects": train_subjects,
                    "test_subjects": test_subjects,
                    "subject_overlap": overlap,
                    "preprocessor_fit_partition": "outer_train",
                    "estimator_fit_partition": "outer_train",
                }
                if numeric:
                    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
                    audit_row["scaler_mean"] = {
                        column: float(value)
                        for column, value in zip(numeric, scaler.mean_)
                    }
                    audit_row["outer_train_numeric_mean"] = {
                        column: float(train[column].mean()) for column in numeric
                    }
                fit_audit.append(audit_row)

    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    if not predictions["prediction_id"].is_unique:
        raise RuntimeError("Baseline prediction IDs are not unique")
    expected_samples = set(frame["sample_id"].tolist())
    overall_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    identity_reference: pd.DataFrame | None = None
    for keys, group in predictions.groupby(
        ["diagnostic_set", "model"], sort=True, observed=True
    ):
        diagnostic_set, model_name = keys
        if set(group["sample_id"].tolist()) != expected_samples or not group["sample_id"].is_unique:
            raise RuntimeError(
                f"{diagnostic_set}/{model_name} does not predict every sample exactly once"
            )
        identity = group[["sample_id", "fold", "subject_id", "y_true"]].sort_values(
            "sample_id", kind="mergesort"
        ).reset_index(drop=True)
        if identity_reference is None:
            identity_reference = identity
        elif not identity.equals(identity_reference):
            raise RuntimeError("Diagnostic variants have misaligned folds or targets")
        feature_columns = DIAGNOSTIC_FEATURES[str(diagnostic_set)]
        overall_rows.append(
            {
                "diagnostic_set": str(diagnostic_set),
                "model": str(model_name),
                "feature_columns": feature_columns,
                "metrics": _selected_metrics(
                    group["y_true"].to_numpy(dtype=int),
                    group["y_pred"].to_numpy(dtype=int),
                ),
            }
        )
        for row in metrics_by_group(group, "source"):
            source_rows.append(
                {
                    "diagnostic_set": str(diagnostic_set),
                    "model": str(model_name),
                    **row,
                }
            )
        for row in metrics_by_group(group, "subject_id"):
            subject_rows.append(
                {
                    "diagnostic_set": str(diagnostic_set),
                    "model": str(model_name),
                    **row,
                }
            )

    return predictions, {
        "feature_policy": {
            "D0": "outer-train class mode",
            "D1": DIAGNOSTIC_FEATURES["D1"],
            "D2": DIAGNOSTIC_FEATURES["D2"],
            "D3": DIAGNOSTIC_FEATURES["D3"],
            "forbidden": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "eeg_features": [],
            "pow_features": [],
        },
        "overall": overall_rows,
        "by_fold": fold_metrics,
        "fold_summary": _summarize_folds(fold_metrics),
        "by_source": source_rows,
        "by_subject": subject_rows,
        "fit_audit": fit_audit,
        "identity_alignment": {
            "columns": ["sample_id", "fold", "subject_id", "y_true"],
            "variants": int(len(overall_rows)),
            "rows_per_variant": int(len(frame)),
            "exact_match": True,
        },
    }


__all__ = [
    "DIAGNOSTIC_FEATURES",
    "FORBIDDEN_FEATURE_COLUMNS",
    "POSITION_COLUMNS",
    "align_with_canonical_predictions",
    "assign_subject_folds",
    "run_diagnostic_baselines",
]
