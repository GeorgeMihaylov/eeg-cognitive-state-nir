"""Preregistered LOW-vs-HIGH classification for seven continuous PM targets.

Only the fixed causal previous-window alignment ``EEG(t-10 s) -> PM(t)`` is
allowed.  For each outer fold and PM, the lower and upper tertile thresholds
are fitted on continuous outer-train targets only.  The strict middle tertile
is excluded, and the unchanged train thresholds are applied to outer-test.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    roc_auc_score,
)

from bench.experiments.pm_eeg_lag_confirmatory import (
    PM_NAMES,
    build_previous_window_pairing,
    stable_hash,
    validate_cache_contract,
)
from bench.features.cogstate_feature_cache import load_feature_cache
from cogstate.model_zoo import build_model


SCHEMA_VERSION = "pm-low-high-q3-extremes-confirmatory-v1"
FIXED_LAG_SECONDS = -10
FIXED_LABELS = (0, 1)
PRIMARY_METRICS = ("balanced_accuracy", "f1", "roc_auc")
SECONDARY_METRICS = (
    "pr_auc",
    "low_recall",
    "high_recall",
    "precision",
    "accuracy",
)
METRICS = (*PRIMARY_METRICS, *SECONDARY_METRICS)


def _sample_hash(values: Sequence[Any]) -> str:
    return stable_hash([str(value) for value in values])


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen scientific specification."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if config.get("task") != "binary_classification":
        raise ValueError("The protocol task must be binary_classification")
    if tuple(config.get("pm_names", ())) != PM_NAMES:
        raise ValueError("The protocol must contain exactly seven PM in canonical order")
    expected_targets = tuple(f"target_{pm}" for pm in PM_NAMES)
    if tuple(config.get("target_ids", ())) != expected_targets:
        raise ValueError("Only the seven canonical continuous target_* PM are allowed")

    alignment = config.get("alignment", {})
    expected_alignment = {
        "condition_id": "lag_minus_10s",
        "lag_seconds": FIXED_LAG_SECONDS,
        "mapping": "EEG(t-10s) -> PM(t)",
        "time_column": "t_start",
        "step_seconds": 10.0,
        "require_same_record_id": True,
        "require_same_subject_id": True,
        "require_same_outer_fold": True,
        "require_exact_previous_window": True,
        "gap_policy": "exclude_no_substitution",
    }
    if alignment != expected_alignment:
        raise ValueError("Alignment is frozen at the exact record-local lag -10 contract")
    if "conditions" in config or "candidate_lags_seconds" in config:
        raise ValueError("Lag comparison and lag search fields are forbidden")

    expected_transform = {
        "name": "outer_train_q33_q67_extremes",
        "fit_scope": "outer_train_continuous_complete_cases",
        "q_low": 1.0 / 3.0,
        "q_high": 2.0 / 3.0,
        "low_rule": "y <= q_low",
        "high_rule": "y >= q_high",
        "middle_rule": "q_low < y < q_high",
        "middle_policy": "exclude",
        "labels": {"LOW": 0, "HIGH": 1},
    }
    if config.get("target_transform") != expected_transform:
        raise ValueError("Binary target must use outer-train q33/q67 extremes exactly")

    evaluation = config.get("evaluation", {})
    if evaluation.get("folds") != [1, 2, 3, 4, 5]:
        raise ValueError("Fixed outer folds must be [1, 2, 3, 4, 5]")
    if evaluation.get("group_column") != "subject_id":
        raise ValueError("Outer grouping must use subject_id")
    if evaluation.get("precomputed_fold_column") != "outer_fold":
        raise ValueError("The canonical precomputed outer_fold column is required")
    if tuple(evaluation.get("primary_metrics", ())) != tuple(
        f"participant_macro_{metric}" for metric in PRIMARY_METRICS
    ):
        raise ValueError("Primary metrics are frozen")
    if tuple(evaluation.get("secondary_metrics", ())) != tuple(
        f"participant_macro_{metric}" for metric in SECONDARY_METRICS
    ):
        raise ValueError("Secondary metrics are frozen")
    if evaluation.get("probability_metrics_source") != "predict_proba_high":
        raise ValueError("ROC-AUC and PR-AUC must use HIGH probabilities")
    if (
        evaluation.get("single_class_participant_auc_policy")
        != "undefined_exclude_metric_only"
    ):
        raise ValueError("Single-class participant AUC policy is frozen")
    expected_metric_definitions = {
        "balanced_accuracy": "mean_of_defined_LOW_and_HIGH_recalls",
        "macro_f1": "fixed_LOW_HIGH_labels_zero_division_0",
        "roc_auc": "roc_auc_score_of_predict_proba_HIGH",
        "pr_auc": "average_precision_score_of_predict_proba_HIGH",
        "low_recall": "undefined_when_participant_has_no_LOW_samples",
        "high_recall": "undefined_when_participant_has_no_HIGH_samples",
        "precision": "macro_LOW_HIGH_precision_zero_division_0",
        "accuracy": "window_accuracy",
    }
    if evaluation.get("metric_definitions") != expected_metric_definitions:
        raise ValueError("Participant metric definitions are frozen")

    model = config.get("model", {})
    expected_model = {
        "name": "xgboost",
        "task_type": "classification",
        "estimator": "XGBClassifier",
        "seed": 42,
        "params": {"n_estimators": 200, "n_jobs": 4, "random_state": 42},
    }
    if model != expected_model:
        raise ValueError("The XGBoost classifier and its parameters are frozen")
    identity = config.get("feature_cache_identity", {})
    if int(identity.get("n_features", -1)) != 371:
        raise ValueError("Canonical feature count must be 371")
    if bool(identity.get("target_columns_present", True)):
        raise ValueError("Target columns must not enter X")
    if any(key in config for key in ("target_overrides", "per_target", "focus_override")):
        raise ValueError("Target-specific and Focus-specific overrides are forbidden")
    serialized = json.dumps(config, sort_keys=True).lower()
    if any(token in serialized for token in ("median", "label_q5", '"q2"', "lag_0")):
        raise ValueError("Median/Q2, Q5 and lag-0 paths are forbidden")
    return config


def _target_table(path: Path) -> pd.DataFrame:
    columns = ["subject_id", "record_id", *(f"target_{pm}" for pm in PM_NAMES)]
    frame = pd.read_parquet(path, columns=columns)
    if "sample_id" not in frame.columns:
        frame = frame.copy()
        frame.insert(0, "sample_id", frame.index.to_numpy())
    frame = frame.reset_index(drop=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Processed target table contains duplicate sample_id")
    return frame


@dataclass(frozen=True)
class ExtremeThresholds:
    """Two outer-train thresholds defining LOW, excluded middle and HIGH."""

    q_low: float
    q_high: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.q_low, self.q_high]).all():
            raise ValueError("Extreme thresholds must be finite")
        if not self.q_low < self.q_high:
            raise ValueError("q_low must be strictly below q_high")

    def transform(self, values: Sequence[float]) -> np.ndarray:
        return apply_extreme_labels(values, q_low=self.q_low, q_high=self.q_high)


def fit_extreme_thresholds(values: Sequence[float]) -> ExtremeThresholds:
    """Fit q33/q67 using only values explicitly passed by the caller."""
    continuous = np.asarray(values, dtype=float).reshape(-1)
    continuous = continuous[np.isfinite(continuous)]
    if not len(continuous):
        raise ValueError("Cannot fit thresholds without finite outer-train targets")
    q_low, q_high = np.quantile(continuous, [1.0 / 3.0, 2.0 / 3.0])
    return ExtremeThresholds(q_low=float(q_low), q_high=float(q_high))


def apply_extreme_labels(
    values: Sequence[float], *, q_low: float, q_high: float
) -> np.ndarray:
    """Return 0 for LOW, 1 for HIGH and NaN for middle/missing targets."""
    thresholds = ExtremeThresholds(float(q_low), float(q_high))
    continuous = np.asarray(values, dtype=float).reshape(-1)
    labels = np.full(continuous.shape, np.nan, dtype=float)
    finite = np.isfinite(continuous)
    labels[finite & (continuous <= thresholds.q_low)] = 0.0
    labels[finite & (continuous >= thresholds.q_high)] = 1.0
    return labels


def build_pm_temporal_cohorts(
    full: pd.DataFrame,
    temporal_pairing: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build PM complete-case cohorts after exact previous-window pairing."""
    target_lookup = full.set_index("sample_id")
    target_ids = temporal_pairing["target_sample_id"].to_numpy()
    cohorts: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    for pm in PM_NAMES:
        target_id = f"target_{pm}"
        canonical_values = pd.to_numeric(full[target_id], errors="coerce").to_numpy(
            dtype=float
        )
        paired_values = pd.to_numeric(
            target_lookup.loc[target_ids, target_id], errors="coerce"
        ).to_numpy(dtype=float)
        keep = np.isfinite(paired_values)
        cohort = temporal_pairing.loc[keep].copy().reset_index(drop=True)
        cohort["continuous_target"] = paired_values[keep]
        if cohort.empty:
            raise ValueError(f"{target_id} has no exact-lag complete cases")
        if cohort["target_sample_id"].duplicated().any():
            raise RuntimeError(f"{target_id}: duplicate matched target sample IDs")
        if not np.allclose(
            cohort["target_time"].to_numpy(dtype=float)
            - cohort["feature_time_lag_minus_10s"].to_numpy(dtype=float),
            10.0,
        ):
            raise RuntimeError(f"{target_id}: non-exact -10 second pairing")
        complete_rows = int(np.isfinite(canonical_values).sum())
        summaries[pm] = {
            "pm": pm,
            "target_id": target_id,
            "canonical_complete_case_rows": complete_rows,
            "canonical_missing_target_rows": int(len(full) - complete_rows),
            "temporal_matched_rows": int(len(cohort)),
            "lost_without_exact_previous_window": int(complete_rows - len(cohort)),
            "subjects": int(cohort["subject_id"].nunique()),
            "records": int(cohort["record_id"].nunique()),
            "target_sample_hash": _sample_hash(cohort["target_sample_id"]),
            "lag_minus_10s_feature_sample_hash": _sample_hash(
                cohort["lag_minus_10s_feature_sample_id"]
            ),
        }
        cohorts[pm] = cohort
    if tuple(cohorts) != PM_NAMES:
        raise RuntimeError("Exactly seven PM cohorts are required")
    return cohorts, summaries


def build_threshold_audit(
    full: pd.DataFrame,
    cohorts: Mapping[str, pd.DataFrame],
    folds: Sequence[int],
) -> tuple[dict[tuple[int, str], ExtremeThresholds], pd.DataFrame]:
    """Fit outer-train thresholds and audit paired train/test label counts."""
    transforms: dict[tuple[int, str], ExtremeThresholds] = {}
    rows: list[dict[str, Any]] = []
    for fold in folds:
        full_train = full["outer_fold"].astype(int).ne(int(fold))
        for pm in PM_NAMES:
            target_id = f"target_{pm}"
            fit_values = pd.to_numeric(
                full.loc[full_train, target_id], errors="coerce"
            ).to_numpy(dtype=float)
            fit_valid = np.isfinite(fit_values)
            thresholds = fit_extreme_thresholds(fit_values[fit_valid])
            transforms[(int(fold), pm)] = thresholds
            fit_sample_ids = full.loc[full_train, "sample_id"].to_numpy()[fit_valid]

            cohort = cohorts[pm]
            labels = thresholds.transform(cohort["continuous_target"].to_numpy())
            train_before = cohort["outer_fold"].astype(int).ne(int(fold)).to_numpy()
            test_before = cohort["outer_fold"].astype(int).eq(int(fold)).to_numpy()
            train_low = train_before & (labels == 0)
            train_high = train_before & (labels == 1)
            train_middle = train_before & ~np.isfinite(labels)
            test_low = test_before & (labels == 0)
            test_high = test_before & (labels == 1)
            test_middle = test_before & ~np.isfinite(labels)
            train_keep = train_low | train_high
            test_keep = test_low | test_high
            if not train_before.any() or not test_before.any():
                raise ValueError(f"fold {fold} {pm}: empty paired train or test cohort")
            if sorted(np.unique(labels[train_keep]).astype(int).tolist()) != [0, 1]:
                raise ValueError(f"fold {fold} {pm}: outer train is not class-complete")

            train_subjects = sorted(
                cohort.loc[train_before, "subject_id"].astype(str).unique().tolist()
            )
            test_subjects = sorted(
                cohort.loc[test_before, "subject_id"].astype(str).unique().tolist()
            )
            overlap = sorted(set(train_subjects) & set(test_subjects))
            if overlap:
                raise RuntimeError(f"fold {fold} {pm}: subject leakage: {overlap}")
            threshold_payload = {
                "outer_fold": int(fold),
                "pm": pm,
                "target_id": target_id,
                "fit_scope": "outer_train_continuous_complete_cases",
                "fit_sample_count": int(fit_valid.sum()),
                "fit_sample_hash": _sample_hash(fit_sample_ids),
                "q_low": thresholds.q_low,
                "q_high": thresholds.q_high,
                "low_rule": "y <= q_low",
                "high_rule": "y >= q_high",
                "middle_rule": "q_low < y < q_high",
            }
            threshold_hash = stable_hash(threshold_payload)
            rows.append({
                **threshold_payload,
                "threshold_hash": threshold_hash,
                "n_train_before_exclusion": int(train_before.sum()),
                "n_train_low": int(train_low.sum()),
                "n_train_high": int(train_high.sum()),
                "n_train_excluded_middle": int(train_middle.sum()),
                "n_test_before_exclusion": int(test_before.sum()),
                "n_test_low": int(test_low.sum()),
                "n_test_high": int(test_high.sum()),
                "n_test_excluded_middle": int(test_middle.sum()),
                "n_train_retained": int(train_keep.sum()),
                "n_test_retained": int(test_keep.sum()),
                "n_train_subjects": len(train_subjects),
                "n_test_subjects": len(test_subjects),
                "n_test_subjects_retained": int(
                    cohort.loc[test_keep, "subject_id"].nunique()
                ),
                "train_subjects": "|".join(train_subjects),
                "test_subjects": "|".join(test_subjects),
                "subject_overlap_count": 0,
                "subject_overlap": "",
                "train_before_sample_hash": _sample_hash(
                    cohort.loc[train_before, "target_sample_id"]
                ),
                "test_before_sample_hash": _sample_hash(
                    cohort.loc[test_before, "target_sample_id"]
                ),
                "train_retained_sample_hash": _sample_hash(
                    cohort.loc[train_keep, "target_sample_id"]
                ),
                "test_retained_sample_hash": _sample_hash(
                    cohort.loc[test_keep, "target_sample_id"]
                ),
                "train_low_fraction_retained": float(train_low.sum() / train_keep.sum()),
                "train_high_fraction_retained": float(train_high.sum() / train_keep.sum()),
                "test_low_fraction_retained": float(test_low.sum() / test_keep.sum()),
                "test_high_fraction_retained": float(test_high.sum() / test_keep.sum()),
            })
    audit = pd.DataFrame(rows)
    expected = len(PM_NAMES) * len(folds)
    if len(audit) != expected:
        raise RuntimeError(f"Expected {expected} PM-fold threshold rows, got {len(audit)}")
    if not audit["subject_overlap_count"].eq(0).all():
        raise RuntimeError("Outer subject leakage detected")
    return transforms, audit


@dataclass
class ProtocolContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    matrix: np.ndarray
    feature_index: pd.DataFrame
    feature_names: list[str]
    cache_manifest: dict[str, Any]
    cache_identity: dict[str, Any]
    full: pd.DataFrame
    temporal_pairing: pd.DataFrame
    temporal_pairing_summary: dict[str, Any]
    cohorts: dict[str, pd.DataFrame]
    cohort_summary: dict[str, Any]
    transforms: dict[tuple[int, str], ExtremeThresholds]
    threshold_audit: pd.DataFrame
    protocol: dict[str, Any]
    run_matrix: pd.DataFrame


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> ProtocolContext:
    """Load canonical inputs and freeze all 35 PM-fold specifications."""
    root_path = Path(root).resolve()
    cache_path = Path(feature_cache_dir).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root_path / output
    matrix, feature_index, feature_names, cache_manifest = load_feature_cache(cache_path)
    identity = validate_cache_contract(
        matrix,
        feature_index,
        feature_names,
        cache_manifest,
        config["feature_cache_identity"],
    )
    if tuple(matrix.shape) != (int(identity["rows"]), 371):
        raise ValueError("The protocol requires exactly 371 canonical features")

    targets = _target_table(root_path / config["data"]["processed_targets"])
    full = feature_index.merge(
        targets,
        on="sample_id",
        how="left",
        suffixes=("", "_target"),
        validate="one_to_one",
    )
    if len(full) != len(feature_index):
        raise RuntimeError("Feature/target join changed the canonical row count")
    for column in ("subject_id", "record_id"):
        target_column = f"{column}_target"
        if not full[column].astype(str).eq(full[target_column].astype(str)).all():
            raise RuntimeError(f"Feature/target {column} identity mismatch")
    folds = [int(value) for value in config["evaluation"]["folds"]]
    if sorted(full["outer_fold"].astype(int).unique().tolist()) != folds:
        raise ValueError("Feature cache outer folds differ from configured folds")

    temporal_pairing, temporal_summary = build_previous_window_pairing(
        feature_index,
        step_seconds=float(config["alignment"]["step_seconds"]),
        time_column=str(config["alignment"]["time_column"]),
    )
    if any(int(temporal_summary[key]) for key in (
        "cross_record_pairs", "cross_subject_pairs", "cross_fold_pairs"
    )):
        raise RuntimeError("Temporal pairing crossed a protected boundary")
    cohorts, cohort_summary = build_pm_temporal_cohorts(full, temporal_pairing)
    transforms, threshold_audit = build_threshold_audit(full, cohorts, folds)

    fixed_fold_hash = stable_hash(
        feature_index[["sample_id", "subject_id", "outer_fold"]]
        .sort_values("sample_id", kind="stable")
        .astype(str)
        .to_dict("records")
    )
    scientific_config = {key: value for key, value in config.items() if key != "output_dir"}
    threshold_hashes = {
        f"fold_{int(row.outer_fold):02d}__{row.pm}": row.threshold_hash
        for row in threshold_audit.itertuples(index=False)
    }
    cohort_hashes = {
        pm: cohort_summary[pm]["target_sample_hash"] for pm in PM_NAMES
    }
    protocol_hash = stable_hash({
        "schema_version": SCHEMA_VERSION,
        "scientific_config": scientific_config,
        "feature_cache_identity": identity,
        "fixed_fold_hash": fixed_fold_hash,
        "temporal_pairing_hash": temporal_summary["matched_target_sample_hash"],
        "cohort_hashes": cohort_hashes,
        "threshold_hashes": threshold_hashes,
    })
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "confirmatory_preregistered_candidate",
        "task": "binary_classification",
        "training_executed": False,
        "alignment": config["alignment"],
        "fixed_lag_seconds": FIXED_LAG_SECONDS,
        "lag_search_performed": False,
        "preregistration_statement": config["preregistration_statement"],
        "feature_cache_identity": identity,
        "feature_count": len(feature_names),
        "target_ids": [f"target_{pm}" for pm in PM_NAMES],
        "target_transform": config["target_transform"],
        "fold_ids": folds,
        "fixed_fold_hash": fixed_fold_hash,
        "temporal_pairing_hash": temporal_summary["matched_target_sample_hash"],
        "matched_target_sample_hashes": cohort_hashes,
        "threshold_hashes": threshold_hashes,
        "model": config["model"],
        "metrics": {
            "primary": config["evaluation"]["primary_metrics"],
            "secondary": config["evaluation"]["secondary_metrics"],
            "probability_source": "predict_proba_high",
            "single_class_participant_auc_policy": "undefined_exclude_metric_only",
            "definitions": config["evaluation"]["metric_definitions"],
        },
        "git_commit": _git_head(root_path),
        "protocol_hash": protocol_hash,
    }

    audit_by_key = threshold_audit.set_index(["outer_fold", "pm"])
    specs: list[dict[str, Any]] = []
    for fold in folds:
        for pm in PM_NAMES:
            audit = audit_by_key.loc[(fold, pm)]
            spec = {
                "outer_fold": int(fold),
                "pm": pm,
                "target_id": f"target_{pm}",
                "task": "binary_classification",
                "condition": "lag_minus_10s",
                "lag_seconds": FIXED_LAG_SECONDS,
                "model": "xgboost",
                "seed": int(config["model"]["seed"]),
                "q_low": float(audit["q_low"]),
                "q_high": float(audit["q_high"]),
                "threshold_hash": str(audit["threshold_hash"]),
                "n_train": int(audit["n_train_retained"]),
                "n_test": int(audit["n_test_retained"]),
                "n_test_participants": int(audit["n_test_subjects_retained"]),
                "train_sample_hash": str(audit["train_retained_sample_hash"]),
                "test_sample_hash": str(audit["test_retained_sample_hash"]),
                "matched_target_sample_hash": cohort_hashes[pm],
            }
            spec_hash = stable_hash({"protocol_hash": protocol_hash, "run_spec": spec})
            spec["specification_hash"] = spec_hash
            spec["run_id"] = f"fold_{fold:02d}__{pm}__low_high__{spec_hash[:12]}"
            specs.append(spec)
    run_matrix = pd.DataFrame(specs)
    if len(run_matrix) != 35 or run_matrix["run_id"].duplicated().any():
        raise RuntimeError("Exactly 35 unique PM-fold fits are required")
    if set(run_matrix["lag_seconds"].astype(int)) != {FIXED_LAG_SECONDS}:
        raise RuntimeError("Only lag -10 is allowed")
    if tuple(run_matrix["pm"].drop_duplicates()) != PM_NAMES:
        raise RuntimeError("All seven PM must follow one identical run path")
    specification_by_key = {
        (int(row.outer_fold), str(row.pm)): str(row.specification_hash)
        for row in run_matrix.itertuples(index=False)
    }
    threshold_audit = threshold_audit.copy()
    threshold_audit["protocol_hash"] = protocol_hash
    threshold_audit["specification_hash"] = [
        specification_by_key[(int(row.outer_fold), str(row.pm))]
        for row in threshold_audit.itertuples(index=False)
    ]
    return ProtocolContext(
        root=root_path,
        output_dir=output,
        config=dict(config),
        matrix=matrix,
        feature_index=feature_index,
        feature_names=list(feature_names),
        cache_manifest=cache_manifest,
        cache_identity=identity,
        full=full,
        temporal_pairing=temporal_pairing,
        temporal_pairing_summary=temporal_summary,
        cohorts=cohorts,
        cohort_summary=cohort_summary,
        transforms=transforms,
        threshold_audit=threshold_audit,
        protocol=protocol,
        run_matrix=run_matrix,
    )


def _cohort_summary_frame(context: ProtocolContext) -> pd.DataFrame:
    return pd.DataFrame([context.cohort_summary[pm] for pm in PM_NAMES])


def write_dry_run(context: ProtocolContext) -> dict[str, Any]:
    """Write compact preregistration artifacts without constructing a model."""
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(context.output_dir / "cohort_summary.csv", _cohort_summary_frame(context))
    _write_csv(context.output_dir / "thresholds_by_fold.csv", context.threshold_audit)
    _write_csv(context.output_dir / "run_matrix.csv", context.run_matrix)

    class_balance_by_pm: dict[str, Any] = {}
    for pm in PM_NAMES:
        group = context.threshold_audit.loc[context.threshold_audit["pm"].eq(pm)]
        test_low = int(group["n_test_low"].sum())
        test_high = int(group["n_test_high"].sum())
        test_retained = test_low + test_high
        class_balance_by_pm[pm] = {
            "test_low": test_low,
            "test_high": test_high,
            "test_retained": test_retained,
            "test_low_fraction": float(test_low / test_retained),
            "test_high_fraction": float(test_high / test_retained),
        }
    try:
        output_reference = context.output_dir.relative_to(context.root).as_posix()
    except ValueError:
        output_reference = str(context.output_dir)
    summary = {
        "canonical_feature_rows": int(context.matrix.shape[0]),
        "feature_count": int(context.matrix.shape[1]),
        "feature_dtype": str(context.matrix.dtype),
        "feature_cache_identity_hash": context.cache_identity["cache_identity_hash"],
        "feature_hash": context.cache_identity["feature_hash"],
        "sample_id_universe_hash": context.cache_identity["sample_id_universe_hash"],
        "target_ids": [f"target_{pm}" for pm in PM_NAMES],
        "complete_case_counts_by_pm": {
            pm: int(context.cohort_summary[pm]["canonical_complete_case_rows"])
            for pm in PM_NAMES
        },
        "temporal_matched_counts_by_pm": {
            pm: int(context.cohort_summary[pm]["temporal_matched_rows"])
            for pm in PM_NAMES
        },
        "temporal_pairing": context.temporal_pairing_summary,
        "fixed_lag_seconds": FIXED_LAG_SECONDS,
        "lag_search_performed": False,
        "cross_record_pairs": int(context.temporal_pairing_summary["cross_record_pairs"]),
        "cross_subject_pairs": int(context.temporal_pairing_summary["cross_subject_pairs"]),
        "cross_fold_pairs": int(context.temporal_pairing_summary["cross_fold_pairs"]),
        "counts_by_pm_fold": context.threshold_audit.to_dict("records"),
        "class_balance_by_pm": class_balance_by_pm,
        "model": context.config["model"],
        "planned_fits": int(len(context.run_matrix)),
        "training_executed": False,
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hashes": context.run_matrix[
            ["outer_fold", "pm", "specification_hash"]
        ].to_dict("records"),
        "output_dir": output_reference,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    readme = f"""# PM LOW-vs-HIGH q3 extremes confirmatory v1

This preregistered protocol evaluates whether canonical EEG features distinguish
strongly LOW from strongly HIGH continuous PM states. The middle outer-train
tertile is excluded. Thresholds are fitted on continuous outer-train targets only
and applied unchanged to the paired outer-test targets.

- alignment: `EEG(t-10 s) -> PM(t)` only
- pairing: exact 10-second, record-local; gaps are excluded without substitution
- protocol hash: `{context.protocol['protocol_hash']}`
- canonical matrix: `{len(context.feature_index)} x {len(context.feature_names)}`
- targets/folds/planned fits: `7 / 5 / {len(context.run_matrix)}`
- model: `XGBClassifier`
- hyperparameters: `{json.dumps(context.config['model']['params'], sort_keys=True)}`
- participant-macro primary metrics: balanced accuracy, Macro-F1, ROC-AUC
- participant AUC policy: one-class subsets are undefined and excluded only from that metric
- training executed by dry-run: `false`

`results_by_fold.csv`, `summary_by_pm.csv` and `pooled_summary.csv` are created
only by an explicitly requested full run. Per-run predictions remain under
`runs/` as runtime artifacts.
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def participant_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_high: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compute equal-weight participant metrics with metric-local NaN handling."""
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    probability = np.asarray(probability_high, dtype=float).reshape(-1)
    subjects = np.asarray(subject_ids).astype(str).reshape(-1)
    if not (len(truth) == len(prediction) == len(probability) == len(subjects)):
        raise ValueError("truth, prediction, probability and subject_ids lengths differ")
    if not len(truth):
        raise ValueError("Binary metrics require at least one sample")
    if not set(np.unique(truth)).issubset(FIXED_LABELS):
        raise ValueError("y_true contains labels outside LOW=0/HIGH=1")
    if not set(np.unique(prediction)).issubset(FIXED_LABELS):
        raise ValueError("y_pred contains labels outside LOW=0/HIGH=1")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("HIGH probabilities must be finite and within [0, 1]")

    rows: list[dict[str, Any]] = []
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        subject_true = truth[mask]
        subject_pred = prediction[mask]
        subject_probability = probability[mask]
        n_low = int(np.sum(subject_true == 0))
        n_high = int(np.sum(subject_true == 1))
        low_recall = (
            float(np.mean(subject_pred[subject_true == 0] == 0))
            if n_low else float("nan")
        )
        high_recall = (
            float(np.mean(subject_pred[subject_true == 1] == 1))
            if n_high else float("nan")
        )
        available_recalls = [value for value in (low_recall, high_recall) if np.isfinite(value)]
        balanced_accuracy = float(np.mean(available_recalls))
        both_classes = n_low > 0 and n_high > 0
        roc_auc = (
            float(roc_auc_score(subject_true, subject_probability))
            if both_classes else float("nan")
        )
        pr_auc = (
            float(average_precision_score(subject_true, subject_probability))
            if both_classes else float("nan")
        )
        rows.append({
            "subject_id": subject,
            "n_samples": int(mask.sum()),
            "n_low": n_low,
            "n_high": n_high,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": float(
                f1_score(
                    subject_true,
                    subject_pred,
                    labels=list(FIXED_LABELS),
                    average="macro",
                    zero_division=0,
                )
            ),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "low_recall": low_recall,
            "high_recall": high_recall,
            "precision": float(
                precision_score(
                    subject_true,
                    subject_pred,
                    labels=list(FIXED_LABELS),
                    average="macro",
                    zero_division=0,
                )
            ),
            "accuracy": float(accuracy_score(subject_true, subject_pred)),
        })
    frame = pd.DataFrame(rows)
    macro: dict[str, float | int] = {}
    columns = {"f1": "macro_f1", **{metric: metric for metric in METRICS if metric != "f1"}}
    for metric in METRICS:
        values = frame[columns[metric]].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        macro[f"participant_macro_{metric}"] = (
            float(np.mean(finite)) if len(finite) else float("nan")
        )
        macro[f"participant_valid_{metric}"] = int(len(finite))
    macro["n_test_participants"] = int(len(frame))
    return frame, macro


def _run_directory(context: ProtocolContext, spec: Mapping[str, Any]) -> Path:
    return context.output_dir / "runs" / str(spec["run_id"])


def execute_run(
    context: ProtocolContext,
    spec: Mapping[str, Any],
    *,
    model_builder: Callable[..., Any] = build_model,
) -> dict[str, Any]:
    """Execute one frozen PM-fold fit; never called by dry-run."""
    fold = int(spec["outer_fold"])
    pm = str(spec["pm"])
    if pm not in PM_NAMES:
        raise ValueError(f"Unsupported PM: {pm}")
    if str(spec["condition"]) != "lag_minus_10s" or int(spec["lag_seconds"]) != -10:
        raise ValueError("Only the fixed lag-minus-10 condition may execute")
    thresholds = context.transforms[(fold, pm)]
    if thresholds.q_low != float(spec["q_low"]) or thresholds.q_high != float(spec["q_high"]):
        raise RuntimeError("Runtime thresholds differ from the frozen specification")
    cohort = context.cohorts[pm]
    labels = thresholds.transform(cohort["continuous_target"].to_numpy())
    train_mask = (
        cohort["outer_fold"].astype(int).ne(fold).to_numpy() & np.isfinite(labels)
    )
    test_mask = (
        cohort["outer_fold"].astype(int).eq(fold).to_numpy() & np.isfinite(labels)
    )
    train_subjects = set(cohort.loc[train_mask, "subject_id"].astype(str))
    test_subjects = set(cohort.loc[test_mask, "subject_id"].astype(str))
    if train_subjects & test_subjects:
        raise RuntimeError("Outer subject leakage before model fit")
    if int(train_mask.sum()) != int(spec["n_train"]) or int(test_mask.sum()) != int(spec["n_test"]):
        raise RuntimeError("Runtime retained counts differ from frozen specification")
    if _sample_hash(cohort.loc[train_mask, "target_sample_id"]) != spec["train_sample_hash"]:
        raise RuntimeError("Runtime train sample identity differs from frozen specification")
    if _sample_hash(cohort.loc[test_mask, "target_sample_id"]) != spec["test_sample_hash"]:
        raise RuntimeError("Runtime test sample identity differs from frozen specification")
    y_train = labels[train_mask].astype(np.int64)
    y_test = labels[test_mask].astype(np.int64)
    if sorted(np.unique(y_train).tolist()) != list(FIXED_LABELS):
        raise RuntimeError(f"fold {fold} {pm}: outer train is not class-complete")
    positions = cohort["lag_minus_10s_feature_position"].to_numpy(dtype=np.int64)
    x_train = np.asarray(context.matrix[positions[train_mask]], dtype=np.float32)
    x_test = np.asarray(context.matrix[positions[test_mask]], dtype=np.float32)
    if x_train.shape[1] != 371 or x_test.shape[1] != 371:
        raise RuntimeError("Runtime feature count differs from 371")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise RuntimeError("Non-finite features reached model fit")

    started = time.perf_counter()
    model = model_builder(
        "xgboost",
        "classification",
        (len(context.feature_names),),
        2,
        context.config["model"]["params"],
    )
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_test), dtype=np.int64).reshape(-1)
    probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
    elapsed = time.perf_counter() - started
    if probabilities.ndim != 2 or probabilities.shape[0] != len(y_test):
        raise RuntimeError("Classifier returned invalid predict_proba shape")
    classes = np.asarray(getattr(model, "classes_", FIXED_LABELS), dtype=int)
    high_columns = np.flatnonzero(classes == 1)
    if len(high_columns) != 1 or probabilities.shape[1] != len(classes):
        raise RuntimeError("Classifier probabilities lack one HIGH class column")
    probability_high = probabilities[:, int(high_columns[0])]
    if prediction.shape != y_test.shape:
        raise RuntimeError("Classifier returned invalid prediction shape")

    test_cohort = cohort.loc[test_mask].reset_index(drop=True)
    participants, macro = participant_binary_metrics(
        y_test,
        prediction,
        probability_high,
        test_cohort["subject_id"].astype(str).to_numpy(),
    )
    run_dir = _run_directory(context, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_cohort[[
        "target_sample_id", "subject_id", "record_id", "outer_fold"
    ]].copy()
    predictions["feature_sample_id"] = test_cohort[
        "lag_minus_10s_feature_sample_id"
    ].to_numpy()
    predictions["pm"] = pm
    predictions["target_id"] = f"target_{pm}"
    predictions["condition"] = "lag_minus_10s"
    predictions["lag_seconds"] = FIXED_LAG_SECONDS
    predictions["q_low"] = thresholds.q_low
    predictions["q_high"] = thresholds.q_high
    predictions["y_true"] = y_test
    predictions["y_pred"] = prediction
    predictions["probability_high"] = probability_high
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    participants.insert(0, "pm", pm)
    participants.insert(0, "outer_fold", fold)
    _write_csv(run_dir / "participant_metrics.csv", participants)
    summary = {
        "status": "complete",
        "result_status": "confirmatory",
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hash": spec["specification_hash"],
        "threshold_hash": spec["threshold_hash"],
        "run_id": spec["run_id"],
        "outer_fold": fold,
        "pm": pm,
        "target_id": spec["target_id"],
        "condition": "lag_minus_10s",
        "lag_seconds": FIXED_LAG_SECONDS,
        "q_low": thresholds.q_low,
        "q_high": thresholds.q_high,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_test_participants": int(len(participants)),
        "training_time_seconds": float(elapsed),
        **macro,
    }
    _atomic_json(run_dir / "run_summary.json", summary)
    return summary


def load_resumable_summary(
    context: ProtocolContext,
    spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    run_dir = _run_directory(context, spec)
    summary_path = run_dir / "run_summary.json"
    if not all(
        path.is_file()
        for path in (
            summary_path,
            run_dir / "predictions.parquet",
            run_dir / "participant_metrics.csv",
        )
    ):
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    if payload.get("protocol_hash") != context.protocol["protocol_hash"]:
        return None
    if payload.get("specification_hash") != spec["specification_hash"]:
        return None
    if payload.get("threshold_hash") != spec["threshold_hash"]:
        return None
    return payload


def aggregate_results(
    context: ProtocolContext,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    """Aggregate 35 participant-macro PM-fold results without retuning."""
    results = pd.DataFrame(summaries).sort_values(
        ["outer_fold", "pm"], kind="stable"
    )
    if len(results) != 35 or results["run_id"].duplicated().any():
        raise ValueError("Full aggregation requires 35 unique completed runs")
    if set(results["lag_seconds"].astype(int)) != {FIXED_LAG_SECONDS}:
        raise ValueError("Aggregation received a forbidden lag condition")
    _write_csv(context.output_dir / "results_by_fold.csv", results)

    pm_rows: list[dict[str, Any]] = []
    for pm in PM_NAMES:
        group = results.loc[results["pm"].eq(pm)]
        if len(group) != 5:
            raise RuntimeError(f"{pm}: expected five completed folds")
        row: dict[str, Any] = {
            "pm": pm,
            "target_id": f"target_{pm}",
            "n_folds": 5,
        }
        for metric in METRICS:
            values = group[f"participant_macro_{metric}"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"participant_macro_{metric}_mean"] = (
                float(np.mean(finite)) if len(finite) else float("nan")
            )
            row[f"participant_macro_{metric}_std"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
            )
            row[f"valid_folds_{metric}"] = int(len(finite))
            valid_column = f"participant_valid_{metric}"
            row[f"valid_participants_{metric}"] = int(group[valid_column].sum())
        pm_rows.append(row)
    summary_by_pm = pd.DataFrame(pm_rows)
    _write_csv(context.output_dir / "summary_by_pm.csv", summary_by_pm)

    pooled: dict[str, Any] = {
        "n_fold_pm_runs": 35,
        "n_pm": 7,
        "n_folds": 5,
        "lag_seconds": FIXED_LAG_SECONDS,
        "independence_note": "fold-PM rows are descriptive and not independent inferential units",
    }
    for metric in METRICS:
        values = results[f"participant_macro_{metric}"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        pooled[f"participant_macro_{metric}_mean"] = (
            float(np.mean(finite)) if len(finite) else float("nan")
        )
        pooled[f"participant_macro_{metric}_std"] = (
            float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
        )
        pooled[f"participant_macro_{metric}_median"] = (
            float(np.median(finite)) if len(finite) else float("nan")
        )
        pooled[f"valid_fold_pm_{metric}"] = int(len(finite))
        pooled[f"valid_participants_{metric}"] = int(
            results[f"participant_valid_{metric}"].sum()
        )
    _write_csv(context.output_dir / "pooled_summary.csv", pd.DataFrame([pooled]))
    protocol = dict(context.protocol)
    protocol["training_executed"] = True
    protocol["result_status"] = "confirmatory_complete"
    _atomic_json(context.output_dir / "protocol.json", protocol)


def run_experiment(context: ProtocolContext, *, resume: bool) -> dict[str, int]:
    summaries: list[dict[str, Any]] = []
    reused = 0
    trained = 0
    for spec in context.run_matrix.to_dict("records"):
        existing = load_resumable_summary(context, spec) if resume else None
        if existing is not None:
            summaries.append(existing)
            reused += 1
            continue
        run_dir = _run_directory(context, spec)
        if run_dir.exists() and not resume:
            raise FileExistsError(
                f"Run directory exists; use --resume after auditing it: {run_dir}"
            )
        summaries.append(execute_run(context, spec))
        trained += 1
    if len(summaries) != 35:
        raise RuntimeError("Full aggregation requires all 35 fixed runs")
    aggregate_results(context, summaries)
    return {"complete": len(summaries), "trained": trained, "reused": reused}


__all__ = [
    "FIXED_LAG_SECONDS",
    "PM_NAMES",
    "PRIMARY_METRICS",
    "SECONDARY_METRICS",
    "ExtremeThresholds",
    "ProtocolContext",
    "aggregate_results",
    "apply_extreme_labels",
    "build_pm_temporal_cohorts",
    "build_threshold_audit",
    "execute_run",
    "fit_extreme_thresholds",
    "load_config",
    "load_resumable_summary",
    "participant_binary_metrics",
    "prepare_protocol",
    "run_experiment",
    "write_dry_run",
]
