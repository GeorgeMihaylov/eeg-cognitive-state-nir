"""Confirmatory continuous-PM regression for fixed EEG lag ``-10 s``.

The temporal pairing and canonical feature-cache validation are reused from
the preceding classification confirmatory experiment.  Each PM has its own
target-complete matched cohort, shared exactly by ``lag_0`` and
``lag_minus_10s``.
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

from bench.experiments.pm_eeg_lag_confirmatory import (
    CONDITIONS,
    PM_NAMES,
    build_previous_window_pairing,
    condition_target_ids,
    stable_hash,
    validate_cache_contract,
)
from bench.features.cogstate_feature_cache import load_feature_cache
from bench.validation.metrics import MetricsCalculator
from model_zoo import build_model


SCHEMA_VERSION = "pm-eeg-lag-regression-confirmatory-v1"
PRIMARY_METRICS = ("mae", "r2", "pearson")
SECONDARY_METRICS = ("rmse", "spearman")


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
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if config.get("task") != "regression":
        raise ValueError("Confirmatory lag experiment must be continuous regression")
    if tuple(config.get("pm_names", ())) != PM_NAMES:
        raise ValueError("Regression protocol must contain all seven PM in canonical order")
    target_ids = tuple(config.get("target_ids", ()))
    expected_targets = tuple(f"target_{pm}" for pm in PM_NAMES)
    if target_ids != expected_targets:
        raise ValueError("Only the seven canonical continuous target_* PM are allowed")
    condition_ids = tuple(
        str(item["condition_id"]) for item in config.get("conditions", ())
    )
    if condition_ids != tuple(name for name, _ in CONDITIONS):
        raise ValueError("Confirmatory condition IDs are frozen")
    lags = tuple(int(item["lag_seconds"]) for item in config.get("conditions", ()))
    if lags != (0, -10):
        raise ValueError("Confirmatory conditions are frozen at lag 0 and lag -10 seconds")
    if config.get("target_transform") != {"name": "none", "reason": "continuous_pm"}:
        raise ValueError("Regression targets must not be discretized or transformed")
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
        raise ValueError("Primary metrics must be participant-macro MAE, R2 and Pearson")
    model = config.get("model", {})
    if model.get("name") != "xgboost" or model.get("task_type") != "regression":
        raise ValueError("The confirmatory model must be XGBoost regression")
    if int(model.get("seed", -1)) != 42:
        raise ValueError("The confirmatory seed is frozen at 42")
    if int(config.get("feature_cache_identity", {}).get("n_features", -1)) != 371:
        raise ValueError("Canonical feature count must be 371")
    forbidden_text = json.dumps(config, sort_keys=True).lower()
    if any(token in forbidden_text for token in ('"q3"', '"q5"', 'label_q')):
        raise ValueError("Classification labels and quantile classes are forbidden")
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


def build_pm_matched_cohorts(
    full: pd.DataFrame,
    temporal_pairing: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Apply each PM's complete-case mask after exact temporal pairing."""
    target_lookup = full.set_index("sample_id")
    target_ids = temporal_pairing["target_sample_id"].to_numpy()
    cohorts: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    for pm in PM_NAMES:
        column = f"target_{pm}"
        canonical_values = pd.to_numeric(full[column], errors="coerce").to_numpy(
            dtype=float
        )
        paired_values = pd.to_numeric(
            target_lookup.loc[target_ids, column], errors="coerce"
        ).to_numpy(dtype=float)
        cohort = temporal_pairing.loc[np.isfinite(paired_values)].reset_index(drop=True)
        if cohort.empty:
            raise ValueError(f"{column} has no exact-lag complete cases")
        identities = condition_target_ids(cohort)
        if not np.array_equal(identities["lag_0"], identities["lag_minus_10s"]):
            raise RuntimeError(f"{column}: target sample identity differs by condition")
        if cohort["target_sample_id"].duplicated().any():
            raise RuntimeError(f"{column}: duplicate matched target sample IDs")
        complete_rows = int(np.isfinite(canonical_values).sum())
        matched_rows = int(len(cohort))
        cohorts[pm] = cohort
        summaries[pm] = {
            "target_id": column,
            "canonical_complete_case_rows": complete_rows,
            "canonical_missing_target_rows": int(len(full) - complete_rows),
            "matched_rows": matched_rows,
            "lost_complete_cases_without_exact_previous_window": int(
                complete_rows - matched_rows
            ),
            "subjects": int(cohort["subject_id"].nunique()),
            "records": int(cohort["record_id"].nunique()),
            "target_sample_hash": _sample_hash(cohort["target_sample_id"]),
            "identical_target_ids_between_conditions": True,
            "identical_subject_ids_between_conditions": True,
            "identical_fold_membership_between_conditions": True,
        }
    return cohorts, summaries


def build_pm_fold_audit(
    full: pd.DataFrame,
    cohorts: Mapping[str, pd.DataFrame],
    folds: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pm in PM_NAMES:
        cohort = cohorts[pm]
        for fold in folds:
            train_mask = cohort["outer_fold"].astype(int).ne(int(fold))
            test_mask = cohort["outer_fold"].astype(int).eq(int(fold))
            train_subjects = sorted(cohort.loc[train_mask, "subject_id"].astype(str).unique())
            test_subjects = sorted(cohort.loc[test_mask, "subject_id"].astype(str).unique())
            overlap = sorted(set(train_subjects) & set(test_subjects))
            if overlap:
                raise RuntimeError(f"fold {fold} {pm}: outer subject leakage: {overlap}")
            if not train_mask.any() or not test_mask.any():
                raise ValueError(f"fold {fold} {pm}: empty train or test matched cohort")
            rows.append({
                "outer_fold": int(fold),
                "pm": pm,
                "target_id": f"target_{pm}",
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_train_subjects": len(train_subjects),
                "n_test_subjects": len(test_subjects),
                "train_subjects": "|".join(train_subjects),
                "test_subjects": "|".join(test_subjects),
                "subject_overlap_count": 0,
                "train_target_sample_hash": _sample_hash(
                    cohort.loc[train_mask, "target_sample_id"]
                ),
                "test_target_sample_hash": _sample_hash(
                    cohort.loc[test_mask, "target_sample_id"]
                ),
                "conditions_n_train_identical": True,
                "conditions_n_test_identical": True,
                "conditions_subjects_identical": True,
                "conditions_outer_fold_identical": True,
            })
    result = pd.DataFrame(rows)
    expected = len(PM_NAMES) * len(folds)
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} PM-fold audit rows, got {len(result)}")
    return result


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
    fold_audit: pd.DataFrame
    protocol: dict[str, Any]
    run_matrix: pd.DataFrame


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> ProtocolContext:
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
    cohorts, cohort_summary = build_pm_matched_cohorts(full, temporal_pairing)
    fold_audit = build_pm_fold_audit(full, cohorts, folds)
    fixed_fold_hash = stable_hash(
        feature_index[["sample_id", "subject_id", "outer_fold"]]
        .sort_values("sample_id", kind="stable")
        .astype(str)
        .to_dict("records")
    )
    scientific_config = {key: value for key, value in config.items() if key != "output_dir"}
    cohort_hashes = {
        pm: cohort_summary[pm]["target_sample_hash"] for pm in PM_NAMES
    }
    protocol_hash = stable_hash({
        "schema_version": SCHEMA_VERSION,
        "scientific_config": scientific_config,
        "feature_cache_identity": identity,
        "fixed_fold_hash": fixed_fold_hash,
        "matched_target_sample_hashes": cohort_hashes,
    })
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "confirmatory_preregistered_candidate",
        "task": "regression",
        "training_executed": False,
        "candidate_lags_seconds": [0, -10],
        "fixed_candidate_statement": config["preregistration_statement"],
        "interpretation": "fixed causal temporal alignment candidate; not a proven physiological delay",
        "feature_cache_identity": identity,
        "feature_count": len(feature_names),
        "target_ids": [f"target_{pm}" for pm in PM_NAMES],
        "target_transform": {"name": "none", "reason": "continuous_pm"},
        "matched_cohort_policy": config["matched_cohort_policy"],
        "matched_target_sample_hashes": cohort_hashes,
        "fold_ids": folds,
        "fixed_fold_hash": fixed_fold_hash,
        "model_family": "XGBRegressor",
        "model": config["model"],
        "seed": int(config["model"]["seed"]),
        "metrics": config["evaluation"]["primary_metrics"],
        "git_commit": _git_head(root_path),
        "protocol_hash": protocol_hash,
    }
    audit_by_key = fold_audit.set_index(["outer_fold", "pm"])
    specs: list[dict[str, Any]] = []
    for fold in folds:
        for pm in PM_NAMES:
            audit = audit_by_key.loc[(fold, pm)]
            for condition, lag_seconds in CONDITIONS:
                spec = {
                    "outer_fold": int(fold),
                    "pm": pm,
                    "target_id": f"target_{pm}",
                    "task": "regression",
                    "condition": condition,
                    "lag_seconds": int(lag_seconds),
                    "model": "xgboost",
                    "seed": int(config["model"]["seed"]),
                    "n_train": int(audit["n_train"]),
                    "n_test": int(audit["n_test"]),
                    "n_test_participants": int(audit["n_test_subjects"]),
                    "train_target_sample_hash": str(audit["train_target_sample_hash"]),
                    "test_target_sample_hash": str(audit["test_target_sample_hash"]),
                    "matched_target_sample_hash": cohort_hashes[pm],
                }
                spec_hash = stable_hash({"protocol_hash": protocol_hash, "run_spec": spec})
                spec["specification_hash"] = spec_hash
                spec["run_id"] = f"fold_{fold:02d}__{pm}__{condition}__{spec_hash[:12]}"
                specs.append(spec)
    run_matrix = pd.DataFrame(specs)
    if len(run_matrix) != 70:
        raise RuntimeError(f"Expected 70 fixed runs, got {len(run_matrix)}")
    if not run_matrix.groupby(["outer_fold", "pm"]).size().eq(2).all():
        raise RuntimeError("Each PM-fold must contain exactly both lag conditions")
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
        fold_audit=fold_audit,
        protocol=protocol,
        run_matrix=run_matrix,
    )


def write_dry_run(context: ProtocolContext) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    matched_summary = {
        "temporal_pairing": context.temporal_pairing_summary,
        "by_pm": context.cohort_summary,
    }
    _atomic_json(context.output_dir / "matched_cohort_summary.json", matched_summary)
    _write_csv(context.output_dir / "matched_cohort_by_fold.csv", context.fold_audit)
    _write_csv(context.output_dir / "run_matrix.csv", context.run_matrix)
    readme = f"""# PM EEG lag regression confirmatory v1

This protocol compares continuous-PM regression from `X(t)` and `X(t-10s)`
on one exact, PM-specific matched target cohort per outcome.

The -10 s lag was fixed from the preceding classification lag analysis before
inspecting regression results. No regression-specific lag selection is performed.

- protocol hash: `{context.protocol['protocol_hash']}`
- canonical features: `{len(context.feature_names)}`
- canonical rows: `{len(context.feature_index)}`
- targets: `{len(PM_NAMES)}` continuous PM
- model: `XGBRegressor`
- hyperparameters: `{json.dumps(context.config['model']['params'], sort_keys=True)}`
- seed: `{context.config['model']['seed']}`
- planned fits: `{len(context.run_matrix)}`
- training executed by dry-run: `false`
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
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
        "target_names": [f"target_{pm}" for pm in PM_NAMES],
        "complete_case_counts_by_pm": {
            pm: int(context.cohort_summary[pm]["canonical_complete_case_rows"])
            for pm in PM_NAMES
        },
        "matched_counts_by_pm": {
            pm: int(context.cohort_summary[pm]["matched_rows"]) for pm in PM_NAMES
        },
        "lost_complete_cases_by_pm": {
            pm: int(
                context.cohort_summary[pm][
                    "lost_complete_cases_without_exact_previous_window"
                ]
            )
            for pm in PM_NAMES
        },
        "matched_counts_by_fold": context.fold_audit.to_dict("records"),
        "subjects": int(context.temporal_pairing["subject_id"].nunique()),
        "records": int(context.temporal_pairing["record_id"].nunique()),
        "temporal_pairing": context.temporal_pairing_summary,
        "cross_record_pairs": int(context.temporal_pairing_summary["cross_record_pairs"]),
        "cross_subject_pairs": int(context.temporal_pairing_summary["cross_subject_pairs"]),
        "cross_fold_pairs": int(context.temporal_pairing_summary["cross_fold_pairs"]),
        "conditions": [
            {"condition_id": name, "lag_seconds": lag} for name, lag in CONDITIONS
        ],
        "model": context.config["model"],
        "seed": int(context.config["model"]["seed"]),
        "identical_target_ids_between_conditions": True,
        "identical_subject_ids_between_conditions": True,
        "identical_fold_membership_between_conditions": True,
        "identical_train_test_counts_between_conditions": True,
        "planned_fits": int(len(context.run_matrix)),
        "training_executed": False,
        "protocol_hash": context.protocol["protocol_hash"],
        "output_dir": output_reference,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    return summary


def participant_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    subjects = np.asarray(subject_ids).astype(str)
    if not (len(truth) == len(prediction) == len(subjects)):
        raise ValueError("truth, prediction and subject_ids must have equal lengths")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression metrics require finite truth and predictions")
    rows: list[dict[str, Any]] = []
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        metrics = MetricsCalculator.calculate_regression_metrics(
            truth[mask], prediction[mask]
        )
        rows.append({
            "subject_id": subject,
            "n_samples": int(mask.sum()),
            **{metric: float(metrics[metric]) for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS)},
        })
    frame = pd.DataFrame(rows)
    macro: dict[str, float | int] = {}
    for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS):
        values = frame[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        macro[f"participant_macro_{metric}"] = (
            float(np.mean(finite)) if len(finite) else float("nan")
        )
        macro[f"participant_valid_{metric}"] = int(len(finite))
    return frame, macro


def _run_directory(context: ProtocolContext, spec: Mapping[str, Any]) -> Path:
    return context.output_dir / "runs" / str(spec["run_id"])


def execute_run(
    context: ProtocolContext,
    spec: Mapping[str, Any],
    *,
    model_builder: Callable[..., Any] = build_model,
) -> dict[str, Any]:
    fold = int(spec["outer_fold"])
    pm = str(spec["pm"])
    condition = str(spec["condition"])
    if condition not in {name for name, _ in CONDITIONS}:
        raise ValueError(f"Unsupported condition: {condition}")
    cohort = context.cohorts[pm]
    train_mask = cohort["outer_fold"].astype(int).ne(fold).to_numpy()
    test_mask = cohort["outer_fold"].astype(int).eq(fold).to_numpy()
    train_subjects = set(cohort.loc[train_mask, "subject_id"].astype(str))
    test_subjects = set(cohort.loc[test_mask, "subject_id"].astype(str))
    if train_subjects & test_subjects:
        raise RuntimeError("Outer subject leakage before model fit")
    if int(train_mask.sum()) != int(spec["n_train"]) or int(test_mask.sum()) != int(spec["n_test"]):
        raise RuntimeError("Runtime matched-cohort counts differ from frozen run spec")
    if _sample_hash(cohort.loc[train_mask, "target_sample_id"]) != spec["train_target_sample_hash"]:
        raise RuntimeError("Runtime train sample identity differs from frozen run spec")
    if _sample_hash(cohort.loc[test_mask, "target_sample_id"]) != spec["test_target_sample_hash"]:
        raise RuntimeError("Runtime test sample identity differs from frozen run spec")
    position_column = (
        "lag_0_feature_position"
        if condition == "lag_0"
        else "lag_minus_10s_feature_position"
    )
    x_positions = cohort[position_column].to_numpy(dtype=np.int64)
    x_train = np.asarray(context.matrix[x_positions[train_mask]], dtype=np.float32)
    x_test = np.asarray(context.matrix[x_positions[test_mask]], dtype=np.float32)
    target_lookup = context.full.set_index("sample_id")
    values = pd.to_numeric(
        target_lookup.loc[cohort["target_sample_id"], f"target_{pm}"], errors="raise"
    ).to_numpy(dtype=np.float32)
    y_train = values[train_mask]
    y_test = values[test_mask]
    if not all(np.isfinite(array).all() for array in (x_train, x_test, y_train, y_test)):
        raise RuntimeError("Non-finite regression inputs reached model fit")
    started = time.perf_counter()
    model = model_builder(
        "xgboost",
        "regression",
        (len(context.feature_names),),
        1,
        context.config["model"]["params"],
    )
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
    elapsed = time.perf_counter() - started
    if prediction.shape != y_test.shape or not np.isfinite(prediction).all():
        raise RuntimeError("Regressor returned invalid predictions")
    test_cohort = cohort.loc[test_mask].reset_index(drop=True)
    participants, macro = participant_regression_metrics(
        y_test,
        prediction,
        test_cohort["subject_id"].astype(str).to_numpy(),
    )
    run_dir = _run_directory(context, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_cohort[[
        "target_sample_id", "subject_id", "record_id", "outer_fold"
    ]].copy()
    feature_id_column = (
        "lag_0_feature_sample_id"
        if condition == "lag_0"
        else "lag_minus_10s_feature_sample_id"
    )
    predictions["feature_sample_id"] = test_cohort[feature_id_column].to_numpy()
    predictions["pm"] = pm
    predictions["target_id"] = f"target_{pm}"
    predictions["condition"] = condition
    predictions["lag_seconds"] = int(spec["lag_seconds"])
    predictions["y_true"] = y_test
    predictions["y_pred"] = prediction
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    participants.insert(0, "condition", condition)
    participants.insert(0, "pm", pm)
    participants.insert(0, "outer_fold", fold)
    _write_csv(run_dir / "participant_metrics.csv", participants)
    summary = {
        "status": "complete",
        "result_status": "confirmatory",
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hash": spec["specification_hash"],
        "run_id": spec["run_id"],
        "outer_fold": fold,
        "pm": pm,
        "condition": condition,
        "lag_seconds": int(spec["lag_seconds"]),
        "target_id": spec["target_id"],
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_test_participants": int(participants["subject_id"].nunique()),
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
    path = run_dir / "run_summary.json"
    predictions = run_dir / "predictions.parquet"
    if not path.is_file() or not predictions.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    if payload.get("protocol_hash") != context.protocol["protocol_hash"]:
        return None
    if payload.get("specification_hash") != spec["specification_hash"]:
        return None
    return payload


def aggregate_results(
    context: ProtocolContext,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    results = pd.DataFrame(summaries).sort_values(
        ["outer_fold", "pm", "lag_seconds"], kind="stable"
    )
    if len(results) != 70 or results["run_id"].duplicated().any():
        raise ValueError("Full aggregation requires 70 unique completed runs")
    _write_csv(context.output_dir / "results_by_fold.csv", results)
    baseline = results.loc[results["condition"].eq("lag_0")].copy()
    lagged = results.loc[results["condition"].eq("lag_minus_10s")].copy()
    paired = baseline.merge(
        lagged,
        on=["outer_fold", "pm", "target_id"],
        suffixes=("_lag0", "_lag_minus_10s"),
        validate="one_to_one",
    )
    if len(paired) != 35:
        raise RuntimeError("Expected 35 paired fold-PM comparisons")
    for count in ("n_train", "n_test", "n_test_participants"):
        if not paired[f"{count}_lag0"].eq(paired[f"{count}_lag_minus_10s"]).all():
            raise RuntimeError(f"Matched conditions differ in {count}")
    for metric in (*PRIMARY_METRICS, *SECONDARY_METRICS):
        name = f"participant_macro_{metric}"
        paired[f"delta_{metric}"] = (
            paired[f"{name}_lag_minus_10s"] - paired[f"{name}_lag0"]
        )
    _write_csv(context.output_dir / "paired_delta_by_fold.csv", paired)
    pm_rows: list[dict[str, Any]] = []
    for pm in PM_NAMES:
        group = paired.loc[paired["pm"].eq(pm)]
        row: dict[str, Any] = {"pm": pm, "target_id": f"target_{pm}", "n_folds": len(group)}
        for metric in PRIMARY_METRICS:
            base = group[f"participant_macro_{metric}_lag0"]
            lag = group[f"participant_macro_{metric}_lag_minus_10s"]
            delta = group[f"delta_{metric}"]
            row.update({
                f"lag0_{metric}_mean": float(base.mean()),
                f"lag0_{metric}_std": float(base.std(ddof=1)),
                f"lag_minus_10s_{metric}_mean": float(lag.mean()),
                f"lag_minus_10s_{metric}_std": float(lag.std(ddof=1)),
                f"delta_{metric}_mean": float(delta.mean()),
                f"delta_{metric}_std": float(delta.std(ddof=1)),
            })
        pm_rows.append(row)
    summary_by_pm = pd.DataFrame(pm_rows)
    _write_csv(context.output_dir / "summary_by_pm.csv", summary_by_pm)
    pooled: dict[str, Any] = {
        "n_fold_pm_pairs": int(len(paired)),
        "independence_note": "fold-PM rows are paired descriptive comparisons, not fully independent inferential units",
    }
    favorable = {"mae": lambda values: values < 0, "r2": lambda values: values > 0, "pearson": lambda values: values > 0}
    for metric in PRIMARY_METRICS:
        values = paired[f"delta_{metric}"]
        pm_means = summary_by_pm[f"delta_{metric}_mean"]
        mask = favorable[metric](values)
        pooled.update({
            f"mean_delta_{metric}": float(values.mean()),
            f"std_delta_{metric}": float(values.std(ddof=1)),
            f"median_delta_{metric}": float(values.median()),
            f"favorable_fold_pm_{metric}": int(mask.sum()),
            f"favorable_fold_pm_fraction_{metric}": float(mask.mean()),
            f"favorable_pm_mean_{metric}": int(favorable[metric](pm_means).sum()),
        })
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
    if len(summaries) != 70:
        raise RuntimeError("Full aggregation requires all 70 fixed runs")
    aggregate_results(context, summaries)
    return {"complete": len(summaries), "trained": trained, "reused": reused}


__all__ = [
    "CONDITIONS",
    "PM_NAMES",
    "PRIMARY_METRICS",
    "ProtocolContext",
    "aggregate_results",
    "build_pm_fold_audit",
    "build_pm_matched_cohorts",
    "execute_run",
    "load_config",
    "load_resumable_summary",
    "participant_regression_metrics",
    "prepare_protocol",
    "run_experiment",
    "write_dry_run",
]
