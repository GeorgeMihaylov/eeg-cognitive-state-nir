"""Preregistered nested AutoML for the canonical seven-PM LOW/HIGH task.

The default path is a zero-fit dry-run. Candidate search and final outer
evaluation are separate explicit stages so outer-test information cannot enter
candidate generation, inner threshold fitting, scoring, or selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.automl.scientific.nested_extremes import (
    MODEL_FAMILIES,
    build_candidate_portfolio,
    build_inner_run_matrix,
    build_inner_subject_splits,
    build_nested_threshold_provenance,
    candidate_evaluation_matrix,
    candidate_matrix_hash,
    inner_split_hash,
    participant_first_objective,
    resumable_inner_summary,
    select_best_candidate,
    threshold_provenance_hash,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    PM_NAMES,
    _atomic_json,
    _git_head,
    _sample_hash,
    _write_csv,
    apply_extreme_labels,
    load_config as load_reference_config,
    participant_binary_metrics,
    prepare_protocol as prepare_reference_protocol,
    stable_hash,
)
from cogstate.model_zoo import build_model


SCHEMA_VERSION = "pm-low-high-nested-automl-v1"
EXPERIMENT_ID = "pm_low_high_nested_automl_v1"
EXPECTED_CONFIG_HASH = (
    "b038ada95e7d65235891f05440687f2ed9f5fa035c1485e2960912aa01fe02ad"
)
XGBOOST_REFERENCE_HASH = (
    "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
)
LIGHTGBM_REFERENCE_HASH = (
    "a2c51deb4d94e33d71863f1f1c7927470266b7352647664f3a411fb5db819b4e"
)
EXPECTED_INNER_CELLS = 105
EXPECTED_CANDIDATES = 26
EXPECTED_CANDIDATE_EVALUATIONS = 130
EXPECTED_INNER_FITS = 2730
EXPECTED_FINAL_FITS = 35
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42

REFERENCE_CONFIG = (
    "experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json"
)
XGBOOST_OUTPUT = (
    "reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1"
)
LIGHTGBM_OUTPUT = "reports/diagnostics/pm_low_high_model_robustness_v1"
DEFAULT_OUTPUT = "reports/diagnostics/pm_low_high_nested_automl_v1"


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the byte-independent frozen design and reject any mutation."""
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if stable_hash(config) != EXPECTED_CONFIG_HASH:
        raise ValueError("Frozen nested AutoML config hash changed")
    if tuple(config["scientific_contract"]["pm_names"]) != PM_NAMES:
        raise ValueError("Exactly seven PM in canonical order are required")
    if not all(config["forbidden"].values()):
        raise ValueError("All forbidden scientific switches must remain enabled")
    return config


def _completed_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("result_status") != "confirmatory_complete":
        raise ValueError(f"Reference protocol is not complete: {path}")
    return protocol


def _reference_audit(
    *,
    root: Path,
    config: Mapping[str, Any],
    reference: Any,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    xgb = _completed_protocol(root / XGBOOST_OUTPUT)
    lightgbm = _completed_protocol(root / LIGHTGBM_OUTPUT)
    protocols = {"xgboost": xgb, "lightgbm": lightgbm}
    expected = {
        "xgboost": XGBOOST_REFERENCE_HASH,
        "lightgbm": LIGHTGBM_REFERENCE_HASH,
    }
    rows = []
    for family in MODEL_FAMILIES:
        protocol = protocols[family]
        actual = str(protocol["protocol_hash"])
        configured = str(config["models"][family]["reference_protocol_hash"])
        valid = actual == expected[family] == configured
        if not valid:
            raise ValueError(f"{family}: frozen reference hash mismatch")
        if protocol["feature_cache_identity"]["feature_hash"] != (
            reference.cache_identity["feature_hash"]
        ):
            raise ValueError(f"{family}: feature hash mismatch")
        if protocol["fixed_fold_hash"] != reference.protocol["fixed_fold_hash"]:
            raise ValueError(f"{family}: fixed fold hash mismatch")
        rows.append({
            "reference": family,
            "output_dir": XGBOOST_OUTPUT if family == "xgboost" else LIGHTGBM_OUTPUT,
            "expected_protocol_hash": expected[family],
            "actual_protocol_hash": actual,
            "feature_hash": protocol["feature_cache_identity"]["feature_hash"],
            "cache_identity_hash": protocol["feature_cache_identity"][
                "cache_identity_hash"
            ],
            "fixed_fold_hash": protocol["fixed_fold_hash"],
            "valid": valid,
        })
    return pd.DataFrame(rows), protocols


def _anchors(protocols: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "xgboost": dict(protocols["xgboost"]["model"]["params"]),
        "lightgbm": dict(
            protocols["lightgbm"]["candidate_models"]["lightgbm"]["params"]
        ),
    }


@dataclass
class NestedAutoMLContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    reference: Any
    reference_protocols: dict[str, dict[str, Any]]
    reference_audit: pd.DataFrame
    candidates: pd.DataFrame
    inner_splits: pd.DataFrame
    thresholds: pd.DataFrame
    candidate_evaluations: pd.DataFrame
    inner_run_matrix: pd.DataFrame
    final_outer_plan: pd.DataFrame
    protocol: dict[str, Any]


def _final_outer_plan(reference: Any, *, protocol_hash: str) -> pd.DataFrame:
    rows = []
    audit = reference.threshold_audit.sort_values(
        ["outer_fold", "pm"], kind="stable"
    )
    for row in audit.to_dict("records"):
        payload = {
            "outer_fold": int(row["outer_fold"]),
            "pm": str(row["pm"]),
            "target_id": str(row["target_id"]),
            "selected_candidate_id": "PENDING_INNER_SELECTION_FREEZE",
            "outer_threshold_source": "full_outer_train_only",
            "outer_threshold_hash": str(row["threshold_hash"]),
            "q_low": float(row["q_low"]),
            "q_high": float(row["q_high"]),
            "n_train": int(row["n_train_retained"]),
            "n_test": int(row["n_test_retained"]),
        }
        payload["plan_hash"] = stable_hash({
            "protocol_hash": protocol_hash,
            "final_outer_plan": payload,
        })
        rows.append(payload)
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_FINAL_FITS:
        raise RuntimeError("Expected exactly 35 final outer fit plans")
    return frame


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> NestedAutoMLContext:
    """Reconstruct all identities and zero-fit nested specifications."""
    if stable_hash(dict(config)) != EXPECTED_CONFIG_HASH:
        raise ValueError("Frozen nested AutoML config changed")
    root_path = Path(root).resolve()
    output = Path(output_dir or DEFAULT_OUTPUT)
    if not output.is_absolute():
        output = root_path / output
    reference_config = load_reference_config(root_path / REFERENCE_CONFIG)
    reference = prepare_reference_protocol(
        reference_config,
        root=root_path,
        feature_cache_dir=feature_cache_dir,
        output_dir=root_path / XGBOOST_OUTPUT,
    )
    identities = config["identities"]
    for key in ("feature_hash", "cache_identity_hash"):
        if reference.cache_identity[key] != identities[key]:
            raise ValueError(f"Canonical {key} changed")
    if reference.protocol["fixed_fold_hash"] != identities["fixed_fold_hash"]:
        raise ValueError("Fixed outer-fold hash changed")
    reference_audit, protocols = _reference_audit(
        root=root_path,
        config=config,
        reference=reference,
    )
    candidates = build_candidate_portfolio(
        config,
        anchors=_anchors(protocols),
    )
    inner_splits = build_inner_subject_splits(
        reference.feature_index[["sample_id", "subject_id", "outer_fold"]],
        outer_folds=config["scientific_contract"]["outer_folds"],
        inner_folds=int(config["scientific_contract"]["inner_folds"]),
    )
    thresholds = build_nested_threshold_provenance(
        full=reference.full,
        cohorts=reference.cohorts,
        inner_splits=inner_splits,
        outer_threshold_audit=reference.threshold_audit,
    )
    candidate_hash = candidate_matrix_hash(candidates)
    split_hash = inner_split_hash(inner_splits)
    provenance_hash = threshold_provenance_hash(thresholds)
    protocol_hash = stable_hash({
        "schema_version": SCHEMA_VERSION,
        "frozen_config_hash": EXPECTED_CONFIG_HASH,
        "feature_cache_identity": reference.cache_identity,
        "fixed_fold_hash": reference.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": reference.protocol["temporal_pairing_hash"],
        "reference_protocol_hashes": {
            family: protocols[family]["protocol_hash"]
            for family in MODEL_FAMILIES
        },
        "candidate_matrix_hash": candidate_hash,
        "inner_split_hash": split_hash,
        "threshold_provenance_hash": provenance_hash,
    })
    candidate_evaluations = candidate_evaluation_matrix(
        candidates,
        outer_folds=config["scientific_contract"]["outer_folds"],
        protocol_hash=protocol_hash,
    )
    run_matrix = build_inner_run_matrix(
        candidates=candidates,
        thresholds=thresholds,
        protocol_hash=protocol_hash,
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "result_status": "preregistered_dry_run",
        "frozen_config_hash": EXPECTED_CONFIG_HASH,
        "scientific_contract": config["scientific_contract"],
        "nested_target_contract": config["nested_target_contract"],
        "candidate_generation": config["candidate_generation"],
        "inner_selection": config["inner_selection"],
        "final_outer_training": config["final_outer_training"],
        "evaluation": config["evaluation"],
        "forbidden": config["forbidden"],
        "feature_cache_identity": reference.cache_identity,
        "fixed_fold_hash": reference.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": reference.protocol["temporal_pairing_hash"],
        "reference_protocol_hashes": {
            family: protocols[family]["protocol_hash"]
            for family in MODEL_FAMILIES
        },
        "candidate_matrix_hash": candidate_hash,
        "inner_split_hash": split_hash,
        "threshold_provenance_hash": provenance_hash,
        "inner_run_matrix_hash": stable_hash({
            "inner_runs": run_matrix.to_dict("records")
        }),
        "candidate_model_training_executed": False,
        "candidate_model_inference_executed": False,
        "outer_model_training_executed": False,
        "outer_model_inference_executed": False,
        "performance_evaluation_executed": False,
        "git_commit": _git_head(root_path),
        "protocol_hash": protocol_hash,
    }
    final_plan = _final_outer_plan(reference, protocol_hash=protocol_hash)
    return NestedAutoMLContext(
        root=root_path,
        output_dir=output,
        config=dict(config),
        reference=reference,
        reference_protocols=protocols,
        reference_audit=reference_audit,
        candidates=candidates,
        inner_splits=inner_splits,
        thresholds=thresholds,
        candidate_evaluations=candidate_evaluations,
        inner_run_matrix=run_matrix,
        final_outer_plan=final_plan,
        protocol=protocol,
    )


def write_dry_run(context: NestedAutoMLContext) -> dict[str, Any]:
    """Write preregistration artifacts without model construction or inference."""
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(context.output_dir / "reference_audit.csv", context.reference_audit)
    _write_csv(context.output_dir / "candidate_matrix.csv", context.candidates)
    _write_csv(context.output_dir / "inner_splits.csv", context.inner_splits)
    _write_csv(
        context.output_dir / "threshold_provenance.csv", context.thresholds
    )
    _write_csv(
        context.output_dir / "candidate_evaluation_matrix.csv",
        context.candidate_evaluations,
    )
    _write_csv(
        context.output_dir / "inner_run_matrix.csv", context.inner_run_matrix
    )
    _write_csv(
        context.output_dir / "final_outer_plan.csv", context.final_outer_plan
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": context.protocol["protocol_hash"],
        "candidate_matrix_hash": context.protocol["candidate_matrix_hash"],
        "inner_split_hash": context.protocol["inner_split_hash"],
        "threshold_provenance_hash": context.protocol[
            "threshold_provenance_hash"
        ],
        "feature_count": int(context.reference.matrix.shape[1]),
        "fixed_lag_seconds": -10,
        "outer_folds": 5,
        "inner_folds_per_outer": 3,
        "pm_count": len(PM_NAMES),
        "valid_inner_fold_pm_cells": int(len(context.thresholds)),
        "class_complete_inner_train_cells": int(
            context.thresholds["class_complete_train"].sum()
        ),
        "class_complete_inner_validation_cells": int(
            context.thresholds["class_complete_validation"].sum()
        ),
        "candidate_count": int(len(context.candidates)),
        "candidates_per_model": {
            family: int(context.candidates["model_family"].eq(family).sum())
            for family in MODEL_FAMILIES
        },
        "candidates_per_outer_fold": 26,
        "candidate_evaluations": int(len(context.candidate_evaluations)),
        "planned_inner_fits": int(len(context.inner_run_matrix)),
        "planned_final_outer_fits": int(len(context.final_outer_plan)),
        "inner_subject_overlap_count": int(
            context.inner_splits["subject_overlap_count"].sum()
        ),
        "outer_test_inner_leakage_count": int(
            context.inner_splits["outer_test_leakage_count"].sum()
        ),
        "validation_labels_used_for_thresholds": False,
        "outer_thresholds_reused_inside_inner_cv": False,
        "reference_hashes_valid": bool(context.reference_audit["valid"].all()),
        "candidate_model_training_executed": False,
        "candidate_model_inference_executed": False,
        "outer_model_training_executed": False,
        "outer_model_inference_executed": False,
        "performance_evaluation_executed": False,
    }
    if summary["valid_inner_fold_pm_cells"] != EXPECTED_INNER_CELLS:
        raise RuntimeError("Dry-run did not create 105 valid inner cells")
    if summary["planned_inner_fits"] != EXPECTED_INNER_FITS:
        raise RuntimeError("Dry-run did not create 2730 inner fits")
    if summary["planned_final_outer_fits"] != EXPECTED_FINAL_FITS:
        raise RuntimeError("Dry-run did not create 35 final fits")
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    readme = f"""# PM LOW/HIGH nested AutoML v1

Preregistered nested model-family/hyperparameter selection protocol.

- seven PM, one shared selected candidate per outer fold
- exact `EEG(t-10s) -> PM(t)` pairing and 371 canonical features
- five frozen subject-disjoint outer folds
- three deterministic subject-disjoint inner folds
- inner Q33/Q67 fitted on inner-train continuous labels only
- 13 distinct XGBoost + 13 distinct LightGBM candidates
- 2730 planned inner fits and 35 planned final outer fits
- dry-run training, inference and performance evaluation: false

Protocol hash: `{context.protocol['protocol_hash']}`
Candidate matrix hash: `{context.protocol['candidate_matrix_hash']}`
Inner split hash: `{context.protocol['inner_split_hash']}`
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def _inner_run_directory(
    context: NestedAutoMLContext, spec: Mapping[str, Any]
) -> Path:
    return context.output_dir / "inner_runs" / str(spec["run_id"])


def _known_inner_spec(
    context: NestedAutoMLContext, spec: Mapping[str, Any]
) -> None:
    matches = context.inner_run_matrix.loc[
        context.inner_run_matrix["run_id"].eq(str(spec["run_id"]))
    ]
    if len(matches) != 1:
        raise ValueError("Inner fit is not in the frozen run matrix")
    expected = matches.iloc[0]
    for key in ("specification_hash", "candidate_hash", "threshold_hash"):
        if str(spec[key]) != str(expected[key]):
            raise ValueError(f"Inner fit changed at {key}")


def _inner_arrays(
    context: NestedAutoMLContext,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    outer_fold = int(spec["outer_fold"])
    inner_fold = int(spec["inner_fold"])
    pm = str(spec["pm"])
    split = context.inner_splits.set_index(["outer_fold", "inner_fold"]).loc[
        (outer_fold, inner_fold)
    ]
    train_subjects = set(str(split["train_subjects"]).split("|"))
    validation_subjects = set(str(split["validation_subjects"]).split("|"))
    cohort = context.reference.cohorts[pm]
    protected = cohort["outer_fold"].astype(int).ne(outer_fold).to_numpy()
    train_before = protected & cohort["subject_id"].astype(str).isin(
        train_subjects
    ).to_numpy()
    validation_before = protected & cohort["subject_id"].astype(str).isin(
        validation_subjects
    ).to_numpy()
    labels = apply_extreme_labels(
        cohort["continuous_target"],
        q_low=float(spec["q_low"]),
        q_high=float(spec["q_high"]),
    )
    train = train_before & np.isfinite(labels)
    validation = validation_before & np.isfinite(labels)
    if set(cohort.loc[train, "subject_id"].astype(str)) & set(
        cohort.loc[validation, "subject_id"].astype(str)
    ):
        raise RuntimeError("Inner execution subject leakage")
    if int(train.sum()) != int(spec["n_train"]):
        raise RuntimeError("Inner train count changed")
    if int(validation.sum()) != int(spec["n_validation"]):
        raise RuntimeError("Inner validation count changed")
    if _sample_hash(cohort.loc[train, "target_sample_id"]) != str(
        spec["train_sample_hash"]
    ):
        raise RuntimeError("Inner train sample hash changed")
    if _sample_hash(cohort.loc[validation, "target_sample_id"]) != str(
        spec["validation_sample_hash"]
    ):
        raise RuntimeError("Inner validation sample hash changed")
    positions = cohort["lag_minus_10s_feature_position"].to_numpy(dtype=int)
    x_train = np.asarray(context.reference.matrix[positions[train]], dtype=np.float32)
    x_validation = np.asarray(
        context.reference.matrix[positions[validation]], dtype=np.float32
    )
    return (
        x_train,
        labels[train].astype(np.int64),
        cohort.loc[train].reset_index(drop=True),
        x_validation,
        labels[validation].astype(np.int64),
        cohort.loc[validation].reset_index(drop=True),
    )


def execute_inner_fit(
    context: NestedAutoMLContext,
    spec: Mapping[str, Any],
    *,
    model_builder: Callable[..., Any] = build_model,
) -> dict[str, Any]:
    """Execute one inner fit; this function is never called by dry-run."""
    _known_inner_spec(context, spec)
    x_train, y_train, _, x_validation, y_validation, metadata = _inner_arrays(
        context, spec
    )
    if x_train.shape[1] != 371 or x_validation.shape[1] != 371:
        raise RuntimeError("Inner execution feature count changed")
    params = json.loads(str(spec["params_json"]))
    started = time.perf_counter()
    model = model_builder(
        str(spec["model_family"]),
        "classification",
        (371,),
        2,
        params,
    )
    model.fit(x_train, y_train)
    probabilities = np.asarray(model.predict_proba(x_validation), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]), dtype=int)
    high_columns = np.flatnonzero(classes == 1)
    if probabilities.shape != (len(y_validation), len(classes)):
        raise RuntimeError("Inner predict_proba shape changed")
    if len(high_columns) != 1:
        raise RuntimeError("Inner classifier lacks HIGH probability")
    probability_high = probabilities[:, int(high_columns[0])]
    prediction = (probability_high >= 0.5).astype(np.int64)
    elapsed = time.perf_counter() - started
    predictions = metadata[[
        "target_sample_id", "subject_id", "record_id", "outer_fold"
    ]].copy()
    predictions["inner_fold"] = int(spec["inner_fold"])
    predictions["pm"] = str(spec["pm"])
    predictions["candidate_id"] = str(spec["candidate_id"])
    predictions["y_true"] = y_validation
    predictions["y_pred"] = prediction
    predictions["probability_high"] = probability_high
    directory = _inner_run_directory(context, spec)
    _atomic_parquet(directory / "predictions.parquet", predictions)
    summary = {
        "status": "complete",
        "result_status": "inner_selection_runtime",
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hash": spec["specification_hash"],
        "run_id": spec["run_id"],
        "candidate_id": spec["candidate_id"],
        "model_family": spec["model_family"],
        "outer_fold": int(spec["outer_fold"]),
        "inner_fold": int(spec["inner_fold"]),
        "pm": spec["pm"],
        "threshold_hash": spec["threshold_hash"],
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "validation_sample_hash": spec["validation_sample_hash"],
        "training_time_seconds": float(elapsed),
        "candidate_model_training_executed": True,
        "candidate_model_inference_executed": True,
        "outer_model_training_executed": False,
        "outer_model_inference_executed": False,
        "performance_evaluation_executed": False,
    }
    _atomic_json(directory / "run_summary.json", summary)
    return summary


def score_inner_candidates(
    context: NestedAutoMLContext,
    summaries: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if len(summaries) != EXPECTED_INNER_FITS:
        raise RuntimeError("Candidate scoring requires all 2730 inner fits")
    rows = []
    for evaluation in context.candidate_evaluations.to_dict("records"):
        specs = context.inner_run_matrix.loc[
            context.inner_run_matrix["outer_fold"].eq(
                int(evaluation["outer_fold"])
            )
            & context.inner_run_matrix["candidate_id"].eq(
                str(evaluation["candidate_id"])
            )
        ]
        if len(specs) != 21:
            raise RuntimeError("Each candidate evaluation requires 21 inner fits")
        predictions = pd.concat([
            pd.read_parquet(
                _inner_run_directory(context, spec) / "predictions.parquet"
            )
            for spec in specs.to_dict("records")
        ], ignore_index=True)
        objective = participant_first_objective(predictions)
        rows.append({
            **evaluation,
            "participant_first_balanced_accuracy": objective[
                "participant_first_balanced_accuracy"
            ],
            "participant_first_macro_f1": objective[
                "participant_first_macro_f1"
            ],
            "participants": objective["participants"],
            "participant_pm_rows": objective["participant_pm_rows"],
            "completed_inner_fits": 21,
        })
    frame = pd.DataFrame(rows).sort_values(
        ["outer_fold", "candidate_id"], kind="stable"
    )
    _write_csv(context.output_dir / "candidate_scores.csv", frame)
    return frame


def run_inner_search(
    context: NestedAutoMLContext,
    *,
    resume: bool,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    trained = 0
    reused = 0
    for spec in context.inner_run_matrix.to_dict("records"):
        directory = _inner_run_directory(context, spec)
        existing = (
            resumable_inner_summary(
                directory,
                specification=spec,
                protocol_hash=context.protocol["protocol_hash"],
            )
            if resume
            else None
        )
        if existing is not None:
            summaries.append(existing)
            reused += 1
            continue
        if directory.exists() and not resume:
            raise FileExistsError(f"Inner run exists; use --resume: {directory}")
        summaries.append(execute_inner_fit(context, spec))
        trained += 1
    scores = score_inner_candidates(context, summaries)
    return {
        "status": "inner_search_complete",
        "protocol_hash": context.protocol["protocol_hash"],
        "completed_inner_fits": len(summaries),
        "trained": trained,
        "reused": reused,
        "candidate_evaluations": len(scores),
        "outer_test_performance_evaluated": False,
    }


def freeze_selection(
    context: NestedAutoMLContext,
    scores: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Freeze one family+parameter candidate per outer fold using inner OOF only."""
    if scores is None:
        scores = pd.read_csv(context.output_dir / "candidate_scores.csv")
    if len(scores) != EXPECTED_CANDIDATE_EVALUATIONS:
        raise RuntimeError("Selection requires all 130 candidate evaluations")
    selected = []
    for outer_fold in range(1, 6):
        fold_scores = scores.loc[scores["outer_fold"].eq(outer_fold)]
        if len(fold_scores) != EXPECTED_CANDIDATES:
            raise RuntimeError("Each outer fold must compare 26 candidates")
        best = select_best_candidate(fold_scores)
        selected.append({
            "outer_fold": outer_fold,
            "candidate_id": str(best["candidate_id"]),
            "model_family": str(best["model_family"]),
            "candidate_hash": str(best["candidate_hash"]),
            "participant_first_balanced_accuracy": float(
                best["participant_first_balanced_accuracy"]
            ),
            "participant_first_macro_f1": float(
                best["participant_first_macro_f1"]
            ),
            "selection_scope": "one_shared_candidate_all_seven_pm",
        })
    selection_hash = stable_hash({
        "protocol_hash": context.protocol["protocol_hash"],
        "selected": selected,
    })
    payload = {
        "status": "inner_selection_frozen",
        "protocol_hash": context.protocol["protocol_hash"],
        "selection_hash": selection_hash,
        "outer_test_used": False,
        "selected_candidates": selected,
    }
    _atomic_json(context.output_dir / "selected_candidates.json", payload)
    return payload


def build_final_run_matrix(
    context: NestedAutoMLContext,
    selection: Mapping[str, Any],
) -> pd.DataFrame:
    if selection.get("status") != "inner_selection_frozen":
        raise ValueError("Final outer stage requires frozen inner selection")
    if selection.get("protocol_hash") != context.protocol["protocol_hash"]:
        raise ValueError("Selection protocol hash mismatch")
    selected = {
        int(row["outer_fold"]): row
        for row in selection["selected_candidates"]
    }
    if set(selected) != set(range(1, 6)):
        raise ValueError("Exactly five selected outer-fold candidates are required")
    candidates = context.candidates.set_index("candidate_id")
    audit = context.reference.threshold_audit.set_index(["outer_fold", "pm"])
    rows = []
    for outer_fold in range(1, 6):
        choice = selected[outer_fold]
        candidate = candidates.loc[str(choice["candidate_id"])]
        if str(candidate["candidate_hash"]) != str(choice["candidate_hash"]):
            raise ValueError("Selected candidate hash mismatch")
        for pm in PM_NAMES:
            threshold = audit.loc[(outer_fold, pm)]
            spec = {
                "outer_fold": outer_fold,
                "pm": pm,
                "target_id": f"target_{pm}",
                "candidate_id": str(candidate.name),
                "model_family": str(candidate["model_family"]),
                "candidate_hash": str(candidate["candidate_hash"]),
                "params_json": str(candidate["params_json"]),
                "selection_hash": str(selection["selection_hash"]),
                "threshold_source": "full_outer_train_only",
                "threshold_hash": str(threshold["threshold_hash"]),
                "q_low": float(threshold["q_low"]),
                "q_high": float(threshold["q_high"]),
                "n_train": int(threshold["n_train_retained"]),
                "n_test": int(threshold["n_test_retained"]),
                "train_sample_hash": str(threshold["train_retained_sample_hash"]),
                "test_sample_hash": str(threshold["test_retained_sample_hash"]),
            }
            spec_hash = stable_hash({
                "protocol_hash": context.protocol["protocol_hash"],
                "final_outer_fit": spec,
            })
            spec["specification_hash"] = spec_hash
            spec["run_id"] = (
                f"outer_{outer_fold:02d}__{candidate.name}__{pm}__"
                f"{spec_hash[:12]}"
            )
            rows.append(spec)
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_FINAL_FITS:
        raise RuntimeError("Expected 35 final outer fit specifications")
    shared = frame.groupby("outer_fold")["candidate_id"].nunique()
    if not shared.eq(1).all():
        raise RuntimeError("A selected candidate differs across PM")
    return frame


def _final_run_directory(
    context: NestedAutoMLContext, spec: Mapping[str, Any]
) -> Path:
    return context.output_dir / "final_runs" / str(spec["run_id"])


def execute_final_outer_fit(
    context: NestedAutoMLContext,
    spec: Mapping[str, Any],
    *,
    model_builder: Callable[..., Any] = build_model,
) -> dict[str, Any]:
    """Fit one selected PM model and use its frozen outer test exactly once."""
    fold = int(spec["outer_fold"])
    pm = str(spec["pm"])
    cohort = context.reference.cohorts[pm]
    labels = apply_extreme_labels(
        cohort["continuous_target"],
        q_low=float(spec["q_low"]),
        q_high=float(spec["q_high"]),
    )
    train = cohort["outer_fold"].astype(int).ne(fold).to_numpy() & np.isfinite(labels)
    test = cohort["outer_fold"].astype(int).eq(fold).to_numpy() & np.isfinite(labels)
    if set(cohort.loc[train, "subject_id"].astype(str)) & set(
        cohort.loc[test, "subject_id"].astype(str)
    ):
        raise RuntimeError("Final outer subject leakage")
    if int(train.sum()) != int(spec["n_train"]) or int(test.sum()) != int(
        spec["n_test"]
    ):
        raise RuntimeError("Final outer cohort count changed")
    if _sample_hash(cohort.loc[train, "target_sample_id"]) != str(
        spec["train_sample_hash"]
    ) or _sample_hash(cohort.loc[test, "target_sample_id"]) != str(
        spec["test_sample_hash"]
    ):
        raise RuntimeError("Final outer sample identity changed")
    positions = cohort["lag_minus_10s_feature_position"].to_numpy(dtype=int)
    x_train = np.asarray(context.reference.matrix[positions[train]], dtype=np.float32)
    x_test = np.asarray(context.reference.matrix[positions[test]], dtype=np.float32)
    y_train = labels[train].astype(np.int64)
    y_test = labels[test].astype(np.int64)
    started = time.perf_counter()
    model = model_builder(
        str(spec["model_family"]),
        "classification",
        (371,),
        2,
        json.loads(str(spec["params_json"])),
    )
    model.fit(x_train, y_train)
    probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]), dtype=int)
    high_columns = np.flatnonzero(classes == 1)
    if len(high_columns) != 1 or probabilities.shape[0] != len(y_test):
        raise RuntimeError("Final predict_proba output changed")
    probability_high = probabilities[:, int(high_columns[0])]
    prediction = (probability_high >= 0.5).astype(np.int64)
    elapsed = time.perf_counter() - started
    metadata = cohort.loc[test].reset_index(drop=True)
    predictions = metadata[[
        "target_sample_id", "subject_id", "record_id", "outer_fold"
    ]].copy()
    predictions["pm"] = pm
    predictions["candidate_id"] = str(spec["candidate_id"])
    predictions["model_family"] = str(spec["model_family"])
    predictions["y_true"] = y_test
    predictions["y_pred"] = prediction
    predictions["probability_high"] = probability_high
    directory = _final_run_directory(context, spec)
    _atomic_parquet(directory / "predictions.parquet", predictions)
    summary = {
        "status": "complete",
        "result_status": "final_outer_runtime",
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hash": spec["specification_hash"],
        "selection_hash": spec["selection_hash"],
        "run_id": spec["run_id"],
        "outer_fold": fold,
        "pm": pm,
        "candidate_id": spec["candidate_id"],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "training_time_seconds": float(elapsed),
        "outer_model_training_executed": True,
        "outer_model_inference_executed": True,
        "performance_evaluation_executed": False,
    }
    _atomic_json(directory / "run_summary.json", summary)
    return summary


def _reference_prediction_paths(
    context: NestedAutoMLContext,
) -> dict[tuple[str, int, str], Path]:
    paths: dict[tuple[str, int, str], Path] = {}
    xgb = pd.read_csv(context.root / XGBOOST_OUTPUT / "run_matrix.csv")
    lightgbm = pd.read_csv(context.root / LIGHTGBM_OUTPUT / "run_matrix.csv")
    for row in xgb.to_dict("records"):
        paths[("frozen_xgboost", int(row["outer_fold"]), str(row["pm"]))] = (
            context.root / XGBOOST_OUTPUT / "runs" / row["run_id"] / "predictions.parquet"
        )
    for row in lightgbm.loc[lightgbm["model"].eq("lightgbm")].to_dict("records"):
        paths[("frozen_lightgbm", int(row["outer_fold"]), str(row["pm"]))] = (
            context.root / LIGHTGBM_OUTPUT / "runs" / row["run_id"] / "predictions.parquet"
        )
    if len(paths) != 70:
        raise RuntimeError("Expected 70 frozen XGBoost/LightGBM prediction sources")
    return paths


def aggregate_final_outer(
    context: NestedAutoMLContext,
    final_matrix: pd.DataFrame,
) -> dict[str, Any]:
    """Apply the preregistered participant-first comparison after 35 fits."""
    reference_paths = _reference_prediction_paths(context)
    participant_rows = []
    for spec in final_matrix.to_dict("records"):
        fold = int(spec["outer_fold"])
        pm = str(spec["pm"])
        sources = {
            "automl": _final_run_directory(context, spec) / "predictions.parquet",
            "frozen_lightgbm": reference_paths[("frozen_lightgbm", fold, pm)],
            "frozen_xgboost": reference_paths[("frozen_xgboost", fold, pm)],
        }
        expected_ids: list[str] | None = None
        for source, path in sources.items():
            predictions = pd.read_parquet(path)
            predictions["target_sample_id"] = predictions[
                "target_sample_id"
            ].astype(str)
            predictions = predictions.sort_values("target_sample_id", kind="stable")
            ids = predictions["target_sample_id"].tolist()
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise RuntimeError("Final reference evaluation cohort changed")
            participants, _ = participant_binary_metrics(
                predictions["y_true"].to_numpy(int),
                predictions["y_pred"].to_numpy(int),
                predictions["probability_high"].to_numpy(float),
                predictions["subject_id"].astype(str).to_numpy(),
            )
            participants.insert(0, "source", source)
            participants.insert(1, "outer_fold", fold)
            participants.insert(2, "pm", pm)
            participant_rows.append(participants)
    participant_pm = pd.concat(participant_rows, ignore_index=True)
    metrics = ["balanced_accuracy", "macro_f1", "roc_auc", "pr_auc"]
    participant = participant_pm.groupby(
        ["source", "subject_id"], sort=True
    )[metrics].mean().reset_index()
    pooled = participant.groupby("source", sort=True)[metrics].mean().reset_index()
    contrasts = participant.loc[participant["source"].eq("automl")].set_index(
        "subject_id"
    )[metrics]
    contrast_rows = []
    bootstrap_rows = []
    for reference in ("frozen_lightgbm", "frozen_xgboost"):
        base = participant.loc[participant["source"].eq(reference)].set_index(
            "subject_id"
        )[metrics]
        aligned = contrasts.join(base, lsuffix="_automl", rsuffix="_reference")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for metric in metrics:
            delta = (
                aligned[f"{metric}_automl"]
                - aligned[f"{metric}_reference"]
            ).dropna().to_numpy(float)
            contrast_rows.append({
                "reference": reference,
                "metric": metric,
                "participant_first_delta": float(delta.mean()),
                "valid_participants": int(len(delta)),
            })
            indices = rng.integers(
                0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta))
            )
            sampled = delta[indices].mean(axis=1)
            bootstrap_rows.append({
                "reference": reference,
                "metric": metric,
                "observed_delta": float(delta.mean()),
                "ci95_low": float(np.quantile(sampled, 0.025)),
                "ci95_high": float(np.quantile(sampled, 0.975)),
                "bootstrap_unit": "subject_id",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
    _write_csv(context.output_dir / "final_participant_pm.csv", participant_pm)
    _write_csv(context.output_dir / "final_participant_first.csv", participant)
    _write_csv(context.output_dir / "final_pooled_summary.csv", pooled)
    _write_csv(context.output_dir / "final_contrasts.csv", pd.DataFrame(contrast_rows))
    _write_csv(context.output_dir / "final_bootstrap.csv", pd.DataFrame(bootstrap_rows))
    summary = {
        "status": "final_outer_complete",
        "protocol_hash": context.protocol["protocol_hash"],
        "completed_final_outer_fits": int(len(final_matrix)),
        "primary_metric": "participant_first_balanced_accuracy",
        "primary_reference": "frozen_lightgbm",
        "secondary_reference": "frozen_xgboost",
        "outer_model_training_executed": True,
        "outer_model_inference_executed": True,
        "performance_evaluation_executed": True,
    }
    _atomic_json(context.output_dir / "final_summary.json", summary)
    return summary


def run_final_outer(
    context: NestedAutoMLContext,
    *,
    resume: bool,
) -> dict[str, Any]:
    selection = json.loads(
        (context.output_dir / "selected_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    matrix = build_final_run_matrix(context, selection)
    _write_csv(context.output_dir / "final_run_matrix.csv", matrix)
    summaries = []
    for spec in matrix.to_dict("records"):
        directory = _final_run_directory(context, spec)
        summary_path = directory / "run_summary.json"
        prediction_path = directory / "predictions.parquet"
        if resume and summary_path.is_file() and prediction_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            valid = (
                summary.get("status") == "complete"
                and summary.get("protocol_hash") == context.protocol["protocol_hash"]
                and summary.get("specification_hash") == spec["specification_hash"]
            )
            if valid:
                summaries.append(summary)
                continue
        if directory.exists() and not resume:
            raise FileExistsError(f"Final run exists; use --resume: {directory}")
        summaries.append(execute_final_outer_fit(context, spec))
    if len(summaries) != EXPECTED_FINAL_FITS:
        raise RuntimeError("Final outer stage did not complete 35 fits")
    return aggregate_final_outer(context, matrix)


__all__ = [
    "NestedAutoMLContext",
    "aggregate_final_outer",
    "build_final_run_matrix",
    "execute_final_outer_fit",
    "execute_inner_fit",
    "freeze_selection",
    "load_config",
    "prepare_protocol",
    "run_final_outer",
    "run_inner_search",
    "score_inner_candidates",
    "write_dry_run",
]
