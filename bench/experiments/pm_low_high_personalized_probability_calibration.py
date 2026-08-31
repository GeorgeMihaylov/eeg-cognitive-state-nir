"""Personalized probability calibration at the frozen 30-minute budget.

Candidate: one-parameter logit offset, p' = sigmoid(logit(p) + b).
The slope is fixed to one; b is fit from the participant-PM 1800-second
calibration prefix only. Eligibility is prospectively fixed at full 1800-second
coverage plus >=10 LOW and >=10 HIGH labels.

Classification reference: the already frozen 1800-second median-midpoint
policy. On logit-calibration-ineligible cells, candidate classification falls
back exactly to that reference policy. For probability metrics, ineligible
cells keep the original zero-shot probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bench.experiments.pm_low_high_personalization_duration_response import (
    METRICS,
    _atomic_json,
    _completed_protocol,
    _frame_hash,
    _git_head,
    _write_csv,
)
from bench.experiments.pm_low_high_personalization_long_duration_feasibility import (
    load_config as load_long_feasibility_config,
    prepare_protocol as prepare_long_feasibility_protocol,
)
from bench.experiments.pm_low_high_personalization_feasibility import (
    _subject_pm_timeline,
)
from bench.experiments.pm_low_high_personalized_threshold import (
    _audit_prediction_sources,
    _load_prediction_lookup,
    _metric_row,
    _prediction_source_matrix,
    _state_to_y,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    PM_NAMES,
    stable_hash,
)

SCHEMA_VERSION = "pm-low-high-personalized-probability-calibration-v1"
MODELS = ("xgboost", "lightgbm")
CALIBRATION_BUDGET_SECONDS = 1800
MIN_EACH = 10
EXPECTED_EVAL_READY = 345
EXPECTED_ELIGIBLE = 285
EXPECTED_RESULT_ROWS = EXPECTED_EVAL_READY * len(MODELS)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
EPS = 1e-6
SOLVER_LOW = -40.0
SOLVER_HIGH = 40.0
SOLVER_ITERATIONS = 80

LONG_FEASIBILITY_HASH = (
    "34e0aa3350f84198383cd0e6a1d213711983132dcc30aa14fcc9edaafbc1095f"
)
LONG_RESPONSE_HASH = (
    "7fdc10bccad792c1f2d113ee063469230f229bd3dabf0cf0cb21e4f9b88e5caf"
)
XGBOOST_HASH = (
    "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
)
LIGHTGBM_HASH = (
    "a2c51deb4d94e33d71863f1f1c7927470266b7352647664f3a411fb5db819b4e"
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    q = _clip_probability(p)
    return np.log(q) - np.log1p(-q)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit_offset(
    probability_high: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Fit the unique intercept-only logistic score root deterministically."""
    p = _clip_probability(probability_high)
    y = np.asarray(y_true, dtype=int)
    if len(p) != len(y) or len(p) == 0:
        raise ValueError("Calibration arrays must be non-empty and aligned")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both calibration classes are required")

    z = _logit(p)

    def score(offset: float) -> float:
        return float(np.sum(_sigmoid(z + offset) - y))

    lo = SOLVER_LOW
    hi = SOLVER_HIGH
    slo = score(lo)
    shi = score(hi)
    if not (slo < 0.0 and shi > 0.0):
        raise RuntimeError(
            f"Offset root not bracketed: score({lo})={slo}, score({hi})={shi}"
        )

    for _ in range(SOLVER_ITERATIONS):
        mid = (lo + hi) / 2.0
        smid = score(mid)
        if smid > 0.0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2.0)


def apply_logit_offset(
    probability_high: np.ndarray,
    offset: float,
) -> np.ndarray:
    return _sigmoid(_logit(probability_high) + float(offset))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip_probability(p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        n = int(mask.sum())
        if n:
            value += (n / total) * abs(float(p[mask].mean() - y[mask].mean()))
    return float(value)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")

    refs = cfg.get("references", {})
    expected_refs = {
        "long_duration_feasibility": LONG_FEASIBILITY_HASH,
        "long_duration_response": LONG_RESPONSE_HASH,
        "xgboost": XGBOOST_HASH,
        "lightgbm": LIGHTGBM_HASH,
    }
    for key, expected in expected_refs.items():
        if refs.get(key, {}).get("protocol_hash") != expected:
            raise ValueError(f"{key} reference protocol changed")

    contract = cfg.get("scientific_contract", {})
    if contract.get("pm_names") != list(PM_NAMES):
        raise ValueError("PM contract changed")
    if contract.get("models") != list(MODELS):
        raise ValueError("Model contract changed")
    fixed = {
        "alignment": "EEG(t-10s) -> PM(t)",
        "lag_seconds": -10,
        "target_transform": "outer_train_q33_q67_extremes",
        "calibration_budget_seconds": 1800,
        "common_evaluation_boundary_seconds": 1800,
        "calibration_record_policy":
            "earliest_logical_record_by_selected_record_start_utc",
        "calibration_cross_record_policy": "forbidden",
        "budget_fully_available_rule":
            "source_duration_and_feature_grid_span_cover_budget",
        "minimum_calibration_low": 10,
        "minimum_calibration_high": 10,
        "probability_calibrator": "logit_offset_intercept_only",
        "logit_slope": 1.0,
        "probability_clip_epsilon": 1e-6,
        "offset_solver": "deterministic_bisection_score_root",
        "offset_solver_bounds": [-40.0, 40.0],
        "offset_solver_iterations": 80,
        "classification_threshold_after_calibration": 0.5,
        "classification_reference_policy": "frozen_1800s_median_midpoint",
        "classification_ineligible_fallback":
            "frozen_1800s_median_midpoint_policy",
        "probability_ineligible_fallback": "zero_shot_probability",
        "probability_source": "stored_predict_proba_high",
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise ValueError(f"Scientific contract changed at {key}")

    evaluation = cfg.get("evaluation", {})
    expected_eval = {
        "primary_metric": "brier_score",
        "primary_probability_contrast":
            "zero_shot_minus_personalized_logit_offset",
        "secondary_probability_metric": "log_loss",
        "secondary_probability_contrast":
            "zero_shot_minus_personalized_logit_offset",
        "classification_metric": "balanced_accuracy",
        "classification_secondary_metric": "macro_f1",
        "classification_contrast":
            "logit_offset_policy_minus_frozen_median_midpoint_policy",
        "aggregation": "mean_pm_within_participant_then_mean_participants",
        "primary_estimand": "operational_all_post1800_eval_ready",
        "secondary_estimand":
            "logit_calibration_applied_only_participant_first",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "bootstrap_unit": "subject_id",
        "ece_role": "descriptive_only",
        "ece_bins": 10,
    }
    if evaluation != expected_eval:
        raise ValueError("Evaluation contract changed")

    feasibility = cfg.get("feasibility_lock", {})
    if feasibility != {
        "post1800_eval_ready_participant_pm": 345,
        "full1800_and_min10_each_participant_pm": 285,
        "full1800_and_min10_each_fraction": 0.8260869565217391,
        "eligibility_selected_without_performance_evaluation": True,
    }:
        raise ValueError("Feasibility lock changed")

    forbidden = cfg.get("forbidden", {})
    if not forbidden or not all(value is True for value in forbidden.values()):
        raise ValueError("All forbidden switches must remain true")
    return cfg


@dataclass
class ProbabilityCalibrationContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    feasibility: Any
    feasibility_detail: pd.DataFrame
    frozen_response: pd.DataFrame
    source_matrix: pd.DataFrame
    source_audit: pd.DataFrame
    protocol: dict[str, Any]


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> ProbabilityCalibrationContext:
    root = Path(root).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root / output

    refs = config["references"]
    _completed_protocol(
        root, refs["long_duration_feasibility"]["output_dir"],
        LONG_FEASIBILITY_HASH, allowed_statuses={"feasibility_audit_complete"},
    )
    _completed_protocol(
        root, refs["long_duration_response"]["output_dir"],
        LONG_RESPONSE_HASH, allowed_statuses={"confirmatory_complete"},
    )
    _completed_protocol(
        root, refs["xgboost"]["output_dir"],
        XGBOOST_HASH, allowed_statuses={"confirmatory_complete"},
    )
    _completed_protocol(
        root, refs["lightgbm"]["output_dir"],
        LIGHTGBM_HASH, allowed_statuses={"confirmatory_complete"},
    )

    fcfg = load_long_feasibility_config(
        root / refs["long_duration_feasibility"]["config"]
    )
    feasibility = prepare_long_feasibility_protocol(
        fcfg,
        root=root,
        feature_cache_dir=feature_cache_dir,
        output_dir=root / refs["long_duration_feasibility"]["output_dir"],
    )
    detail = pd.read_csv(
        root / refs["long_duration_feasibility"]["output_dir"]
        / "participant_pm_budget_feasibility.csv"
    )
    detail = detail.loc[detail["budget_seconds"].astype(int).eq(1800)].copy()
    detail["subject_id"] = detail["subject_id"].astype(str)

    frozen = pd.read_csv(
        root / refs["long_duration_response"]["output_dir"]
        / "participant_pm_results.csv"
    )
    frozen["subject_id"] = frozen["subject_id"].astype(str)
    frozen = frozen.loc[frozen["budget_seconds"].astype(int).eq(1800)].copy()
    if len(frozen) != EXPECTED_RESULT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_RESULT_ROWS} frozen 1800s model rows, got {len(frozen)}"
        )

    source_cfg = {"references": {
        "xgboost": refs["xgboost"],
        "lightgbm": refs["lightgbm"],
    }}
    source_matrix = _prediction_source_matrix(root, source_cfg)
    source_audit = _audit_prediction_sources(root, source_matrix)
    if len(source_matrix) != 70 or len(source_audit) != 70:
        raise RuntimeError("Expected 70 stored prediction sources")
    if not source_audit["valid"].all():
        raise RuntimeError("Stored prediction source audit failed")

    ready = detail.loc[
        detail["fixed_evaluation_ready_min20_both_classes"]
        .astype(str).str.lower().eq("true")
    ]
    if len(ready) != EXPECTED_EVAL_READY:
        raise RuntimeError(
            f"Expected {EXPECTED_EVAL_READY} eval-ready cells, got {len(ready)}"
        )
    eligible = ready.loc[
        ready["budget_fully_available"].astype(str).str.lower().eq("true")
        & (ready["calibration_low"].astype(int) >= MIN_EACH)
        & (ready["calibration_high"].astype(int) >= MIN_EACH)
    ]
    if len(eligible) != EXPECTED_ELIGIBLE:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE} min10 eligible cells, got {len(eligible)}"
        )

    scientific_payload = {
        "schema_version": SCHEMA_VERSION,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "evaluation": config["evaluation"],
        "feasibility_lock": config["feasibility_lock"],
        "forbidden": config["forbidden"],
        "feasibility_1800_hash": _frame_hash(detail),
        "frozen_response_1800_hash": _frame_hash(frozen),
        "prediction_source_matrix_hash": _frame_hash(
            source_matrix.drop(columns=["source_output_dir"], errors="ignore")
        ),
        "prediction_source_audit_hash": _frame_hash(
            source_audit.drop(
                columns=["prediction_file", "run_summary_file"], errors="ignore"
            )
        ),
        "fixed_fold_hash": feasibility.base.low_high.protocol["fixed_fold_hash"],
        "threshold_hashes": feasibility.base.low_high.protocol["threshold_hashes"],
    }
    protocol_hash = stable_hash(scientific_payload)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "preregistered_candidate",
        "probability_calibration_executed": False,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": False,
        "git_commit": _git_head(root),
        "protocol_hash": protocol_hash,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "evaluation": config["evaluation"],
        "feasibility_lock": config["feasibility_lock"],
        "expected_eval_ready_participant_pm": EXPECTED_EVAL_READY,
        "expected_logit_eligible_participant_pm": EXPECTED_ELIGIBLE,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "prediction_source_rows": int(len(source_matrix)),
        "prediction_source_audit_mismatches":
            int((~source_audit["valid"]).sum()),
        "fixed_fold_hash": feasibility.base.low_high.protocol["fixed_fold_hash"],
        "threshold_hashes": feasibility.base.low_high.protocol["threshold_hashes"],
    }
    return ProbabilityCalibrationContext(
        root=root,
        output_dir=output,
        config=dict(config),
        feasibility=feasibility,
        feasibility_detail=detail,
        frozen_response=frozen,
        source_matrix=source_matrix,
        source_audit=source_audit,
        protocol=protocol,
    )


def write_dry_run(context: ProbabilityCalibrationContext) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(
        context.output_dir / "prediction_source_matrix.csv",
        context.source_matrix,
    )
    _write_csv(
        context.output_dir / "prediction_source_audit.csv",
        context.source_audit,
    )
    summary = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "models": list(MODELS),
        "calibration_budget_seconds": CALIBRATION_BUDGET_SECONDS,
        "minimum_calibration_each_class": MIN_EACH,
        "eval_ready_participant_pm": EXPECTED_EVAL_READY,
        "logit_eligible_participant_pm": EXPECTED_ELIGIBLE,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "probability_calibrator": "logit_offset_intercept_only",
        "classification_reference": "frozen_1800s_median_midpoint",
        "prediction_source_rows": 70,
        "prediction_source_audit_mismatches": 0,
        "probability_calibration_executed": False,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": False,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    (context.output_dir / "README.md").write_text(
        f"""# PM LOW/HIGH personalized probability calibration v1

- 30-minute calibration budget only
- all seven PM
- XGBoost + LightGBM stored probabilities
- no base-model training or new inference
- candidate: intercept-only logit offset
- slope fixed to 1
- eligibility: full 1800 s + >=10 LOW + >=10 HIGH
- eligibility fixed from feasibility only: {EXPECTED_ELIGIBLE}/{EXPECTED_EVAL_READY}
- classification reference: frozen 30-minute median-midpoint policy
- classification fallback when logit-ineligible: frozen median-midpoint policy
- probability fallback when logit-ineligible: zero-shot probability
- primary metric: Brier score
- secondary probability metric: log loss
- secondary classification metrics: balanced accuracy and Macro-F1
- participant-first aggregation
- subject-clustered bootstrap: 10,000 replicates

Protocol hash: `{context.protocol['protocol_hash']}`
""",
        encoding="utf-8",
    )
    return summary


def _participant_aggregate(results: pd.DataFrame, applied_only: bool) -> pd.DataFrame:
    frame = (
        results.loc[results["logit_calibration_applied"].astype(bool)].copy()
        if applied_only else results.copy()
    )
    rows = []
    for (model, subject_id), group in frame.groupby(
        ["model", "subject_id"], sort=True
    ):
        row = {
            "model": model,
            "subject_id": str(subject_id),
            "n_pm": int(group["pm"].nunique()),
            "brier_improvement": float(group["brier_improvement"].mean()),
            "log_loss_improvement": float(group["log_loss_improvement"].mean()),
            "delta_balanced_accuracy_vs_median": float(
                group["delta_balanced_accuracy_vs_median"].mean()
            ),
            "delta_macro_f1_vs_median": float(
                group["delta_macro_f1_vs_median"].mean()
            ),
            "delta_balanced_accuracy_vs_zero": float(
                group["delta_balanced_accuracy_vs_zero"].mean()
            ),
            "delta_macro_f1_vs_zero": float(
                group["delta_macro_f1_vs_zero"].mean()
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap(participant: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("brier_score", "brier_improvement"),
        ("log_loss", "log_loss_improvement"),
        ("balanced_accuracy_vs_median",
         "delta_balanced_accuracy_vs_median"),
        ("macro_f1_vs_median", "delta_macro_f1_vs_median"),
    ]
    rows = []
    for model_idx, model in enumerate(MODELS):
        part = participant.loc[participant["model"].eq(model)]
        for metric_idx, (metric, column) in enumerate(specs):
            values = part[column].to_numpy(float)
            n = len(values)
            seed = BOOTSTRAP_SEED + model_idx * 10000 + metric_idx * 1000
            rng = np.random.default_rng(seed)
            boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for i in range(BOOTSTRAP_REPLICATES):
                idx = rng.integers(0, n, size=n)
                boot[i] = float(values[idx].mean())
            rows.append({
                "model": model,
                "metric": metric,
                "contrast": (
                    "zero_shot_minus_logit_offset"
                    if metric in {"brier_score", "log_loss"}
                    else "logit_offset_policy_minus_frozen_median_midpoint"
                ),
                "observed_mean_delta": float(values.mean()),
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": seed,
                "resampling_unit": "subject_id",
                "n_subjects": n,
            })
    return pd.DataFrame(rows)


def run_experiment(context: ProbabilityCalibrationContext) -> dict[str, Any]:
    predictions = _load_prediction_lookup(context)
    detail = context.feasibility_detail.copy()
    detail["subject_id"] = detail["subject_id"].astype(str)
    frozen = context.frozen_response.copy()
    frozen["subject_id"] = frozen["subject_id"].astype(str)

    ready = detail.loc[
        detail["fixed_evaluation_ready_min20_both_classes"]
        .astype(str).str.lower().eq("true")
    ].copy()

    base = context.feasibility.base
    results = []

    for row in ready.itertuples(index=False):
        subject_id = str(row.subject_id)
        pm = str(row.pm)
        fold = int(row.outer_fold)

        timeline = _subject_pm_timeline(
            base, subject_id=subject_id, pm=pm
        ).sort_values(
            ["absolute_target_epoch_seconds", "target_sample_id"],
            kind="stable",
        )
        subject = base.subject_chronology.loc[
            base.subject_chronology["subject_id"].astype(str).eq(subject_id)
        ]
        if len(subject) != 1:
            raise RuntimeError("Expected one subject chronology row")
        subject = subject.iloc[0]
        cal_group = str(subject["calibration_record_group_id"])
        start = float(
            pd.Timestamp(subject["calibration_record_start_utc"]).timestamp()
        )

        relative = timeline["target_relative_seconds"].to_numpy(float)
        calibration = timeline.loc[
            timeline["state"].isin(["low", "high"])
            & timeline["record_group_id"].astype(str).eq(cal_group)
            & (relative > 0.0)
            & (relative <= 1800.0)
        ].copy()
        evaluation = timeline.loc[
            timeline["state"].isin(["low", "high"])
            & (
                timeline["absolute_target_epoch_seconds"].to_numpy(float)
                > start + 1800.0
            )
        ].copy()

        y_cal = _state_to_y(calibration["state"])
        y_eval = _state_to_y(evaluation["state"])
        eval_ids = evaluation["target_sample_id"].astype(str).tolist()
        eval_hash = stable_hash(eval_ids)
        if str(row.evaluation_extreme_sample_hash) != eval_hash:
            raise RuntimeError("Evaluation sample hash changed")

        n_low = int(row.calibration_low)
        n_high = int(row.calibration_high)
        if int(np.sum(y_cal == 0)) != n_low or int(np.sum(y_cal == 1)) != n_high:
            raise RuntimeError("Calibration counts changed")
        eligible = bool(
            _as_bool(row.budget_fully_available)
            and n_low >= MIN_EACH
            and n_high >= MIN_EACH
        )

        for model in MODELS:
            source = predictions[(model, fold, pm)]
            if set(eval_ids) - set(source.index.astype(str)):
                raise RuntimeError("Missing stored evaluation predictions")
            p_eval = source.loc[eval_ids, "probability_high"].to_numpy(float)
            y_source = source.loc[eval_ids, "y_true"].to_numpy(int)
            if not np.array_equal(y_source, y_eval):
                raise RuntimeError("Stored evaluation labels changed")

            frozen_row = frozen.loc[
                frozen["model"].eq(model)
                & frozen["subject_id"].eq(subject_id)
                & frozen["pm"].astype(str).eq(pm)
            ]
            if len(frozen_row) != 1:
                raise RuntimeError("Missing frozen 1800s median reference row")
            frozen_row = frozen_row.iloc[0]
            if str(frozen_row.evaluation_sample_hash) != eval_hash:
                raise RuntimeError("Frozen median evaluation hash changed")

            zero_metrics = _metric_row(y_eval, p_eval, 0.5, subject_id)
            median_threshold = float(frozen_row.personalized_threshold)
            median_metrics = _metric_row(
                y_eval, p_eval, median_threshold, subject_id
            )
            if not np.isclose(
                median_metrics["balanced_accuracy"],
                float(frozen_row.personalized_balanced_accuracy),
                atol=1e-15, rtol=0.0,
            ):
                raise RuntimeError("Frozen median BA did not reproduce")

            offset = 0.0
            applied = False
            candidate_p = p_eval.copy()
            candidate_threshold_original_scale = median_threshold

            if eligible:
                cal_ids = calibration["target_sample_id"].astype(str).tolist()
                if set(cal_ids) - set(source.index.astype(str)):
                    raise RuntimeError("Missing stored calibration predictions")
                p_cal = source.loc[
                    cal_ids, "probability_high"
                ].to_numpy(float)
                offset = fit_logit_offset(p_cal, y_cal)
                candidate_p = apply_logit_offset(p_eval, offset)
                applied = True
                candidate_threshold_original_scale = float(_sigmoid(
                    np.array([-offset])
                )[0])

            if applied:
                candidate_class_metrics = _metric_row(
                    y_eval, candidate_p, 0.5, subject_id
                )
            else:
                candidate_class_metrics = median_metrics

            zero_brier = _brier(y_eval, p_eval)
            personalized_brier = _brier(y_eval, candidate_p)
            zero_ll = _log_loss(y_eval, p_eval)
            personalized_ll = _log_loss(y_eval, candidate_p)

            results.append({
                "model": model,
                "outer_fold": fold,
                "subject_id": subject_id,
                "pm": pm,
                "calibration_low": n_low,
                "calibration_high": n_high,
                "calibration_extreme": n_low + n_high,
                "budget_fully_available": _as_bool(row.budget_fully_available),
                "logit_calibration_eligible": eligible,
                "logit_calibration_applied": applied,
                "logit_offset": float(offset),
                "equivalent_original_probability_threshold":
                    candidate_threshold_original_scale,
                "frozen_median_threshold": median_threshold,
                "evaluation_low": int(np.sum(y_eval == 0)),
                "evaluation_high": int(np.sum(y_eval == 1)),
                "evaluation_extreme": int(len(y_eval)),
                "evaluation_sample_hash": eval_hash,
                "zero_shot_brier": zero_brier,
                "personalized_brier": personalized_brier,
                "brier_improvement": zero_brier - personalized_brier,
                "zero_shot_log_loss": zero_ll,
                "personalized_log_loss": personalized_ll,
                "log_loss_improvement": zero_ll - personalized_ll,
                "zero_shot_ece10": _ece(y_eval, p_eval, 10),
                "personalized_ece10": _ece(y_eval, candidate_p, 10),
                "zero_shot_balanced_accuracy":
                    zero_metrics["balanced_accuracy"],
                "frozen_median_balanced_accuracy":
                    median_metrics["balanced_accuracy"],
                "candidate_balanced_accuracy":
                    candidate_class_metrics["balanced_accuracy"],
                "delta_balanced_accuracy_vs_median":
                    candidate_class_metrics["balanced_accuracy"]
                    - median_metrics["balanced_accuracy"],
                "delta_balanced_accuracy_vs_zero":
                    candidate_class_metrics["balanced_accuracy"]
                    - zero_metrics["balanced_accuracy"],
                "zero_shot_macro_f1": zero_metrics["macro_f1"],
                "frozen_median_macro_f1": median_metrics["macro_f1"],
                "candidate_macro_f1": candidate_class_metrics["macro_f1"],
                "delta_macro_f1_vs_median":
                    candidate_class_metrics["macro_f1"]
                    - median_metrics["macro_f1"],
                "delta_macro_f1_vs_zero":
                    candidate_class_metrics["macro_f1"]
                    - zero_metrics["macro_f1"],
            })

    results = pd.DataFrame(results)
    if len(results) != EXPECTED_RESULT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_RESULT_ROWS} results, got {len(results)}"
        )
    if results.duplicated(["model", "subject_id", "pm"]).any():
        raise RuntimeError("Duplicate model-participant-PM rows")

    applied_per_model = (
        results.groupby("model")["logit_calibration_applied"].sum().to_dict()
    )
    if any(int(v) != EXPECTED_ELIGIBLE for v in applied_per_model.values()):
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE} applied cells/model, got {applied_per_model}"
        )

    ineligible = results.loc[~results["logit_calibration_applied"].astype(bool)]
    if not np.allclose(ineligible["brier_improvement"], 0.0, atol=1e-15):
        raise RuntimeError("Probability fallback changed Brier score")
    if not np.allclose(ineligible["log_loss_improvement"], 0.0, atol=1e-15):
        raise RuntimeError("Probability fallback changed log loss")
    if not np.allclose(
        ineligible["delta_balanced_accuracy_vs_median"], 0.0, atol=1e-15
    ):
        raise RuntimeError("Classification fallback differs from frozen median")
    if not np.allclose(
        ineligible["delta_macro_f1_vs_median"], 0.0, atol=1e-15
    ):
        raise RuntimeError("Classification fallback differs from frozen median")

    context.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(context.output_dir / "participant_pm_results.csv", results)

    participant = _participant_aggregate(results, applied_only=False)
    applied_participant = _participant_aggregate(results, applied_only=True)
    _write_csv(
        context.output_dir / "participant_aggregate.csv", participant
    )
    _write_csv(
        context.output_dir / "participant_aggregate_applied_only.csv",
        applied_participant,
    )

    summary_rows = []
    for model, group in participant.groupby("model", sort=True):
        raw = results.loc[results["model"].eq(model)]
        summary_rows.append({
            "model": model,
            "participants": int(group["subject_id"].nunique()),
            "participant_pm_rows": int(len(raw)),
            "logit_eligible": int(raw["logit_calibration_eligible"].sum()),
            "logit_applied": int(raw["logit_calibration_applied"].sum()),
            "mean_brier_improvement": float(group["brier_improvement"].mean()),
            "mean_log_loss_improvement":
                float(group["log_loss_improvement"].mean()),
            "mean_delta_balanced_accuracy_vs_median":
                float(group["delta_balanced_accuracy_vs_median"].mean()),
            "mean_delta_macro_f1_vs_median":
                float(group["delta_macro_f1_vs_median"].mean()),
            "mean_delta_balanced_accuracy_vs_zero":
                float(group["delta_balanced_accuracy_vs_zero"].mean()),
            "mean_delta_macro_f1_vs_zero":
                float(group["delta_macro_f1_vs_zero"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    _write_csv(context.output_dir / "summary_operational.csv", summary)

    pm_rows = []
    for (model, pm), group in results.groupby(["model", "pm"], sort=True):
        pm_rows.append({
            "model": model,
            "pm": pm,
            "participant_rows": int(len(group)),
            "eligible": int(group["logit_calibration_eligible"].sum()),
            "applied": int(group["logit_calibration_applied"].sum()),
            "mean_brier_improvement": float(group["brier_improvement"].mean()),
            "mean_log_loss_improvement":
                float(group["log_loss_improvement"].mean()),
            "mean_delta_balanced_accuracy_vs_median":
                float(group["delta_balanced_accuracy_vs_median"].mean()),
            "mean_delta_macro_f1_vs_median":
                float(group["delta_macro_f1_vs_median"].mean()),
            "median_logit_offset": float(
                group.loc[
                    group["logit_calibration_applied"].astype(bool),
                    "logit_offset",
                ].median()
            ),
        })
    _write_csv(
        context.output_dir / "summary_by_pm.csv", pd.DataFrame(pm_rows)
    )

    bootstrap = _bootstrap(participant)
    _write_csv(context.output_dir / "bootstrap_operational.csv", bootstrap)

    applied_bootstrap = _bootstrap(applied_participant)
    applied_bootstrap["estimand"] = "applied_only"
    _write_csv(
        context.output_dir / "bootstrap_applied_only.csv",
        applied_bootstrap,
    )

    offset_summary = (
        results.loc[results["logit_calibration_applied"].astype(bool)]
        .groupby("model")
        .agg(
            applied=("logit_offset", "size"),
            median_offset=("logit_offset", "median"),
            q25_offset=("logit_offset", lambda x: x.quantile(0.25)),
            q75_offset=("logit_offset", lambda x: x.quantile(0.75)),
            median_equivalent_threshold=(
                "equivalent_original_probability_threshold", "median"
            ),
        )
        .reset_index()
    )
    _write_csv(context.output_dir / "offset_summary.csv", offset_summary)

    protocol = dict(context.protocol)
    protocol.update({
        "result_status": "confirmatory_complete",
        "probability_calibration_executed": True,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": True,
        "participant_pm_result_rows": int(len(results)),
        "eligible_participant_pm_per_model": EXPECTED_ELIGIBLE,
        "result_hash": _frame_hash(results),
    })
    _atomic_json(context.output_dir / "protocol.json", protocol)

    pooled = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "result_status": "confirmatory_complete",
        "models": list(MODELS),
        "calibration_budget_seconds": 1800,
        "minimum_calibration_each_class": MIN_EACH,
        "eval_ready_participant_pm": EXPECTED_EVAL_READY,
        "eligible_participant_pm_per_model": EXPECTED_ELIGIBLE,
        "participant_pm_result_rows": int(len(results)),
        "primary_metric": "brier_score",
        "classification_reference": "frozen_1800s_median_midpoint",
        "probability_calibration_executed": True,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": True,
    }
    _atomic_json(context.output_dir / "pooled_summary.json", pooled)
    return pooled


__all__ = [
    "ProbabilityCalibrationContext",
    "apply_logit_offset",
    "fit_logit_offset",
    "load_config",
    "prepare_protocol",
    "run_experiment",
    "write_dry_run",
]
