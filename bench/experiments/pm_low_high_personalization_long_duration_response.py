"""Confirmatory 15/20/30-minute LOW/HIGH personalization response.

Reuses stored XGBoost and LightGBM outer-test probabilities. No base-model
training or inference is performed. Budgets 900/1200/1800 s share one common
evaluation suffix strictly after +1800 s. Full-budget eligibility requires both
source duration and canonical feature-grid span to cover the budget, plus at
least 2 LOW and 2 HIGH calibration labels. Otherwise threshold 0.5 is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bench.experiments.pm_low_high_personalization_duration_response import (
    METRICS,
    MODELS,
    _atomic_json,
    _bootstrap_vs_zero_shot,
    _completed_protocol,
    _frame_hash,
    _git_head,
    _participant_first,
    _summary_from_participant,
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
    _median_midpoint,
    _metric_row,
    _prediction_source_matrix,
    _state_to_y,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    PM_NAMES,
    stable_hash,
)

SCHEMA_VERSION = "pm-low-high-personalization-long-duration-response-v1"
BUDGETS = (900, 1200, 1800)
BOOTSTRAP_METRICS = ("balanced_accuracy", "macro_f1")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42

LONG_FEASIBILITY_HASH = (
    "34e0aa3350f84198383cd0e6a1d213711983132dcc30aa14fcc9edaafbc1095f"
)
PRIOR_DURATION_RESPONSE_HASH = (
    "14f6fb28ebd748a1f897df0df6d8e7e5a03302733cddde8c33017524d6335035"
)
XGBOOST_HASH = (
    "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
)
LIGHTGBM_HASH = (
    "a2c51deb4d94e33d71863f1f1c7927470266b7352647664f3a411fb5db819b4e"
)
EXPECTED_EVAL_READY = 345
EXPECTED_RESULT_ROWS = EXPECTED_EVAL_READY * len(MODELS) * len(BUDGETS)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")

    refs = cfg.get("references", {})
    expected_refs = {
        "long_duration_feasibility": LONG_FEASIBILITY_HASH,
        "prior_duration_response": PRIOR_DURATION_RESPONSE_HASH,
        "xgboost": XGBOOST_HASH,
        "lightgbm": LIGHTGBM_HASH,
    }
    for key, expected in expected_refs.items():
        if refs.get(key, {}).get("protocol_hash") != expected:
            raise ValueError(f"{key} reference protocol changed")

    expected_contract = {
        "pm_names": list(PM_NAMES),
        "models": list(MODELS),
        "alignment": "EEG(t-10s) -> PM(t)",
        "lag_seconds": -10,
        "target_transform": "outer_train_q33_q67_extremes",
        "threshold_fit_scope": "outer_train_continuous_complete_cases",
        "middle_policy": "exclude",
        "calibration_budgets_seconds": list(BUDGETS),
        "calibration_record_policy":
            "earliest_logical_record_by_selected_record_start_utc",
        "calibration_cross_record_policy": "forbidden",
        "calibration_interval_rule":
            "0 < target_relative_seconds <= budget_seconds",
        "budget_fully_available_rule":
            "source_duration_and_feature_grid_span_cover_budget",
        "common_evaluation_boundary_seconds": 1800,
        "fixed_evaluation_policy":
            "absolute_target_utc > earliest_record_start_utc + 1800s",
        "cross_record_overlap_policy":
            "earlier_record_precedence_trim_later_overlapping_prefix_by_feature_grid_utc",
        "minimum_calibration_per_class": 2,
        "minimum_fixed_evaluation_extremes": 20,
        "require_both_evaluation_classes": True,
        "ineligible_policy": "zero_shot_fallback_no_budget_extension",
        "threshold_strategy": "median_midpoint",
        "nonseparated_medians_policy": "zero_shot_fallback",
        "probability_source": "stored_predict_proba_high",
        "base_decision_threshold": 0.5,
    }
    if cfg.get("scientific_contract") != expected_contract:
        raise ValueError("Scientific contract changed")

    expected_eval = {
        "primary_metric": "balanced_accuracy",
        "secondary_metrics": [
            "macro_f1", "low_recall", "high_recall", "precision", "accuracy",
        ],
        "aggregation": "mean_pm_within_participant_then_mean_participants",
        "primary_estimand":
            "operational_all_post1800_eval_ready_with_zero_shot_fallback",
        "secondary_estimand":
            "adaptation_applied_only_participant_first",
        "primary_duration_contrast": "1800s_minus_900s",
        "secondary_duration_contrast": "1200s_minus_900s",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "bootstrap_unit": "subject_id",
    }
    if cfg.get("evaluation") != expected_eval:
        raise ValueError("Evaluation contract changed")

    selection = cfg.get("selection_context", {})
    if selection != {
        "reason":
            "exploratory_posthoc_900s_minus_600s_positive_for_both_models_and_metrics",
        "no_budget_search_inside_experiment": True,
    }:
        raise ValueError("Selection context changed")

    forbidden = cfg.get("forbidden", {})
    if not forbidden or not all(value is True for value in forbidden.values()):
        raise ValueError("All forbidden switches must remain true")
    return cfg


@dataclass
class LongDurationResponseContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    long_feasibility: Any
    long_detail: pd.DataFrame
    source_matrix: pd.DataFrame
    source_audit: pd.DataFrame
    protocol: dict[str, Any]


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> LongDurationResponseContext:
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
        root, refs["prior_duration_response"]["output_dir"],
        PRIOR_DURATION_RESPONSE_HASH, allowed_statuses={"confirmatory_complete"},
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
    if feasibility.protocol["protocol_hash"] != LONG_FEASIBILITY_HASH:
        raise RuntimeError("Recomputed long feasibility hash changed")

    detail = pd.read_csv(
        root / refs["long_duration_feasibility"]["output_dir"]
        / "participant_pm_budget_feasibility.csv"
    )
    if len(detail) != 1134:
        raise RuntimeError(f"Expected 1134 feasibility rows, got {len(detail)}")

    required = {
        "subject_id", "outer_fold", "pm", "budget_seconds",
        "budget_fully_available", "calibration_low", "calibration_high",
        "fixed_evaluation_ready_min20_both_classes",
        "evaluation_extreme_sample_hash",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise RuntimeError(f"Feasibility detail missing columns: {missing}")

    source_cfg = {
        "references": {
            "xgboost": refs["xgboost"],
            "lightgbm": refs["lightgbm"],
        }
    }
    source_matrix = _prediction_source_matrix(root, source_cfg)
    source_audit = _audit_prediction_sources(root, source_matrix)
    if len(source_matrix) != 70 or len(source_audit) != 70:
        raise RuntimeError("Expected 70 stored prediction sources")
    if not source_audit["valid"].all():
        raise RuntimeError("Stored prediction source audit failed")

    scientific_payload = {
        "schema_version": SCHEMA_VERSION,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "evaluation": config["evaluation"],
        "selection_context": config["selection_context"],
        "forbidden": config["forbidden"],
        "long_feasibility_detail_hash": _frame_hash(detail),
        "prediction_source_matrix_hash": _frame_hash(
            source_matrix.drop(columns=["source_output_dir"], errors="ignore")
        ),
        "prediction_source_audit_hash": _frame_hash(
            source_audit.drop(
                columns=["prediction_file", "run_summary_file"], errors="ignore"
            )
        ),
        "fixed_fold_hash":
            feasibility.base.low_high.protocol["fixed_fold_hash"],
        "threshold_hashes":
            feasibility.base.low_high.protocol["threshold_hashes"],
    }
    protocol_hash = stable_hash(scientific_payload)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "preregistered_candidate",
        "threshold_calibration_executed": False,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": False,
        "git_commit": _git_head(root),
        "protocol_hash": protocol_hash,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "evaluation": config["evaluation"],
        "selection_context": config["selection_context"],
        "long_feasibility_detail_hash":
            scientific_payload["long_feasibility_detail_hash"],
        "prediction_source_rows": int(len(source_matrix)),
        "prediction_source_audit_mismatches":
            int((~source_audit["valid"]).sum()),
        "fixed_fold_hash":
            feasibility.base.low_high.protocol["fixed_fold_hash"],
        "threshold_hashes":
            feasibility.base.low_high.protocol["threshold_hashes"],
        "expected_fixed_evaluation_ready_participant_pm": EXPECTED_EVAL_READY,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
    }
    return LongDurationResponseContext(
        root=root,
        output_dir=output,
        config=dict(config),
        long_feasibility=feasibility,
        long_detail=detail,
        source_matrix=source_matrix,
        source_audit=source_audit,
        protocol=protocol,
    )


def write_dry_run(context: LongDurationResponseContext) -> dict[str, Any]:
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
        "budgets_seconds": list(BUDGETS),
        "threshold_strategy": "median_midpoint",
        "budget_fully_available_rule":
            "source_duration_and_feature_grid_span_cover_budget",
        "common_evaluation_boundary_seconds": 1800,
        "long_feasibility_rows": int(len(context.long_detail)),
        "prediction_source_rows": int(len(context.source_matrix)),
        "prediction_source_audit_mismatches": 0,
        "expected_fixed_evaluation_ready_participant_pm": EXPECTED_EVAL_READY,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "threshold_calibration_executed": False,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": False,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    (context.output_dir / "README.md").write_text(
        f"""# PM LOW/HIGH long-duration personalization response v1

- models: XGBoost + LightGBM
- no base-model training or inference
- stored outer-test probabilities only
- budgets: 900 / 1200 / 1800 seconds
- common evaluation: strictly after +1800 seconds
- full budget: source duration AND canonical feature-grid span
- calibration: >=2 LOW and >=2 HIGH
- threshold: median_midpoint only
- fallback: threshold 0.5
- expected evaluation-ready participant-PM: {EXPECTED_EVAL_READY}
- expected result rows: {EXPECTED_RESULT_ROWS}
- primary contrast: 1800 - 900 s
- secondary contrast: 1200 - 900 s
- participant-first aggregation
- subject-clustered bootstrap, 10,000 replicates

Protocol hash: `{context.protocol['protocol_hash']}`
""",
        encoding="utf-8",
    )
    return summary


def _ready_keys(detail: pd.DataFrame) -> list[tuple[str, str]]:
    frame = detail.copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    baseline = frame.loc[frame["budget_seconds"].astype(int).eq(900)]
    ready = baseline.loc[
        baseline["fixed_evaluation_ready_min20_both_classes"]
        .astype(str).str.lower().eq("true")
    ]
    keys = list(
        ready[["subject_id", "pm"]].itertuples(index=False, name=None)
    )
    if len(keys) != EXPECTED_EVAL_READY:
        raise RuntimeError(
            f"Expected {EXPECTED_EVAL_READY} ready cells, got {len(keys)}"
        )
    return [(str(s), str(pm)) for s, pm in keys]


def _duration_contrasts(
    participant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    bootstrap_rows = []

    for model in MODELS:
        mf = participant.loc[participant["model"].eq(model)].copy()
        reference = (
            mf.loc[mf["budget_seconds"].astype(int).eq(900)]
            .set_index("subject_id").sort_index()
        )
        for target_budget, label in (
            (1200, "1200s_minus_900s"),
            (1800, "1800s_minus_900s"),
        ):
            target = (
                mf.loc[mf["budget_seconds"].astype(int).eq(target_budget)]
                .set_index("subject_id").sort_index()
            )
            if not reference.index.equals(target.index):
                raise RuntimeError(f"{model}/{label}: participant sets differ")

            rows = []
            for subject in reference.index.astype(str):
                row = {
                    "model": model,
                    "contrast": label,
                    "target_budget_seconds": target_budget,
                    "reference_budget_seconds": 900,
                    "subject_id": subject,
                }
                for metric in METRICS:
                    row[f"delta_{metric}"] = (
                        float(target.loc[subject, f"personalized_{metric}"])
                        - float(reference.loc[subject, f"personalized_{metric}"])
                    )
                rows.append(row)
                detail_rows.append(row)

            contrast = pd.DataFrame(rows)
            for mi, metric in enumerate(BOOTSTRAP_METRICS):
                values = contrast[f"delta_{metric}"].to_numpy(dtype=float)
                n = len(values)
                seed = (
                    BOOTSTRAP_SEED + target_budget
                    + (100 if model == "lightgbm" else 0)
                    + mi * 10000
                )
                rng = np.random.default_rng(seed)
                boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
                for i in range(BOOTSTRAP_REPLICATES):
                    idx = rng.integers(0, n, size=n)
                    boot[i] = float(np.mean(values[idx]))
                bootstrap_rows.append({
                    "model": model,
                    "contrast": label,
                    "target_budget_seconds": target_budget,
                    "reference_budget_seconds": 900,
                    "metric": metric,
                    "observed_mean_delta": float(np.mean(values)),
                    "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": seed,
                    "resampling_unit": "subject_id",
                    "n_subjects": n,
                })

    return pd.DataFrame(detail_rows), pd.DataFrame(bootstrap_rows)


def run_experiment(context: LongDurationResponseContext) -> dict[str, Any]:
    predictions = _load_prediction_lookup(context)
    detail = context.long_detail.copy()
    detail["subject_id"] = detail["subject_id"].astype(str)
    ready_keys = _ready_keys(detail)
    base = context.long_feasibility.base
    results = []

    for subject_id, pm in ready_keys:
        f900 = detail.loc[
            detail["subject_id"].eq(subject_id)
            & detail["pm"].astype(str).eq(pm)
            & detail["budget_seconds"].astype(int).eq(900)
        ]
        if len(f900) != 1:
            raise RuntimeError("Missing 900-second feasibility row")
        fold = int(f900.iloc[0]["outer_fold"])

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

        evaluation = timeline.loc[
            timeline["state"].isin(["low", "high"])
            & (
                timeline["absolute_target_epoch_seconds"].to_numpy(dtype=float)
                > start + 1800.0
            )
        ].copy()
        y_eval = _state_to_y(evaluation["state"])
        if len(evaluation) < 20 or set(np.unique(y_eval)) != {0, 1}:
            raise RuntimeError("Post-1800 evaluation readiness mismatch")
        eval_ids = evaluation["target_sample_id"].astype(str).tolist()
        eval_hash = stable_hash(eval_ids)

        expected_hashes = detail.loc[
            detail["subject_id"].eq(subject_id)
            & detail["pm"].astype(str).eq(pm),
            "evaluation_extreme_sample_hash",
        ].astype(str).unique()
        if len(expected_hashes) != 1 or expected_hashes[0] != eval_hash:
            raise RuntimeError("Evaluation hash differs from feasibility")

        for model in MODELS:
            source = predictions[(model, fold, pm)]
            if set(eval_ids) - set(source.index.astype(str)):
                raise RuntimeError("Stored evaluation predictions missing")
            p_eval = source.loc[eval_ids, "probability_high"].to_numpy(float)
            y_source = source.loc[eval_ids, "y_true"].to_numpy(int)
            if not np.array_equal(y_source, y_eval):
                raise RuntimeError("Stored evaluation labels changed")
            zero = _metric_row(y_eval, p_eval, 0.5, subject_id)

            for budget in BUDGETS:
                fr = detail.loc[
                    detail["subject_id"].eq(subject_id)
                    & detail["pm"].astype(str).eq(pm)
                    & detail["budget_seconds"].astype(int).eq(budget)
                ]
                if len(fr) != 1:
                    raise RuntimeError("Missing feasibility budget row")
                fr = fr.iloc[0]

                full = str(fr["budget_fully_available"]).lower() == "true"
                n_low = int(fr["calibration_low"])
                n_high = int(fr["calibration_high"])
                eligible = bool(full and n_low >= 2 and n_high >= 2)

                relative = timeline["target_relative_seconds"].to_numpy(float)
                calibration = timeline.loc[
                    timeline["state"].isin(["low", "high"])
                    & timeline["record_group_id"].astype(str).eq(cal_group)
                    & (relative > 0.0)
                    & (relative <= float(budget))
                ].copy()

                if int(calibration["state"].eq("low").sum()) != n_low:
                    raise RuntimeError("Calibration LOW mismatch")
                if int(calibration["state"].eq("high").sum()) != n_high:
                    raise RuntimeError("Calibration HIGH mismatch")

                threshold = 0.5
                applied = False
                reason = "ineligible_zero_shot_fallback"
                med_low = np.nan
                med_high = np.nan

                if eligible:
                    cal_ids = calibration["target_sample_id"].astype(str).tolist()
                    if set(cal_ids) - set(source.index.astype(str)):
                        raise RuntimeError("Stored calibration predictions missing")
                    p_cal = source.loc[cal_ids, "probability_high"].to_numpy(float)
                    y_cal = _state_to_y(calibration["state"])
                    threshold, applied, reason, extra = _median_midpoint(
                        p_cal, y_cal
                    )
                    med_low = float(extra["median_probability_low"])
                    med_high = float(extra["median_probability_high"])

                personal = _metric_row(
                    y_eval, p_eval, float(threshold), subject_id
                )
                row = {
                    "model": model,
                    "outer_fold": fold,
                    "pm": pm,
                    "subject_id": subject_id,
                    "budget_seconds": budget,
                    "budget_minutes": budget / 60.0,
                    "budget_fully_available": full,
                    "calibration_low": n_low,
                    "calibration_high": n_high,
                    "calibration_extreme": n_low + n_high,
                    "calibration_class_eligible": eligible,
                    "adaptation_applied": bool(applied),
                    "adaptation_reason": str(reason),
                    "personalized_threshold": float(threshold),
                    "threshold_shift_from_0_5": float(threshold - 0.5),
                    "median_probability_low": med_low,
                    "median_probability_high": med_high,
                    "evaluation_low": int(np.sum(y_eval == 0)),
                    "evaluation_high": int(np.sum(y_eval == 1)),
                    "evaluation_extreme": int(len(y_eval)),
                    "evaluation_sample_hash": eval_hash,
                }
                for metric in METRICS:
                    row[f"zero_shot_{metric}"] = zero[metric]
                    row[f"personalized_{metric}"] = personal[metric]
                    row[f"delta_{metric}"] = personal[metric] - zero[metric]
                results.append(row)

    results = pd.DataFrame(results)
    if len(results) != EXPECTED_RESULT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_RESULT_ROWS} rows, got {len(results)}"
        )
    if results.duplicated(
        ["model", "subject_id", "pm", "budget_seconds"]
    ).any():
        raise RuntimeError("Duplicate result rows")

    for _, g in results.groupby(["subject_id", "pm"], sort=False):
        assert g["evaluation_sample_hash"].nunique() == 1
        assert g["evaluation_low"].nunique() == 1
        assert g["evaluation_high"].nunique() == 1

    for _, g in results.groupby(
        ["model", "subject_id", "pm"], sort=False
    ):
        ordered = g.sort_values("budget_seconds")
        assert np.all(np.diff(ordered["calibration_low"].to_numpy(int)) >= 0)
        assert np.all(np.diff(ordered["calibration_high"].to_numpy(int)) >= 0)
        for metric in METRICS:
            z = ordered[f"zero_shot_{metric}"].to_numpy(float)
            assert np.allclose(z, z[0], rtol=0.0, atol=1e-15)

    fallback = results.loc[~results["adaptation_applied"].astype(bool)]
    assert np.allclose(
        fallback["personalized_threshold"], 0.5, rtol=0.0, atol=1e-15
    )
    for metric in METRICS:
        assert np.allclose(
            fallback[f"delta_{metric}"], 0.0, rtol=0.0, atol=1e-15
        )

    context.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(context.output_dir / "participant_pm_results.csv", results)

    participant = _participant_first(results, applied_only=False)
    _write_csv(context.output_dir / "participant_aggregate.csv", participant)
    operational = _summary_from_participant(
        participant, results, applied_only=False
    )
    _write_csv(context.output_dir / "summary_operational.csv", operational)

    applied_participant = _participant_first(results, applied_only=True)
    _write_csv(
        context.output_dir / "participant_aggregate_applied_only.csv",
        applied_participant,
    )
    applied_summary = _summary_from_participant(
        applied_participant, results, applied_only=True
    )
    _write_csv(
        context.output_dir / "summary_adaptation_applied_only.csv",
        applied_summary,
    )

    pm_rows = []
    for (model, budget, pm), g in results.groupby(
        ["model", "budget_seconds", "pm"], sort=True
    ):
        row = {
            "model": model,
            "budget_seconds": budget,
            "pm": pm,
            "participant_rows": len(g),
            "budget_fully_available": int(
                g["budget_fully_available"].astype(bool).sum()
            ),
            "class_eligible": int(
                g["calibration_class_eligible"].astype(bool).sum()
            ),
            "adaptation_applied": int(
                g["adaptation_applied"].astype(bool).sum()
            ),
        }
        for metric in METRICS:
            row[f"delta_{metric}_mean"] = float(
                g[f"delta_{metric}"].mean()
            )
        pm_rows.append(row)
    _write_csv(
        context.output_dir / "summary_by_pm.csv", pd.DataFrame(pm_rows)
    )

    _write_csv(
        context.output_dir / "bootstrap_vs_zero_shot.csv",
        _bootstrap_vs_zero_shot(participant),
    )
    contrast_detail, contrast_bootstrap = _duration_contrasts(participant)
    _write_csv(
        context.output_dir / "participant_duration_contrasts.csv",
        contrast_detail,
    )
    _write_csv(
        context.output_dir / "bootstrap_duration_contrasts.csv",
        contrast_bootstrap,
    )

    threshold_rows = []
    for (model, budget), g in results.groupby(
        ["model", "budget_seconds"], sort=True
    ):
        adapted = g.loc[g["adaptation_applied"].astype(bool)]
        threshold_rows.append({
            "model": model,
            "budget_seconds": budget,
            "participant_pm_rows": len(g),
            "budget_fully_available": int(
                g["budget_fully_available"].astype(bool).sum()
            ),
            "class_eligible": int(
                g["calibration_class_eligible"].astype(bool).sum()
            ),
            "adaptation_applied": int(
                g["adaptation_applied"].astype(bool).sum()
            ),
            "median_threshold_adapted": (
                float(adapted["personalized_threshold"].median())
                if len(adapted) else np.nan
            ),
            "median_abs_shift_adapted": (
                float(adapted["threshold_shift_from_0_5"].abs().median())
                if len(adapted) else np.nan
            ),
        })
    _write_csv(
        context.output_dir / "threshold_summary.csv",
        pd.DataFrame(threshold_rows),
    )

    protocol = dict(context.protocol)
    protocol.update({
        "result_status": "confirmatory_complete",
        "threshold_calibration_executed": True,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": True,
        "fixed_evaluation_ready_participant_pm": EXPECTED_EVAL_READY,
        "participant_pm_result_rows": len(results),
        "result_hash": _frame_hash(results),
    })
    _atomic_json(context.output_dir / "protocol.json", protocol)

    pooled = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "result_status": "confirmatory_complete",
        "threshold_calibration_executed": True,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": True,
        "models": list(MODELS),
        "budgets_seconds": list(BUDGETS),
        "common_evaluation_boundary_seconds": 1800,
        "fixed_evaluation_ready_participant_pm": EXPECTED_EVAL_READY,
        "participant_pm_result_rows": len(results),
        "primary_duration_contrast": "1800s_minus_900s",
        "secondary_duration_contrast": "1200s_minus_900s",
    }
    _atomic_json(context.output_dir / "pooled_summary.json", pooled)
    return pooled


__all__ = [
    "BUDGETS",
    "LongDurationResponseContext",
    "_participant_first",
    "_ready_keys",
    "load_config",
    "prepare_protocol",
    "run_experiment",
    "write_dry_run",
]
