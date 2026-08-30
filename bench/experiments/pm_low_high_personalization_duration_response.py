"""Confirmatory 5/10/15-minute personalization duration response.

This experiment reuses completed XGBoost and LightGBM outer-test probabilities.
No base model is trained and no model inference is repeated.

All calibration budgets (300, 600, 900 s) are evaluated on the exact same
participant-PM suffix strictly after +900 s. Threshold adaptation uses only the
participant's labeled LOW/HIGH calibration prefix in the earliest logical
record. The only threshold strategy is the previously supported robust
median-midpoint rule. Ineligible or non-separated calibration falls back to
the zero-shot threshold 0.5 without extending the budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bench.experiments.pm_low_high_personalization_duration_feasibility import (
    load_config as load_duration_feasibility_config,
    prepare_protocol as prepare_duration_feasibility_protocol,
)
from bench.experiments.pm_low_high_personalized_threshold import (
    _audit_prediction_sources,
    _load_prediction_lookup,
    _median_midpoint,
    _metric_row,
    _prediction_source_matrix,
    _state_to_y,
)
from bench.experiments.pm_low_high_personalization_feasibility import (
    _subject_pm_timeline,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    PM_NAMES,
    stable_hash,
)

SCHEMA_VERSION = "pm-low-high-personalization-duration-response-v1"
MODELS = ("xgboost", "lightgbm")
BUDGETS = (300, 600, 900)
METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "low_recall",
    "high_recall",
    "precision",
    "accuracy",
)
BOOTSTRAP_METRICS = ("balanced_accuracy", "macro_f1")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42

DURATION_FEASIBILITY_HASH = (
    "6bd91b39eef1869125e3f2c57125cfbef017f3b0c0a5538b135b4e32563075bb"
)
PERSONALIZED_5MIN_HASH = (
    "578c359c6c56115aff8ccea29af18bf755989641464dc6672e22753349018af0"
)
XGBOOST_HASH = (
    "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
)
LIGHTGBM_HASH = (
    "a2c51deb4d94e33d71863f1f1c7927470266b7352647664f3a411fb5db819b4e"
)


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def _frame_hash(frame: pd.DataFrame) -> str:
    return stable_hash(
        frame.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    )


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")

    refs = cfg.get("references", {})
    expected_refs = {
        "duration_feasibility": DURATION_FEASIBILITY_HASH,
        "personalized_threshold_5min": PERSONALIZED_5MIN_HASH,
        "xgboost": XGBOOST_HASH,
        "lightgbm": LIGHTGBM_HASH,
    }
    for key, phash in expected_refs.items():
        if refs.get(key, {}).get("protocol_hash") != phash:
            raise ValueError(f"{key} reference protocol changed")

    c = cfg.get("scientific_contract", {})
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
        "common_evaluation_boundary_seconds": 900,
        "fixed_evaluation_policy":
            "absolute_target_utc > earliest_record_start_utc + 900s",
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
    if c != expected_contract:
        raise ValueError("Scientific contract changed")

    ev = cfg.get("evaluation", {})
    if ev != {
        "primary_metric": "balanced_accuracy",
        "secondary_metrics": [
            "macro_f1", "low_recall", "high_recall",
            "precision", "accuracy",
        ],
        "unchanged_ranking_metrics": ["roc_auc", "pr_auc"],
        "aggregation": "mean_pm_within_participant_then_mean_participants",
        "primary_estimand":
            "operational_all_post900_eval_ready_with_zero_shot_fallback",
        "secondary_estimand":
            "adaptation_applied_only_participant_first",
        "primary_duration_contrast": "900s_minus_300s",
        "secondary_duration_contrast": "600s_minus_300s",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "bootstrap_unit": "subject_id",
    }:
        raise ValueError("Evaluation contract changed")

    forbidden = cfg.get("forbidden", {})
    if not forbidden or not all(value is True for value in forbidden.values()):
        raise ValueError("All forbidden switches must remain true")
    return cfg


def _completed_protocol(
    root: Path,
    output_dir: str,
    expected_hash: str,
    *,
    allowed_statuses: set[str],
) -> dict[str, Any]:
    path = root / output_dir / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_hash") != expected_hash:
        raise RuntimeError(f"{path}: protocol hash changed")
    if value.get("result_status") not in allowed_statuses:
        raise RuntimeError(f"{path}: unexpected result status")
    return value


@dataclass
class DurationResponseContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    duration_feasibility: Any
    duration_detail: pd.DataFrame
    source_matrix: pd.DataFrame
    source_audit: pd.DataFrame
    protocol: dict[str, Any]


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> DurationResponseContext:
    root = Path(root).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root / output

    refs = config["references"]
    _completed_protocol(
        root,
        refs["duration_feasibility"]["output_dir"],
        DURATION_FEASIBILITY_HASH,
        allowed_statuses={"feasibility_audit_complete"},
    )
    _completed_protocol(
        root,
        refs["personalized_threshold_5min"]["output_dir"],
        PERSONALIZED_5MIN_HASH,
        allowed_statuses={"confirmatory_complete"},
    )
    _completed_protocol(
        root,
        refs["xgboost"]["output_dir"],
        XGBOOST_HASH,
        allowed_statuses={"confirmatory_complete"},
    )
    _completed_protocol(
        root,
        refs["lightgbm"]["output_dir"],
        LIGHTGBM_HASH,
        allowed_statuses={"confirmatory_complete"},
    )

    fcfg = load_duration_feasibility_config(
        root / refs["duration_feasibility"]["config"]
    )
    duration = prepare_duration_feasibility_protocol(
        fcfg,
        root=root,
        feature_cache_dir=feature_cache_dir,
        output_dir=root / refs["duration_feasibility"]["output_dir"],
    )
    if duration.protocol["protocol_hash"] != DURATION_FEASIBILITY_HASH:
        raise RuntimeError("Recomputed duration feasibility hash changed")

    detail_path = (
        root
        / refs["duration_feasibility"]["output_dir"]
        / "participant_pm_budget_feasibility.csv"
    )
    detail = pd.read_csv(detail_path)
    if len(detail) != 1134:
        raise RuntimeError("Duration feasibility detail must contain 1134 rows")

    # Reuse the already audited source-prediction discovery implementation,
    # but build it from this experiment's refs.
    source_cfg = {
        "references": {
            "xgboost": refs["xgboost"],
            "lightgbm": refs["lightgbm"],
        }
    }
    source_matrix = _prediction_source_matrix(root, source_cfg)
    source_audit = _audit_prediction_sources(root, source_matrix)
    if len(source_audit) != 70 or not source_audit["valid"].all():
        raise RuntimeError("Stored prediction source audit failed")

    scientific_payload = {
        "schema_version": SCHEMA_VERSION,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "evaluation": config["evaluation"],
        "forbidden": config["forbidden"],
        "duration_feasibility_detail_hash": _frame_hash(detail),
        "prediction_source_matrix_hash": _frame_hash(
            source_matrix.drop(columns=["source_output_dir"])
        ),
        "prediction_source_audit_hash": _frame_hash(
            source_audit.drop(
                columns=["prediction_file", "run_summary_file"]
            )
        ),
        "fixed_fold_hash":
            duration.base.low_high.protocol["fixed_fold_hash"],
        "threshold_hashes":
            duration.base.low_high.protocol["threshold_hashes"],
    }
    phash = stable_hash(scientific_payload)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "preregistered_candidate",
        "threshold_calibration_executed": False,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": False,
        "git_commit": _git_head(root),
        "protocol_hash": phash,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "evaluation": config["evaluation"],
        "duration_feasibility_detail_hash":
            scientific_payload["duration_feasibility_detail_hash"],
        "prediction_source_rows": int(len(source_matrix)),
        "prediction_source_audit_mismatches":
            int((~source_audit["valid"]).sum()),
        "fixed_fold_hash":
            duration.base.low_high.protocol["fixed_fold_hash"],
        "threshold_hashes":
            duration.base.low_high.protocol["threshold_hashes"],
        "expected_fixed_evaluation_ready_participant_pm": 364,
        "expected_result_rows": 364 * len(MODELS) * len(BUDGETS),
    }
    return DurationResponseContext(
        root=root,
        output_dir=output,
        config=dict(config),
        duration_feasibility=duration,
        duration_detail=detail,
        source_matrix=source_matrix,
        source_audit=source_audit,
        protocol=protocol,
    )


def write_dry_run(context: DurationResponseContext) -> dict[str, Any]:
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
        "common_evaluation_boundary_seconds": 900,
        "duration_feasibility_rows": int(len(context.duration_detail)),
        "prediction_source_rows": int(len(context.source_matrix)),
        "prediction_source_audit_mismatches": 0,
        "expected_fixed_evaluation_ready_participant_pm": 364,
        "expected_result_rows": 364 * len(MODELS) * len(BUDGETS),
        "threshold_calibration_executed": False,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": False,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    readme = f"""# PM LOW/HIGH personalization duration response v1

Confirmatory duration-response experiment.

- models: XGBoost + LightGBM
- base-model training: none
- base-model inference: none
- stored outer-test HIGH probabilities are reused
- calibration budgets: 300 / 600 / 900 seconds
- threshold strategy: median_midpoint only
- calibration eligibility: full budget and >=2 LOW + >=2 HIGH
- ineligible/non-separated calibration: zero-shot threshold 0.5 fallback
- common evaluation for every budget: strictly after +900 seconds
- fixed evaluation readiness: >=20 extremes and both classes
- expected evaluation-ready participant-PM cells: 364
- expected result rows: {364 * len(MODELS) * len(BUDGETS)}
- primary contrast: 900 s minus 300 s
- secondary contrast: 600 s minus 300 s
- aggregation: PM within participant, then participants
- bootstrap: subject-clustered, 10,000 replicates

Protocol hash:
`{context.protocol['protocol_hash']}`
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def _duration_ready_keys(detail: pd.DataFrame) -> list[tuple[str, str]]:
    d = detail.copy()
    d["subject_id"] = d["subject_id"].astype(str)
    baseline = d.loc[d["budget_seconds"].astype(int).eq(300)].copy()
    ready = baseline.loc[
        baseline["fixed_evaluation_ready_min20_both_classes"]
        .astype(str).str.lower().eq("true")
    ]
    keys = list(
        ready[["subject_id", "pm"]]
        .itertuples(index=False, name=None)
    )
    if len(keys) != 364:
        raise RuntimeError(
            f"Expected 364 post-900 evaluation-ready cells, got {len(keys)}"
        )
    return [(str(s), str(pm)) for s, pm in keys]


def _participant_first(
    results: pd.DataFrame,
    *,
    applied_only: bool,
) -> pd.DataFrame:
    frame = (
        results.loc[results["adaptation_applied"]].copy()
        if applied_only
        else results.copy()
    )
    rows = []
    for keys, group in frame.groupby(
        ["model", "budget_seconds", "subject_id"],
        sort=True,
    ):
        model, budget, subject = keys
        row = {
            "model": model,
            "budget_seconds": int(budget),
            "subject_id": str(subject),
            "n_pm": int(group["pm"].nunique()),
        }
        for metric in METRICS:
            row[f"zero_shot_{metric}"] = float(
                group[f"zero_shot_{metric}"].mean()
            )
            row[f"personalized_{metric}"] = float(
                group[f"personalized_{metric}"].mean()
            )
            row[f"delta_{metric}"] = float(
                group[f"delta_{metric}"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_from_participant(
    participant: pd.DataFrame,
    results: pd.DataFrame,
    *,
    applied_only: bool,
) -> pd.DataFrame:
    rows = []
    for keys, group in participant.groupby(
        ["model", "budget_seconds"], sort=True
    ):
        model, budget = keys
        raw = results.loc[
            results["model"].eq(model)
            & results["budget_seconds"].astype(int).eq(int(budget))
        ]
        if applied_only:
            raw = raw.loc[raw["adaptation_applied"]]
        row = {
            "model": model,
            "budget_seconds": int(budget),
            "participants": int(group["subject_id"].nunique()),
            "participant_pm_rows": int(len(raw)),
            "mean_pm_per_participant": float(group["n_pm"].mean()),
            "min_pm_per_participant": int(group["n_pm"].min()),
            "max_pm_per_participant": int(group["n_pm"].max()),
        }
        if not applied_only:
            row["class_eligible_participant_pm"] = int(
                raw["calibration_class_eligible"].sum()
            )
            row["adaptation_applied_participant_pm"] = int(
                raw["adaptation_applied"].sum()
            )
        for metric in METRICS:
            row[f"zero_shot_{metric}_mean"] = float(
                group[f"zero_shot_{metric}"].mean()
            )
            row[f"personalized_{metric}_mean"] = float(
                group[f"personalized_{metric}"].mean()
            )
            row[f"delta_{metric}_mean"] = float(
                group[f"delta_{metric}"].mean()
            )
            row[f"delta_{metric}_median"] = float(
                group[f"delta_{metric}"].median()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_vs_zero_shot(
    participant: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for keys, group in participant.groupby(
        ["model", "budget_seconds"], sort=True
    ):
        model, budget = keys
        for metric in BOOTSTRAP_METRICS:
            values = group[f"delta_{metric}"].to_numpy(dtype=float)
            observed = float(np.mean(values))
            n = len(values)
            samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for i in range(BOOTSTRAP_REPLICATES):
                idx = rng.integers(0, n, size=n)
                samples[i] = float(np.mean(values[idx]))
            rows.append({
                "model": model,
                "budget_seconds": int(budget),
                "contrast": "personalized_minus_zero_shot",
                "metric": metric,
                "observed_mean_delta": observed,
                "bootstrap_ci_low":
                    float(np.quantile(samples, 0.025)),
                "bootstrap_ci_high":
                    float(np.quantile(samples, 0.975)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "resampling_unit": "subject_id",
                "n_subjects": int(n),
            })
    return pd.DataFrame(rows)


def _duration_contrasts(
    participant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    bootstrap_rows = []

    for model in MODELS:
        model_frame = participant.loc[
            participant["model"].eq(model)
        ].copy()

        for target_budget, label in (
            (600, "600s_minus_300s"),
            (900, "900s_minus_300s"),
        ):
            base = model_frame.loc[
                model_frame["budget_seconds"].astype(int).eq(300)
            ].set_index("subject_id")
            target = model_frame.loc[
                model_frame["budget_seconds"].astype(int).eq(target_budget)
            ].set_index("subject_id")
            if set(base.index) != set(target.index):
                raise RuntimeError(
                    f"{model}/{label}: participant sets differ"
                )
            subjects = sorted(base.index.astype(str))
            for subject in subjects:
                row = {
                    "model": model,
                    "contrast": label,
                    "target_budget_seconds": int(target_budget),
                    "reference_budget_seconds": 300,
                    "subject_id": subject,
                }
                for metric in METRICS:
                    row[f"delta_{metric}"] = (
                        float(target.loc[subject, f"personalized_{metric}"])
                        - float(base.loc[subject, f"personalized_{metric}"])
                    )
                detail_rows.append(row)

            detail = pd.DataFrame(
                [
                    row for row in detail_rows
                    if row["model"] == model and row["contrast"] == label
                ]
            )
            rng = np.random.default_rng(
                BOOTSTRAP_SEED
                + (0 if model == "xgboost" else 100)
                + int(target_budget)
            )
            for metric in BOOTSTRAP_METRICS:
                values = detail[f"delta_{metric}"].to_numpy(dtype=float)
                n = len(values)
                samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
                for i in range(BOOTSTRAP_REPLICATES):
                    idx = rng.integers(0, n, size=n)
                    samples[i] = float(np.mean(values[idx]))
                bootstrap_rows.append({
                    "model": model,
                    "contrast": label,
                    "target_budget_seconds": int(target_budget),
                    "reference_budget_seconds": 300,
                    "metric": metric,
                    "observed_mean_delta": float(np.mean(values)),
                    "bootstrap_ci_low":
                        float(np.quantile(samples, 0.025)),
                    "bootstrap_ci_high":
                        float(np.quantile(samples, 0.975)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed":
                        BOOTSTRAP_SEED
                        + (0 if model == "xgboost" else 100)
                        + int(target_budget),
                    "resampling_unit": "subject_id",
                    "n_subjects": int(n),
                })

    return pd.DataFrame(detail_rows), pd.DataFrame(bootstrap_rows)


def run_experiment(context: DurationResponseContext) -> dict[str, Any]:
    predictions = _load_prediction_lookup(context)
    detail = context.duration_detail.copy()
    detail["subject_id"] = detail["subject_id"].astype(str)
    ready_keys = _duration_ready_keys(detail)

    base = context.duration_feasibility.base
    results = []

    for subject_id, pm in ready_keys:
        f300 = detail.loc[
            detail["subject_id"].eq(subject_id)
            & detail["pm"].astype(str).eq(pm)
            & detail["budget_seconds"].astype(int).eq(300)
        ]
        if len(f300) != 1:
            raise RuntimeError("Missing duration feasibility reference row")
        fold = int(f300.iloc[0]["outer_fold"])

        timeline = _subject_pm_timeline(
            base,
            subject_id=subject_id,
            pm=pm,
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
                > start + 900.0
            )
        ].copy()
        y_eval = _state_to_y(evaluation["state"])
        if len(evaluation) < 20 or set(np.unique(y_eval)) != {0, 1}:
            raise RuntimeError("Runtime post-900 evaluation readiness mismatch")
        eval_hash = stable_hash(
            [str(value) for value in evaluation["target_sample_id"].tolist()]
        )

        for model in MODELS:
            source = predictions[(model, fold, pm)]
            missing_eval = sorted(
                set(evaluation["target_sample_id"].astype(str))
                - set(source.index.astype(str))
            )
            if missing_eval:
                raise RuntimeError(
                    f"{model}/{fold}/{pm}/{subject_id}: "
                    "missing evaluation predictions"
                )
            p_eval = source.loc[
                evaluation["target_sample_id"].astype(str),
                "probability_high",
            ].to_numpy(dtype=float)
            y_source = source.loc[
                evaluation["target_sample_id"].astype(str),
                "y_true",
            ].to_numpy(dtype=int)
            if not np.array_equal(y_source, y_eval):
                raise RuntimeError("Stored evaluation labels changed")
            zero = _metric_row(y_eval, p_eval, 0.5, subject_id)

            for budget in BUDGETS:
                frow = detail.loc[
                    detail["subject_id"].eq(subject_id)
                    & detail["pm"].astype(str).eq(pm)
                    & detail["budget_seconds"].astype(int).eq(budget)
                ]
                if len(frow) != 1:
                    raise RuntimeError("Missing participant-PM-budget feasibility row")
                frow = frow.iloc[0]

                fully_available = (
                    str(frow["budget_fully_available"]).lower() == "true"
                )
                n_low = int(frow["calibration_low"])
                n_high = int(frow["calibration_high"])
                eligible = fully_available and n_low >= 2 and n_high >= 2

                calibration = timeline.loc[
                    timeline["state"].isin(["low", "high"])
                    & timeline["record_group_id"].astype(str).eq(cal_group)
                    & (
                        timeline["target_relative_seconds"].to_numpy(dtype=float)
                        > 0.0
                    )
                    & (
                        timeline["target_relative_seconds"].to_numpy(dtype=float)
                        <= float(budget)
                    )
                ].copy()
                if int(calibration["state"].eq("low").sum()) != n_low:
                    raise RuntimeError("Calibration LOW count mismatch")
                if int(calibration["state"].eq("high").sum()) != n_high:
                    raise RuntimeError("Calibration HIGH count mismatch")

                threshold = 0.5
                applied = False
                reason = "ineligible_zero_shot_fallback"
                median_low = np.nan
                median_high = np.nan

                if eligible:
                    missing_cal = sorted(
                        set(calibration["target_sample_id"].astype(str))
                        - set(source.index.astype(str))
                    )
                    if missing_cal:
                        raise RuntimeError(
                            f"{model}/{fold}/{pm}/{subject_id}: "
                            "missing calibration predictions"
                        )
                    p_cal = source.loc[
                        calibration["target_sample_id"].astype(str),
                        "probability_high",
                    ].to_numpy(dtype=float)
                    y_cal = _state_to_y(calibration["state"])
                    threshold, applied, reason, extra = _median_midpoint(
                        p_cal, y_cal
                    )
                    median_low = float(
                        extra["median_probability_low"]
                    )
                    median_high = float(
                        extra["median_probability_high"]
                    )

                personal = _metric_row(
                    y_eval, p_eval, threshold, subject_id
                )
                row = {
                    "model": model,
                    "outer_fold": fold,
                    "pm": pm,
                    "subject_id": subject_id,
                    "budget_seconds": int(budget),
                    "budget_minutes": float(budget / 60.0),
                    "budget_fully_available": fully_available,
                    "calibration_low": n_low,
                    "calibration_high": n_high,
                    "calibration_extreme": n_low + n_high,
                    "calibration_class_eligible": eligible,
                    "adaptation_applied": applied,
                    "adaptation_reason": reason,
                    "personalized_threshold": float(threshold),
                    "threshold_shift_from_0_5":
                        float(threshold - 0.5),
                    "median_probability_low": median_low,
                    "median_probability_high": median_high,
                    "evaluation_low": int(np.sum(y_eval == 0)),
                    "evaluation_high": int(np.sum(y_eval == 1)),
                    "evaluation_extreme": int(len(y_eval)),
                    "evaluation_sample_hash": eval_hash,
                }
                for metric in METRICS:
                    row[f"zero_shot_{metric}"] = zero[metric]
                    row[f"personalized_{metric}"] = personal[metric]
                    row[f"delta_{metric}"] = (
                        personal[metric] - zero[metric]
                    )
                row["zero_shot_roc_auc"] = zero["roc_auc"]
                row["zero_shot_pr_auc"] = zero["pr_auc"]
                results.append(row)

    results = pd.DataFrame(results)
    expected = 364 * len(MODELS) * len(BUDGETS)
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(results)}")
    if results.duplicated(
        ["model", "subject_id", "pm", "budget_seconds"]
    ).any():
        raise RuntimeError("Duplicate duration-response rows")

    # Common evaluation must be identical across budgets and models.
    for _, group in results.groupby(["subject_id", "pm"], sort=False):
        if group["evaluation_sample_hash"].nunique() != 1:
            raise RuntimeError("Evaluation sample set changed")
        if group["evaluation_low"].nunique() != 1:
            raise RuntimeError("Evaluation LOW count changed")
        if group["evaluation_high"].nunique() != 1:
            raise RuntimeError("Evaluation HIGH count changed")

    # Zero-shot is budget invariant.
    for _, group in results.groupby(
        ["model", "subject_id", "pm"], sort=False
    ):
        for metric in METRICS:
            values = group[f"zero_shot_{metric}"].to_numpy(dtype=float)
            if not np.allclose(
                values, values[0], rtol=0.0, atol=1e-15
            ):
                raise RuntimeError("Zero-shot metric changed across budgets")

    _write_csv(
        context.output_dir / "participant_pm_results.csv",
        results,
    )

    participant = _participant_first(results, applied_only=False)
    _write_csv(
        context.output_dir / "participant_aggregate.csv",
        participant,
    )
    summary = _summary_from_participant(
        participant, results, applied_only=False
    )
    _write_csv(
        context.output_dir / "summary_operational.csv",
        summary,
    )

    applied_participant = _participant_first(
        results, applied_only=True
    )
    _write_csv(
        context.output_dir / "participant_aggregate_applied_only.csv",
        applied_participant,
    )
    applied_summary = _summary_from_participant(
        applied_participant,
        results,
        applied_only=True,
    )
    _write_csv(
        context.output_dir / "summary_adaptation_applied_only.csv",
        applied_summary,
    )

    pm_rows = []
    for keys, group in results.groupby(
        ["model", "budget_seconds", "pm"], sort=True
    ):
        model, budget, pm = keys
        row = {
            "model": model,
            "budget_seconds": int(budget),
            "pm": pm,
            "participant_rows": int(len(group)),
            "class_eligible": int(
                group["calibration_class_eligible"].sum()
            ),
            "adaptation_applied": int(
                group["adaptation_applied"].sum()
            ),
            "nonseparated_median_fallback": int(
                group["adaptation_reason"]
                .astype(str).eq("nonseparated_medians").sum()
            ),
            "median_personalized_threshold": float(
                group["personalized_threshold"].median()
            ),
            "median_abs_threshold_shift": float(
                group["threshold_shift_from_0_5"].abs().median()
            ),
        }
        for metric in METRICS:
            row[f"delta_{metric}_mean"] = float(
                group[f"delta_{metric}"].mean()
            )
        pm_rows.append(row)
    _write_csv(
        context.output_dir / "summary_by_pm.csv",
        pd.DataFrame(pm_rows),
    )

    bootstrap_zero = _bootstrap_vs_zero_shot(participant)
    _write_csv(
        context.output_dir / "bootstrap_vs_zero_shot.csv",
        bootstrap_zero,
    )

    contrast_detail, contrast_bootstrap = _duration_contrasts(
        participant
    )
    _write_csv(
        context.output_dir / "participant_duration_contrasts.csv",
        contrast_detail,
    )
    _write_csv(
        context.output_dir / "bootstrap_duration_contrasts.csv",
        contrast_bootstrap,
    )

    threshold_summary_rows = []
    for keys, group in results.groupby(
        ["model", "budget_seconds"], sort=True
    ):
        model, budget = keys
        adapted = group.loc[group["adaptation_applied"]]
        threshold_summary_rows.append({
            "model": model,
            "budget_seconds": int(budget),
            "participant_pm_rows": int(len(group)),
            "class_eligible": int(
                group["calibration_class_eligible"].sum()
            ),
            "adaptation_applied": int(
                group["adaptation_applied"].sum()
            ),
            "nonseparated_median_fallback": int(
                group["adaptation_reason"]
                .astype(str).eq("nonseparated_medians").sum()
            ),
            "median_threshold_adapted": (
                float(adapted["personalized_threshold"].median())
                if len(adapted) else np.nan
            ),
            "q25_threshold_adapted": (
                float(adapted["personalized_threshold"].quantile(0.25))
                if len(adapted) else np.nan
            ),
            "q75_threshold_adapted": (
                float(adapted["personalized_threshold"].quantile(0.75))
                if len(adapted) else np.nan
            ),
            "median_abs_shift_adapted": (
                float(adapted["threshold_shift_from_0_5"].abs().median())
                if len(adapted) else np.nan
            ),
            "threshold_below_0_5_adapted": int(
                (adapted["personalized_threshold"] < 0.5).sum()
            ),
            "threshold_above_0_5_adapted": int(
                (adapted["personalized_threshold"] > 0.5).sum()
            ),
        })
    _write_csv(
        context.output_dir / "threshold_summary.csv",
        pd.DataFrame(threshold_summary_rows),
    )

    protocol = dict(context.protocol)
    protocol.update({
        "result_status": "confirmatory_complete",
        "threshold_calibration_executed": True,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "performance_evaluation_executed": True,
        "fixed_evaluation_ready_participant_pm": 364,
        "participant_pm_result_rows": int(len(results)),
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
        "common_evaluation_boundary_seconds": 900,
        "fixed_evaluation_ready_participant_pm": 364,
        "participant_pm_result_rows": int(len(results)),
        "primary_duration_contrast": "900s_minus_300s",
        "secondary_duration_contrast": "600s_minus_300s",
    }
    _atomic_json(context.output_dir / "pooled_summary.json", pooled)
    return pooled


__all__ = [
    "BUDGETS",
    "MODELS",
    "DurationResponseContext",
    "_duration_ready_keys",
    "_participant_first",
    "load_config",
    "prepare_protocol",
    "run_experiment",
    "write_dry_run",
]
