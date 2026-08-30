"""No-model feasibility gate for 15/20/30-minute PM personalization.

The next duration question was selected after the completed 5/10/15-minute
experiment showed positive 15-minus-10-minute exploratory bootstrap contrasts
for both XGBoost and LightGBM on balanced accuracy and Macro-F1.

This module does not evaluate model performance. It freezes:
- calibration budgets 900 / 1200 / 1800 s;
- earliest logical record only, with no record stitching;
- one common evaluation suffix strictly after +1800 s for every budget;
- unchanged EEG(t-10s) -> PM(t) and outer-train Q33/Q67 LOW/HIGH contract;
- unchanged chronology overlap trimming from the completed base feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bench.experiments.pm_low_high_personalization_feasibility import (
    FeasibilityContext,
    _subject_pm_timeline,
    load_config as load_base_config,
    prepare_protocol as prepare_base_protocol,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    FIXED_LAG_SECONDS,
    PM_NAMES,
    stable_hash,
)

SCHEMA_VERSION = "pm-low-high-personalization-long-duration-feasibility-v1"
BUDGETS_SECONDS = (900, 1200, 1800)
MAX_BUDGET_SECONDS = 1800
BASE_FEASIBILITY_HASH = (
    "94c568d7e41344478c0550f573b0abf8893783831f6c7241b92c8e4fdd25c9cd"
)
DURATION_RESPONSE_HASH = (
    "14f6fb28ebd748a1f897df0df6d8e7e5a03302733cddde8c33017524d6335035"
)
FUTURE_MODELS = ("xgboost", "lightgbm")


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
            payload, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False, default=str,
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


def _sample_hash(values: list[Any] | pd.Series) -> str:
    return stable_hash([str(value) for value in list(values)])


def _budget_fully_available(
    source_duration_seconds: float,
    feature_grid_duration_seconds: float,
    budget_seconds: int,
) -> bool:
    """Require both source span and canonical feature-grid span."""
    return bool(
        float(source_duration_seconds) >= float(budget_seconds)
        and float(feature_grid_duration_seconds) >= float(budget_seconds)
    )


def _state_counts(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    counts = frame["state"].value_counts()
    low = int(counts.get("low", 0))
    high = int(counts.get("high", 0))
    middle = int(counts.get("middle", 0))
    missing = int(counts.get("missing", 0))
    available = low + high + middle
    extreme = low + high
    return {
        f"{prefix}_exact_lag_slots": int(len(frame)),
        f"{prefix}_pm_available": available,
        f"{prefix}_missing_pm": missing,
        f"{prefix}_low": low,
        f"{prefix}_high": high,
        f"{prefix}_middle": middle,
        f"{prefix}_extreme": extreme,
        f"{prefix}_has_any_extreme": bool(extreme >= 1),
        f"{prefix}_has_both_classes": bool(low >= 1 and high >= 1),
        f"{prefix}_min1_each": bool(low >= 1 and high >= 1),
        f"{prefix}_min2_each": bool(low >= 2 and high >= 2),
        f"{prefix}_min3_each": bool(low >= 3 and high >= 3),
        f"{prefix}_min5_each": bool(low >= 5 and high >= 5),
        f"{prefix}_min10_each": bool(low >= 10 and high >= 10),
        f"{prefix}_extreme_fraction_of_available": (
            float(extreme / available) if available else None
        ),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    refs = cfg.get("references", {})
    if refs.get("base_feasibility", {}).get("protocol_hash") != BASE_FEASIBILITY_HASH:
        raise ValueError("Base feasibility reference changed")
    if refs.get("duration_response", {}).get("protocol_hash") != DURATION_RESPONSE_HASH:
        raise ValueError("Duration-response reference changed")

    expected_contract = {
        "pm_names": list(PM_NAMES),
        "alignment": "EEG(t-10s) -> PM(t)",
        "lag_seconds": FIXED_LAG_SECONDS,
        "calibration_budgets_seconds": list(BUDGETS_SECONDS),
        "maximum_calibration_budget_seconds": MAX_BUDGET_SECONDS,
        "control_budget_seconds": 900,
        "calibration_record_policy":
            "earliest_logical_record_by_selected_record_start_utc",
        "calibration_cross_record_policy": "forbidden",
        "calibration_interval_rule":
            "0 < target_relative_seconds <= budget_seconds",
        "budget_measurement":
            "elapsed_recording_time_not_extreme_sample_count",
        "budget_fully_available_rule":
            "source_duration_and_feature_grid_span_cover_budget",
        "fixed_evaluation_policy":
            "all_exact_lag_targets_strictly_after_max_budget_utc_boundary",
        "fixed_evaluation_boundary_rule":
            "absolute_target_utc > earliest_record_start_utc + 1800s",
        "reserved_interval_policy":
            "targets_after_current_budget_and_not_after_1800s_are_unused",
        "target_transform": "outer_train_q33_q67_extremes",
        "threshold_fit_scope": "outer_train_continuous_complete_cases",
        "middle_policy":
            "exclude_from_binary_training_and_evaluation_but_count_in_feasibility",
        "missing_pm_policy": "count_as_missing_not_middle",
        "outer_group": "subject_id",
        "folds": [1, 2, 3, 4, 5],
        "cross_record_overlap_policy":
            "earlier_record_precedence_trim_later_overlapping_prefix_by_feature_grid_utc",
        "future_personalization_models": list(FUTURE_MODELS),
        "future_threshold_strategy": "median_midpoint",
        "minimum_calibration_per_class_for_future_run": 2,
        "future_ineligible_policy": "zero_shot_fallback_no_budget_extension",
    }
    if cfg.get("scientific_contract") != expected_contract:
        raise ValueError("Scientific contract changed")

    if cfg.get("feasibility_criteria") != {
        "report_any_extreme": True,
        "report_both_extreme_classes": True,
        "report_minimum_per_class": [1, 2, 3, 5, 10],
        "minimum_fixed_evaluation_extremes_descriptive": 20,
        "minimum_fixed_evaluation_both_classes": True,
        "report_joint_min2_and_fixed_evaluation_ready": True,
        "criteria_role":
            "descriptive_prerun_gate_for_15_20_30min_duration_response",
    }:
        raise ValueError("Feasibility criteria changed")

    planned = cfg.get("planned_duration_comparison", {})
    expected_planned = {
        "control_budget_seconds": 900,
        "intermediate_budget_seconds": 1200,
        "primary_budget_seconds": 1800,
        "primary_contrast": "1800s_minus_900s",
        "secondary_contrast": "1200s_minus_900s",
        "common_evaluation_boundary_seconds": 1800,
        "performance_run_not_executed_by_this_audit": True,
        "selection_context":
            "prior exploratory 900s_minus_600s bootstrap was positive for both models and both primary metrics",
    }
    if planned != expected_planned:
        raise ValueError("Planned duration comparison changed")

    forbidden = cfg.get("forbidden", {})
    if not forbidden or not all(value is True for value in forbidden.values()):
        raise ValueError("All forbidden switches must remain true")
    return cfg


def _validate_references(root: Path, cfg: Mapping[str, Any]) -> None:
    refs = cfg["references"]
    checks = (
        (
            refs["base_feasibility"]["output_dir"],
            BASE_FEASIBILITY_HASH,
            "feasibility_audit_complete",
        ),
        (
            refs["duration_response"]["output_dir"],
            DURATION_RESPONSE_HASH,
            "confirmatory_complete",
        ),
    )
    for output_dir, expected_hash, expected_status in checks:
        path = root / output_dir / "protocol.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol_hash") != expected_hash:
            raise RuntimeError(f"{path}: protocol hash changed")
        if value.get("result_status") != expected_status:
            raise RuntimeError(f"{path}: expected status {expected_status}")


@dataclass
class LongDurationFeasibilityContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    base: FeasibilityContext
    subject_duration_availability: pd.DataFrame
    protocol: dict[str, Any]


def _subject_availability(base: FeasibilityContext) -> pd.DataFrame:
    frame = base.subject_chronology[
        [
            "subject_id", "outer_fold",
            "calibration_record_group_id", "calibration_record_id",
            "calibration_record_start_utc",
            "calibration_record_duration_seconds",
            "calibration_record_feature_grid_duration_seconds",
        ]
    ].copy()
    source_duration = frame[
        "calibration_record_duration_seconds"
    ].to_numpy(dtype=float)
    feature_duration = frame[
        "calibration_record_feature_grid_duration_seconds"
    ].to_numpy(dtype=float)
    for budget in BUDGETS_SECONDS:
        frame[f"source_full_{budget}s_available"] = (
            source_duration >= float(budget)
        )
        frame[f"feature_grid_full_{budget}s_available"] = (
            feature_duration >= float(budget)
        )
    return frame.sort_values("subject_id", kind="stable").reset_index(drop=True)


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> LongDurationFeasibilityContext:
    root = Path(root).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root / output

    _validate_references(root, config)
    ref = config["references"]["base_feasibility"]
    base_cfg = load_base_config(root / ref["config"])
    base = prepare_base_protocol(
        base_cfg,
        root=root,
        feature_cache_dir=feature_cache_dir,
        output_dir=root / ref["output_dir"],
    )
    if base.protocol["protocol_hash"] != BASE_FEASIBILITY_HASH:
        raise RuntimeError("Recomputed base feasibility hash changed")

    availability = _subject_availability(base)
    if len(availability) != 54:
        raise RuntimeError(f"Expected 54 subjects, got {len(availability)}")

    scientific_payload = {
        "schema_version": SCHEMA_VERSION,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "feasibility_criteria": config["feasibility_criteria"],
        "planned_duration_comparison": config["planned_duration_comparison"],
        "forbidden": config["forbidden"],
        "feature_cache_identity": base.low_high.cache_identity,
        "fixed_fold_hash": base.low_high.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": base.low_high.protocol["temporal_pairing_hash"],
        "threshold_hashes": base.low_high.protocol["threshold_hashes"],
        "record_chronology_hash": base.protocol["record_chronology_hash"],
        "subject_chronology_hash": base.protocol["subject_chronology_hash"],
        "subject_duration_availability_hash": _frame_hash(availability),
    }
    phash = stable_hash(scientific_payload)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "preregistered_candidate",
        "audit_executed": False,
        "model_training_executed": False,
        "model_inference_executed": False,
        "performance_evaluation_executed": False,
        "git_commit": _git_head(root),
        "protocol_hash": phash,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "feasibility_criteria": config["feasibility_criteria"],
        "planned_duration_comparison": config["planned_duration_comparison"],
        "fixed_fold_hash": base.low_high.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": base.low_high.protocol["temporal_pairing_hash"],
        "threshold_hashes": base.low_high.protocol["threshold_hashes"],
        "record_chronology_hash": base.protocol["record_chronology_hash"],
        "subject_chronology_hash": base.protocol["subject_chronology_hash"],
        "subject_duration_availability_hash":
            scientific_payload["subject_duration_availability_hash"],
        "n_subjects": 54,
        "n_logical_records": int(len(base.record_chronology)),
        "paired_target_rows_before_subject_overlap_trim":
            int(len(base.paired_timeline)),
        "budgets_seconds": list(BUDGETS_SECONDS),
        "max_budget_seconds": MAX_BUDGET_SECONDS,
    }
    return LongDurationFeasibilityContext(
        root=root,
        output_dir=output,
        config=dict(config),
        base=base,
        subject_duration_availability=availability,
        protocol=protocol,
    )


def write_dry_run(context: LongDurationFeasibilityContext) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(
        context.output_dir / "subject_duration_availability.csv",
        context.subject_duration_availability,
    )
    a = context.subject_duration_availability
    summary = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "audit_executed": False,
        "model_training_executed": False,
        "model_inference_executed": False,
        "performance_evaluation_executed": False,
        "subjects": 54,
        "logical_records": int(context.protocol["n_logical_records"]),
        "budgets_seconds": list(BUDGETS_SECONDS),
        "common_evaluation_boundary_seconds": MAX_BUDGET_SECONDS,
        "subjects_source_full_900s":
            int(a["source_full_900s_available"].sum()),
        "subjects_source_full_1200s":
            int(a["source_full_1200s_available"].sum()),
        "subjects_source_full_1800s":
            int(a["source_full_1800s_available"].sum()),
        "subjects_feature_grid_full_900s":
            int(a["feature_grid_full_900s_available"].sum()),
        "subjects_feature_grid_full_1200s":
            int(a["feature_grid_full_1200s_available"].sum()),
        "subjects_feature_grid_full_1800s":
            int(a["feature_grid_full_1800s_available"].sum()),
        "subjects_full_900s": int(
            (
                a["source_full_900s_available"]
                & a["feature_grid_full_900s_available"]
            ).sum()
        ),
        "subjects_full_1200s": int(
            (
                a["source_full_1200s_available"]
                & a["feature_grid_full_1200s_available"]
            ).sum()
        ),
        "subjects_full_1800s": int(
            (
                a["source_full_1800s_available"]
                & a["feature_grid_full_1800s_available"]
            ).sum()
        ),
        "fixed_lag_seconds": FIXED_LAG_SECONDS,
        "future_models": list(FUTURE_MODELS),
        "future_threshold_strategy": "median_midpoint",
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    readme = f"""# PM LOW/HIGH long-duration personalization feasibility v1

No model training, inference, or performance evaluation.

Frozen question:
- budgets: 900 / 1200 / 1800 seconds (15 / 20 / 30 minutes)
- control: 900 seconds
- common evaluation: strictly after +1800 seconds
- earliest logical record only; no record stitching
- a budget is fully available only when both source duration and canonical
  feature-grid span cover the complete interval
- threshold contract remains outer-train Q33/Q67
- temporal alignment remains EEG(t-10s) -> PM(t)
- future threshold strategy remains median_midpoint
- future models remain XGBoost + LightGBM
- calibration support reports 2+2, 3+3, 5+5, and 10+10 LOW/HIGH

Protocol hash:
`{context.protocol['protocol_hash']}`
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def run_audit(context: LongDurationFeasibilityContext) -> dict[str, Any]:
    subjects = context.base.subject_chronology.copy()
    subjects["subject_id"] = subjects["subject_id"].astype(str)
    lookup = subjects.set_index("subject_id")
    rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []

    for subject_id in sorted(lookup.index):
        subject = lookup.loc[subject_id]
        cal_group = str(subject["calibration_record_group_id"])
        start = float(
            pd.Timestamp(subject["calibration_record_start_utc"]).timestamp()
        )
        source_duration = float(
            subject["calibration_record_duration_seconds"]
        )
        feature_duration = float(
            subject["calibration_record_feature_grid_duration_seconds"]
        )
        boundary = start + MAX_BUDGET_SECONDS

        for pm in PM_NAMES:
            timeline = _subject_pm_timeline(
                context.base, subject_id=subject_id, pm=pm
            ).sort_values(
                ["absolute_target_epoch_seconds", "target_sample_id"],
                kind="stable",
            )
            cal_record = timeline.loc[
                timeline["record_group_id"].astype(str).eq(cal_group)
            ].copy()
            absolute = timeline[
                "absolute_target_epoch_seconds"
            ].to_numpy(dtype=float)
            fixed_eval = timeline.loc[absolute > boundary].copy()
            eval_counts = _state_counts(fixed_eval, "evaluation")
            eval_extreme = fixed_eval.loc[
                fixed_eval["state"].isin(["low", "high"])
            ]
            eval_hash = _sample_hash(
                eval_extreme["target_sample_id"].tolist()
            )
            eval_ready = bool(
                eval_counts["evaluation_extreme"] >= 20
                and eval_counts["evaluation_has_both_classes"]
            )

            relative = cal_record[
                "target_relative_seconds"
            ].to_numpy(dtype=float)
            for budget in BUDGETS_SECONDS:
                calibration = cal_record.loc[
                    (relative > 0.0) & (relative <= float(budget))
                ].copy()
                cal_counts = _state_counts(calibration, "calibration")
                source_full = source_duration >= float(budget)
                feature_full = feature_duration >= float(budget)
                budget_fully_available = _budget_fully_available(
                    source_duration,
                    feature_duration,
                    budget,
                )
                operational_eligible = bool(
                    budget_fully_available
                    and cal_counts["calibration_min2_each"]
                )
                joint_ready = bool(operational_eligible and eval_ready)

                reserved = timeline.loc[
                    (absolute > start + float(budget))
                    & (absolute <= boundary)
                ].copy()
                reserved_counts = _state_counts(reserved, "reserved")
                cal_extreme = calibration.loc[
                    calibration["state"].isin(["low", "high"])
                ]

                rows.append({
                    "subject_id": subject_id,
                    "outer_fold": int(subject["outer_fold"]),
                    "pm": pm,
                    "budget_seconds": int(budget),
                    "budget_minutes": float(budget / 60.0),
                    "calibration_record_group_id": cal_group,
                    "calibration_record_id":
                        str(subject["calibration_record_id"]),
                    "calibration_record_start_utc":
                        str(subject["calibration_record_start_utc"]),
                    "calibration_record_duration_seconds": source_duration,
                    "calibration_record_feature_grid_duration_seconds":
                        feature_duration,
                    "budget_source_fully_available": bool(source_full),
                    "budget_feature_grid_fully_available": bool(feature_full),
                    "budget_fully_available": budget_fully_available,
                    "operational_calibration_eligible_min2_each":
                        operational_eligible,
                    "fixed_evaluation_boundary_utc": pd.to_datetime(
                        boundary, unit="s", utc=True
                    ).isoformat(),
                    "fixed_evaluation_ready_min20_both_classes": eval_ready,
                    "joint_min2_calibration_and_fixed_evaluation_ready":
                        joint_ready,
                    "calibration_extreme_sample_hash":
                        _sample_hash(cal_extreme["target_sample_id"].tolist()),
                    "evaluation_extreme_sample_hash": eval_hash,
                    **cal_counts,
                    **reserved_counts,
                    **eval_counts,
                })

            timeline_rows.append({
                "subject_id": subject_id,
                "pm": pm,
                "outer_fold": int(subject["outer_fold"]),
                "full_subject_pm_timeline_hash":
                    _sample_hash(timeline["target_sample_id"].tolist()),
                "post_1800s_evaluation_extreme_sample_hash": eval_hash,
                "post_1800s_evaluation_extreme":
                    int(eval_counts["evaluation_extreme"]),
                "post_1800s_evaluation_ready_min20_both_classes":
                    eval_ready,
            })

    detail = pd.DataFrame(rows)
    expected = 54 * len(PM_NAMES) * len(BUDGETS_SECONDS)
    if len(detail) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(detail)}")
    if detail.duplicated(["subject_id", "pm", "budget_seconds"]).any():
        raise RuntimeError("Duplicate participant-PM-budget rows")

    for _, group in detail.groupby(["subject_id", "pm"], sort=False):
        if group["evaluation_extreme_sample_hash"].nunique() != 1:
            raise RuntimeError("Evaluation set changed across budgets")
        if group[
            "fixed_evaluation_ready_min20_both_classes"
        ].nunique() != 1:
            raise RuntimeError("Evaluation readiness changed across budgets")

    summary_rows = []
    for budget, group in detail.groupby("budget_seconds", sort=True):
        summary_rows.append({
            "budget_seconds": int(budget),
            "budget_minutes": float(budget / 60.0),
            "participant_pm_rows": int(len(group)),
            "subjects": int(group["subject_id"].nunique()),
            "source_fully_available_rows":
                int(group["budget_source_fully_available"].sum()),
            "source_fully_available_subjects": int(
                group.loc[
                    group["budget_source_fully_available"], "subject_id"
                ].nunique()
            ),
            "feature_grid_fully_available_rows":
                int(group["budget_feature_grid_fully_available"].sum()),
            "feature_grid_fully_available_subjects": int(
                group.loc[
                    group["budget_feature_grid_fully_available"], "subject_id"
                ].nunique()
            ),
            "fully_available_rows":
                int(group["budget_fully_available"].sum()),
            "fully_available_subjects": int(
                group.loc[
                    group["budget_fully_available"], "subject_id"
                ].nunique()
            ),
            "has_both_classes_rows":
                int(group["calibration_has_both_classes"].sum()),
            "min2_each_rows": int(group["calibration_min2_each"].sum()),
            "min3_each_rows": int(group["calibration_min3_each"].sum()),
            "min5_each_rows": int(group["calibration_min5_each"].sum()),
            "min10_each_rows": int(group["calibration_min10_each"].sum()),
            "operational_calibration_eligible_min2_each_rows":
                int(group[
                    "operational_calibration_eligible_min2_each"
                ].sum()),
            "fixed_evaluation_ready_min20_both_classes_rows":
                int(group[
                    "fixed_evaluation_ready_min20_both_classes"
                ].sum()),
            "joint_min2_calibration_and_fixed_evaluation_ready_rows":
                int(group[
                    "joint_min2_calibration_and_fixed_evaluation_ready"
                ].sum()),
            "median_calibration_extremes":
                float(group["calibration_extreme"].median()),
            "median_calibration_low":
                float(group["calibration_low"].median()),
            "median_calibration_high":
                float(group["calibration_high"].median()),
            "median_post_1800s_evaluation_extremes":
                float(group["evaluation_extreme"].median()),
        })
    summary = pd.DataFrame(summary_rows)

    pm_rows = []
    for (pm, budget), group in detail.groupby(
        ["pm", "budget_seconds"], sort=True
    ):
        pm_rows.append({
            "pm": str(pm),
            "budget_seconds": int(budget),
            "participant_rows": int(len(group)),
            "source_fully_available":
                int(group["budget_source_fully_available"].sum()),
            "feature_grid_fully_available":
                int(group["budget_feature_grid_fully_available"].sum()),
            "fully_available":
                int(group["budget_fully_available"].sum()),
            "has_both_classes":
                int(group["calibration_has_both_classes"].sum()),
            "min2_each": int(group["calibration_min2_each"].sum()),
            "min3_each": int(group["calibration_min3_each"].sum()),
            "min5_each": int(group["calibration_min5_each"].sum()),
            "min10_each": int(group["calibration_min10_each"].sum()),
            "operational_eligible_min2_each":
                int(group[
                    "operational_calibration_eligible_min2_each"
                ].sum()),
            "fixed_eval_ready_min20_both":
                int(group[
                    "fixed_evaluation_ready_min20_both_classes"
                ].sum()),
            "joint_min2_and_fixed_eval_ready":
                int(group[
                    "joint_min2_calibration_and_fixed_evaluation_ready"
                ].sum()),
            "median_extreme_count":
                float(group["calibration_extreme"].median()),
            "median_post_1800s_evaluation_extremes":
                float(group["evaluation_extreme"].median()),
        })
    pm_summary = pd.DataFrame(pm_rows)
    timeline = pd.DataFrame(timeline_rows)

    context.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        context.output_dir / "participant_pm_budget_feasibility.csv",
        detail,
    )
    _write_csv(context.output_dir / "summary_by_budget.csv", summary)
    _write_csv(context.output_dir / "summary_by_pm_budget.csv", pm_summary)
    _write_csv(context.output_dir / "timeline_hash_audit.csv", timeline)

    protocol = dict(context.protocol)
    protocol.update({
        "result_status": "feasibility_audit_complete",
        "audit_executed": True,
        "model_training_executed": False,
        "model_inference_executed": False,
        "performance_evaluation_executed": False,
        "participant_pm_budget_rows": int(len(detail)),
        "detail_hash": _frame_hash(detail),
        "timeline_hash_audit_hash": _frame_hash(timeline),
    })
    _atomic_json(context.output_dir / "protocol.json", protocol)

    pooled = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "result_status": "feasibility_audit_complete",
        "audit_executed": True,
        "model_training_executed": False,
        "model_inference_executed": False,
        "performance_evaluation_executed": False,
        "subjects": 54,
        "participant_pm_budget_rows": int(len(detail)),
        "budgets_seconds": list(BUDGETS_SECONDS),
        "common_evaluation_boundary_seconds": MAX_BUDGET_SECONDS,
        "summary_by_budget": summary.to_dict("records"),
    }
    _atomic_json(context.output_dir / "pooled_summary.json", pooled)
    return pooled


__all__ = [
    "BUDGETS_SECONDS",
    "LongDurationFeasibilityContext",
    "_budget_fully_available",
    "_state_counts",
    "load_config",
    "prepare_protocol",
    "run_audit",
    "write_dry_run",
]
