"""Preregistered subject-weighted augmentation for the frozen LOW/HIGH task.

The dry-run path reconstructs the frozen cohort, eligibility and deterministic
570-fit matrix without constructing a model, fitting, inference or performance
evaluation.  The full-run path is intentionally separate and must be requested
explicitly by the CLI ``--run`` switch.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from bench.experiments.pm_low_high_personalization_duration_response import (
    _atomic_json,
    _completed_protocol,
    _frame_hash,
    _git_head,
    _write_csv,
)
from bench.experiments.pm_low_high_personalization_feasibility import (
    _subject_pm_timeline,
)
from bench.experiments.pm_low_high_personalization_long_duration_feasibility import (
    load_config as load_long_feasibility_config,
    prepare_protocol as prepare_long_feasibility_protocol,
)
from bench.experiments.pm_low_high_personalized_threshold import (
    _audit_prediction_sources,
    _prediction_source_matrix,
    _state_to_y,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    PM_NAMES,
    _sample_hash,
    stable_hash,
)
from cogstate.model_zoo import build_model


SCHEMA_VERSION = "pm-low-high-personalized-weighted-augmentation-v1"
EXPERIMENT_ID = "pm_low_high_personalized_weighted_augmentation_v1"
MODELS = ("xgboost", "lightgbm")
CALIBRATION_BUDGET_SECONDS = 1800
MINIMUM_LOW = 10
MINIMUM_HIGH = 10
EXPECTED_EVAL_READY = 345
EXPECTED_ELIGIBLE = 285
EXPECTED_ELIGIBLE_PARTICIPANTS = 48
EXPECTED_PERSONALIZED_FITS = 570
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42

XGBOOST_HASH = (
    "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
)
LIGHTGBM_HASH = (
    "a2c51deb4d94e33d71863f1f1c7927470266b7352647664f3a411fb5db819b4e"
)
LONG_FEASIBILITY_HASH = (
    "34e0aa3350f84198383cd0e6a1d213711983132dcc30aa14fcc9edaafbc1095f"
)
LONG_RESPONSE_HASH = (
    "7fdc10bccad792c1f2d113ee063469230f229bd3dabf0cf0cb21e4f9b88e5caf"
)
PROBABILITY_CALIBRATION_HASH = (
    "d0a2e21e333a1ec70c9d1f5ca28604961d184ac8254f7f5fc93a8af62c70ba58"
)

REFERENCE_PATHS = {
    "xgboost": "reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1",
    "lightgbm": "reports/diagnostics/pm_low_high_model_robustness_v1",
    "long_duration_feasibility": (
        "reports/diagnostics/"
        "pm_low_high_personalization_long_duration_feasibility_v1"
    ),
    "long_duration_response": (
        "reports/diagnostics/"
        "pm_low_high_personalization_long_duration_response_v1"
    ),
    "probability_calibration": (
        "reports/diagnostics/pm_low_high_personalized_probability_calibration_v1"
    ),
    "base_reconstruction": (
        "reports/diagnostics/pm_low_high_base_model_reconstruction_audit_v1"
    ),
}
LONG_FEASIBILITY_CONFIG = (
    "experiments/pm_diagnostics/"
    "pm_low_high_personalization_long_duration_feasibility_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    "reports/diagnostics/pm_low_high_personalized_weighted_augmentation_v1"
)

CONTRAST_COLUMNS = (
    "delta_roc_auc",
    "delta_pr_auc",
    "brier_improvement",
    "log_loss_improvement",
    "delta_balanced_accuracy_vs_frozen_median",
    "delta_macro_f1_vs_frozen_median",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _strict_expected_config() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "preregistered_design",
        "scientific_contract": {
            "pm_names": list(PM_NAMES),
            "alignment": "EEG(t-10s) -> PM(t)",
            "lag_seconds": -10,
            "feature_count": 371,
            "target_transform": "outer_train_q33_q67_extremes",
            "middle_policy": "exclude",
            "outer_group": "subject_id",
            "folds": [1, 2, 3, 4, 5],
        },
        "models": {
            "xgboost": {
                "reference_protocol_hash": XGBOOST_HASH,
                "reuse_exact_frozen_hyperparameters": True,
            },
            "lightgbm": {
                "reference_protocol_hash": LIGHTGBM_HASH,
                "reuse_exact_frozen_hyperparameters": True,
            },
        },
        "calibration": {
            "budget_seconds": 1800,
            "minimum_low": 10,
            "minimum_high": 10,
            "budget_extension": False,
            "record_stitching": False,
            "middle_consumes_time": True,
            "middle_used_for_training": False,
            "eligibility_source": "feasibility_only",
            "expected_post1800_eval_ready_participant_pm": 345,
            "expected_eligible_participant_pm": 285,
            "expected_eligible_participants": 48,
        },
        "adaptation": {
            "method": "subject_equivalent_class_balanced_weighted_augmentation",
            "training_mode": "fresh_fit_from_frozen_base_hyperparameters",
            "base_outer_train_sample_weight": 1.0,
            "personal_total_mass_formula": (
                "n_outer_train / n_outer_train_subjects"
            ),
            "personal_low_mass_fraction": 0.5,
            "personal_high_mass_fraction": 0.5,
            "personal_low_sample_weight_formula": (
                "personal_total_mass / (2 * n_calibration_low)"
            ),
            "personal_high_sample_weight_formula": (
                "personal_total_mass / (2 * n_calibration_high)"
            ),
            "weight_cap": None,
            "weight_multiplier": None,
            "hyperparameter_search": False,
            "expected_new_fits": 570,
        },
        "prediction_policy": {
            "eligible_probability_source": (
                "personalized_model_predict_proba_high"
            ),
            "eligible_classification_threshold": 0.5,
            "ineligible_probability_fallback": "frozen_zero_shot_probability",
            "ineligible_classification_fallback": (
                "frozen_1800s_median_midpoint_policy"
            ),
        },
        "evaluation": {
            "primary_metric": "participant_first_roc_auc",
            "primary_contrast": "personalized_minus_zero_shot",
            "secondary_metrics": [
                "pr_auc",
                "brier_score",
                "log_loss",
                "balanced_accuracy",
                "macro_f1",
            ],
            "classification_reference": "frozen_1800s_median_midpoint",
            "probability_reference": "frozen_zero_shot",
            "ranking_reference": "frozen_zero_shot",
            "aggregation": (
                "mean_pm_within_participant_then_mean_participants"
            ),
            "bootstrap_unit": "subject_id",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 42,
            "operational_estimand": "all_post1800_eval_ready_with_fallback",
            "applied_only_secondary": True,
        },
        "references": {
            "long_duration_feasibility_protocol_hash": LONG_FEASIBILITY_HASH,
            "long_duration_response_protocol_hash": LONG_RESPONSE_HASH,
            "probability_calibration_protocol_hash": PROBABILITY_CALIBRATION_HASH,
        },
        "forbidden": {
            "evaluation_label_use_during_training": True,
            "evaluation_probability_use_for_selection": True,
            "target_specific_hyperparameters": True,
            "focus_specific_logic": True,
            "weight_multiplier_search": True,
            "weight_cap_search": True,
            "model_hyperparameter_search": True,
            "post_adaptation_threshold_search": True,
            "additional_probability_calibration": True,
            "calibration_budget_search": True,
        },
    }


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the frozen config and reject every scientific deviation."""
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    expected = _strict_expected_config()
    if config != expected:
        differing = sorted(
            key
            for key in set(config) | set(expected)
            if config.get(key) != expected.get(key)
        )
        raise ValueError(
            "Frozen weighted-augmentation config changed at top-level sections: "
            f"{differing}"
        )
    return config


def subject_equivalent_class_balanced_weights(
    *,
    n_outer_train: int,
    n_outer_train_subjects: int,
    n_calibration_low: int,
    n_calibration_high: int,
) -> dict[str, float]:
    """Return the preregistered personal mass and per-row class weights."""
    values = (
        n_outer_train,
        n_outer_train_subjects,
        n_calibration_low,
        n_calibration_high,
    )
    if any(int(value) != value or int(value) <= 0 for value in values):
        raise ValueError("All weighting counts must be positive integers")
    personal_total_mass = float(n_outer_train / n_outer_train_subjects)
    low_weight = float(personal_total_mass / (2.0 * n_calibration_low))
    high_weight = float(personal_total_mass / (2.0 * n_calibration_high))
    low_mass = float(low_weight * n_calibration_low)
    high_mass = float(high_weight * n_calibration_high)
    if not np.isclose(low_mass, personal_total_mass / 2.0, atol=1e-12):
        raise RuntimeError("Calibration LOW mass is not one half")
    if not np.isclose(high_mass, personal_total_mass / 2.0, atol=1e-12):
        raise RuntimeError("Calibration HIGH mass is not one half")
    return {
        "personal_total_mass": personal_total_mass,
        "calibration_low_sample_weight": low_weight,
        "calibration_high_sample_weight": high_weight,
        "calibration_low_total_mass": low_mass,
        "calibration_high_total_mass": high_mass,
    }


def is_weighted_augmentation_eligible(
    *,
    evaluation_ready: bool,
    budget_fully_available: bool,
    n_calibration_low: int,
    n_calibration_high: int,
) -> bool:
    return bool(
        evaluation_ready
        and budget_fully_available
        and n_calibration_low >= MINIMUM_LOW
        and n_calibration_high >= MINIMUM_HIGH
    )


def validate_sample_firewall(
    *,
    outer_train_sample_ids: Sequence[Any],
    calibration_sample_ids: Sequence[Any],
    evaluation_sample_ids: Sequence[Any],
    outer_train_subjects: Sequence[Any],
    personalized_subject_id: str,
) -> None:
    train = set(map(str, outer_train_sample_ids))
    calibration = set(map(str, calibration_sample_ids))
    evaluation = set(map(str, evaluation_sample_ids))
    if train & calibration or train & evaluation or calibration & evaluation:
        raise RuntimeError("Outer-train, calibration and evaluation samples overlap")
    if str(personalized_subject_id) in set(map(str, outer_train_subjects)):
        raise RuntimeError("Personalized outer-test subject leaked into outer-train")


def resolve_prediction_policy(
    *,
    eligible: bool,
    personalized_probability: np.ndarray | None,
    zero_shot_probability: np.ndarray,
    frozen_median_threshold: float,
) -> tuple[np.ndarray, float, str, str]:
    """Resolve frozen eligible/ineligible probability and class policies."""
    zero = np.asarray(zero_shot_probability, dtype=float).reshape(-1)
    if eligible:
        if personalized_probability is None:
            raise ValueError("Eligible cells require personalized probabilities")
        probability = np.asarray(personalized_probability, dtype=float).reshape(-1)
        if probability.shape != zero.shape:
            raise ValueError("Personalized and zero-shot probabilities differ in shape")
        return (
            probability,
            0.5,
            "personalized_model_predict_proba_high",
            "fixed_probability_threshold_0.5",
        )
    return (
        zero.copy(),
        float(frozen_median_threshold),
        "frozen_zero_shot_probability",
        "frozen_1800s_median_midpoint_policy",
    )


@dataclass
class WeightedAugmentationContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    feasibility: Any
    feasibility_detail: pd.DataFrame
    eligibility_audit: pd.DataFrame
    outer_train_audit: pd.DataFrame
    frozen_response: pd.DataFrame
    source_matrix: pd.DataFrame
    source_audit: pd.DataFrame
    reference_audit: pd.DataFrame
    model_contracts: dict[str, dict[str, Any]]
    protocol: dict[str, Any]
    run_matrix: pd.DataFrame


def _reference_protocol(
    root: Path,
    *,
    label: str,
    expected_hash: str,
    allowed_statuses: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = REFERENCE_PATHS[label]
    protocol = _completed_protocol(
        root,
        relative,
        expected_hash,
        allowed_statuses=allowed_statuses,
    )
    audit = {
        "reference": label,
        "output_dir": relative,
        "expected_protocol_hash": expected_hash,
        "actual_protocol_hash": protocol["protocol_hash"],
        "result_status": protocol["result_status"],
        "valid": True,
    }
    return protocol, audit


def _timeline_slices(
    feasibility: Any,
    *,
    subject_id: str,
    pm: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    timeline = _subject_pm_timeline(
        feasibility.base,
        subject_id=str(subject_id),
        pm=str(pm),
    ).sort_values(
        ["absolute_target_epoch_seconds", "target_sample_id"],
        kind="stable",
    )
    subject_rows = feasibility.base.subject_chronology.loc[
        feasibility.base.subject_chronology["subject_id"]
        .astype(str)
        .eq(str(subject_id))
    ]
    if len(subject_rows) != 1:
        raise RuntimeError(f"Expected one chronology row for {subject_id}")
    subject = subject_rows.iloc[0]
    calibration_group = str(subject["calibration_record_group_id"])
    relative = timeline["target_relative_seconds"].to_numpy(dtype=float)
    calibration_all = timeline.loc[
        timeline["record_group_id"].astype(str).eq(calibration_group)
        & (relative > 0.0)
        & (relative <= float(CALIBRATION_BUDGET_SECONDS))
    ].copy()
    calibration_extreme = calibration_all.loc[
        calibration_all["state"].isin(["low", "high"])
    ].copy()
    boundary = (
        float(pd.Timestamp(subject["calibration_record_start_utc"]).timestamp())
        + CALIBRATION_BUDGET_SECONDS
    )
    evaluation = timeline.loc[
        timeline["state"].isin(["low", "high"])
        & (
            timeline["absolute_target_epoch_seconds"].to_numpy(dtype=float)
            > boundary
        )
    ].copy()
    return calibration_all, calibration_extreme, evaluation, subject


def _build_eligibility_audit(
    feasibility: Any,
    detail: pd.DataFrame,
    frozen_response: pd.DataFrame,
) -> pd.DataFrame:
    frozen_lookup = frozen_response.set_index(["model", "subject_id", "pm"])
    rows: list[dict[str, Any]] = []
    for frozen in detail.sort_values(
        ["outer_fold", "pm", "subject_id"], kind="stable"
    ).to_dict("records"):
        subject_id = str(frozen["subject_id"])
        pm = str(frozen["pm"])
        calibration_all, calibration, evaluation, subject = _timeline_slices(
            feasibility,
            subject_id=subject_id,
            pm=pm,
        )
        n_low = int(calibration["state"].eq("low").sum())
        n_high = int(calibration["state"].eq("high").sum())
        n_middle = int(calibration_all["state"].eq("middle").sum())
        y_evaluation = _state_to_y(evaluation["state"])
        evaluation_ready = bool(
            len(evaluation) >= 20 and set(np.unique(y_evaluation)) == {0, 1}
        )
        fully_available = _as_bool(frozen["budget_fully_available"])
        expected_ready = _as_bool(
            frozen["fixed_evaluation_ready_min20_both_classes"]
        )
        if evaluation_ready != expected_ready:
            raise RuntimeError(f"{subject_id}/{pm}: evaluation readiness changed")
        if n_low != int(frozen["calibration_low"]):
            raise RuntimeError(f"{subject_id}/{pm}: calibration LOW changed")
        if n_high != int(frozen["calibration_high"]):
            raise RuntimeError(f"{subject_id}/{pm}: calibration HIGH changed")
        if n_middle != int(frozen["calibration_middle"]):
            raise RuntimeError(f"{subject_id}/{pm}: calibration MIDDLE changed")
        calibration_hash = _sample_hash(
            calibration["target_sample_id"].astype(str).tolist()
        )
        evaluation_hash = _sample_hash(
            evaluation["target_sample_id"].astype(str).tolist()
        )
        if calibration_hash != str(frozen["calibration_extreme_sample_hash"]):
            raise RuntimeError(f"{subject_id}/{pm}: calibration sample hash changed")
        if evaluation_hash != str(frozen["evaluation_extreme_sample_hash"]):
            raise RuntimeError(f"{subject_id}/{pm}: evaluation sample hash changed")
        eligible = is_weighted_augmentation_eligible(
            evaluation_ready=evaluation_ready,
            budget_fully_available=fully_available,
            n_calibration_low=n_low,
            n_calibration_high=n_high,
        )
        for model in MODELS:
            if evaluation_ready:
                key = (model, subject_id, pm)
                if key not in frozen_lookup.index:
                    raise RuntimeError(f"{key}: frozen response row missing")
                response = frozen_lookup.loc[key]
                if str(response["evaluation_sample_hash"]) != evaluation_hash:
                    raise RuntimeError(f"{key}: frozen evaluation hash changed")
                if int(response["outer_fold"]) != int(frozen["outer_fold"]):
                    raise RuntimeError(f"{key}: frozen outer fold changed")
        calibration_ids = calibration["target_sample_id"].astype(str).tolist()
        evaluation_ids = evaluation["target_sample_id"].astype(str).tolist()
        if set(calibration_ids) & set(evaluation_ids):
            raise RuntimeError(f"{subject_id}/{pm}: calibration/evaluation overlap")
        rows.append({
            "subject_id": subject_id,
            "outer_fold": int(frozen["outer_fold"]),
            "pm": pm,
            "target_id": f"target_{pm}",
            "calibration_record_group_id": str(
                subject["calibration_record_group_id"]
            ),
            "calibration_budget_seconds": CALIBRATION_BUDGET_SECONDS,
            "budget_fully_available": fully_available,
            "calibration_exact_lag_slots": int(len(calibration_all)),
            "calibration_low": n_low,
            "calibration_high": n_high,
            "calibration_middle": n_middle,
            "calibration_extreme": int(len(calibration)),
            "calibration_extreme_sample_hash": calibration_hash,
            "evaluation_low": int(np.sum(y_evaluation == 0)),
            "evaluation_high": int(np.sum(y_evaluation == 1)),
            "evaluation_extreme": int(len(evaluation)),
            "evaluation_extreme_sample_hash": evaluation_hash,
            "evaluation_ready": evaluation_ready,
            "eligible": eligible,
            "eligibility_reason": (
                "full1800_min10_each_eval_ready"
                if eligible
                else "operational_fallback"
            ),
            "calibration_evaluation_overlap": 0,
            "record_stitching_used": False,
            "middle_used_for_training": False,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 378 or frame.duplicated(["subject_id", "pm"]).any():
        raise RuntimeError("Eligibility audit must contain 54 x 7 unique rows")
    ready = frame.loc[frame["evaluation_ready"]]
    eligible = frame.loc[frame["eligible"]]
    if len(ready) != EXPECTED_EVAL_READY:
        raise RuntimeError(f"Expected {EXPECTED_EVAL_READY} eval-ready rows")
    if len(eligible) != EXPECTED_ELIGIBLE:
        raise RuntimeError(f"Expected {EXPECTED_ELIGIBLE} eligible rows")
    if eligible["subject_id"].nunique() != EXPECTED_ELIGIBLE_PARTICIPANTS:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_PARTICIPANTS} eligible participants"
        )
    return frame


def _build_outer_train_audit(feasibility: Any) -> pd.DataFrame:
    low_high = feasibility.base.low_high
    rows: list[dict[str, Any]] = []
    for reference in low_high.run_matrix.sort_values(
        ["outer_fold", "pm"], kind="stable"
    ).to_dict("records"):
        fold = int(reference["outer_fold"])
        pm = str(reference["pm"])
        cohort = low_high.cohorts[pm]
        labels = low_high.transforms[(fold, pm)].transform(
            cohort["continuous_target"].to_numpy(dtype=float)
        )
        train_mask = (
            cohort["outer_fold"].astype(int).ne(fold).to_numpy()
            & np.isfinite(labels)
        )
        subjects = sorted(
            cohort.loc[train_mask, "subject_id"].astype(str).unique().tolist()
        )
        sample_hash = _sample_hash(
            cohort.loc[train_mask, "target_sample_id"].astype(str).tolist()
        )
        if int(train_mask.sum()) != int(reference["n_train"]):
            raise RuntimeError(f"fold {fold}/{pm}: outer-train count changed")
        if sample_hash != str(reference["train_sample_hash"]):
            raise RuntimeError(f"fold {fold}/{pm}: outer-train hash changed")
        rows.append({
            "outer_fold": fold,
            "pm": pm,
            "target_id": f"target_{pm}",
            "n_outer_train": int(train_mask.sum()),
            "n_outer_train_subjects": int(len(subjects)),
            "outer_train_subjects_hash": stable_hash(subjects),
            "outer_train_sample_hash": sample_hash,
            "threshold_hash": str(reference["threshold_hash"]),
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 35:
        raise RuntimeError("Outer-train audit must contain 35 fold-PM rows")
    return frame


def build_adaptation_matrix(
    *,
    eligibility_audit: pd.DataFrame,
    outer_train_audit: pd.DataFrame,
    model_contracts: Mapping[str, Mapping[str, Any]],
    protocol_hash: str,
) -> pd.DataFrame:
    """Build deterministic model x eligible-participant x PM specifications."""
    train_lookup = outer_train_audit.set_index(["outer_fold", "pm"])
    rows: list[dict[str, Any]] = []
    eligible = eligibility_audit.loc[eligibility_audit["eligible"].astype(bool)]
    eligible = eligible.sort_values(
        ["outer_fold", "pm", "subject_id"], kind="stable"
    )
    for model in MODELS:
        model_contract = model_contracts[model]
        params_hash = stable_hash(model_contract["params"])
        for item in eligible.to_dict("records"):
            fold = int(item["outer_fold"])
            pm = str(item["pm"])
            train = train_lookup.loc[(fold, pm)]
            weights = subject_equivalent_class_balanced_weights(
                n_outer_train=int(train["n_outer_train"]),
                n_outer_train_subjects=int(train["n_outer_train_subjects"]),
                n_calibration_low=int(item["calibration_low"]),
                n_calibration_high=int(item["calibration_high"]),
            )
            spec: dict[str, Any] = {
                "model": model,
                "estimator": str(model_contract["estimator"]),
                "model_params_hash": params_hash,
                "outer_fold": fold,
                "subject_id": str(item["subject_id"]),
                "pm": pm,
                "target_id": f"target_{pm}",
                "lag_seconds": -10,
                "feature_count": 371,
                "calibration_budget_seconds": CALIBRATION_BUDGET_SECONDS,
                "n_outer_train": int(train["n_outer_train"]),
                "n_outer_train_subjects": int(train["n_outer_train_subjects"]),
                "outer_train_sample_hash": str(train["outer_train_sample_hash"]),
                "threshold_hash": str(train["threshold_hash"]),
                "n_calibration_low": int(item["calibration_low"]),
                "n_calibration_high": int(item["calibration_high"]),
                "n_calibration_extreme": int(item["calibration_extreme"]),
                "calibration_sample_hash": str(
                    item["calibration_extreme_sample_hash"]
                ),
                "evaluation_low": int(item["evaluation_low"]),
                "evaluation_high": int(item["evaluation_high"]),
                "n_evaluation_extreme": int(item["evaluation_extreme"]),
                "evaluation_sample_hash": str(
                    item["evaluation_extreme_sample_hash"]
                ),
                "base_outer_train_sample_weight": 1.0,
                **weights,
            }
            specification_hash = stable_hash({
                "protocol_hash": protocol_hash,
                "run_spec": spec,
            })
            spec["specification_hash"] = specification_hash
            spec["run_id"] = (
                f"{model}__fold_{fold:02d}__{pm}__{item['subject_id']}__"
                f"weighted_aug__{specification_hash[:12]}"
            )
            rows.append(spec)
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_PERSONALIZED_FITS:
        raise RuntimeError(
            f"Expected {EXPECTED_PERSONALIZED_FITS} personalized fits"
        )
    if frame["run_id"].duplicated().any():
        raise RuntimeError("Adaptation run IDs are not unique")
    if set(frame["model"]) != set(MODELS) or set(frame["pm"]) != set(PM_NAMES):
        raise RuntimeError("Both models and all seven PM must use one path")
    if not frame.groupby("model").size().eq(EXPECTED_ELIGIBLE).all():
        raise RuntimeError("Each model must contain 285 personalized fits")
    if not np.allclose(
        frame["calibration_low_total_mass"],
        frame["calibration_high_total_mass"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("LOW/HIGH calibration mass is not balanced")
    return frame


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> WeightedAugmentationContext:
    root_path = Path(root).resolve()
    output = Path(output_dir or DEFAULT_OUTPUT_DIR)
    if not output.is_absolute():
        output = root_path / output

    references: list[dict[str, Any]] = []
    xgboost_protocol, audit = _reference_protocol(
        root_path,
        label="xgboost",
        expected_hash=XGBOOST_HASH,
        allowed_statuses={"confirmatory_complete"},
    )
    references.append(audit)
    lightgbm_protocol, audit = _reference_protocol(
        root_path,
        label="lightgbm",
        expected_hash=LIGHTGBM_HASH,
        allowed_statuses={"confirmatory_complete"},
    )
    references.append(audit)
    _, audit = _reference_protocol(
        root_path,
        label="long_duration_feasibility",
        expected_hash=LONG_FEASIBILITY_HASH,
        allowed_statuses={"feasibility_audit_complete"},
    )
    references.append(audit)
    _, audit = _reference_protocol(
        root_path,
        label="long_duration_response",
        expected_hash=LONG_RESPONSE_HASH,
        allowed_statuses={"confirmatory_complete"},
    )
    references.append(audit)
    _, audit = _reference_protocol(
        root_path,
        label="probability_calibration",
        expected_hash=PROBABILITY_CALIBRATION_HASH,
        allowed_statuses={"confirmatory_complete"},
    )
    references.append(audit)

    reconstruction_path = (
        root_path / REFERENCE_PATHS["base_reconstruction"] / "summary.json"
    )
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    reconstruction_required = {
        "all_exact": True,
        "completed_fits": 70,
        "expected_fits": 70,
        "maximum_probability_difference": 0.0,
        "model_robustness_protocol_hash": LIGHTGBM_HASH,
        "xgboost_protocol_hash": XGBOOST_HASH,
    }
    for key, expected in reconstruction_required.items():
        if reconstruction.get(key) != expected:
            raise RuntimeError(f"Base reconstruction audit changed at {key}")
    references.append({
        "reference": "base_reconstruction",
        "output_dir": REFERENCE_PATHS["base_reconstruction"],
        "expected_protocol_hash": "70_of_70_exact_saved_probabilities",
        "actual_protocol_hash": "70_of_70_exact_saved_probabilities",
        "result_status": "complete",
        "valid": True,
    })

    model_contracts = {
        "xgboost": dict(xgboost_protocol["model"]),
        "lightgbm": dict(lightgbm_protocol["candidate_models"]["lightgbm"]),
    }
    expected_params = {
        "n_estimators": 200,
        "n_jobs": 4,
        "random_state": 42,
    }
    for model, estimator in (
        ("xgboost", "XGBClassifier"),
        ("lightgbm", "LGBMClassifier"),
    ):
        if model_contracts[model]["params"] != expected_params:
            raise RuntimeError(f"{model}: frozen hyperparameters changed")
        if model_contracts[model]["estimator"] != estimator:
            raise RuntimeError(f"{model}: frozen estimator changed")

    feasibility_config = load_long_feasibility_config(
        root_path / LONG_FEASIBILITY_CONFIG
    )
    feasibility = prepare_long_feasibility_protocol(
        feasibility_config,
        root=root_path,
        feature_cache_dir=feature_cache_dir,
        output_dir=root_path / REFERENCE_PATHS["long_duration_feasibility"],
    )
    if feasibility.protocol["protocol_hash"] != LONG_FEASIBILITY_HASH:
        raise RuntimeError("Recomputed long-duration feasibility hash changed")
    low_high = feasibility.base.low_high
    if low_high.protocol["protocol_hash"] != XGBOOST_HASH:
        raise RuntimeError("Recomputed LOW/HIGH protocol hash changed")
    if int(low_high.matrix.shape[1]) != 371:
        raise RuntimeError("Canonical feature count changed")

    detail = pd.read_csv(
        root_path
        / REFERENCE_PATHS["long_duration_feasibility"]
        / "participant_pm_budget_feasibility.csv"
    )
    detail = detail.loc[detail["budget_seconds"].astype(int).eq(1800)].copy()
    detail["subject_id"] = detail["subject_id"].astype(str)
    if len(detail) != 378:
        raise RuntimeError("Expected 378 participant-PM feasibility rows")

    frozen_columns = [
        "model",
        "outer_fold",
        "pm",
        "subject_id",
        "budget_seconds",
        "budget_fully_available",
        "calibration_low",
        "calibration_high",
        "calibration_class_eligible",
        "adaptation_applied",
        "personalized_threshold",
        "evaluation_low",
        "evaluation_high",
        "evaluation_extreme",
        "evaluation_sample_hash",
    ]
    frozen_response = pd.read_csv(
        root_path
        / REFERENCE_PATHS["long_duration_response"]
        / "participant_pm_results.csv",
        usecols=frozen_columns,
    )
    frozen_response = frozen_response.loc[
        frozen_response["budget_seconds"].astype(int).eq(1800)
    ].copy()
    frozen_response["subject_id"] = frozen_response["subject_id"].astype(str)
    if len(frozen_response) != EXPECTED_EVAL_READY * len(MODELS):
        raise RuntimeError("Frozen 1800-s response cohort changed")
    if frozen_response.duplicated(["model", "subject_id", "pm"]).any():
        raise RuntimeError("Frozen response contains duplicate cells")

    source_config = {"references": {
        "xgboost": {"output_dir": REFERENCE_PATHS["xgboost"]},
        "lightgbm": {"output_dir": REFERENCE_PATHS["lightgbm"]},
    }}
    source_matrix = _prediction_source_matrix(root_path, source_config)
    source_audit = _audit_prediction_sources(root_path, source_matrix)
    if len(source_audit) != 70 or not source_audit["valid"].all():
        raise RuntimeError("Frozen base prediction source audit failed")

    eligibility = _build_eligibility_audit(
        feasibility,
        detail,
        frozen_response,
    )
    outer_train = _build_outer_train_audit(feasibility)
    reference_audit = pd.DataFrame(references)

    fallback_lock = frozen_response[
        [
            "model",
            "outer_fold",
            "subject_id",
            "pm",
            "personalized_threshold",
            "evaluation_low",
            "evaluation_high",
            "evaluation_extreme",
            "evaluation_sample_hash",
        ]
    ].sort_values(["model", "outer_fold", "pm", "subject_id"], kind="stable")
    source_hash_columns = [
        "model",
        "outer_fold",
        "pm",
        "run_id",
        "threshold_hash",
        "test_sample_hash",
        "source_output_dir",
    ]
    scientific_payload = {
        "schema_version": SCHEMA_VERSION,
        "frozen_config": dict(config),
        "reference_hashes": {
            "xgboost": XGBOOST_HASH,
            "lightgbm": LIGHTGBM_HASH,
            "long_duration_feasibility": LONG_FEASIBILITY_HASH,
            "long_duration_response": LONG_RESPONSE_HASH,
            "probability_calibration": PROBABILITY_CALIBRATION_HASH,
        },
        "model_contracts": model_contracts,
        "base_reconstruction_lock": reconstruction_required,
        "feature_cache_identity": low_high.cache_identity,
        "fixed_fold_hash": low_high.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": low_high.protocol["temporal_pairing_hash"],
        "threshold_hashes": low_high.protocol["threshold_hashes"],
        "feasibility_1800_hash": _frame_hash(detail),
        "eligibility_audit_hash": _frame_hash(eligibility),
        "outer_train_audit_hash": _frame_hash(outer_train),
        "fixed_evaluation_cohort_hash": _frame_hash(
            eligibility.loc[eligibility["evaluation_ready"], [
                "subject_id",
                "outer_fold",
                "pm",
                "evaluation_low",
                "evaluation_high",
                "evaluation_extreme_sample_hash",
            ]]
        ),
        "frozen_median_midpoint_lock_hash": _frame_hash(fallback_lock),
        "prediction_source_matrix_hash": _frame_hash(
            source_matrix[source_hash_columns]
        ),
        "prediction_source_audit_hash": _frame_hash(
            source_audit.drop(
                columns=["prediction_file", "run_summary_file"],
                errors="ignore",
            )
        ),
    }
    protocol_hash = stable_hash(scientific_payload)
    run_matrix = build_adaptation_matrix(
        eligibility_audit=eligibility,
        outer_train_audit=outer_train,
        model_contracts=model_contracts,
        protocol_hash=protocol_hash,
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "result_status": "preregistered_candidate",
        "git_commit": _git_head(root_path),
        "protocol_hash": protocol_hash,
        "base_model_training_executed": False,
        "personalized_training_executed": False,
        "base_model_inference_executed": False,
        "personalized_inference_executed": False,
        "performance_evaluation_executed": False,
        "scientific_contract": config["scientific_contract"],
        "calibration": config["calibration"],
        "adaptation": config["adaptation"],
        "prediction_policy": config["prediction_policy"],
        "evaluation": config["evaluation"],
        "forbidden": config["forbidden"],
        "references": scientific_payload["reference_hashes"],
        "model_contracts": model_contracts,
        "feature_cache_identity": low_high.cache_identity,
        "fixed_fold_hash": low_high.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": low_high.protocol["temporal_pairing_hash"],
        "threshold_hashes": low_high.protocol["threshold_hashes"],
        "eligibility_audit_hash": scientific_payload["eligibility_audit_hash"],
        "fixed_evaluation_cohort_hash": scientific_payload[
            "fixed_evaluation_cohort_hash"
        ],
        "frozen_median_midpoint_lock_hash": scientific_payload[
            "frozen_median_midpoint_lock_hash"
        ],
        "base_reconstruction_verified_fits": 70,
        "eval_ready_participant_pm": EXPECTED_EVAL_READY,
        "eligible_participant_pm": EXPECTED_ELIGIBLE,
        "eligible_participants": EXPECTED_ELIGIBLE_PARTICIPANTS,
        "planned_personalized_fits": EXPECTED_PERSONALIZED_FITS,
        "adaptation_matrix_hash": _frame_hash(run_matrix),
    }
    return WeightedAugmentationContext(
        root=root_path,
        output_dir=output,
        config=dict(config),
        feasibility=feasibility,
        feasibility_detail=detail,
        eligibility_audit=eligibility,
        outer_train_audit=outer_train,
        frozen_response=frozen_response,
        source_matrix=source_matrix,
        source_audit=source_audit,
        reference_audit=reference_audit,
        model_contracts=model_contracts,
        protocol=protocol,
        run_matrix=run_matrix,
    )


def write_dry_run(context: WeightedAugmentationContext) -> dict[str, Any]:
    """Write preregistration artifacts without any fit, inference or metrics."""
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(
        context.output_dir / "eligibility_audit.csv",
        context.eligibility_audit,
    )
    _write_csv(
        context.output_dir / "outer_train_weight_basis.csv",
        context.outer_train_audit,
    )
    _write_csv(context.output_dir / "adaptation_matrix.csv", context.run_matrix)
    _write_csv(context.output_dir / "reference_audit.csv", context.reference_audit)
    eligible = context.eligibility_audit.loc[
        context.eligibility_audit["eligible"].astype(bool)
    ]
    counts_by_pm = {
        pm: {
            "eval_ready_participant_pm": int(
                context.eligibility_audit.loc[
                    context.eligibility_audit["pm"].eq(pm),
                    "evaluation_ready",
                ].sum()
            ),
            "eligible_participant_pm": int(eligible["pm"].eq(pm).sum()),
            "planned_personalized_fits": int(
                context.run_matrix["pm"].eq(pm).sum()
            ),
        }
        for pm in PM_NAMES
    }
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": context.protocol["protocol_hash"],
        "models": list(MODELS),
        "pm_names": list(PM_NAMES),
        "feature_count": 371,
        "fixed_lag_seconds": -10,
        "calibration_budget_seconds": CALIBRATION_BUDGET_SECONDS,
        "minimum_calibration_low": MINIMUM_LOW,
        "minimum_calibration_high": MINIMUM_HIGH,
        "eval_ready_participant_pm": int(
            context.eligibility_audit["evaluation_ready"].sum()
        ),
        "eligible_participant_pm": int(
            context.eligibility_audit["eligible"].sum()
        ),
        "eligible_participants": int(eligible["subject_id"].nunique()),
        "planned_personalized_fits": int(len(context.run_matrix)),
        "counts_by_pm": counts_by_pm,
        "reference_hashes_valid": bool(context.reference_audit["valid"].all()),
        "base_reconstruction_exact_fits": 70,
        "weight_formula_valid": True,
        "personal_low_high_mass_balanced": bool(np.allclose(
            context.run_matrix["calibration_low_total_mass"],
            context.run_matrix["calibration_high_total_mass"],
            rtol=0.0,
            atol=1e-12,
        )),
        "calibration_evaluation_overlap_count": int(
            context.eligibility_audit["calibration_evaluation_overlap"].sum()
        ),
        "record_stitching_count": int(
            context.eligibility_audit["record_stitching_used"].sum()
        ),
        "adaptation_matrix_hash": context.protocol["adaptation_matrix_hash"],
        "fixed_evaluation_cohort_hash": context.protocol[
            "fixed_evaluation_cohort_hash"
        ],
        "base_model_training_executed": False,
        "personalized_training_executed": False,
        "base_model_inference_executed": False,
        "personalized_inference_executed": False,
        "performance_evaluation_executed": False,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    (context.output_dir / "README.md").write_text(
        f"""# PM LOW/HIGH personalized weighted augmentation v1

Preregistered model-level personalization protocol. Dry-run only at this stage.

- all seven PM through one identical path
- exact `EEG(t-10s) -> PM(t)` temporal pairing
- canonical 371 engineered features
- fixed five subject-disjoint outer folds
- original outer-train Q33/Q67 LOW/HIGH thresholds
- fixed 1800-second earliest-record calibration; no stitching or extension
- middle consumes elapsed time and never enters binary fitting
- eligibility: full 1800 s + LOW>=10 + HIGH>=10 + fixed suffix ready
- method: subject-equivalent class-balanced weighted augmentation
- models: XGBoost and LightGBM with exact frozen hyperparameters
- expected operational / eligible / participants: 345 / 285 / 48
- planned personalized fits: 570
- dry-run fit, inference and performance evaluation: false

Protocol hash: `{context.protocol['protocol_hash']}`
""",
        encoding="utf-8",
    )
    return summary


def _known_spec(
    context: WeightedAugmentationContext,
    spec: Mapping[str, Any],
) -> None:
    matches = context.run_matrix.loc[
        context.run_matrix["run_id"].astype(str).eq(str(spec["run_id"]))
    ]
    if len(matches) != 1:
        raise ValueError("Run specification is not in the frozen matrix")
    expected = matches.iloc[0]
    for key in ("specification_hash", "model", "subject_id", "pm", "outer_fold"):
        if str(spec[key]) != str(expected[key]):
            raise ValueError(f"Run specification changed at {key}")


def _outer_train_arrays(
    context: WeightedAugmentationContext,
    *,
    fold: int,
    pm: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    low_high = context.feasibility.base.low_high
    cohort = low_high.cohorts[pm]
    labels = low_high.transforms[(fold, pm)].transform(
        cohort["continuous_target"].to_numpy(dtype=float)
    )
    train_mask = (
        cohort["outer_fold"].astype(int).ne(fold).to_numpy()
        & np.isfinite(labels)
    )
    metadata = cohort.loc[train_mask].reset_index(drop=True)
    positions = metadata["lag_minus_10s_feature_position"].to_numpy(dtype=int)
    matrix = np.asarray(low_high.matrix[positions], dtype=np.float32)
    return matrix, labels[train_mask].astype(np.int64), metadata


def _run_directory(
    context: WeightedAugmentationContext,
    spec: Mapping[str, Any],
) -> Path:
    return context.output_dir / "runs" / str(spec["run_id"])


def execute_run(
    context: WeightedAugmentationContext,
    spec: Mapping[str, Any],
    *,
    model_builder: Callable[..., Any] = build_model,
) -> dict[str, Any]:
    """Execute one eligible model-subject-PM fresh weighted fit."""
    _known_spec(context, spec)
    model_name = str(spec["model"])
    subject_id = str(spec["subject_id"])
    pm = str(spec["pm"])
    fold = int(spec["outer_fold"])
    if model_name not in MODELS or pm not in PM_NAMES:
        raise ValueError("Unsupported model or PM")

    x_outer, y_outer, outer_meta = _outer_train_arrays(
        context,
        fold=fold,
        pm=pm,
    )
    calibration_all, calibration, _, _ = _timeline_slices(
        context.feasibility,
        subject_id=subject_id,
        pm=pm,
    )
    if not calibration_all["record_group_id"].astype(str).nunique() == 1:
        raise RuntimeError("Calibration record stitching detected")
    y_calibration = _state_to_y(calibration["state"])
    if int(np.sum(y_calibration == 0)) != int(spec["n_calibration_low"]):
        raise RuntimeError("Runtime calibration LOW count changed")
    if int(np.sum(y_calibration == 1)) != int(spec["n_calibration_high"]):
        raise RuntimeError("Runtime calibration HIGH count changed")
    if _sample_hash(calibration["target_sample_id"].astype(str).tolist()) != str(
        spec["calibration_sample_hash"]
    ):
        raise RuntimeError("Runtime calibration sample hash changed")
    low_high = context.feasibility.base.low_high
    calibration_positions = calibration[
        "lag_minus_10s_feature_position"
    ].to_numpy(dtype=int)
    x_calibration = np.asarray(
        low_high.matrix[calibration_positions], dtype=np.float32
    )
    validate_sample_firewall(
        outer_train_sample_ids=outer_meta["target_sample_id"],
        calibration_sample_ids=calibration["target_sample_id"],
        # Evaluation rows are intentionally not materialized until after fit.
        # Their concrete sample IDs are checked below before inference.
        evaluation_sample_ids=(),
        outer_train_subjects=outer_meta["subject_id"],
        personalized_subject_id=subject_id,
    )
    # The full evaluation IDs are intentionally not loaded until after fit.
    if x_outer.shape[1] != 371 or x_calibration.shape[1] != 371:
        raise RuntimeError("Weighted fit feature count differs from 371")
    weights = np.concatenate([
        np.ones(len(y_outer), dtype=float),
        np.where(
            y_calibration == 0,
            float(spec["calibration_low_sample_weight"]),
            float(spec["calibration_high_sample_weight"]),
        ),
    ])
    x_train = np.concatenate([x_outer, x_calibration], axis=0)
    y_train = np.concatenate([y_outer, y_calibration], axis=0)
    if not np.isclose(weights[: len(y_outer)].sum(), len(y_outer), atol=1e-12):
        raise RuntimeError("Outer-train mass changed")
    if not np.isclose(
        weights[len(y_outer) :][y_calibration == 0].sum(),
        float(spec["calibration_low_total_mass"]),
        atol=1e-12,
    ):
        raise RuntimeError("Calibration LOW mass changed")
    if not np.isclose(
        weights[len(y_outer) :][y_calibration == 1].sum(),
        float(spec["calibration_high_total_mass"]),
        atol=1e-12,
    ):
        raise RuntimeError("Calibration HIGH mass changed")

    contract = context.model_contracts[model_name]
    started = time.perf_counter()
    model = model_builder(
        model_name,
        "classification",
        (371,),
        2,
        contract["params"],
    )
    model.fit(x_train, y_train, sample_weight=weights)
    training_seconds = time.perf_counter() - started

    _, _, evaluation, _ = _timeline_slices(
        context.feasibility,
        subject_id=subject_id,
        pm=pm,
    )
    evaluation_ids = evaluation["target_sample_id"].astype(str).tolist()
    if set(outer_meta["target_sample_id"].astype(str)) & set(evaluation_ids):
        raise RuntimeError("Outer-train/evaluation overlap after fit")
    if set(calibration["target_sample_id"].astype(str)) & set(evaluation_ids):
        raise RuntimeError("Calibration/evaluation overlap after fit")
    if _sample_hash(evaluation_ids) != str(spec["evaluation_sample_hash"]):
        raise RuntimeError("Runtime evaluation sample hash changed")
    y_evaluation = _state_to_y(evaluation["state"])
    evaluation_positions = evaluation[
        "lag_minus_10s_feature_position"
    ].to_numpy(dtype=int)
    x_evaluation = np.asarray(
        low_high.matrix[evaluation_positions], dtype=np.float32
    )
    probabilities = np.asarray(model.predict_proba(x_evaluation), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]), dtype=int)
    high_columns = np.flatnonzero(classes == 1)
    if probabilities.shape != (len(y_evaluation), len(classes)):
        raise RuntimeError("Personalized predict_proba shape changed")
    if len(high_columns) != 1:
        raise RuntimeError("Personalized model lacks HIGH probability")
    probability_high = probabilities[:, int(high_columns[0])]
    if not np.isfinite(probability_high).all():
        raise RuntimeError("Personalized probability contains NaN/Inf")
    prediction = (probability_high >= 0.5).astype(np.int64)

    run_dir = _run_directory(context, spec)
    prediction_frame = evaluation[[
        "target_sample_id",
        "subject_id",
        "record_id",
        "outer_fold",
    ]].copy()
    prediction_frame["feature_sample_id"] = evaluation[
        "lag_minus_10s_feature_sample_id"
    ].astype(str).to_numpy()
    prediction_frame["model"] = model_name
    prediction_frame["pm"] = pm
    prediction_frame["target_id"] = f"target_{pm}"
    prediction_frame["y_true"] = y_evaluation
    prediction_frame["y_pred"] = prediction
    prediction_frame["probability_high"] = probability_high
    _atomic_parquet(run_dir / "predictions.parquet", prediction_frame)
    summary = {
        "status": "complete",
        "result_status": "confirmatory_runtime",
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hash": spec["specification_hash"],
        "run_id": spec["run_id"],
        "model": model_name,
        "estimator": spec["estimator"],
        "model_params_hash": spec["model_params_hash"],
        "outer_fold": fold,
        "subject_id": subject_id,
        "pm": pm,
        "target_id": f"target_{pm}",
        "n_outer_train": int(len(y_outer)),
        "n_calibration": int(len(y_calibration)),
        "n_evaluation": int(len(y_evaluation)),
        "calibration_sample_hash": spec["calibration_sample_hash"],
        "evaluation_sample_hash": spec["evaluation_sample_hash"],
        "personal_total_mass": float(spec["personal_total_mass"]),
        "calibration_low_total_mass": float(
            spec["calibration_low_total_mass"]
        ),
        "calibration_high_total_mass": float(
            spec["calibration_high_total_mass"]
        ),
        "training_time_seconds": float(training_seconds),
        "personalized_training_executed": True,
        "personalized_inference_executed": True,
        "performance_evaluation_executed": False,
    }
    _atomic_json(run_dir / "run_summary.json", summary)
    return summary


def load_resumable_summary(
    context: WeightedAugmentationContext,
    spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a completed run only after strict artifact validation."""
    run_dir = _run_directory(context, spec)
    summary_path = run_dir / "run_summary.json"
    prediction_path = run_dir / "predictions.parquet"
    if not summary_path.is_file() or not prediction_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            return None
        for key in (
            "protocol_hash",
            "specification_hash",
            "run_id",
            "model",
            "subject_id",
            "pm",
        ):
            expected = (
                context.protocol["protocol_hash"]
                if key == "protocol_hash"
                else spec[key]
            )
            if str(summary.get(key)) != str(expected):
                return None
        predictions = pd.read_parquet(prediction_path)
        required = {
            "target_sample_id",
            "subject_id",
            "pm",
            "model",
            "y_true",
            "probability_high",
        }
        if required - set(predictions.columns):
            return None
        if len(predictions) != int(spec["n_evaluation_extreme"]):
            return None
        if predictions["target_sample_id"].astype(str).duplicated().any():
            return None
        if _sample_hash(
            predictions["target_sample_id"].astype(str).tolist()
        ) != str(spec["evaluation_sample_hash"]):
            return None
        probability = predictions["probability_high"].to_numpy(dtype=float)
        if not np.isfinite(probability).all():
            return None
        if np.any((probability < 0.0) | (probability > 1.0)):
            return None
        return summary
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _load_base_prediction_lookup(
    context: WeightedAugmentationContext,
) -> dict[tuple[str, int, str], pd.DataFrame]:
    lookup: dict[tuple[str, int, str], pd.DataFrame] = {}
    for source in context.source_matrix.to_dict("records"):
        path = (
            context.root
            / source["source_output_dir"]
            / "runs"
            / source["run_id"]
            / "predictions.parquet"
        )
        frame = pd.read_parquet(path).copy()
        frame["target_sample_id"] = frame["target_sample_id"].astype(str)
        lookup[(
            str(source["model"]),
            int(source["outer_fold"]),
            str(source["pm"]),
        )] = frame.set_index("target_sample_id", drop=False)
    return lookup


def _probability_metric_row(
    y_true: np.ndarray,
    probability_high: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability_high, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Operational evaluation requires both classes")
    prediction = (probability >= float(threshold)).astype(np.int64)
    low_recall = float(np.mean(prediction[y == 0] == 0))
    high_recall = float(np.mean(prediction[y == 1] == 1))
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier_score": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(
            y * np.log(clipped) + (1 - y) * np.log1p(-clipped)
        )),
        "balanced_accuracy": float((low_recall + high_recall) / 2.0),
        "macro_f1": float(f1_score(
            y,
            prediction,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )),
    }


def participant_first_aggregate(
    results: pd.DataFrame,
    *,
    applied_only: bool,
) -> pd.DataFrame:
    """Mean across available PM within participant before any grand mean."""
    frame = (
        results.loc[results["adaptation_applied"].astype(bool)].copy()
        if applied_only
        else results.copy()
    )
    rows: list[dict[str, Any]] = []
    for (model, subject_id), group in frame.groupby(
        ["model", "subject_id"], sort=True
    ):
        row: dict[str, Any] = {
            "model": model,
            "subject_id": str(subject_id),
            "n_pm": int(group["pm"].nunique()),
            "n_adaptation_applied_pm": int(
                group["adaptation_applied"].astype(bool).sum()
            ),
        }
        for column in CONTRAST_COLUMNS:
            row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_participant_first(
    participant: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        group = participant.loc[participant["model"].eq(model)]
        if group.empty or group["subject_id"].duplicated().any():
            raise ValueError(f"{model}: participant-first rows are invalid")
        for column in CONTRAST_COLUMNS:
            values = group[column].to_numpy(dtype=float)
            rng = np.random.default_rng(seed)
            indices = rng.integers(
                0, len(values), size=(replicates, len(values))
            )
            samples = values[indices].mean(axis=1)
            rows.append({
                "model": model,
                "metric_contrast": column,
                "observed_mean": float(values.mean()),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
                "participants": int(len(values)),
                "bootstrap_unit": "subject_id",
                "bootstrap_replicates": int(replicates),
                "bootstrap_seed": int(seed),
            })
    return pd.DataFrame(rows)


def aggregate_results(
    context: WeightedAugmentationContext,
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(summaries) != EXPECTED_PERSONALIZED_FITS:
        raise RuntimeError("Aggregation requires all 570 personalized fits")
    if len({str(item["run_id"]) for item in summaries}) != len(summaries):
        raise RuntimeError("Aggregation received duplicate runs")
    base_predictions = _load_base_prediction_lookup(context)
    frozen_lookup = context.frozen_response.set_index(
        ["model", "subject_id", "pm"]
    )
    spec_lookup = context.run_matrix.set_index(["model", "subject_id", "pm"])
    result_rows: list[dict[str, Any]] = []
    ready = context.eligibility_audit.loc[
        context.eligibility_audit["evaluation_ready"].astype(bool)
    ].sort_values(["outer_fold", "pm", "subject_id"], kind="stable")
    for eligibility in ready.to_dict("records"):
        subject_id = str(eligibility["subject_id"])
        pm = str(eligibility["pm"])
        fold = int(eligibility["outer_fold"])
        _, _, evaluation, _ = _timeline_slices(
            context.feasibility,
            subject_id=subject_id,
            pm=pm,
        )
        evaluation_ids = evaluation["target_sample_id"].astype(str).tolist()
        y_evaluation = _state_to_y(evaluation["state"])
        for model in MODELS:
            base = base_predictions[(model, fold, pm)]
            if set(evaluation_ids) - set(base.index.astype(str)):
                raise RuntimeError("Frozen zero-shot evaluation predictions missing")
            zero_probability = base.loc[
                evaluation_ids, "probability_high"
            ].to_numpy(dtype=float)
            if not np.array_equal(
                base.loc[evaluation_ids, "y_true"].to_numpy(dtype=int),
                y_evaluation,
            ):
                raise RuntimeError("Frozen zero-shot evaluation labels changed")
            frozen = frozen_lookup.loc[(model, subject_id, pm)]
            frozen_threshold = float(frozen["personalized_threshold"])
            eligible = bool(eligibility["eligible"])
            personalized_probability: np.ndarray | None = None
            run_id = ""
            specification_hash = ""
            if eligible:
                spec = spec_lookup.loc[(model, subject_id, pm)]
                run_id = str(spec["run_id"])
                specification_hash = str(spec["specification_hash"])
                personalized = pd.read_parquet(
                    context.output_dir / "runs" / run_id / "predictions.parquet"
                )
                personalized["target_sample_id"] = personalized[
                    "target_sample_id"
                ].astype(str)
                personalized = personalized.set_index("target_sample_id")
                if set(evaluation_ids) != set(personalized.index.astype(str)):
                    raise RuntimeError("Personalized evaluation cohort changed")
                personalized_probability = personalized.loc[
                    evaluation_ids, "probability_high"
                ].to_numpy(dtype=float)
                if not np.array_equal(
                    personalized.loc[evaluation_ids, "y_true"].to_numpy(int),
                    y_evaluation,
                ):
                    raise RuntimeError("Personalized evaluation labels changed")
            candidate_probability, candidate_threshold, probability_source, class_policy = (
                resolve_prediction_policy(
                    eligible=eligible,
                    personalized_probability=personalized_probability,
                    zero_shot_probability=zero_probability,
                    frozen_median_threshold=frozen_threshold,
                )
            )
            zero_metrics = _probability_metric_row(
                y_evaluation, zero_probability, 0.5
            )
            median_metrics = _probability_metric_row(
                y_evaluation, zero_probability, frozen_threshold
            )
            candidate_metrics = _probability_metric_row(
                y_evaluation, candidate_probability, candidate_threshold
            )
            result_rows.append({
                "model": model,
                "outer_fold": fold,
                "subject_id": subject_id,
                "pm": pm,
                "target_id": f"target_{pm}",
                "adaptation_eligible": eligible,
                "adaptation_applied": eligible,
                "run_id": run_id,
                "specification_hash": specification_hash,
                "probability_source": probability_source,
                "classification_policy": class_policy,
                "candidate_classification_threshold": candidate_threshold,
                "frozen_median_threshold": frozen_threshold,
                "evaluation_low": int(np.sum(y_evaluation == 0)),
                "evaluation_high": int(np.sum(y_evaluation == 1)),
                "evaluation_extreme": int(len(y_evaluation)),
                "evaluation_sample_hash": eligibility[
                    "evaluation_extreme_sample_hash"
                ],
                "zero_shot_roc_auc": zero_metrics["roc_auc"],
                "personalized_roc_auc": candidate_metrics["roc_auc"],
                "delta_roc_auc": (
                    candidate_metrics["roc_auc"] - zero_metrics["roc_auc"]
                ),
                "zero_shot_pr_auc": zero_metrics["pr_auc"],
                "personalized_pr_auc": candidate_metrics["pr_auc"],
                "delta_pr_auc": (
                    candidate_metrics["pr_auc"] - zero_metrics["pr_auc"]
                ),
                "zero_shot_brier": zero_metrics["brier_score"],
                "personalized_brier": candidate_metrics["brier_score"],
                "brier_improvement": (
                    zero_metrics["brier_score"]
                    - candidate_metrics["brier_score"]
                ),
                "zero_shot_log_loss": zero_metrics["log_loss"],
                "personalized_log_loss": candidate_metrics["log_loss"],
                "log_loss_improvement": (
                    zero_metrics["log_loss"]
                    - candidate_metrics["log_loss"]
                ),
                "frozen_median_balanced_accuracy": median_metrics[
                    "balanced_accuracy"
                ],
                "personalized_balanced_accuracy": candidate_metrics[
                    "balanced_accuracy"
                ],
                "delta_balanced_accuracy_vs_frozen_median": (
                    candidate_metrics["balanced_accuracy"]
                    - median_metrics["balanced_accuracy"]
                ),
                "frozen_median_macro_f1": median_metrics["macro_f1"],
                "personalized_macro_f1": candidate_metrics["macro_f1"],
                "delta_macro_f1_vs_frozen_median": (
                    candidate_metrics["macro_f1"] - median_metrics["macro_f1"]
                ),
            })
    results = pd.DataFrame(result_rows)
    if len(results) != EXPECTED_EVAL_READY * len(MODELS):
        raise RuntimeError("Operational result matrix must contain 690 rows")
    fallback = results.loc[~results["adaptation_applied"].astype(bool)]
    if not np.allclose(fallback["delta_roc_auc"], 0.0, atol=1e-15):
        raise RuntimeError("Ineligible ROC-AUC fallback is not exact")
    if not np.allclose(
        fallback["delta_balanced_accuracy_vs_frozen_median"],
        0.0,
        atol=1e-15,
    ):
        raise RuntimeError("Ineligible classification fallback is not exact")
    _write_csv(context.output_dir / "participant_pm_results.csv", results)

    participant = participant_first_aggregate(results, applied_only=False)
    applied = participant_first_aggregate(results, applied_only=True)
    _write_csv(context.output_dir / "participant_aggregate.csv", participant)
    _write_csv(
        context.output_dir / "participant_aggregate_applied_only.csv", applied
    )
    summaries_by_estimand = []
    for estimand, frame in (
        ("operational_all_eval_ready_with_fallback", participant),
        ("adaptation_applied_only", applied),
    ):
        for model in MODELS:
            group = frame.loc[frame["model"].eq(model)]
            row: dict[str, Any] = {
                "estimand": estimand,
                "model": model,
                "participants": int(group["subject_id"].nunique()),
                "participant_pm_rows": int(
                    len(results.loc[
                        results["model"].eq(model)
                        & (
                            results["adaptation_applied"].astype(bool)
                            if estimand == "adaptation_applied_only"
                            else True
                        )
                    ])
                ),
            }
            for column in CONTRAST_COLUMNS:
                row[f"{column}_mean"] = float(group[column].mean())
                row[f"{column}_median"] = float(group[column].median())
            summaries_by_estimand.append(row)
    summary = pd.DataFrame(summaries_by_estimand)
    _write_csv(context.output_dir / "summary_by_estimand.csv", summary)

    pm_rows = []
    for (model, pm), group in results.groupby(["model", "pm"], sort=True):
        row: dict[str, Any] = {
            "model": model,
            "pm": pm,
            "participant_rows": int(len(group)),
            "adaptation_applied": int(group["adaptation_applied"].sum()),
        }
        for column in CONTRAST_COLUMNS:
            row[f"{column}_mean"] = float(group[column].mean())
        pm_rows.append(row)
    _write_csv(context.output_dir / "summary_by_pm.csv", pd.DataFrame(pm_rows))
    _write_csv(
        context.output_dir / "bootstrap_operational.csv",
        bootstrap_participant_first(participant),
    )
    _write_csv(
        context.output_dir / "bootstrap_applied_only.csv",
        bootstrap_participant_first(applied),
    )

    completed_protocol = dict(context.protocol)
    completed_protocol.update({
        "result_status": "confirmatory_complete",
        "personalized_training_executed": True,
        "personalized_inference_executed": True,
        "performance_evaluation_executed": True,
        "completed_personalized_fits": EXPECTED_PERSONALIZED_FITS,
        "operational_participant_pm_model_rows": int(len(results)),
        "result_hash": _frame_hash(results),
    })
    _atomic_json(context.output_dir / "protocol.json", completed_protocol)
    pooled = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": context.protocol["protocol_hash"],
        "result_status": "confirmatory_complete",
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "personalized_training_executed": True,
        "personalized_inference_executed": True,
        "performance_evaluation_executed": True,
        "eval_ready_participant_pm": EXPECTED_EVAL_READY,
        "eligible_participant_pm": EXPECTED_ELIGIBLE,
        "eligible_participants": EXPECTED_ELIGIBLE_PARTICIPANTS,
        "completed_personalized_fits": EXPECTED_PERSONALIZED_FITS,
        "operational_participant_pm_model_rows": int(len(results)),
    }
    _atomic_json(context.output_dir / "pooled_summary.json", pooled)
    return pooled


def run_experiment(
    context: WeightedAugmentationContext,
    *,
    resume: bool,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    trained = 0
    reused = 0
    for spec in context.run_matrix.to_dict("records"):
        existing = load_resumable_summary(context, spec) if resume else None
        if existing is not None:
            summaries.append(existing)
            reused += 1
            continue
        run_dir = _run_directory(context, spec)
        if run_dir.exists() and not resume:
            raise FileExistsError(
                f"Run directory exists; use --resume after audit: {run_dir}"
            )
        summaries.append(execute_run(context, spec))
        trained += 1
    pooled = aggregate_results(context, summaries)
    return {**pooled, "trained": trained, "reused": reused}


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "EXPECTED_ELIGIBLE",
    "EXPECTED_ELIGIBLE_PARTICIPANTS",
    "EXPECTED_EVAL_READY",
    "EXPECTED_PERSONALIZED_FITS",
    "MODELS",
    "WeightedAugmentationContext",
    "aggregate_results",
    "bootstrap_participant_first",
    "build_adaptation_matrix",
    "execute_run",
    "is_weighted_augmentation_eligible",
    "load_config",
    "load_resumable_summary",
    "participant_first_aggregate",
    "prepare_protocol",
    "resolve_prediction_policy",
    "run_experiment",
    "subject_equivalent_class_balanced_weights",
    "validate_sample_firewall",
    "write_dry_run",
]
