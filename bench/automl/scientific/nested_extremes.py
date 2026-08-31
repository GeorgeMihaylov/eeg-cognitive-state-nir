"""Reusable deterministic nested-CV primitives for LOW/HIGH PM experiments."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    PM_NAMES,
    _sample_hash,
    fit_extreme_thresholds,
    stable_hash,
)
from bench.validation.cross_val import deterministic_group_kfold_indices


MODEL_FAMILIES = ("xgboost", "lightgbm")


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _split_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item for item in value.split("|") if item)
    return tuple(str(item) for item in value)


def build_candidate_portfolio(
    config: Mapping[str, Any],
    *,
    anchors: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Build one frozen anchor plus 12 sampled Cartesian candidates per model."""
    generation = config["candidate_generation"]
    seed = int(generation["sampler_seed"])
    sampled_count = int(generation["sampled_candidates_per_model"])
    expected_per_model = int(generation["total_candidates_per_model"])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        model_config = config["models"][family]
        fixed = dict(model_config["fixed_parameters"])
        anchor = dict(anchors[family])
        anchor_signature = stable_hash({"model": family, "params": anchor})
        rows.append({
            "candidate_id": f"{family}__anchor",
            "model_family": family,
            "candidate_kind": "frozen_anchor",
            "sample_order": 0,
            "params_json": _json(anchor),
            "candidate_hash": anchor_signature,
            "reference_protocol_hash": model_config[
                "reference_protocol_hash"
            ],
        })

        search_space = model_config["search_space"]
        parameter_names = tuple(search_space)
        combinations = [
            dict(zip(parameter_names, values))
            for values in product(*(search_space[name] for name in parameter_names))
        ]
        if len(combinations) < sampled_count:
            raise ValueError(f"{family}: search space is smaller than sample count")
        order = rng.permutation(len(combinations))
        accepted: list[tuple[dict[str, Any], str]] = []
        seen = {anchor_signature}
        for index in order:
            params = {**fixed, **combinations[int(index)]}
            signature = stable_hash({"model": family, "params": params})
            if signature in seen:
                continue
            seen.add(signature)
            accepted.append((params, signature))
            if len(accepted) == sampled_count:
                break
        if len(accepted) != sampled_count:
            raise RuntimeError(f"{family}: could not create 12 distinct candidates")
        for sample_order, (params, signature) in enumerate(accepted, start=1):
            rows.append({
                "candidate_id": f"{family}__sample_{sample_order:03d}",
                "model_family": family,
                "candidate_kind": "sampled_cartesian",
                "sample_order": sample_order,
                "params_json": _json(params),
                "candidate_hash": signature,
                "reference_protocol_hash": model_config[
                    "reference_protocol_hash"
                ],
            })
    frame = pd.DataFrame(rows).sort_values(
        ["model_family", "sample_order"], kind="stable"
    ).reset_index(drop=True)
    counts = frame.groupby("model_family").size().to_dict()
    if counts != {family: expected_per_model for family in MODEL_FAMILIES}:
        raise RuntimeError(f"Candidate counts changed: {counts}")
    if frame["candidate_id"].duplicated().any():
        raise RuntimeError("Candidate IDs are not unique")
    if frame["candidate_hash"].duplicated().any():
        raise RuntimeError("Candidate configurations are not distinct")
    return frame


def candidate_matrix_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("candidate_id", kind="stable")
    return stable_hash({"candidates": ordered.to_dict("records")})


def build_inner_subject_splits(
    rows: pd.DataFrame,
    *,
    outer_folds: Sequence[int],
    inner_folds: int,
) -> pd.DataFrame:
    """Create one deterministic subject split shared by all PM per outer fold."""
    required = {"sample_id", "subject_id", "outer_fold"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"Split rows are missing columns: {sorted(missing)}")
    result: list[dict[str, Any]] = []
    for outer_fold in outer_folds:
        outer_train = rows.loc[
            rows["outer_fold"].astype(int).ne(int(outer_fold))
        ].copy()
        outer_test = rows.loc[
            rows["outer_fold"].astype(int).eq(int(outer_fold))
        ].copy()
        outer_test_subjects = set(outer_test["subject_id"].astype(str))
        splits = deterministic_group_kfold_indices(
            outer_train["subject_id"].astype(str).to_numpy(),
            n_splits=int(inner_folds),
        )
        validation_counts: dict[str, int] = {}
        for inner_fold, (train_index, validation_index) in enumerate(
            splits, start=1
        ):
            train = outer_train.iloc[train_index]
            validation = outer_train.iloc[validation_index]
            train_subjects = sorted(train["subject_id"].astype(str).unique())
            validation_subjects = sorted(
                validation["subject_id"].astype(str).unique()
            )
            overlap = set(train_subjects) & set(validation_subjects)
            outer_leakage = (
                set(train_subjects) | set(validation_subjects)
            ) & outer_test_subjects
            if overlap or outer_leakage:
                raise RuntimeError("Nested subject leakage detected")
            for subject in validation_subjects:
                validation_counts[subject] = validation_counts.get(subject, 0) + 1
            payload = {
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "train_subjects": train_subjects,
                "validation_subjects": validation_subjects,
                "outer_test_subjects": sorted(outer_test_subjects),
                "train_sample_hash": _sample_hash(train["sample_id"]),
                "validation_sample_hash": _sample_hash(validation["sample_id"]),
            }
            result.append({
                **payload,
                "n_train_subjects": len(train_subjects),
                "n_validation_subjects": len(validation_subjects),
                "n_outer_test_subjects": len(outer_test_subjects),
                "n_train_rows": int(len(train)),
                "n_validation_rows": int(len(validation)),
                "train_subjects": "|".join(train_subjects),
                "validation_subjects": "|".join(validation_subjects),
                "outer_test_subjects": "|".join(sorted(outer_test_subjects)),
                "subject_overlap_count": 0,
                "outer_test_leakage_count": 0,
                "split_hash": stable_hash(payload),
            })
        expected_subjects = set(outer_train["subject_id"].astype(str))
        if set(validation_counts) != expected_subjects:
            raise RuntimeError("Inner validation does not cover outer-train subjects")
        if set(validation_counts.values()) != {1}:
            raise RuntimeError("An outer-train subject validates more than once")
    frame = pd.DataFrame(result).sort_values(
        ["outer_fold", "inner_fold"], kind="stable"
    ).reset_index(drop=True)
    if len(frame) != len(tuple(outer_folds)) * int(inner_folds):
        raise RuntimeError("Unexpected inner split count")
    return frame


def inner_split_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["outer_fold", "inner_fold"], kind="stable"
    )
    return stable_hash({"inner_splits": ordered.to_dict("records")})


def build_nested_threshold_provenance(
    *,
    full: pd.DataFrame,
    cohorts: Mapping[str, pd.DataFrame],
    inner_splits: pd.DataFrame,
    outer_threshold_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Fit every Q33/Q67 transform exclusively on inner-train targets."""
    outer_lookup = outer_threshold_audit.set_index(["outer_fold", "pm"])
    records: list[dict[str, Any]] = []
    for split in inner_splits.to_dict("records"):
        outer_fold = int(split["outer_fold"])
        inner_fold = int(split["inner_fold"])
        train_subjects = set(_split_values(split["train_subjects"]))
        validation_subjects = set(_split_values(split["validation_subjects"]))
        outer_train_full = full["outer_fold"].astype(int).ne(outer_fold)
        for pm in PM_NAMES:
            target_id = f"target_{pm}"
            fit_mask = (
                outer_train_full
                & full["subject_id"].astype(str).isin(train_subjects)
            )
            fit_values = pd.to_numeric(
                full.loc[fit_mask, target_id], errors="coerce"
            )
            fit_array = fit_values.to_numpy(dtype=float)
            fit_valid = np.isfinite(fit_array)
            thresholds = fit_extreme_thresholds(fit_array[fit_valid])
            fit_ids = full.loc[fit_mask, "sample_id"].to_numpy()[fit_valid]

            cohort = cohorts[pm]
            protected_outer_train = cohort["outer_fold"].astype(int).ne(outer_fold)
            train_before = (
                protected_outer_train
                & cohort["subject_id"].astype(str).isin(train_subjects)
            ).to_numpy()
            validation_before = (
                protected_outer_train
                & cohort["subject_id"].astype(str).isin(validation_subjects)
            ).to_numpy()
            labels = thresholds.transform(cohort["continuous_target"])
            train_low = train_before & (labels == 0)
            train_high = train_before & (labels == 1)
            validation_low = validation_before & (labels == 0)
            validation_high = validation_before & (labels == 1)
            train_keep = train_low | train_high
            validation_keep = validation_low | validation_high
            if sorted(np.unique(labels[train_keep]).astype(int)) != [0, 1]:
                raise ValueError("An inner-train cell is not class-complete")
            if sorted(np.unique(labels[validation_keep]).astype(int)) != [0, 1]:
                raise ValueError("An inner-validation cell is not class-complete")
            if set(cohort.loc[train_before, "subject_id"].astype(str)) & set(
                cohort.loc[validation_before, "subject_id"].astype(str)
            ):
                raise RuntimeError("Inner threshold cohort has subject leakage")
            outer = outer_lookup.loc[(outer_fold, pm)]
            threshold_payload = {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "pm": pm,
                "fit_scope": "inner_train_continuous_complete_cases",
                "fit_sample_hash": _sample_hash(fit_ids),
                "q_low": thresholds.q_low,
                "q_high": thresholds.q_high,
                "train_subject_hash": _sample_hash(sorted(train_subjects)),
                "validation_subject_hash": _sample_hash(
                    sorted(validation_subjects)
                ),
            }
            records.append({
                **threshold_payload,
                "target_id": target_id,
                "threshold_source": "inner_train_only",
                "validation_threshold_policy": "reuse_inner_train_unchanged",
                "threshold_hash": stable_hash(threshold_payload),
                "outer_threshold_hash": str(outer["threshold_hash"]),
                "outer_threshold_reused_inside_inner_cv": False,
                "validation_labels_used_for_thresholds": False,
                "outer_test_labels_used": False,
                "n_threshold_fit": int(fit_valid.sum()),
                "n_inner_train_before": int(train_before.sum()),
                "n_inner_train_low": int(train_low.sum()),
                "n_inner_train_high": int(train_high.sum()),
                "n_inner_validation_before": int(validation_before.sum()),
                "n_inner_validation_low": int(validation_low.sum()),
                "n_inner_validation_high": int(validation_high.sum()),
                "inner_train_retained_sample_hash": _sample_hash(
                    cohort.loc[train_keep, "target_sample_id"]
                ),
                "inner_validation_retained_sample_hash": _sample_hash(
                    cohort.loc[validation_keep, "target_sample_id"]
                ),
                "split_hash": split["split_hash"],
                "class_complete_train": True,
                "class_complete_validation": True,
                "subject_overlap_count": 0,
            })
    frame = pd.DataFrame(records).sort_values(
        ["outer_fold", "inner_fold", "pm"], kind="stable"
    ).reset_index(drop=True)
    if len(frame) != 105:
        raise RuntimeError(f"Expected 105 inner fold-PM cells, got {len(frame)}")
    if not frame["class_complete_train"].all():
        raise RuntimeError("An inner-train cell is missing a class")
    if not frame["class_complete_validation"].all():
        raise RuntimeError("An inner-validation cell is missing a class")
    return frame


def threshold_provenance_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["outer_fold", "inner_fold", "pm"], kind="stable"
    )
    return stable_hash({"nested_thresholds": ordered.to_dict("records")})


def build_inner_run_matrix(
    *,
    candidates: pd.DataFrame,
    thresholds: pd.DataFrame,
    protocol_hash: str,
) -> pd.DataFrame:
    """Freeze one deterministic specification per candidate/fold/inner-fold/PM."""
    rows: list[dict[str, Any]] = []
    for threshold in thresholds.to_dict("records"):
        for candidate in candidates.to_dict("records"):
            spec = {
                "outer_fold": int(threshold["outer_fold"]),
                "inner_fold": int(threshold["inner_fold"]),
                "pm": str(threshold["pm"]),
                "target_id": str(threshold["target_id"]),
                "candidate_id": str(candidate["candidate_id"]),
                "model_family": str(candidate["model_family"]),
                "candidate_hash": str(candidate["candidate_hash"]),
                "params_json": str(candidate["params_json"]),
                "threshold_hash": str(threshold["threshold_hash"]),
                "q_low": float(threshold["q_low"]),
                "q_high": float(threshold["q_high"]),
                "n_train": int(
                    threshold["n_inner_train_low"]
                    + threshold["n_inner_train_high"]
                ),
                "n_validation": int(
                    threshold["n_inner_validation_low"]
                    + threshold["n_inner_validation_high"]
                ),
                "train_sample_hash": str(
                    threshold["inner_train_retained_sample_hash"]
                ),
                "validation_sample_hash": str(
                    threshold["inner_validation_retained_sample_hash"]
                ),
                "split_hash": str(threshold["split_hash"]),
            }
            specification_hash = stable_hash({
                "protocol_hash": protocol_hash,
                "inner_fit": spec,
            })
            spec["specification_hash"] = specification_hash
            spec["run_id"] = (
                f"outer_{spec['outer_fold']:02d}__{spec['candidate_id']}__"
                f"inner_{spec['inner_fold']:02d}__{spec['pm']}__"
                f"{specification_hash[:12]}"
            )
            rows.append(spec)
    frame = pd.DataFrame(rows).sort_values(
        ["outer_fold", "candidate_id", "inner_fold", "pm"], kind="stable"
    ).reset_index(drop=True)
    if len(frame) != 2730 or frame["run_id"].duplicated().any():
        raise RuntimeError("Expected exactly 2730 unique inner-fit specifications")
    return frame


def candidate_evaluation_matrix(
    candidates: pd.DataFrame,
    *,
    outer_folds: Sequence[int],
    protocol_hash: str,
) -> pd.DataFrame:
    rows = []
    for outer_fold in outer_folds:
        for candidate in candidates.to_dict("records"):
            payload = {
                "outer_fold": int(outer_fold),
                "candidate_id": candidate["candidate_id"],
                "model_family": candidate["model_family"],
                "candidate_hash": candidate["candidate_hash"],
                "expected_inner_fits": 21,
            }
            payload["evaluation_hash"] = stable_hash({
                "protocol_hash": protocol_hash,
                "candidate_evaluation": payload,
            })
            rows.append(payload)
    frame = pd.DataFrame(rows)
    if len(frame) != 130:
        raise RuntimeError("Expected 130 outer-fold candidate evaluations")
    if set(frame.groupby("outer_fold").size()) != {26}:
        raise RuntimeError("Every outer fold must evaluate 26 candidates")
    return frame


def participant_first_objective(predictions: pd.DataFrame) -> dict[str, Any]:
    """Score OOF predictions per participant-PM, then participant, then cohort."""
    required = {"subject_id", "pm", "y_true", "y_pred"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    pm_rows: list[dict[str, Any]] = []
    for (subject_id, pm), group in predictions.groupby(
        ["subject_id", "pm"], sort=True
    ):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_pred = group["y_pred"].to_numpy(dtype=int)
        low = y_true == 0
        high = y_true == 1
        recalls = [
            float(np.mean(y_pred[low] == 0)) if low.any() else np.nan,
            float(np.mean(y_pred[high] == 1)) if high.any() else np.nan,
        ]
        pm_rows.append({
            "subject_id": str(subject_id),
            "pm": str(pm),
            "balanced_accuracy": float(np.nanmean(recalls)),
            "macro_f1": float(f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )),
            "n_rows": int(len(group)),
        })
    per_pm = pd.DataFrame(pm_rows)
    per_participant = (
        per_pm.groupby("subject_id", sort=True)
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            n_pm=("pm", "nunique"),
        )
        .reset_index()
    )
    return {
        "participant_first_balanced_accuracy": float(
            per_participant["balanced_accuracy"].mean()
        ),
        "participant_first_macro_f1": float(
            per_participant["macro_f1"].mean()
        ),
        "participants": int(len(per_participant)),
        "participant_pm_rows": int(len(per_pm)),
        "per_participant_pm": per_pm,
        "per_participant": per_participant,
    }


def select_best_candidate(scores: pd.DataFrame) -> pd.Series:
    """Apply the preregistered primary, secondary and lexical tie break."""
    required = {
        "candidate_id",
        "participant_first_balanced_accuracy",
        "participant_first_macro_f1",
    }
    if required - set(scores):
        raise ValueError("Candidate score table is incomplete")
    ordered = scores.sort_values(
        [
            "participant_first_balanced_accuracy",
            "participant_first_macro_f1",
            "candidate_id",
        ],
        ascending=[False, False, True],
        kind="stable",
    )
    if ordered.empty:
        raise ValueError("Cannot select from an empty candidate score table")
    return ordered.iloc[0]


def resumable_inner_summary(
    run_dir: str | Path,
    *,
    specification: Mapping[str, Any],
    protocol_hash: str,
) -> dict[str, Any] | None:
    """Accept only a complete summary plus matching prediction artifact."""
    directory = Path(run_dir)
    summary_path = directory / "run_summary.json"
    prediction_path = directory / "predictions.parquet"
    if not summary_path.is_file() or not prediction_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "protocol_hash": protocol_hash,
            "specification_hash": specification["specification_hash"],
            "run_id": specification["run_id"],
        }
        if summary.get("status") != "complete":
            return None
        if any(str(summary.get(key)) != str(value) for key, value in expected.items()):
            return None
        predictions = pd.read_parquet(prediction_path)
        columns = {
            "target_sample_id",
            "subject_id",
            "pm",
            "y_true",
            "y_pred",
            "probability_high",
        }
        if columns - set(predictions):
            return None
        if len(predictions) != int(specification["n_validation"]):
            return None
        if predictions["target_sample_id"].astype(str).duplicated().any():
            return None
        if _sample_hash(predictions["target_sample_id"].astype(str)) != str(
            specification["validation_sample_hash"]
        ):
            return None
        probability = predictions["probability_high"].to_numpy(dtype=float)
        if not np.isfinite(probability).all():
            return None
        if np.any((probability < 0.0) | (probability > 1.0)):
            return None
        return summary
    except (OSError, ValueError, KeyError, TypeError):
        return None


__all__ = [
    "MODEL_FAMILIES",
    "build_candidate_portfolio",
    "build_inner_run_matrix",
    "build_inner_subject_splits",
    "build_nested_threshold_provenance",
    "candidate_evaluation_matrix",
    "candidate_matrix_hash",
    "inner_split_hash",
    "participant_first_objective",
    "resumable_inner_summary",
    "select_best_candidate",
    "threshold_provenance_hash",
]
