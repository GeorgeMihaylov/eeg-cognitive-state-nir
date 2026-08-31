from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bench.automl.scientific import nested_extremes as nested
from bench.experiments import pm_low_high_nested_automl as experiment
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    PM_NAMES,
    _sample_hash,
)


CONFIG_PATH = (
    "experiments/pm_diagnostics/pm_low_high_nested_automl_v1.json"
)


def _anchors() -> dict[str, dict]:
    return {
        "xgboost": {
            "n_estimators": 200,
            "n_jobs": 4,
            "random_state": 42,
        },
        "lightgbm": {
            "n_estimators": 200,
            "n_jobs": 4,
            "random_state": 42,
        },
    }


def _synthetic_nested_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    for subject_index in range(15):
        subject = f"s{subject_index:02d}"
        outer_fold = subject_index % 5 + 1
        for value_index, value in enumerate((0.0, 1.0, 2.0, 3.0, 4.0, 5.0)):
            row = {
                "sample_id": f"{subject}_{value_index}",
                "subject_id": subject,
                "outer_fold": outer_fold,
            }
            for pm in PM_NAMES:
                row[f"target_{pm}"] = value
            rows.append(row)
    full = pd.DataFrame(rows)
    cohorts = {}
    for pm in PM_NAMES:
        cohort = full[["sample_id", "subject_id", "outer_fold"]].copy()
        cohort["target_sample_id"] = cohort["sample_id"]
        cohort["continuous_target"] = full[f"target_{pm}"].to_numpy()
        cohorts[pm] = cohort
    audit = pd.DataFrame([
        {
            "outer_fold": fold,
            "pm": pm,
            "target_id": f"target_{pm}",
            "q_low": 1.5,
            "q_high": 3.5,
            "threshold_hash": f"outer-{fold}-{pm}",
        }
        for fold in range(1, 6)
        for pm in PM_NAMES
    ])
    return full, cohorts, audit


def _nested_artifacts():
    config = experiment.load_config(CONFIG_PATH)
    candidates = nested.build_candidate_portfolio(config, anchors=_anchors())
    full, cohorts, outer_audit = _synthetic_nested_inputs()
    splits = nested.build_inner_subject_splits(
        full[["sample_id", "subject_id", "outer_fold"]],
        outer_folds=range(1, 6),
        inner_folds=3,
    )
    thresholds = nested.build_nested_threshold_provenance(
        full=full,
        cohorts=cohorts,
        inner_splits=splits,
        outer_threshold_audit=outer_audit,
    )
    return config, candidates, full, cohorts, outer_audit, splits, thresholds


def test_frozen_config_is_strict_and_forbidden_switches_stay_enabled(
    tmp_path: Path,
) -> None:
    config = experiment.load_config(CONFIG_PATH)
    assert all(config["forbidden"].values())
    changed = deepcopy(config)
    changed["candidate_generation"]["sampler_seed"] = 43
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="config hash changed"):
        experiment.load_config(path)


def test_candidate_portfolio_is_deterministic_distinct_and_includes_anchors() -> None:
    config = experiment.load_config(CONFIG_PATH)
    first = nested.build_candidate_portfolio(config, anchors=_anchors())
    second = nested.build_candidate_portfolio(config, anchors=_anchors())

    pd.testing.assert_frame_equal(first, second)
    assert nested.candidate_matrix_hash(first) == nested.candidate_matrix_hash(second)
    assert first.groupby("model_family").size().to_dict() == {
        "lightgbm": 13,
        "xgboost": 13,
    }
    assert first["candidate_hash"].nunique() == 26
    assert set(first.loc[first["candidate_kind"].eq("frozen_anchor"), "model_family"]) == {
        "xgboost",
        "lightgbm",
    }


def test_inner_subject_folds_cover_each_outer_train_subject_once_without_leakage() -> None:
    full, _, _ = _synthetic_nested_inputs()
    splits = nested.build_inner_subject_splits(
        full[["sample_id", "subject_id", "outer_fold"]],
        outer_folds=range(1, 6),
        inner_folds=3,
    )

    assert len(splits) == 15
    assert splits["subject_overlap_count"].eq(0).all()
    assert splits["outer_test_leakage_count"].eq(0).all()
    for outer_fold, group in splits.groupby("outer_fold"):
        expected = set(
            full.loc[full["outer_fold"].ne(outer_fold), "subject_id"].astype(str)
        )
        validation = [
            subject
            for value in group["validation_subjects"]
            for subject in value.split("|")
        ]
        assert set(validation) == expected
        assert len(validation) == len(set(validation))


def test_nested_thresholds_are_inner_train_only_and_all_105_cells_are_valid() -> None:
    _, _, _, _, _, _, thresholds = _nested_artifacts()

    assert len(thresholds) == 105
    assert set(thresholds["pm"]) == set(PM_NAMES)
    assert thresholds["threshold_source"].eq("inner_train_only").all()
    assert not thresholds["validation_labels_used_for_thresholds"].any()
    assert not thresholds["outer_threshold_reused_inside_inner_cv"].any()
    assert thresholds["class_complete_train"].all()
    assert thresholds["class_complete_validation"].all()


def test_validation_labels_cannot_change_their_own_inner_thresholds() -> None:
    _, _, full, cohorts, outer_audit, splits, original = _nested_artifacts()
    split = splits.iloc[0]
    validation_subjects = split["validation_subjects"].split("|")
    changed_full = full.copy()
    changed_cohorts = {pm: frame.copy() for pm, frame in cohorts.items()}
    for pm in PM_NAMES:
        mask = changed_full["subject_id"].isin(validation_subjects)
        pattern = np.tile([0.0, 0.0, 0.0, 5.0, 5.0, 5.0], len(validation_subjects))
        changed_full.loc[mask, f"target_{pm}"] = pattern
        cohort_mask = changed_cohorts[pm]["subject_id"].isin(validation_subjects)
        changed_cohorts[pm].loc[cohort_mask, "continuous_target"] = pattern
    changed = nested.build_nested_threshold_provenance(
        full=changed_full,
        cohorts=changed_cohorts,
        inner_splits=splits,
        outer_threshold_audit=outer_audit,
    )
    key = (
        original["outer_fold"].eq(int(split["outer_fold"]))
        & original["inner_fold"].eq(int(split["inner_fold"]))
    )
    columns = ["pm", "q_low", "q_high", "fit_sample_hash", "threshold_hash"]
    pd.testing.assert_frame_equal(
        original.loc[key, columns].reset_index(drop=True),
        changed.loc[key, columns].reset_index(drop=True),
    )


def test_run_matrices_have_2730_inner_fits_130_evaluations_and_seven_pm() -> None:
    _, candidates, _, _, _, _, thresholds = _nested_artifacts()
    runs = nested.build_inner_run_matrix(
        candidates=candidates,
        thresholds=thresholds,
        protocol_hash="protocol",
    )
    evaluations = nested.candidate_evaluation_matrix(
        candidates,
        outer_folds=range(1, 6),
        protocol_hash="protocol",
    )

    assert len(runs) == 2730
    assert runs["run_id"].nunique() == 2730
    assert set(runs["pm"]) == set(PM_NAMES)
    assert set(runs.groupby(["outer_fold", "candidate_id"]).size()) == {21}
    assert len(evaluations) == 130
    assert set(evaluations.groupby("outer_fold").size()) == {26}


def test_participant_first_objective_means_pm_before_participants() -> None:
    predictions = pd.DataFrame({
        "subject_id": ["s1"] * 8 + ["s2"] * 4,
        "pm": ["attention"] * 4 + ["focus"] * 4 + ["attention"] * 4,
        "y_true": [0, 0, 1, 1] * 3,
        "y_pred": [0, 0, 1, 1] + [1, 1, 0, 0] + [0, 0, 1, 1],
    })
    result = nested.participant_first_objective(predictions)

    # s1 mean PM BA=(1+0)/2=.5, s2 BA=1; participant-first grand mean=.75.
    assert result["participant_first_balanced_accuracy"] == pytest.approx(0.75)
    assert result["participants"] == 2
    assert result["participant_pm_rows"] == 3


def test_tie_breaking_is_primary_then_secondary_then_candidate_id() -> None:
    scores = pd.DataFrame([
        {"candidate_id": "z", "participant_first_balanced_accuracy": 0.8,
         "participant_first_macro_f1": 0.9},
        {"candidate_id": "b", "participant_first_balanced_accuracy": 0.8,
         "participant_first_macro_f1": 0.91},
        {"candidate_id": "a", "participant_first_balanced_accuracy": 0.8,
         "participant_first_macro_f1": 0.91},
    ])
    assert nested.select_best_candidate(scores)["candidate_id"] == "a"


def test_one_selected_candidate_is_shared_across_all_pm_and_final_fits_are_35() -> None:
    _, candidates, _, _, outer_audit, _, _ = _nested_artifacts()
    selected_rows = []
    for fold in range(1, 6):
        candidate = candidates.iloc[fold]
        selected_rows.append({
            "outer_fold": fold,
            "candidate_id": candidate["candidate_id"],
            "model_family": candidate["model_family"],
            "candidate_hash": candidate["candidate_hash"],
        })
    selection = {
        "status": "inner_selection_frozen",
        "protocol_hash": "protocol",
        "selection_hash": "selection",
        "selected_candidates": selected_rows,
    }
    expanded_audit = outer_audit.copy()
    expanded_audit["n_train_retained"] = 10
    expanded_audit["n_test_retained"] = 5
    expanded_audit["train_retained_sample_hash"] = "train"
    expanded_audit["test_retained_sample_hash"] = "test"
    context = SimpleNamespace(
        protocol={"protocol_hash": "protocol"},
        candidates=candidates,
        reference=SimpleNamespace(threshold_audit=expanded_audit),
    )
    matrix = experiment.build_final_run_matrix(context, selection)

    assert len(matrix) == 35
    assert matrix.groupby("outer_fold")["candidate_id"].nunique().eq(1).all()
    assert matrix.groupby("outer_fold")["pm"].nunique().eq(7).all()


def test_resume_rejects_partial_or_hash_mismatched_inner_artifacts(
    tmp_path: Path,
) -> None:
    spec = {
        "specification_hash": "spec",
        "run_id": "run",
        "n_validation": 2,
        "validation_sample_hash": _sample_hash(["a", "b"]),
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = pd.DataFrame({
        "target_sample_id": ["a", "b"],
        "subject_id": ["s1", "s2"],
        "pm": ["focus", "focus"],
        "y_true": [0, 1],
        "y_pred": [0, 1],
        "probability_high": [0.1, 0.9],
    })
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    assert nested.resumable_inner_summary(
        run_dir, specification=spec, protocol_hash="protocol"
    ) is None
    summary = {
        "status": "complete",
        "protocol_hash": "protocol",
        "specification_hash": "spec",
        "run_id": "run",
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    assert nested.resumable_inner_summary(
        run_dir, specification=spec, protocol_hash="protocol"
    ) == summary
    summary["specification_hash"] = "wrong"
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    assert nested.resumable_inner_summary(
        run_dir, specification=spec, protocol_hash="protocol"
    ) is None


def test_hashes_are_deterministic() -> None:
    _, candidates, _, _, _, splits, thresholds = _nested_artifacts()
    assert nested.candidate_matrix_hash(candidates) == nested.candidate_matrix_hash(
        candidates.copy()
    )
    assert nested.inner_split_hash(splits) == nested.inner_split_hash(splits.copy())
    assert nested.threshold_provenance_hash(
        thresholds
    ) == nested.threshold_provenance_hash(thresholds.copy())


def test_dry_run_writer_executes_zero_fit_inference_or_performance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, candidates, _, _, _, splits, thresholds = _nested_artifacts()
    evaluations = nested.candidate_evaluation_matrix(
        candidates, outer_folds=range(1, 6), protocol_hash="protocol"
    )
    runs = nested.build_inner_run_matrix(
        candidates=candidates,
        thresholds=thresholds,
        protocol_hash="protocol",
    )
    final_plan = pd.DataFrame([{"outer_fold": fold, "pm": pm}
                               for fold in range(1, 6) for pm in PM_NAMES])
    context = experiment.NestedAutoMLContext(
        root=tmp_path,
        output_dir=tmp_path / "out",
        config={},
        reference=SimpleNamespace(matrix=np.empty((1, 371))),
        reference_protocols={},
        reference_audit=pd.DataFrame([{"valid": True}]),
        candidates=candidates,
        inner_splits=splits,
        thresholds=thresholds,
        candidate_evaluations=evaluations,
        inner_run_matrix=runs,
        final_outer_plan=final_plan,
        protocol={
            "protocol_hash": "protocol",
            "candidate_matrix_hash": nested.candidate_matrix_hash(candidates),
            "inner_split_hash": nested.inner_split_hash(splits),
            "threshold_provenance_hash": nested.threshold_provenance_hash(thresholds),
        },
    )
    monkeypatch.setattr(
        experiment,
        "build_model",
        lambda *args, **kwargs: pytest.fail("dry-run constructed a model"),
    )

    summary = experiment.write_dry_run(context)

    assert summary["planned_inner_fits"] == 2730
    assert summary["planned_final_outer_fits"] == 35
    assert summary["valid_inner_fold_pm_cells"] == 105
    assert summary["candidate_model_training_executed"] is False
    assert summary["candidate_model_inference_executed"] is False
    assert summary["outer_model_training_executed"] is False
    assert summary["outer_model_inference_executed"] is False
    assert summary["performance_evaluation_executed"] is False
    assert not (context.output_dir / "inner_runs").exists()
    assert not (context.output_dir / "final_runs").exists()
