"""Tests for the unified personalization protocol planner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.personalization_calibration import (
    PLAN_FILES,
    PersonalizationCalibrationPlanner,
    PlanFilters,
    aggregate_participant_metrics,
    build_model_compatibility,
    build_participant_calibration_plan,
    build_run_matrix,
    fit_outer_train_q3,
    validate_protocol_config,
)

CONFIG_PATH = Path("experiments/calibration/personalization_calibration_v1.json")


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def target_frame(pm: str = "attention", windows: int = 100) -> pd.DataFrame:
    rows, sample_id = [], 0
    for fold in range(1, 6):
        for participant in range(2):
            subject = f"subject_{fold}_{participant}"
            for record in range(2):
                record_id = f"{subject}_record_{record}"
                for window in range(windows):
                    rows.append({
                        "sample_id": f"{pm}_{sample_id}", "source": "synthetic",
                        "subject_id": subject, "record_id": record_id,
                        "record_group_id": record_id, "t_start": window * 10.0,
                        "t_end": (window + 1) * 10.0, "outer_fold": fold,
                        "absolute_t_start": (
                            fold * 1000000 + participant * 100000
                            + record * 10000 + window * 10.0
                        ),
                        "target_value": fold * .1 + participant * .01 + window / 1000,
                    })
                    sample_id += 1
    return pd.DataFrame(rows)


def participant_plan(pm: str = "attention", windows: int = 100):
    document = config()
    return build_participant_calibration_plan(
        target_frame(pm, windows), pm=pm,
        budgets=document["calibration"]["budgets_fraction"],
        protocol=document["protocol"],
    )


def test_config_covers_all_pm_and_classification_task() -> None:
    resolved = validate_protocol_config(config())
    assert resolved["pms"] == [
        "attention", "engagement", "excitement", "stress",
        "relaxation", "interest", "focus",
    ]
    assert resolved["task_types"] == ["classification"]
    assert resolved["protocol"]["n_outer_folds"] == 5
    assert resolved["calibration"]["budgets_fraction"] == [
        0.0, 0.01, 0.05, 0.1, 0.2,
    ]
    assert resolved["calibration"]["modes"] == [
        "zero_shot", "head_only", "full_model",
    ]
    assert resolved["protocol"]["fraction_allocation"] == "global_prefix"
    assert resolved["protocol"]["q3_fit_scope"] == "outer_train_only"
    assert resolved["execution"]["feature_scaling"]["strategy"] == "standard_clip"
    assert resolved["analysis"] == {
        "formal_accuracy_threshold": 0.75,
        "aggregation": "participant_macro",
        "threshold_role": "report_only_not_for_selection",
    }
    assert set(resolved["execution"]["model_params"]) == {
        "torch_shallow_convnet", "torch_eegnet", "torch_mlp"
    }


def test_validator_retains_global_regression_support() -> None:
    document = config()
    document["task_types"] = ["classification", "regression"]
    assert validate_protocol_config(document)["task_types"] == [
        "classification", "regression"
    ]

    document["task_types"] = ["regression"]
    assert validate_protocol_config(document)["task_types"] == ["regression"]

    document["task_types"] = ["regression", "classification"]
    with pytest.raises(ValueError, match="registry order"):
        validate_protocol_config(document)


def test_formal_accuracy_threshold_cannot_be_used_for_selection() -> None:
    document = config()
    document["analysis"]["threshold_role"] = "hyperparameter_selection"
    with pytest.raises(ValueError, match="report-only"):
        validate_protocol_config(document)


def test_temporal_split_is_deterministic_disjoint_and_fixed_across_budgets() -> None:
    first, manifests = participant_plan()
    shuffled = target_frame().sample(frac=1, random_state=999)
    second, repeated = build_participant_calibration_plan(
        shuffled, pm="attention", budgets=config()["calibration"]["budgets_fraction"],
        protocol=config()["protocol"],
    )
    pd.testing.assert_frame_equal(first, second)
    assert [x["transform_hash"] for x in manifests.values()] == [
        x["transform_hash"] for x in repeated.values()
    ]
    assert first.outer_train_subject_overlap.max() == 0
    assert first.outer_train_record_group_overlap.max() == 0
    assert first.calibration_evaluation_overlap.max() == 0
    assert first.calibration_before_evaluation.all()
    assert first.groupby(["pm", "outer_fold", "subject_id"])[
        "evaluation_sample_hash"
    ].nunique().max() == 1


def test_q3_fit_uses_only_outer_train_arguments() -> None:
    ids, targets = np.arange(30), np.linspace(0, 1, 30)
    transform, manifest = fit_outer_train_q3(
        pm="stress", outer_fold=2, outer_train_sample_ids=ids,
        outer_train_targets=targets,
    )
    _, repeated = fit_outer_train_q3(
        pm="stress", outer_fold=2, outer_train_sample_ids=ids,
        outer_train_targets=targets,
    )
    assert manifest == repeated
    assert manifest["fit_scope"] == "outer_train_only"
    assert manifest["fit_sample_count"] == 30
    assert transform.transform(np.array([-100., .5, 100.])).tolist() == [0., 1., 2.]


def test_outer_record_group_leakage_is_rejected() -> None:
    frame = target_frame()
    leaked_group = frame.loc[frame["outer_fold"].eq(1), "record_group_id"].iloc[0]
    frame.loc[frame["outer_fold"].eq(2), "record_group_id"] = leaked_group
    with pytest.raises(RuntimeError, match="Outer logical-record leakage"):
        build_participant_calibration_plan(
            frame,
            pm="attention",
            budgets=config()["calibration"]["budgets_fraction"],
            protocol=config()["protocol"],
        )


def test_insufficient_data_never_borrows_evaluation_rows() -> None:
    plan, _ = participant_plan(windows=3)
    positive = plan.loc[plan.budget_fraction.gt(0)]
    assert positive.status.eq("insufficient_data").all()
    assert positive.reason.str.contains("insufficient").all()
    assert positive.calibration_evaluation_overlap.eq(0).all()
    used = (positive.budget_windows + positive.evaluation_windows
            + positive.reserved_windows + positive.purged_windows)
    assert used.le(positive.total_available_windows).all()


def test_factory_supports_regression_but_current_matrix_is_classification_only() -> None:
    document = validate_protocol_config(config())
    compatibility = build_model_compatibility(document["models"])
    assert len(compatibility) == 6
    shallow = compatibility.loc[compatibility.model.eq("torch_shallow_convnet")]
    assert shallow.factory_supported.all() and shallow.head_only_supported.all()
    eegnet_reg = compatibility.loc[
        compatibility.model.eq("torch_eegnet")
        & compatibility.task_type.eq("regression")
    ].iloc[0]
    assert eegnet_reg.factory_supported
    assert eegnet_reg.head_only_supported
    base, _ = participant_plan()
    participants = pd.concat(
        [base.assign(pm=pm) for pm in document["pms"]], ignore_index=True
    )
    matrix = build_run_matrix(
        config=document, compatibility=compatibility,
        participant_plan=participants,
    )
    assert len(matrix) == 945
    assert not matrix.status.eq("unsupported").any()
    assert set(matrix.task_type) == {"classification"}
    assert set(matrix.status) <= {"planned", "insufficient_data"}
    assert matrix.condition_id.is_unique
    filtered = build_run_matrix(
        config=document, compatibility=compatibility,
        participant_plan=participants,
        filters=PlanFilters(1, "attention", "classification", "head_only"),
    )
    assert len(filtered) == 12


def test_participant_macro_does_not_weight_window_count() -> None:
    rows = []
    for subject, score, windows in (("small", .2, 10), ("large", .8, 1000)):
        for mode, budget, gain in (
            ("zero_shot", 0., 0.), ("head_only", .2, .1)
        ):
            rows.append({
                "pm": "focus", "model": "torch_mlp", "outer_fold": 1,
                "subject_id": subject, "mode": mode, "budget_fraction": budget,
                "accuracy": score + gain, "balanced_accuracy": score + gain,
                "macro_f1": score + gain, "weighted_f1": score + gain,
                "evaluation_windows": windows,
            })
    _, summary = aggregate_participant_metrics(
        pd.DataFrame(rows), task_type="classification"
    )
    adapted = summary.loc[summary["mode"].eq("head_only")].iloc[0]
    assert adapted.accuracy_participant_macro == pytest.approx(.6)
    assert (
        adapted.delta_accuracy_vs_zero_shot_participant_macro
        == pytest.approx(.1)
    )


def test_plan_only_artifacts_and_resume(tmp_path: Path, monkeypatch) -> None:
    planner = PersonalizationCalibrationPlanner(
        CONFIG_PATH, data_root=tmp_path, output_dir=tmp_path / "plan"
    )
    monkeypatch.setattr(planner, "_load_target_frame", lambda pm: target_frame(pm))
    manifest = planner.plan()
    assert not manifest["training_executed"]
    assert manifest["run_conditions"] == 945
    assert manifest["unsupported_conditions"] == 0
    assert manifest["leakage_checks"]["outer_record_group_overlap_max"] == 0
    assert manifest["formal_criteria"] == {
        "classification_accuracy_threshold": 0.75,
        "aggregation": "participant_macro",
        "threshold_role": "report_only_not_for_selection",
    }
    assert manifest["protocol_hash"] == (
        "a3723e8f77ec1a9eeef21a2b5a88660d9cd42a717084e6e1aadb12429085d0d4"
    )
    assert manifest["plan_hash"] == (
        "d8c7430e75e692fcf8cf53b7052d48faa2c8f392bb2cd7a049657204d3396412"
    )
    assert all((tmp_path / "plan" / name).is_file() for name in PLAN_FILES)
    with pytest.raises(FileExistsError):
        planner.plan()
    assert planner.plan(resume=True)["protocol_hash"] == manifest["protocol_hash"]
