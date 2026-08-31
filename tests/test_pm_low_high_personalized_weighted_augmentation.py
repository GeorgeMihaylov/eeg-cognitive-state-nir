from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from bench.experiments import pm_low_high_personalized_weighted_augmentation as weighted


CONFIG_PATH = (
    "experiments/pm_diagnostics/"
    "pm_low_high_personalized_weighted_augmentation_v1.json"
)


def _synthetic_matrix_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    participants = [f"s{index:02d}" for index in range(48)]
    cells = []
    for subject_index, subject_id in enumerate(participants):
        for pm in weighted.PM_NAMES[:5]:
            cells.append((subject_id, subject_index % 5 + 1, pm))
    for subject_index, subject_id in enumerate(participants[:45]):
        pm = weighted.PM_NAMES[5 + (subject_index % 2)]
        cells.append((subject_id, subject_index % 5 + 1, pm))
    assert len(cells) == 285
    assert {pm for _, _, pm in cells} == set(weighted.PM_NAMES)
    eligibility = pd.DataFrame([
        {
            "subject_id": subject_id,
            "outer_fold": fold,
            "pm": pm,
            "eligible": True,
            "calibration_low": 10 + index % 7,
            "calibration_high": 12 + index % 5,
            "calibration_extreme": 22 + index % 7 + index % 5,
            "calibration_extreme_sample_hash": f"cal-{index}",
            "evaluation_low": 20,
            "evaluation_high": 21,
            "evaluation_extreme": 41,
            "evaluation_extreme_sample_hash": f"eval-{index}",
        }
        for index, (subject_id, fold, pm) in enumerate(cells)
    ])
    outer_train = pd.DataFrame([
        {
            "outer_fold": fold,
            "pm": pm,
            "target_id": f"target_{pm}",
            "n_outer_train": 16000 + fold,
            "n_outer_train_subjects": 43,
            "outer_train_subjects_hash": f"subjects-{fold}-{pm}",
            "outer_train_sample_hash": f"train-{fold}-{pm}",
            "threshold_hash": f"threshold-{fold}-{pm}",
        }
        for fold in range(1, 6)
        for pm in weighted.PM_NAMES
    ])
    contracts = {
        "xgboost": {
            "estimator": "XGBClassifier",
            "params": {"n_estimators": 200, "n_jobs": 4, "random_state": 42},
        },
        "lightgbm": {
            "estimator": "LGBMClassifier",
            "params": {"n_estimators": 200, "n_jobs": 4, "random_state": 42},
        },
    }
    return eligibility, outer_train, contracts


def test_frozen_config_is_strictly_validated(tmp_path) -> None:
    config = weighted.load_config(CONFIG_PATH)
    assert config["adaptation"]["expected_new_fits"] == 570

    changed = deepcopy(config)
    changed["adaptation"]["weight_cap"] = 2.0
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="adaptation"):
        weighted.load_config(path)


def test_every_forbidden_scientific_switch_is_frozen_on(tmp_path) -> None:
    config = weighted.load_config(CONFIG_PATH)
    assert config["forbidden"]
    assert all(config["forbidden"].values())

    for key in config["forbidden"]:
        changed = deepcopy(config)
        changed["forbidden"][key] = False
        path = tmp_path / f"forbidden_{key}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="forbidden"):
            weighted.load_config(path)


def test_subject_equivalent_weights_have_exact_half_class_mass() -> None:
    result = weighted.subject_equivalent_class_balanced_weights(
        n_outer_train=16400,
        n_outer_train_subjects=41,
        n_calibration_low=25,
        n_calibration_high=40,
    )

    assert result["personal_total_mass"] == pytest.approx(400.0)
    assert result["calibration_low_sample_weight"] == pytest.approx(8.0)
    assert result["calibration_high_sample_weight"] == pytest.approx(5.0)
    assert result["calibration_low_total_mass"] == pytest.approx(200.0)
    assert result["calibration_high_total_mass"] == pytest.approx(200.0)


@pytest.mark.parametrize(
    ("ready", "full", "low", "high", "expected"),
    [
        (True, True, 10, 10, True),
        (True, True, 9, 10, False),
        (True, True, 10, 9, False),
        (True, False, 50, 50, False),
        (False, True, 50, 50, False),
    ],
)
def test_eligibility_requires_eval_ready_full1800_and_min10_each(
    ready: bool,
    full: bool,
    low: int,
    high: int,
    expected: bool,
) -> None:
    assert weighted.is_weighted_augmentation_eligible(
        evaluation_ready=ready,
        budget_fully_available=full,
        n_calibration_low=low,
        n_calibration_high=high,
    ) is expected


def test_sample_firewall_rejects_evaluation_or_subject_leakage() -> None:
    weighted.validate_sample_firewall(
        outer_train_sample_ids=["train"],
        calibration_sample_ids=["cal"],
        evaluation_sample_ids=["eval"],
        outer_train_subjects=["other"],
        personalized_subject_id="heldout",
    )

    with pytest.raises(RuntimeError, match="samples overlap"):
        weighted.validate_sample_firewall(
            outer_train_sample_ids=["train"],
            calibration_sample_ids=["shared"],
            evaluation_sample_ids=["shared"],
            outer_train_subjects=["other"],
            personalized_subject_id="heldout",
        )
    with pytest.raises(RuntimeError, match="subject leaked"):
        weighted.validate_sample_firewall(
            outer_train_sample_ids=["train"],
            calibration_sample_ids=["cal"],
            evaluation_sample_ids=["eval"],
            outer_train_subjects=["heldout"],
            personalized_subject_id="heldout",
        )


def test_prediction_fallbacks_are_exact_and_eligible_threshold_is_fixed() -> None:
    zero = np.asarray([0.2, 0.7])
    probability, threshold, p_source, class_policy = (
        weighted.resolve_prediction_policy(
            eligible=False,
            personalized_probability=None,
            zero_shot_probability=zero,
            frozen_median_threshold=0.63,
        )
    )
    assert np.array_equal(probability, zero)
    assert threshold == pytest.approx(0.63)
    assert p_source == "frozen_zero_shot_probability"
    assert class_policy == "frozen_1800s_median_midpoint_policy"

    personalized = np.asarray([0.1, 0.9])
    probability, threshold, _, _ = weighted.resolve_prediction_policy(
        eligible=True,
        personalized_probability=personalized,
        zero_shot_probability=zero,
        frozen_median_threshold=0.63,
    )
    assert np.array_equal(probability, personalized)
    assert threshold == pytest.approx(0.5)


def test_adaptation_matrix_is_deterministic_570_and_has_no_focus_branch() -> None:
    eligibility, outer_train, contracts = _synthetic_matrix_inputs()

    first = weighted.build_adaptation_matrix(
        eligibility_audit=eligibility,
        outer_train_audit=outer_train,
        model_contracts=contracts,
        protocol_hash="protocol",
    )
    second = weighted.build_adaptation_matrix(
        eligibility_audit=eligibility.sample(frac=1.0, random_state=7),
        outer_train_audit=outer_train.sample(frac=1.0, random_state=8),
        model_contracts=contracts,
        protocol_hash="protocol",
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 570
    assert first["run_id"].nunique() == 570
    assert first.groupby("model").size().to_dict() == {
        "lightgbm": 285,
        "xgboost": 285,
    }
    assert set(first["pm"]) == set(weighted.PM_NAMES)
    assert first.loc[first["pm"].eq("focus"), "feature_count"].eq(371).all()
    assert first.loc[first["pm"].eq("focus"), "lag_seconds"].eq(-10).all()


def _result_rows() -> pd.DataFrame:
    rows = []
    for model in weighted.MODELS:
        for subject_index, subject in enumerate(("s1", "s2")):
            for pm_index, pm in enumerate(("attention", "focus")):
                value = float(subject_index + pm_index + 1) / 10.0
                row = {
                    "model": model,
                    "subject_id": subject,
                    "pm": pm,
                    "adaptation_applied": not (
                        subject == "s2" and pm == "focus"
                    ),
                }
                for column in weighted.CONTRAST_COLUMNS:
                    row[column] = value
                rows.append(row)
    return pd.DataFrame(rows)


def test_participant_first_aggregation_averages_pm_before_participants() -> None:
    participant = weighted.participant_first_aggregate(
        _result_rows(), applied_only=False
    )
    xgboost = participant.loc[participant["model"].eq("xgboost")].set_index(
        "subject_id"
    )

    assert xgboost.loc["s1", "delta_roc_auc"] == pytest.approx(0.15)
    assert xgboost.loc["s2", "delta_roc_auc"] == pytest.approx(0.25)
    assert xgboost["delta_roc_auc"].mean() == pytest.approx(0.20)

    applied = weighted.participant_first_aggregate(
        _result_rows(), applied_only=True
    ).set_index(["model", "subject_id"])
    assert applied.loc[("xgboost", "s2"), "n_pm"] == 1
    assert applied.loc[("xgboost", "s2"), "delta_roc_auc"] == pytest.approx(0.2)


def test_subject_bootstrap_is_deterministic() -> None:
    participant = weighted.participant_first_aggregate(
        _result_rows(), applied_only=False
    )
    first = weighted.bootstrap_participant_first(
        participant, replicates=200, seed=42
    )
    second = weighted.bootstrap_participant_first(
        participant, replicates=200, seed=42
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["bootstrap_unit"].eq("subject_id").all()
    assert first["bootstrap_seed"].eq(42).all()


def test_dry_run_writer_performs_zero_training_inference_or_performance(
    tmp_path, monkeypatch
) -> None:
    eligibility = pd.DataFrame([
        {
            "subject_id": "s1",
            "pm": pm,
            "evaluation_ready": True,
            "eligible": True,
            "calibration_evaluation_overlap": 0,
            "record_stitching_used": False,
        }
        for pm in weighted.PM_NAMES
    ])
    run_matrix = pd.DataFrame([
        {
            "pm": pm,
            "calibration_low_total_mass": 1.0,
            "calibration_high_total_mass": 1.0,
        }
        for pm in weighted.PM_NAMES
    ])
    context = weighted.WeightedAugmentationContext(
        root=tmp_path,
        output_dir=tmp_path / "out",
        config={},
        feasibility=None,
        feasibility_detail=pd.DataFrame(),
        eligibility_audit=eligibility,
        outer_train_audit=pd.DataFrame(),
        frozen_response=pd.DataFrame(),
        source_matrix=pd.DataFrame(),
        source_audit=pd.DataFrame(),
        reference_audit=pd.DataFrame([{"valid": True}]),
        model_contracts={},
        protocol={
            "protocol_hash": "hash",
            "adaptation_matrix_hash": "matrix-hash",
            "fixed_evaluation_cohort_hash": "cohort-hash",
        },
        run_matrix=run_matrix,
    )
    monkeypatch.setattr(
        weighted,
        "build_model",
        lambda *args, **kwargs: pytest.fail("dry-run constructed a model"),
    )

    summary = weighted.write_dry_run(context)

    assert summary["base_model_training_executed"] is False
    assert summary["personalized_training_executed"] is False
    assert summary["base_model_inference_executed"] is False
    assert summary["personalized_inference_executed"] is False
    assert summary["performance_evaluation_executed"] is False
    assert not (context.output_dir / "runs").exists()
    assert not (context.output_dir / "participant_pm_results.csv").exists()
