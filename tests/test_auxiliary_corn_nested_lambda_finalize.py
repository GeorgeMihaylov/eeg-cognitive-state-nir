from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.experiments.auxiliary_corn_nested_lambda_finalize import (
    AuxiliaryCornNestedLambdaFinalizeExperiment,
    load_auxiliary_corn_finalize_spec,
)
from bench.experiments.ordinal_transformer import build_ordinal_transformer_experiment


SPEC = Path("experiments/auxiliary_corn_nested_lambda_finalize.yaml")
FALLBACK_IDS = {
    "eeg_pow_seed42_fold01",
    "eeg_pow_seed123_fold05",
    "eeg_only_seed7_fold01",
    "eeg_only_seed42_fold02",
    "eeg_only_seed123_fold01",
}


def _probabilities(y_true: np.ndarray, confidence: float = 0.8) -> np.ndarray:
    values = np.full((len(y_true), 5), (1.0 - confidence) / 4.0)
    values[np.arange(len(y_true)), y_true] = confidence
    return values


def _categorical_frame(fold: int) -> pd.DataFrame:
    y_true = np.arange(5, dtype=np.int64)
    probabilities = _probabilities(y_true)
    frame = pd.DataFrame({
        "fold": fold,
        "split": f"fold_{fold:02d}",
        "sample_id": np.arange(5) + fold * 100,
        "subject_id": [f"subject-{fold}-{i}" for i in range(5)],
        "record_id": [f"record-{fold}-{i}" for i in range(5)],
        "sequence_id": [f"sequence-{fold:02d}-{i}" for i in range(5)],
        "source": ["synthetic"] * 5,
        "target_sample_id": np.arange(5) + fold * 100,
        "target_time": np.arange(5, dtype=float) + fold * 1000.0,
        "y_true": y_true,
        "y_pred": y_true,
    })
    for index in range(5):
        frame[f"proba_{index}"] = probabilities[:, index]
    return frame


def _joint_frame(fold: int, weight: float = 0.25) -> pd.DataFrame:
    frame = _categorical_frame(fold)
    frame["split"] = "outer_test"
    frame["outer_fold"] = fold
    frame["head_type"] = "categorical_corn"
    frame["auxiliary_weight"] = weight
    probabilities = frame[[f"proba_{index}" for index in range(5)]].to_numpy()
    frame["categorical_expected_rank"] = probabilities @ np.arange(5)
    thresholds = np.asarray([
        [0.8, 0.6, 0.4, 0.2],
        [0.9, 0.7, 0.4, 0.1],
        [0.9, 0.8, 0.5, 0.2],
        [0.95, 0.85, 0.65, 0.3],
        [0.98, 0.9, 0.8, 0.6],
    ])
    aux_probabilities = probabilities.copy()
    for index in range(5):
        frame[f"class_probability_{index}"] = probabilities[:, index]
        frame[f"aux_class_probability_{index}"] = aux_probabilities[:, index]
    for index in range(4):
        frame[f"aux_threshold_probability_{index}"] = thresholds[:, index]
        frame[f"aux_threshold_logit_{index}"] = 0.0
    frame["aux_expected_rank"] = thresholds.sum(axis=1)
    frame["aux_ordinal_prediction"] = frame["y_true"]
    frame["aux_ordinal_argmax"] = frame["y_true"]
    return frame


def _build_fixture(tmp_path: Path) -> Path:
    output_root = tmp_path / "source-output"
    plans = []
    outcomes = []
    for group in ("eeg_pow", "eeg_only"):
        for seed in (7, 42, 123):
            baseline_run = tmp_path / "baseline" / group / f"seed_{seed}"
            unified = baseline_run / "dataset" / "task" / "model" / "group_kfold_subject"
            unified.mkdir(parents=True)
            categorical = pd.concat(
                [_categorical_frame(fold) for fold in range(1, 6)],
                ignore_index=True,
            )
            # Historical categorical artifacts may preserve integer identifiers as
            # floats and timestamps with a sub-nanosecond representation delta.
            categorical["target_sample_id"] = categorical["target_sample_id"].astype(float)
            categorical["target_time"] = categorical["target_time"].astype(float) + 5e-10
            categorical.to_parquet(unified / "predictions.parquet", index=False)
            for fold in range(1, 6):
                selection_id = f"{group}_seed{seed}_fold{fold:02d}"
                candidate_root = (
                    output_root / "candidates" / group / f"seed_{seed}" / f"fold_{fold:02d}"
                )
                selected_root = (
                    output_root / "selected" / group / f"seed_{seed}" / f"fold_{fold:02d}"
                )
                plans.append({
                    "selection_id": selection_id,
                    "feature_group": group,
                    "seed": seed,
                    "outer_fold": fold,
                    "baseline_run_directory": str(baseline_run),
                    "candidate_root": str(candidate_root),
                    "selected_root": str(selected_root),
                })
                for token, weight in (("0p25", 0.25), ("0p5", 0.5), ("1", 1.0)):
                    root = candidate_root / f"lambda_{token}"
                    root.mkdir(parents=True)
                    (root / "model.pt").write_text("model", encoding="utf-8")
                    _joint_frame(fold, weight).to_parquet(
                        root / "validation_predictions.parquet", index=False
                    )
                    (root / "validation_metrics.json").write_text("{}", encoding="utf-8")
                    (root / "candidate_manifest.json").write_text(json.dumps({
                        "status": "completed",
                        "outer_test_used": False,
                        "validation_rows": 5,
                    }), encoding="utf-8")
                if selection_id in FALLBACK_IDS:
                    outcomes.append({
                        "status": "aborted_no_eligible_lambda",
                        "selection_id": selection_id,
                        "reason": "No auxiliary weight satisfies the BA guard",
                        "baseline_metrics": {"balanced_accuracy": 0.5},
                        "candidates": [],
                    })
                else:
                    selected_root.mkdir(parents=True)
                    prediction_path = selected_root / "outer_test_predictions.parquet"
                    _joint_frame(fold, 0.25).to_parquet(prediction_path, index=False)
                    checkpoint = candidate_root / "lambda_0p25" / "model.pt"
                    outcomes.append({
                        "status": "completed",
                        "selection_id": selection_id,
                        "selection": {
                            "selected": {"auxiliary_weight": 0.25},
                            "outer_test_used": False,
                        },
                        "selected_outer": {
                            "selected_checkpoint": str(checkpoint),
                            "artifacts": {"predictions": str(prediction_path)},
                        },
                    })
    source_summary = tmp_path / "source-summary.json"
    source_summary.write_text(json.dumps({
        "schema_version": 1,
        "status": "incomplete",
        "experiment": "auxiliary_corn_nested_lambda",
        "plan": {"folds": plans},
        "outcomes": outcomes,
        "candidate_fold_fits_trained_this_run": 75,
        "candidate_fold_fits_resumed": 0,
        "outer_test_selected_only": True,
    }), encoding="utf-8")
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    document["source"]["summary_path"] = str(source_summary)
    document["experiment"]["output_dir"] = str(tmp_path / "finalized")
    document["experiment"]["summary_path"] = str(tmp_path / "finalized-summary.json")
    document["experiment"]["report_path"] = str(tmp_path / "finalized-report.md")
    document["audit"]["expected_sequences_per_policy_model"] = 25
    document["audit"]["expected_subjects_per_policy_model"] = 25
    spec = tmp_path / "finalize.yaml"
    spec.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return spec


def test_finalize_spec_and_dispatch() -> None:
    document = load_auxiliary_corn_finalize_spec(SPEC)
    assert document["audit"]["expected_candidate_fold_fits"] == 90
    experiment = build_ordinal_transformer_experiment(SPEC)
    assert isinstance(experiment, AuxiliaryCornNestedLambdaFinalizeExperiment)


def test_finalize_plan_preserves_25_joint_and_5_fallback(tmp_path: Path) -> None:
    spec = _build_fixture(tmp_path)
    experiment = AuxiliaryCornNestedLambdaFinalizeExperiment(spec)
    plan = experiment.plan()
    assert len(plan.units) == 30
    assert plan.joint_units == 25
    assert plan.fallback_units == 5
    rendered = experiment.render_plan(plan)
    assert "Joint selections: 25" in rendered
    assert "Categorical fallbacks: 5" in rendered


def test_finalize_materializes_complete_policy_without_training(tmp_path: Path) -> None:
    spec = _build_fixture(tmp_path)
    experiment = AuxiliaryCornNestedLambdaFinalizeExperiment(spec)
    manifest = experiment.execute(experiment.plan())
    assert manifest["status"] == "completed"
    assert manifest["selection_units_completed"] == 30
    assert manifest["selection_units_joint"] == 25
    assert manifest["selection_units_fallback"] == 5
    assert manifest["candidate_fold_fits_completed"] == 90
    assert manifest["model_training_performed"] is False
    assert manifest["ready_for_subject_level_analysis"] is True

    summary = json.loads(Path(manifest["summary"]).read_text(encoding="utf-8"))
    assert summary["candidate_counter_correction"] == 15
    assert summary["selection_units_aborted"] == 0
    assert summary["protocol_amendment"]["outer_test_used_for_selection"] is False
    assert len(summary["cross_policy_outer_alignment"]) == 5
    assert all(item["exact"] for item in summary["cross_policy_outer_alignment"].values())

    fallback = next(
        item for item in summary["outcomes"]
        if item["selection_id"] == "eeg_pow_seed42_fold01"
    )
    fallback_frame = pd.read_parquet(fallback["artifacts"]["predictions"])
    assert set(fallback_frame["policy_branch"]) == {"categorical_fallback"}
    assert fallback_frame["aux_available"].eq(False).all()
    assert fallback_frame["aux_expected_rank"].isna().all()

    joint = next(item for item in summary["outcomes"] if item["policy_branch"] == "joint_selected")
    joint_frame = pd.read_parquet(joint["artifacts"]["predictions"])
    assert joint_frame["aux_available"].eq(True).all()
    assert joint_frame["selected_auxiliary_weight"].notna().all()
    assert Path(summary["artifacts"]["subject_level_analysis_input"]).is_file()
    assert Path(summary["artifacts"]["selection_policy"]).is_file()


def test_finalize_rejects_real_outer_identity_mismatch_with_diagnostics(
    tmp_path: Path,
) -> None:
    spec = _build_fixture(tmp_path)
    baseline_path = next(
        (tmp_path / "baseline" / "eeg_only" / "seed_123").glob(
            "**/group_kfold_subject/predictions.parquet"
        )
    )
    frame = pd.read_parquet(baseline_path)
    mask = frame["fold"].astype(int).eq(1)
    first_index = frame.index[mask][0]
    frame.loc[first_index, "subject_id"] = "different-subject"
    frame.to_parquet(baseline_path, index=False)

    experiment = AuxiliaryCornNestedLambdaFinalizeExperiment(spec)
    with pytest.raises(ValueError, match="diagnostics"):
        experiment.execute(experiment.plan())

    diagnostics = tmp_path / "finalized" / "cross_policy_outer_alignment_failure.json"
    assert diagnostics.is_file()
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert payload["failed_fold"] == 1
    comparison = payload["alignment"]["fold_01"]["comparisons"]
    assert any(
        item.get("mismatches", {}).get("subject_id", 0) > 0
        for item in comparison.values()
    )
