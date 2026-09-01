from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import bench.experiments.pm_all_targets_feature_baseline as baseline
from bench.datasets.base_eeg_data_loader import feature_list_sha256
from bench.experiments.pm_all_targets_feature_baseline import (
    FEATURE_SET_ORDER,
    _feature_comparisons,
    _selected_specs,
    _single_multi_comparisons,
    build_run_matrix,
    load_baseline_config,
    participant_metric_rows,
    prepare_protocol,
    run_baseline,
)
from bench.tasks.target_registry import PM_METRICS, get_target_spec
from cogstate.model_zoo import build_model


CONFIG_PATH = Path("experiments/pm_regression/pm_all_targets_feature_baseline.yaml")


def _synthetic_config(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    subjects = [f"s{index:02d}" for index in range(10)]
    n_per_subject = 6
    n = len(subjects) * n_per_subject
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat(subjects, n_per_subject),
            "record_id": np.repeat([f"r{index:02d}" for index in range(10)], n_per_subject),
            "source": np.tile(["gpn_data", "Old_EEG"], n // 2),
            "EEG.AF3.mean": rng.normal(size=n),
            "EEG.AF4.std": rng.normal(size=n),
            "POW.AF3.alpha": rng.normal(size=n),
            "target_main": rng.uniform(size=n),
            "label_q5": np.arange(n) % 5,
        }
    )
    for offset, metric in enumerate(PM_METRICS):
        frame[f"target_{metric}"] = (
            0.05 * offset + np.linspace(0.0, 1.0, n) + rng.normal(0, 0.01, n)
        )
    dataset_path = tmp_path / "dataset.parquet"
    frame.to_parquet(dataset_path, index=False)
    reference = pd.DataFrame(
        {
            "sample_id": np.arange(n),
            "subject_id": frame["subject_id"],
            "fold": frame["subject_id"].map(
                {subject: index % 5 + 1 for index, subject in enumerate(subjects)}
            ),
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    config = load_baseline_config(CONFIG_PATH)
    config["dataset"]["path"] = str(dataset_path)
    config["dataset"]["reference_predictions"] = str(reference_path)
    config["dataset"]["logical_recording_map"] = str(tmp_path / "unused.parquet")
    config["feature_sets"] = {
        "eeg": {
            "expected_count": 2,
            "expected_hash": feature_list_sha256(["EEG.AF3.mean", "EEG.AF4.std"]),
        },
        "pow": {
            "expected_count": 1,
            "expected_hash": feature_list_sha256(["POW.AF3.alpha"]),
        },
        "eeg_pow": {
            "expected_count": 3,
            "expected_hash": feature_list_sha256(
                ["EEG.AF3.mean", "EEG.AF4.std", "POW.AF3.alpha"]
            ),
        },
    }
    config["output_dir"] = str(tmp_path / "runtime")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_canonical_targets_and_output_order_are_fixed() -> None:
    config = load_baseline_config(CONFIG_PATH)
    assert config["targets"]["single"] == [
        f"pm_{metric}_regression" for metric in PM_METRICS
    ]
    assert config["targets"]["multioutput"] == "pm_multioutput_regression_7"
    assert tuple(config["targets"]["output_order"]) == PM_METRICS
    assert get_target_spec("pm_multioutput_regression_7").output_names == PM_METRICS


def test_feature_sets_counts_and_hashes_match_contract() -> None:
    context = prepare_protocol(CONFIG_PATH)
    assert tuple(context.feature_names) == FEATURE_SET_ORDER
    assert {name: len(values) for name, values in context.feature_names.items()} == {
        "eeg": 168,
        "pow": 280,
        "eeg_pow": 448,
    }
    for names in context.feature_names.values():
        assert not any(
            name.startswith(("PM.", "target_", "label_")) for name in names
        )


def test_real_target_cohorts_match_canonical_contract() -> None:
    context = prepare_protocol(CONFIG_PATH)
    counts = {
        target_id: int(mask.sum()) for target_id, mask in context.target_masks.items()
    }
    assert counts == {
        "pm_attention_regression": 43175,
        "pm_engagement_regression": 48254,
        "pm_excitement_regression": 50983,
        "pm_stress_regression": 45384,
        "pm_relaxation_regression": 45394,
        "pm_interest_regression": 45440,
        "pm_focus_regression": 45384,
        "pm_multioutput_regression_7": 43174,
    }


def test_fixed_folds_do_not_change_between_targets() -> None:
    context = prepare_protocol(CONFIG_PATH)
    assert set(context.folds) == {1, 2, 3, 4, 5}
    assert all(not fold["subject_overlap"] for fold in context.folds.values())
    assert context.cohort_summary.groupby("target_id")["fold"].nunique().eq(5).all()
    assert context.preregistration["reference_predictions_sha256"]


def test_run_matrix_is_deterministic_and_complete() -> None:
    config = load_baseline_config(CONFIG_PATH)
    first = build_run_matrix(config)
    second = build_run_matrix(config)
    assert [item.run_id for item in first] == [item.run_id for item in second]
    assert len(first) == 1125
    assert len({item.run_id for item in first}) == 1125
    assert sum(item.analysis == "main_single" for item in first) == 630
    assert sum(item.analysis == "main_multi" for item in first) == 75
    assert sum(item.analysis == "paired_single" for item in first) == 420


def test_model_and_seed_policy_is_fixed() -> None:
    specs = build_run_matrix(load_baseline_config(CONFIG_PATH))
    assert {item.model for item in specs if item.analysis == "main_single"} == {
        "dummy_mean",
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert {item.seed for item in specs if item.model == "random_forest"} == {
        42,
        123,
        2026,
    }
    assert {item.seed for item in specs if item.model != "random_forest"} == {42}


@pytest.mark.parametrize(
    "name",
    ["dummy_mean", "ridge", "random_forest", "hist_gradient_boosting"],
)
def test_shared_model_factory_builds_required_regressors(name: str) -> None:
    model = build_model(name, "regression", (3,), 1, {})
    assert hasattr(model, "fit") and hasattr(model, "predict")


def test_lightgbm_is_not_added_to_run_matrix() -> None:
    specs = build_run_matrix(load_baseline_config(CONFIG_PATH))
    assert "lightgbm" not in {item.model for item in specs}


def test_smoke_matrix_has_all_targets_and_only_dummy_ridge() -> None:
    context = prepare_protocol(CONFIG_PATH)
    smoke = _selected_specs(context, smoke=True)
    assert len(smoke) == 16
    assert {item.fold for item in smoke} == {1}
    assert {item.feature_set for item in smoke} == {"eeg_pow"}
    assert {item.model for item in smoke} == {"dummy_mean", "ridge"}
    assert {item.target_id for item in smoke} == {
        *context.config["targets"]["single"],
        context.config["targets"]["multioutput"],
    }


def test_participant_metrics_use_train_std_and_preserve_undefined() -> None:
    truth = np.asarray([0.0, 1.0, 2.0, 3.0])
    prediction = np.ones(4)
    frame = participant_metric_rows(
        truth=truth,
        prediction=prediction,
        subjects=np.asarray(["a", "a", "b", "b"]),
        sources=np.asarray(["gpn_data"] * 4),
        target_names=["focus"],
        train_target_std=np.asarray([2.0]),
        metadata={"run_id": "x"},
    )
    assert len(frame) == 2
    assert np.allclose(frame["normalized_mae"], frame["mae"] / 2.0)
    assert frame["pearson"].isna().all()
    assert frame["pearson_undefined_reason"].eq("constant_prediction").all()
    assert not frame["pearson"].eq(0).any()


def _comparison_fixture() -> pd.DataFrame:
    rows = []
    for analysis in ("main_single", "paired_single", "main_multi"):
        for feature_index, feature in enumerate(FEATURE_SET_ORDER):
            rows.append(
                {
                    "analysis": analysis,
                    "target_id": (
                        "pm_multioutput_regression_7"
                        if analysis == "main_multi" else "pm_focus_regression"
                    ),
                    "target_name": "focus",
                    "feature_set": feature,
                    "model": "ridge",
                    "seed": 42,
                    "fold": 1,
                    "subject_id": "s1",
                    "mae": 0.3 - 0.05 * feature_index,
                    "rmse": 0.4,
                    "r2": 0.1,
                    "pearson": 0.2 + 0.05 * feature_index,
                    "spearman": 0.2,
                    "normalized_mae": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_feature_comparisons_are_paired_on_identical_units() -> None:
    comparisons = _feature_comparisons(_comparison_fixture())
    assert len(comparisons) == 3
    assert set(comparisons["comparison"]) == {
        "pow_minus_eeg",
        "eeg_pow_minus_eeg",
        "eeg_pow_minus_pow",
    }
    assert comparisons.loc[
        comparisons["comparison"].eq("eeg_pow_minus_eeg"), "mae_difference"
    ].iloc[0] < 0


def test_single_multi_comparison_is_paired_by_subject_fold_seed_and_feature() -> None:
    comparison = _single_multi_comparisons(_comparison_fixture())
    assert len(comparison) == 3
    assert "mae_multi_minus_single" in comparison


def test_synthetic_plan_creates_preregistration_before_training(tmp_path: Path) -> None:
    config_path = _synthetic_config(tmp_path)
    result = run_baseline(config_path, plan_only=True)
    root = tmp_path / "runtime"
    prereg = root / "preregistration" / "preregistration_manifest.json"
    registry = root / "run_registry" / "run_registry.json"
    assert result["status"] == "protocol_audit_complete"
    assert prereg.is_file() and registry.is_file()
    document = json.loads(prereg.read_text(encoding="utf-8"))
    assert document["created_before_training"] is True
    assert document["protocol_hash"] == result["protocol_hash"]
    assert not (root / "single_target").exists()


def test_synthetic_run_saves_train_only_scaler_and_artifacts(tmp_path: Path) -> None:
    config_path = _synthetic_config(tmp_path)
    context = prepare_protocol(config_path)
    ridge = next(
        spec for spec in context.run_specs
        if spec.analysis == "main_single" and spec.model == "ridge"
    )
    # Limit the matrix deterministically up to and including the first ridge run.
    position = context.run_specs.index(ridge) + 1
    result = run_baseline(config_path, max_runs=position)
    assert result["complete_runs"] == position
    run_dir = context.output_dir / "single_target" / ridge.run_id
    feature = json.loads((run_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert feature["scaler"]["scope"] == "outer_train_only"
    assert feature["scaler"]["n_fit_samples"] == split["train_sample_count"]
    assert feature["imputer"]["used"] is True
    assert feature["imputer"]["scope"] == "outer_train_only"
    assert feature["imputer"]["n_fit_samples"] == split["train_sample_count"]
    for name in (
        "run_specification.json",
        "split_manifest.json",
        "feature_manifest.json",
        "target_manifest.json",
        "training_manifest.json",
        "predictions.parquet",
        "participant_metrics.csv",
        "window_metrics.json",
        "source_metrics.csv",
        "run_summary.json",
        "errors.csv",
    ):
        assert (run_dir / name).is_file()


def test_resume_does_not_retrain_complete_run(tmp_path: Path) -> None:
    config_path = _synthetic_config(tmp_path)
    run_baseline(config_path, max_runs=1)
    registry_path = tmp_path / "runtime" / "run_registry" / "run_registry.json"
    before = json.loads(registry_path.read_text(encoding="utf-8"))
    complete_id = next(
        run_id for run_id, row in before["runs"].items() if row["status"] == "complete"
    )
    assert len(before["runs"][complete_id]["attempts"]) == 1
    run_baseline(config_path, resume=True, max_runs=1)
    after = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(after["runs"][complete_id]["attempts"]) == 1


def test_resume_after_commit_drift_preserves_preregistration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _synthetic_config(tmp_path)
    monkeypatch.setattr(baseline, "_git_head", lambda: "a" * 40)
    run_baseline(config_path, max_runs=1)
    prereg_path = (
        tmp_path / "runtime" / "preregistration" / "preregistration_manifest.json"
    )
    registry_path = tmp_path / "runtime" / "run_registry" / "run_registry.json"
    report_path = tmp_path / "runtime" / "reports" / "benchmark_report.md"
    prereg_before = prereg_path.read_bytes()
    report_before = report_path.read_bytes()
    registry_before = json.loads(registry_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(baseline, "_git_head", lambda: "b" * 40)
    run_baseline(config_path, resume=True, max_runs=1)

    registry_after = json.loads(registry_path.read_text(encoding="utf-8"))
    assert prereg_path.read_bytes() == prereg_before
    assert report_path.read_bytes() == report_before
    assert registry_after == registry_before


def test_incompatible_resume_does_not_overwrite_preregistration(
    tmp_path: Path,
) -> None:
    config_path = _synthetic_config(tmp_path)
    run_baseline(config_path, max_runs=1)
    prereg_path = (
        tmp_path / "runtime" / "preregistration" / "preregistration_manifest.json"
    )
    prereg_before = prereg_path.read_bytes()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["metrics"]["normalized_mae_denominator"] = "incompatible_change"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic inputs differ"):
        run_baseline(config_path, resume=True, max_runs=1)
    assert prereg_path.read_bytes() == prereg_before


def test_tracked_config_contains_no_absolute_paths() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert "F:\\" not in text
    assert "C:\\" not in text
    assert "benchmark_results/pm_all_targets_feature_baseline_v1" in text
