from __future__ import annotations

from pathlib import Path

import numpy as np

from bench.experiments.artifact_removal_ablation import (
    _aggregate,
    _q3_labels,
    _q3_thresholds,
    build_run_matrix,
    load_config,
    protocol_plan,
)
from bench.tasks.target_registry import PM_METRICS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preprocessing" / "artifact_removal_ablation_v1.yaml"


def test_artifact_run_matrix_is_exactly_140_and_smoke_is_four() -> None:
    config = load_config(CONFIG)
    specs = build_run_matrix(config)
    assert len(specs) == 7 * 4 * 5 == 140
    assert {spec.metric for spec in specs} == set(PM_METRICS)
    assert {spec.variant for spec in specs} == {"raw", "faster", "ica", "faster_ica"}
    smoke = [spec for spec in specs if spec.metric == "attention" and spec.fold == 1]
    assert len(smoke) == 4


def test_q3_is_fitted_on_supplied_train_values_only() -> None:
    train = np.arange(90, dtype=float)
    thresholds = _q3_thresholds(train)
    changed_test = np.asarray([-10000.0, 10000.0])
    assert thresholds == _q3_thresholds(train.copy())
    assert _q3_labels(changed_test, thresholds).tolist() == [0, 2]


def test_real_artifact_plan_reuses_raw_cache_and_fixed_subject_folds() -> None:
    plan = protocol_plan(CONFIG)
    assert plan["run_count"] == 140
    assert plan["input_shape"] == [1, 14, 2560]
    assert plan["accepted_deduplicated_windows"] > 0
    assert len(plan["outer_folds"]) == 5
    assert all(not fold["participant_overlap"] for fold in plan["outer_folds"].values())
    assert plan["fixed_fold_reference"]["subject_fold_assignments_match"] is True
    assert plan["fixed_fold_reference"]["unexpected_target_cohort_subjects"] == []
    assert plan["fixed_fold_reference"]["target_cohort_missing_reference_subjects"] == [
        "9192c107"
    ]
    assert plan["preprocessing_contract"]["base"] == (
        "canonical_raw_cache_no_bandpass_no_notch_no_car"
    )


def test_aggregate_writes_pm_macro_fold_summary_and_wins(tmp_path: Path) -> None:
    summaries = []
    for metric in ("attention", "focus"):
        for variant, value in (("raw", 0.30), ("faster", 0.35)):
            summaries.append({
                "metric": metric, "variant": variant, "fold": 1, "seed": 42,
                "train_windows": 10, "test_windows": 4,
                "preprocessing_seconds": 0.1, "training_seconds": 0.2,
                "inference_seconds": 0.01, "epochs": 1,
                "metrics": {
                    "macro_f1": value,
                    "balanced_accuracy": value,
                    "accuracy": value,
                },
            })
    _aggregate(tmp_path, "smoke", summaries)
    fold_macro = __import__("pandas").read_csv(tmp_path / "smoke_pm_macro_by_fold.csv")
    summary = __import__("pandas").read_csv(tmp_path / "smoke_variant_summary.csv")
    faster = fold_macro.loc[fold_macro["variant"] == "faster"].iloc[0]
    faster_summary = summary.loc[summary["variant"] == "faster"].iloc[0]
    assert np.isclose(faster["delta_macro_f1_vs_raw"], 0.05)
    assert faster_summary["folds_better_raw_macro_f1"] == 1
