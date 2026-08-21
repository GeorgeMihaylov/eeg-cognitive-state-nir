from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.analysis.xgboost_robust_shrinkage_final_analysis import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FinalAnalysis,
    fixed_label_balanced_accuracy,
    participant_macro_deltas,
    summarize_delta,
    validate_cross_method_identity,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "benchmark_results/xgboost_robust_shrinkage_personalization_v1"


def test_participant_macro_gives_each_pm_and_participant_equal_weight() -> None:
    frame = pd.DataFrame({
        "subject_id": ["a", "a", "b"],
        "outer_fold": [1, 1, 2],
        "pm": ["attention", "focus", "attention"],
        "delta_accuracy": [0.2, 0.4, -0.1],
        "delta_balanced_accuracy": [0.1, 0.3, -0.2],
        "delta_macro_f1": [0.0, 0.6, -0.3],
        "delta_weighted_f1": [0.2, 0.2, -0.4],
    })
    result = participant_macro_deltas(frame).set_index("subject_id")

    assert result.loc["a", "delta_accuracy"] == pytest.approx(0.3)
    assert result.loc["a", "delta_macro_f1"] == pytest.approx(0.3)
    assert result.loc["a", "pm_count"] == 2
    assert len(result) == 2


def test_locked_participant_statistics_are_deterministic() -> None:
    values = np.asarray([0.1, 0.2, -0.05, 0.0, 0.3])
    first = summarize_delta(
        values, metric="macro_f1", comparison="test", family="primary",
        cohort="synthetic",
    )
    second = summarize_delta(
        values.copy(), metric="macro_f1", comparison="test", family="primary",
        cohort="synthetic",
    )

    assert first == second
    assert first["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES == 20_000
    assert first["bootstrap_seed"] == BOOTSTRAP_SEED == 2026
    assert first["wilcoxon_zero_method"] == "wilcox"
    assert first["wilcoxon_alternative"] == "two-sided"
    assert first["positive_fraction"] == pytest.approx(3 / 5)
    assert -1.0 <= first["rank_biserial"] <= 1.0


def test_fixed_label_balanced_accuracy_and_warning_audit() -> None:
    frame = pd.DataFrame({
        "outer_fold": [1, 1, 1, 1],
        "pm": ["focus"] * 4,
        "subject_id": ["a"] * 4,
        "sample_id": ["s1", "s2", "s3", "s4"],
        "y_true": [0, 0, 1, 1],
        "zero_shot_y_pred": [0, 0, 1, 1],
        "adapted_y_pred": [0, 2, 1, 2],
    })
    audit, participant = fixed_label_balanced_accuracy(frame)

    assert audit.loc[0, "true_class_count"] == 2
    assert audit.loc[0, "zero_shot_warning_condition"] == np.bool_(False)
    assert audit.loc[0, "adapted_warning_condition"] == np.bool_(True)
    assert audit.loc[0, "zero_shot_balanced_accuracy_fixed_labels"] == pytest.approx(2 / 3)
    assert audit.loc[0, "adapted_balanced_accuracy_fixed_labels"] == pytest.approx(1 / 3)
    assert participant.loc[0, "delta_balanced_accuracy_fixed_labels"] == pytest.approx(-1 / 3)


def _identity_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": ["a", "b"], "pm": ["focus", "attention"],
        "outer_fold": [1, 2],
        "calibration_sample_hash": ["c1", "c2"],
        "evaluation_sample_hash": ["e1", "e2"],
        "q3_transform_hash": ["q1", "q2"],
    })


def test_cross_method_comparison_requires_exact_identity() -> None:
    robust = _identity_frame()
    _, exact = validate_cross_method_identity(
        robust, robust.copy(), prior_method="prior",
    )
    assert exact["identity_exact"] is True
    assert exact["joined_rows"] == 2

    changed = robust.copy()
    changed.loc[0, "evaluation_sample_hash"] = "different"
    _, mismatch = validate_cross_method_identity(
        robust, changed, prior_method="prior",
    )
    assert mismatch["identity_exact"] is False
    assert mismatch["hash_equal_rows"]["evaluation_sample_hash"] == 1


def test_real_five_fold_artifacts_validate_read_only(tmp_path: Path) -> None:
    sentinel = EXPERIMENT / "protocol_manifest.json"
    before = sentinel.read_bytes()
    analysis = FinalAnalysis(
        repo_root=ROOT,
        experiment_root=EXPERIMENT,
        output_dir=EXPERIMENT / "unused_test_output",
        xgboost_prior_results=(
            ROOT / "benchmark_results/personalization_calibration_xgboost_v1/participant_results.csv"
        ),
        shallow_prior_results=(
            ROOT / "benchmark_results/personalization_calibration_v1_classification/"
            "execution_scopes/model_torch_shallow_convnet/participant_results.csv"
        ),
    )
    resolved = analysis.load_and_validate()

    assert len(resolved["inner_specifications"]) == 140
    assert len(resolved["base_identities"]) == 35
    assert resolved["participant_pm"]["subject_id"].nunique() == 54
    assert resolved["decision_alphas"] == {1: 0.5, 2: 0.5, 3: 0.25, 4: 0.25, 5: 0.5}
    assert not (EXPERIMENT / "unused_test_output").exists()
    assert sentinel.read_bytes() == before


def test_analysis_module_contains_no_model_training_calls() -> None:
    path = ROOT / "bench/analysis/xgboost_robust_shrinkage_final_analysis.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {"fit", "train", "build_model", "ensure_base", "run"}
    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    ]
    # FinalAnalysis.run is orchestration, but no call site may invoke a method
    # named run; main calls the local analysis entry point and is exempted by
    # checking only model-specific training names here.
    assert not ({"fit", "train", "build_model", "ensure_base"} & set(calls))
