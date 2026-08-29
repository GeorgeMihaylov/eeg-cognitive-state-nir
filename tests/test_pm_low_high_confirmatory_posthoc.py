from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from bench.analysis import pm_low_high_confirmatory_posthoc as posthoc


def _prediction_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": ["s1"] * 4 + ["s2"] * 2,
        "y_true": [0, 0, 1, 1, 1, 1],
        "y_pred": [0, 1, 1, 1, 1, 0],
        "probability_high": [0.1, 0.7, 0.8, 0.9, 0.8, 0.2],
    })


def test_participant_metrics_are_independently_recomputed_from_probabilities() -> None:
    predictions = _prediction_fixture()

    result = posthoc.recompute_participant_metrics(predictions).set_index("subject_id")

    assert len(result) == 2
    assert result.loc["s1", "n_samples"] == 4
    assert result.loc["s1", "n_low"] == 2
    assert result.loc["s1", "n_high"] == 2
    assert result.loc["s1", "low_recall"] == pytest.approx(0.5)
    assert result.loc["s1", "high_recall"] == pytest.approx(1.0)
    assert result.loc["s1", "balanced_accuracy"] == pytest.approx(0.75)
    assert result.loc["s1", "macro_f1"] == pytest.approx(
        f1_score([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1], average="macro")
    )
    assert result.loc["s1", "roc_auc"] == pytest.approx(
        roc_auc_score([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9])
    )
    assert result.loc["s1", "pr_auc"] == pytest.approx(
        average_precision_score([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9])
    )


def test_one_class_auc_is_undefined_without_participant_exclusion() -> None:
    result = posthoc.recompute_participant_metrics(_prediction_fixture()).set_index(
        "subject_id"
    )

    assert "s2" in result.index
    assert result.loc["s2", "n_low"] == 0
    assert result.loc["s2", "n_high"] == 2
    assert result.loc["s2", "balanced_accuracy"] == pytest.approx(0.5)
    assert np.isnan(result.loc["s2", "roc_auc"])
    assert np.isnan(result.loc["s2", "pr_auc"])


def test_participant_uniqueness_rejects_duplicate_pm_subject() -> None:
    duplicated = pd.DataFrame({
        "pm": ["attention", "attention"],
        "subject_id": ["s1", "s1"],
        "outer_fold": [1, 2],
    })

    with pytest.raises(ValueError, match="more than once"):
        posthoc.validate_participant_uniqueness(duplicated)


def test_participant_uniqueness_requires_same_fold_across_pm() -> None:
    inconsistent = pd.DataFrame({
        "pm": ["attention", "focus"],
        "subject_id": ["s1", "s1"],
        "outer_fold": [1, 2],
    })

    with pytest.raises(ValueError, match="inconsistent outer folds"):
        posthoc.validate_participant_uniqueness(inconsistent)


def test_clustered_bootstrap_is_deterministic_and_carries_all_cluster_rows() -> None:
    frame = pd.DataFrame({
        "subject_id": ["s1", "s1", "s2"],
        "balanced_accuracy": [0.0, 1.0, 1.0],
    })

    first = posthoc.clustered_bootstrap_mean_ci(
        frame,
        value_column="balanced_accuracy",
        cluster_column="subject_id",
        n_replicates=200,
        seed=42,
    )
    second = posthoc.clustered_bootstrap_mean_ci(
        frame,
        value_column="balanced_accuracy",
        cluster_column="subject_id",
        n_replicates=200,
        seed=42,
    )

    assert first == second
    assert first["observed_mean"] == pytest.approx(2.0 / 3.0)
    assert first["n_valid_rows"] == 3
    assert first["n_valid_clusters"] == 2
    # Sampling s1 carries both of its rows, so the lower endpoint can reach 0.5.
    assert first["ci95_low"] == pytest.approx(0.5)


def test_recall_asymmetry_keeps_all_participants_with_paired_recalls() -> None:
    rows = []
    for pm in posthoc.PM_NAMES:
        rows.extend([
            {"pm": pm, "low_recall": 0.4, "high_recall": 0.8},
            {"pm": pm, "low_recall": 0.9, "high_recall": 0.7},
            {"pm": pm, "low_recall": 0.5, "high_recall": 0.5},
        ])

    result = posthoc.class_recall_asymmetry_summary(pd.DataFrame(rows))

    assert len(result) == 7
    assert result["n_valid_paired_participants"].eq(3).all()
    assert result["fraction_high_recall_gt_low"].eq(1.0 / 3.0).all()
    assert result["fraction_low_recall_gt_high"].eq(1.0 / 3.0).all()
    assert result["fraction_tied"].eq(1.0 / 3.0).all()


def test_analysis_module_has_no_model_build_or_training_call() -> None:
    source = inspect.getsource(posthoc)

    assert "build_model" not in source
    assert ".fit(" not in source
    assert "model_zoo" not in source
