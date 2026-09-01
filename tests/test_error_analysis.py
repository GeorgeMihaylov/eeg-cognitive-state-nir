from __future__ import annotations

import numpy as np
import pandas as pd

from bench.analysis.error_analysis import calculate_error_analysis, summarize_by_source


def test_severe_error_rate_and_ordinal_mae() -> None:
    frame = pd.DataFrame({
        "y_true": [0, 1, 2, 3, 4],
        "y_pred": [0, 3, 1, 4, 0],
    })
    analysis = calculate_error_analysis(frame)
    distances = np.array([0, 2, 1, 1, 4])
    assert analysis["ordinal_mae"] == distances.mean()
    assert analysis["severe_error_rate"] == 2 / 5
    assert analysis["adjacent_accuracy"] == 3 / 5
    assert np.asarray(analysis["confusion_matrix"]).shape == (5, 5)
    normalized = np.asarray(analysis["row_normalized_confusion_matrix"])
    assert np.allclose(normalized.sum(axis=1), 1.0)


def test_source_summary_does_not_treat_overlap_as_independent_subjects() -> None:
    frame = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2", "S3"],
        "record_id": [
            "gpn_data__a", "Old_EEG__b", "gpn_data__c", "Old_EEG__d"
        ],
        "y_true": [0, 1, 0, 1],
        "y_pred": [0, 1, 1, 1],
    })
    summary = summarize_by_source(frame, model="model")
    assert summary.attrs["unique_subjects_overall"] == 3
    assert summary.attrs["source_subject_counts_are_additive"] is False
    assert summary.attrs["subjects_in_multiple_sources"] == ["S1"]
    assert summary["subjects"].sum() == 4


def test_error_metrics_ignore_marked_calibration_samples() -> None:
    frame = pd.DataFrame({
        "y_true": [4, 0, 1],
        "y_pred": [0, 0, 1],
        "is_calibration_sample": [True, False, False],
    })
    analysis = calculate_error_analysis(frame)
    assert analysis["n_samples"] == 2
    assert analysis["severe_error_rate"] == 0.0
