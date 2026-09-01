from __future__ import annotations

import numpy as np

from bench.analysis.paired_statistics import (
    apply_holm_by_family,
    holm_adjust,
    paired_subject_statistics,
    subject_bootstrap_interval,
)


def test_subject_bootstrap_is_reproducible() -> None:
    first = subject_bootstrap_interval(
        [0.2, 0.3, 0.4, 0.5], n_resamples=500, random_state=42
    )
    second = subject_bootstrap_interval(
        [0.2, 0.3, 0.4, 0.5], n_resamples=500, random_state=42
    )
    assert first == second


def test_paired_bootstrap_preserves_subject_pairs() -> None:
    right = np.array([0.1, 0.4, 0.2, 0.7, 0.5])
    left = right + 0.05
    result = paired_subject_statistics(
        left,
        right,
        n_resamples=500,
        random_state=7,
    )
    assert np.isclose(result["mean_difference"], 0.05)
    assert np.isclose(result["ci_low"], 0.05)
    assert np.isclose(result["ci_high"], 0.05)
    assert result["subjects_improved"] == 5
    assert result["subjects_degraded"] == 0
    assert result["nonzero_differences"] == 5
    assert result["wilcoxon_status"] == "ok"
    assert result["rank_biserial"] == 1.0


def test_wilcoxon_all_zero_status_is_explicit() -> None:
    result = paired_subject_statistics([1, 2, 3], [1, 2, 3], n_resamples=100)
    assert result["wilcoxon_status"] == "undefined_all_differences_zero"
    assert result["sign_test_status"] == "undefined_all_differences_zero"
    assert np.isnan(result["wilcoxon_p_value"])


def test_holm_correction_and_family_boundaries() -> None:
    assert np.allclose(holm_adjust([0.01, 0.03, 0.04]), [0.03, 0.06, 0.06])
    rows = [
        {"family": "a", "wilcoxon_p_value": 0.01},
        {"family": "a", "wilcoxon_p_value": 0.04},
        {"family": "b", "wilcoxon_p_value": 0.04},
    ]
    adjusted = apply_holm_by_family(rows)
    assert adjusted[0]["holm_adjusted_p_value"] == 0.02
    assert adjusted[1]["holm_adjusted_p_value"] == 0.04
    assert adjusted[2]["holm_adjusted_p_value"] == 0.04
