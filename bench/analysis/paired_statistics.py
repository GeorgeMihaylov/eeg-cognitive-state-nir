"""Paired inference with the subject as the independent sampling unit."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, rankdata, spearmanr, wilcoxon


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def subject_bootstrap_interval(
    values: Iterable[float],
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    statistic: str = "mean",
) -> dict[str, float | int | str]:
    """Percentile bootstrap over independent subjects."""

    data = _finite(values)
    if data.size == 0:
        return {
            "status": "undefined_no_finite_subjects",
            "n_subjects": 0,
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    reducer = np.mean if statistic == "mean" else np.median if statistic == "median" else None
    if reducer is None:
        raise ValueError("statistic must be 'mean' or 'median'")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, data.size, size=(n_resamples, data.size))
    estimates = reducer(data[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "status": "ok",
        "n_subjects": int(data.size),
        "estimate": float(reducer(data)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def _paired_finite(
    left: Iterable[float], right: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"Paired values must have equal shape: {left_array.shape} != {right_array.shape}"
        )
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    return left_array[mask], right_array[mask]


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    return 0.0 if total == 0 else (positive - negative) / total


def paired_subject_statistics(
    left: Iterable[float],
    right: Iterable[float],
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare paired subject metrics as ``left - right``.

    Bootstrap resampling selects subject indices once and applies them to both
    model vectors, thereby preserving each model pair.
    """

    left_values, right_values = _paired_finite(left, right)
    differences = left_values - right_values
    n = int(differences.size)
    if n == 0:
        return {
            "status": "undefined_no_finite_pairs",
            "n_subjects": 0,
            "nonzero_differences": 0,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "probability_difference_gt_zero": np.nan,
            "subjects_improved": 0,
            "subjects_degraded": 0,
            "ties": 0,
            "fraction_improved": np.nan,
            "fraction_degraded": np.nan,
            "number_needed_to_improve": np.nan,
            "wilcoxon_status": "undefined_no_finite_pairs",
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p_value": np.nan,
            "sign_test_status": "undefined_no_finite_pairs",
            "sign_test_p_value": np.nan,
            "rank_biserial": np.nan,
        }
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(random_state)
    # Resampling differences is algebraically identical to selecting one set of
    # paired indices from the left and right subject vectors.
    indices = rng.integers(0, n, size=(n_resamples, n))
    bootstrap_means = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    improved = int((differences > 0).sum())
    degraded = int((differences < 0).sum())
    ties = int((differences == 0).sum())
    nonzero = improved + degraded

    if nonzero:
        wilcoxon_result = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_status = "ok"
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
        sign_p = float(
            binomtest(improved, n=nonzero, p=0.5, alternative="two-sided").pvalue
        )
        sign_status = "ok"
    else:
        wilcoxon_status = "undefined_all_differences_zero"
        wilcoxon_statistic = np.nan
        wilcoxon_p = np.nan
        sign_status = "undefined_all_differences_zero"
        sign_p = np.nan

    fraction_improved = improved / n
    fraction_degraded = degraded / n
    return {
        "status": "ok",
        "n_subjects": n,
        "nonzero_differences": nonzero,
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "ci_low": float(np.quantile(bootstrap_means, alpha)),
        "ci_high": float(np.quantile(bootstrap_means, 1.0 - alpha)),
        "probability_difference_gt_zero": float(np.mean(bootstrap_means > 0)),
        "subjects_improved": improved,
        "subjects_degraded": degraded,
        "ties": ties,
        "fraction_improved": float(fraction_improved),
        "fraction_degraded": float(fraction_degraded),
        "number_needed_to_improve": (
            np.nan if fraction_improved == 0 else float(1.0 / fraction_improved)
        ),
        "wilcoxon_status": wilcoxon_status,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_value": wilcoxon_p,
        "sign_test_status": sign_status,
        "sign_test_p_value": sign_p,
        "rank_biserial": float(_rank_biserial(differences)),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment while preserving undefined values."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted.tolist()
    order = finite_indices[np.argsort(values[finite_indices], kind="stable")]
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def apply_holm_by_family(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_key: str = "family",
    p_key: str = "wilcoxon_p_value",
) -> list[dict[str, Any]]:
    """Add Holm-adjusted p-values independently inside logical families."""

    output = [dict(row) for row in rows]
    families: dict[str, list[int]] = {}
    for index, row in enumerate(output):
        families.setdefault(str(row[family_key]), []).append(index)
    for indices in families.values():
        adjusted = holm_adjust([float(output[index].get(p_key, np.nan)) for index in indices])
        for index, value in zip(indices, adjusted):
            output[index]["holm_adjusted_p_value"] = value
    return output


def bootstrap_spearman(
    x: Iterable[float],
    y: Iterable[float],
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """Subject bootstrap interval for a Spearman association."""

    x_values, y_values = _paired_finite(x, y)
    n = len(x_values)
    if n < 3 or np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        return {
            "status": "undefined_insufficient_variation",
            "n_subjects": n,
            "rho": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    observed = spearmanr(x_values, y_values)
    rng = np.random.default_rng(random_state)
    bootstrapped: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        x_sample = x_values[indices]
        y_sample = y_values[indices]
        if np.unique(x_sample).size < 2 or np.unique(y_sample).size < 2:
            continue
        value = float(spearmanr(x_sample, y_sample).statistic)
        if np.isfinite(value):
            bootstrapped.append(value)
    if not bootstrapped:
        return {
            "status": "undefined_bootstrap_variation",
            "n_subjects": n,
            "rho": float(observed.statistic),
            "p_value": float(observed.pvalue),
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "status": "ok",
        "n_subjects": n,
        "rho": float(observed.statistic),
        "p_value": float(observed.pvalue),
        "ci_low": float(np.quantile(bootstrapped, alpha)),
        "ci_high": float(np.quantile(bootstrapped, 1.0 - alpha)),
    }
