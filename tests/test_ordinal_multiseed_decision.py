from __future__ import annotations

from bench.analysis.ordinal_transformer_multiseed_statistics import (
    select_multiseed_decision,
)


def _primary(*, supported: bool) -> list[dict]:
    rows = []
    for group in ("eeg_only", "eeg_pow"):
        for method in ("coral", "corn"):
            for metric in ("ordinal_mae", "severe_error_rate"):
                winning = supported and group == "eeg_pow" and method == "coral"
                rows.append({
                    "feature_group": group,
                    "candidate": f"{method}_{group}",
                    "metric": metric,
                    "holm_adjusted_p_value": 0.01 if winning else 0.5,
                    "bootstrap_ci_low": 0.01 if winning else -0.01,
                    "bootstrap_ci_high": 0.03,
                    "mean_improvement": 0.02 if winning else 0.0,
                })
    return rows


def _secondary(*, quality_loss: bool) -> list[dict]:
    rows = []
    for group in ("eeg_only", "eeg_pow"):
        for method in ("coral", "corn"):
            for metric in ("balanced_accuracy", "macro_f1"):
                losing = quality_loss and group == "eeg_pow" and method == "coral"
                rows.append({
                    "feature_group": group,
                    "candidate": f"{method}_{group}",
                    "metric": metric,
                    "holm_adjusted_p_value": 0.01 if losing else 0.5,
                    "bootstrap_ci_low": -0.03,
                    "bootstrap_ci_high": -0.01 if losing else 0.03,
                })
    return rows


def _consistency() -> list[dict]:
    return [
        {
            "feature_group": group,
            "candidate": method,
            "metric": metric,
            "positive_seeds": 3,
        }
        for group in ("eeg_only", "eeg_pow")
        for method in ("coral", "corn")
        for metric in ("ordinal_mae", "severe_error_rate")
    ]


def test_decision_a_requires_supported_ordinal_gain_without_quality_cost() -> None:
    decision = select_multiseed_decision(
        _primary(supported=True), _secondary(quality_loss=False), _consistency()
    )
    assert decision["selected_decision"] == "A"
    assert decision["selected_head"] == "coral"


def test_decision_b_marks_confirmed_categorical_quality_tradeoff() -> None:
    decision = select_multiseed_decision(
        _primary(supported=True), _secondary(quality_loss=True), _consistency()
    )
    assert decision["selected_decision"] == "B"
    assert "auxiliary ordinal loss" in decision["next_experiment"]


def test_decision_c_is_deterministic_when_primary_support_is_absent() -> None:
    first = select_multiseed_decision(
        _primary(supported=False), _secondary(quality_loss=False), _consistency()
    )
    second = select_multiseed_decision(
        _primary(supported=False), _secondary(quality_loss=False), _consistency()
    )
    assert first == second
    assert first["selected_decision"] == "C"
    assert first["selected_head"] is None
