from __future__ import annotations

import json

import pandas as pd
import pytest

from bench.experiments.pm_low_high_personalization_duration_response import (
    _participant_first,
    load_config,
)


def _real_config(root):
    return root / (
        "experiments/pm_diagnostics/"
        "pm_low_high_personalization_duration_response_v1.json"
    )


def test_real_config_accepts_frozen_contract():
    cfg = load_config(_real_config(__import__("pathlib").Path(".")))
    assert cfg["scientific_contract"]["calibration_budgets_seconds"] == [
        300, 600, 900
    ]
    assert cfg["scientific_contract"]["threshold_strategy"] == "median_midpoint"


def test_config_rejects_budget_change(tmp_path):
    source = json.loads(
        _real_config(__import__("pathlib").Path(".")).read_text(encoding="utf-8")
    )
    source["scientific_contract"]["calibration_budgets_seconds"] = [
        300, 600, 900, 1200
    ]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="Scientific contract"):
        load_config(path)


def test_participant_first_equal_weights_participants():
    rows = []
    for subject, pm, delta in [
        ("s1", "attention", 1.0),
        ("s1", "stress", 1.0),
        ("s2", "focus", -1.0),
    ]:
        row = {
            "model": "xgboost",
            "budget_seconds": 300,
            "subject_id": subject,
            "pm": pm,
            "adaptation_applied": True,
        }
        for metric in (
            "balanced_accuracy", "macro_f1", "low_recall",
            "high_recall", "precision", "accuracy",
        ):
            row[f"zero_shot_{metric}"] = 0.5
            row[f"personalized_{metric}"] = 0.5 + delta
            row[f"delta_{metric}"] = delta
        rows.append(row)

    result = _participant_first(
        pd.DataFrame(rows),
        applied_only=False,
    )
    assert len(result) == 2
    values = result.set_index("subject_id")["delta_balanced_accuracy"]
    assert values["s1"] == pytest.approx(1.0)
    assert values["s2"] == pytest.approx(-1.0)
    assert values.mean() == pytest.approx(0.0)
