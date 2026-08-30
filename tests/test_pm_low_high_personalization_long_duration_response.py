from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bench.experiments.pm_low_high_personalization_long_duration_response import (
    _participant_first,
    load_config,
)


def _config_path() -> Path:
    return Path(
        "experiments/pm_diagnostics/"
        "pm_low_high_personalization_long_duration_response_v1.json"
    )


def test_real_config_accepts_frozen_contract():
    cfg = load_config(_config_path())
    assert cfg["scientific_contract"]["calibration_budgets_seconds"] == [
        900, 1200, 1800
    ]
    assert cfg["scientific_contract"]["budget_fully_available_rule"] == (
        "source_duration_and_feature_grid_span_cover_budget"
    )
    assert cfg["evaluation"]["primary_duration_contrast"] == (
        "1800s_minus_900s"
    )


def test_config_rejects_source_only_availability(tmp_path):
    cfg = json.loads(_config_path().read_text(encoding="utf-8"))
    cfg["scientific_contract"]["budget_fully_available_rule"] = (
        "source_duration_only"
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
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
            "budget_seconds": 900,
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

    aggregate = _participant_first(
        pd.DataFrame(rows), applied_only=False
    )
    assert len(aggregate) == 2
    values = aggregate.set_index("subject_id")["delta_balanced_accuracy"]
    assert values["s1"] == pytest.approx(1.0)
    assert values["s2"] == pytest.approx(-1.0)
    assert values.mean() == pytest.approx(0.0)
