from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bench.experiments.pm_low_high_personalization_long_duration_feasibility import (
    _budget_fully_available,
    _state_counts,
    load_config,
)


def _config_path() -> Path:
    return Path(
        "experiments/pm_diagnostics/"
        "pm_low_high_personalization_long_duration_feasibility_v1.json"
    )


def test_real_config_accepts():
    cfg = load_config(_config_path())
    assert cfg["scientific_contract"]["calibration_budgets_seconds"] == [
        900, 1200, 1800
    ]
    assert (
        cfg["scientific_contract"]["fixed_evaluation_boundary_rule"]
        == "absolute_target_utc > earliest_record_start_utc + 1800s"
    )


def test_config_rejects_added_budget(tmp_path):
    cfg = json.loads(_config_path().read_text(encoding="utf-8"))
    cfg["scientific_contract"]["calibration_budgets_seconds"] = [
        900, 1200, 1500, 1800
    ]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="Scientific contract"):
        load_config(path)


def test_state_counts_reports_min10_each():
    frame = pd.DataFrame({
        "state": ["low"] * 10 + ["high"] * 10 + ["middle", "missing"]
    })
    counts = _state_counts(frame, "calibration")
    assert counts["calibration_min2_each"] is True
    assert counts["calibration_min5_each"] is True
    assert counts["calibration_min10_each"] is True
    assert counts["calibration_extreme"] == 20


def test_budget_availability_requires_source_and_feature_grid():
    assert _budget_fully_available(
        7651.0, 30.0, 900
    ) is False

    assert _budget_fully_available(
        2000.0, 1280.0, 1800
    ) is False

    assert _budget_fully_available(
        2000.0, 1900.0, 1800
    ) is True
