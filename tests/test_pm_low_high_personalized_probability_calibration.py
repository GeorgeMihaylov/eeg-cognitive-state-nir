from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.experiments.pm_low_high_personalized_probability_calibration import (
    apply_logit_offset,
    fit_logit_offset,
    load_config,
)


def _config_path() -> Path:
    return Path(
        "experiments/pm_diagnostics/"
        "pm_low_high_personalized_probability_calibration_v1.json"
    )


def test_real_config_accepts_min10_lock():
    cfg = load_config(_config_path())
    c = cfg["scientific_contract"]
    assert c["minimum_calibration_low"] == 10
    assert c["minimum_calibration_high"] == 10
    assert c["logit_slope"] == 1.0
    assert cfg["feasibility_lock"][
        "full1800_and_min10_each_participant_pm"
    ] == 285


def test_config_rejects_support_change(tmp_path):
    cfg = json.loads(_config_path().read_text(encoding="utf-8"))
    cfg["scientific_contract"]["minimum_calibration_low"] = 5
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="Scientific contract"):
        load_config(path)


def test_logit_offset_zero_when_calibration_is_self_consistent():
    p = np.array([0.2, 0.2, 0.8, 0.8])
    y = np.array([0, 0, 1, 1])
    offset = fit_logit_offset(p, y)
    assert offset == pytest.approx(0.0, abs=1e-10)


def test_logit_offset_matches_calibration_prevalence():
    p = np.array([0.2] * 5 + [0.8] * 5)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    offset = fit_logit_offset(p, y)
    adjusted = apply_logit_offset(p, offset)
    assert adjusted.mean() == pytest.approx(y.mean(), abs=1e-10)


def test_logit_offset_preserves_ranking():
    p = np.array([0.05, 0.2, 0.6, 0.9])
    adjusted = apply_logit_offset(p, 0.7)
    assert np.all(np.diff(adjusted) > 0)
