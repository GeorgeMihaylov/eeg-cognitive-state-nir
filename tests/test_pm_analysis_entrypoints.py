"""Focused guards for PM analysis modules moved behind thin CLI wrappers."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.analysis.pm_eeg_lag_sweep import _lag_positions
from bench.analysis.pm_subject_structure import _entropy3
from bench.analysis.pm_temporal_structure import _paired_lag


@pytest.mark.parametrize(
    "module_name",
    (
        "bench.analysis.pm_target_validity_postprocess",
        "bench.analysis.pm_subject_structure",
        "bench.analysis.pm_temporal_structure",
        "bench.analysis.pm_eeg_lag_sweep",
    ),
)
def test_pm_analysis_library_exposes_main(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert callable(module.main)


def test_lag_pairing_never_crosses_record_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["r1"] * 3 + ["r2"] * 3,
            "t_start": [-5.0, 5.0, 15.0] * 2,
        }
    )

    positions, common, source = _lag_positions(frame, (-1, 0, 1))

    assert source == "per-record t_start/10s grid"
    assert np.flatnonzero(common).tolist() == [1, 4]
    assert positions[-1][common].tolist() == [0, 3]
    assert positions[0][common].tolist() == [1, 4]
    assert positions[1][common].tolist() == [2, 5]


def test_temporal_pairing_drops_nonfinite_pairs() -> None:
    left, right = _paired_lag(np.array([1.0, np.nan, 2.0, 3.0]), 1)
    assert left.tolist() == [2.0]
    assert right.tolist() == [3.0]


def test_subject_entropy_uses_three_class_contract() -> None:
    assert _entropy3(np.array([0.0, 1.0, 2.0])) == pytest.approx(1.0)
    assert _entropy3(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)


def test_pm_analysis_scripts_are_thin_entrypoints() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "postprocess_pm_target_validity.py",
        "analyze_pm_subject_structure.py",
        "analyze_pm_temporal_structure.py",
        "run_pm_eeg_lag_sweep.py",
    )
    for name in scripts:
        source = (root / "scripts" / "analysis" / name).read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 20
        assert "runpy.run_module" in source
