from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import cli
from bench.analysis.alignment import AlignmentError, check_alignment, require_alignment
from bench.analysis.report_builder import StatisticalAnalysis
from bench.analysis.run_inventory import InventoryEntry, select_canonical_runs


def _predictions(ids=(1, 2, 3), *, fold=1) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": list(ids),
        "fold": [fold] * len(ids),
        "subject_id": ["S1", "S1", "S2"][:len(ids)],
        "y_true": [0, 1, 2][:len(ids)],
        "y_pred": [0, 1, 1][:len(ids)],
    })


def test_exact_ids_align_independent_of_row_order() -> None:
    left = _predictions()
    right = _predictions().iloc[::-1].reset_index(drop=True)
    result = check_alignment(
        left,
        right,
        left_model="left",
        right_model="right",
        prediction_unit="window",
    )
    assert result.aligned
    assert result.matched_predictions == 3
    aligned = require_alignment(
        left,
        right,
        left_model="left",
        right_model="right",
        prediction_unit="window",
    )
    assert aligned["sample_id_left"].tolist() == [1, 2, 3]
    assert aligned["sample_id_right"].tolist() == [1, 2, 3]


def test_mismatched_ids_and_duplicate_ids_block_paired_test() -> None:
    left = _predictions()
    right = _predictions(ids=(1, 2, 4))
    result = check_alignment(
        left,
        right,
        left_model="left",
        right_model="right",
        prediction_unit="window",
    )
    assert not result.aligned
    assert "prediction IDs differ" in result.reason
    with pytest.raises(AlignmentError):
        require_alignment(
            left,
            right,
            left_model="left",
            right_model="right",
            prediction_unit="window",
        )

    duplicate = pd.concat([left, left.iloc[[0]]], ignore_index=True)
    duplicate_result = check_alignment(
        duplicate,
        left,
        left_model="left",
        right_model="right",
        prediction_unit="window",
    )
    assert not duplicate_result.aligned
    assert duplicate_result.left_duplicates == 1


def test_window_and_sequence_units_are_never_mixed() -> None:
    result = check_alignment(
        _predictions(),
        _predictions(),
        left_model="window",
        right_model="sequence",
        prediction_unit="window",
        right_prediction_unit="sequence",
    )
    assert not result.aligned
    assert "prediction units differ" in result.reason


def _entry(
    run: str,
    *,
    manifest: str,
    usable: bool,
    smoke: bool = False,
) -> InventoryEntry:
    return InventoryEntry(
        analysis_track="feature_window",
        model="model",
        seed=42,
        run_directory=run,
        config_hash="hash",
        dataset="dataset",
        representation="window",
        preprocessing="feature",
        prediction_unit="window",
        number_of_predictions=10,
        subjects=2,
        folds=5,
        prediction_file=f"{run}/predictions.parquet",
        metrics_file=f"{run}/metrics.json",
        usable=usable,
        reason="ok" if usable else "smoke-limited",
        manifest_status=manifest,
        config_match=True,
        smoke_limited=smoke,
        identity_column="sample_id",
    )


def test_smoke_is_excluded_and_canonical_selection_is_deterministic() -> None:
    entries = [
        _entry("20260101_000000", manifest="legacy_no_manifest", usable=True),
        _entry("20260103_000000", manifest="completed", usable=False, smoke=True),
        _entry("20260102_000000", manifest="completed", usable=True),
    ]
    first = select_canonical_runs(entries)
    second = select_canonical_runs(list(reversed(entries)))
    assert [entry.run_directory for entry in first if entry.canonical] == [
        "20260102_000000"
    ]
    assert [entry.run_directory for entry in second if entry.canonical] == [
        "20260102_000000"
    ]
    assert not next(entry for entry in first if entry.smoke_limited).canonical


def test_plan_only_has_no_output_side_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "analysis-output"
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"
    spec = {
        "analysis": {
            "name": "test",
            "output_dir": str(output_dir),
            "reports_dir": str(reports_dir),
            "figures_dir": str(figures_dir),
            "bootstrap_samples": 10,
        },
        "run_rules": [{
            "analysis_track": "feature_window",
            "model": "missing",
            "search_root": str(tmp_path / "does-not-exist"),
            "representation": "window",
            "prediction_unit": "window",
        }],
    }
    spec_path = tmp_path / "analysis.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    analysis = StatisticalAnalysis(spec_path)
    plan = analysis.plan()
    assert plan["writes_performed"] is False
    assert not output_dir.exists()
    assert not reports_dir.exists()
    assert not figures_dir.exists()
    cli.main(["--statistical-analysis", str(spec_path), "--plan-only"])
    assert "Writes performed: no" in capsys.readouterr().out
    assert not output_dir.exists()
    assert not reports_dir.exists()
    assert not figures_dir.exists()
