from pathlib import Path

import numpy as np
import pandas as pd

from bench.analysis.pm_target_validity_audit import (
    boundary_distance_fraction,
    build_catalog_inventory,
    changed_value_events,
    circular_phase_summary,
    discover_pm_columns,
    interval_summary,
    normalize_is_active,
)
from bench.analysis.pm_target_validity_streaming import _MetricAccumulator


def test_discover_pm_columns_tracks_all_representations():
    columns = [
        "PM.Focus.Raw",
        "PM.Focus.Scaled",
        "PM.Focus.Min",
        "PM.Focus.Max",
        "PM.Focus.IsActive",
        "PM.Stress.Scaled",
    ]
    discovered = discover_pm_columns(columns)
    assert discovered["focus"] == {
        "Raw": True,
        "Scaled": True,
        "Min": True,
        "Max": True,
        "IsActive": True,
    }
    assert discovered["stress"]["Scaled"] is True
    assert discovered["stress"]["Raw"] is False


def test_catalog_inventory_is_record_metric_expansion():
    catalog = pd.DataFrame(
        {
            "source": ["gpn_data"],
            "subject_id": ["subject-a"],
            "main_rel_path": ["record.csv"],
            "pm_columns": [["PM.Focus.Raw", "PM.Focus.Scaled"]],
        }
    )
    inventory = build_catalog_inventory(catalog)
    assert len(inventory) == 7
    focus = inventory.loc[inventory["metric"] == "focus"].iloc[0]
    assert bool(focus["has_raw"])
    assert bool(focus["has_scaled"])
    assert not bool(focus["has_isactive"])


def test_changed_value_events_keeps_only_updates():
    frame = pd.DataFrame(
        {
            "Timestamp": [0.0, 1.0, 10.0, 11.0, 20.0],
            "PM.Focus.Scaled": [0.1, 0.1, 0.2, 0.2, 0.3],
        }
    )
    events = changed_value_events(frame, "PM.Focus.Scaled")
    assert events["Timestamp"].tolist() == [0.0, 10.0, 20.0]


def test_timing_summaries_detect_ten_second_period_and_phase():
    timestamps = np.array([2.0, 12.0, 22.0, 32.0])
    intervals = interval_summary(timestamps)
    phase = circular_phase_summary(timestamps)
    assert intervals["interval_median_seconds"] == 10.0
    assert intervals["near_10s_fraction"] == 1.0
    assert abs(phase["phase_mean_seconds"] - 2.0) < 1e-9
    assert phase["phase_concentration"] > 0.999999


def test_normalize_is_active_handles_text_and_numeric_values():
    text = pd.Series(["true", "false", "ACTIVE", "inactive", "bad"])
    normalized = normalize_is_active(text)
    np.testing.assert_allclose(
        normalized.iloc[:4].to_numpy(dtype=float), [1.0, 0.0, 1.0, 0.0]
    )
    assert np.isnan(normalized.iloc[4])


def test_boundary_distance_fraction_counts_near_internal_edges():
    values = np.array([0.0, 0.30, 0.34, 0.66, 0.70, 1.0])
    boundaries = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    fraction = boundary_distance_fraction(
        values, boundaries, relative_margin=0.05
    )
    assert np.isfinite(fraction)
    assert 0.0 < fraction < 1.0


def test_streaming_accumulator_preserves_events_across_chunk_boundary():
    accumulator = _MetricAccumulator("focus")
    first = pd.DataFrame(
        {
            "Timestamp": [2.0, 3.0, 12.0],
            "PM.Focus.Raw": [10.0, 10.0, 20.0],
            "PM.Focus.Scaled": [0.1, 0.1, 0.2],
            "PM.Focus.IsActive": [1, 1, 1],
        }
    )
    second = pd.DataFrame(
        {
            "Timestamp": [13.0, 22.0, 23.0],
            "PM.Focus.Raw": [20.0, 30.0, 30.0],
            "PM.Focus.Scaled": [0.2, 0.3, 0.3],
            "PM.Focus.IsActive": [1, 0, 0],
        }
    )
    accumulator.update(first)
    accumulator.update(second)
    assert accumulator.event_timestamps == [2.0, 12.0, 22.0]
    result = accumulator.result(
        source="gpn_data",
        subject_id="subject-a",
        path=Path("dummy"),
        rows_read=6,
    )
    assert abs(result["raw_scaled_corr"] - 1.0) < 1e-12
    assert result["scaled_when_inactive_fraction"] == 2.0 / 6.0
    assert result["interval_median_seconds"] == 10.0
