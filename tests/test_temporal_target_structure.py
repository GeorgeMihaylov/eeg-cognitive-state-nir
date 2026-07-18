from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

import cli
from bench.analysis.temporal_target_structure import (
    blocked_time_predictions,
    calculate_runs,
    calculate_temporal_statistics,
    make_lag_pairs,
    prepare_temporal_frame,
    previous_label_predictions,
)


def _temporal_source_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_id = 100
    definitions = [
        ("gpn_data", "s1", "r1", [0, 0, 1, 1, 2, 3, 3, 4]),
        ("gpn_data", "s1", "r2", [4, 4, 4, 3, 2, 2, 1, 0]),
        ("Old_EEG", "s2", "r3", [0, 1, 2, 3, 4, 4, 3, 2]),
    ]
    for source, subject, record, labels in definitions:
        for index, label in enumerate(labels):
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "subject_id": subject,
                    "record_id": record,
                    "t_start": float(index * 10),
                    "t_end": float((index + 1) * 10),
                    "target_focus": float(label) / 4.0 + index * 0.001,
                    "label_q5": label,
                }
            )
            sample_id += 1
    return pd.DataFrame(rows).sample(frac=1.0, random_state=13).reset_index(drop=True)


def test_temporal_order_is_deterministic_and_record_local() -> None:
    source = _temporal_source_frame()
    first = prepare_temporal_frame(source)
    second = prepare_temporal_frame(
        source.sample(frac=1.0, random_state=99).reset_index(drop=True)
    )
    columns = ["source", "subject_id", "record_id", "t_start", "sample_id"]
    pdt.assert_frame_equal(first[columns], second[columns])
    assert first.equals(
        first.sort_values(columns, kind="mergesort").reset_index(drop=True)
    )

    pairs = make_lag_pairs(first, value_col="target_focus", lag=1)
    lookup = first.set_index("sample_id")
    previous = lookup.loc[pairs["previous_sample_id"]]
    for column in ("source", "subject_id", "record_id"):
        assert previous[column].to_numpy().tolist() == pairs[column].tolist()
    assert np.all(previous["t_start"].to_numpy() < pairs["t_start"].to_numpy())
    assert np.all(
        pairs["absolute_window_index"].to_numpy()
        - previous["absolute_window_index"].to_numpy()
        == 1
    )


def test_lag_uses_only_past_and_never_crosses_missing_window() -> None:
    source = _temporal_source_frame()
    source.loc[source["sample_id"] == 103, ["target_focus", "label_q5"]] = np.nan
    prepared = prepare_temporal_frame(source)
    pairs = make_lag_pairs(prepared, value_col="target_focus", lag=2)
    lookup = prepared.set_index("sample_id")
    previous = lookup.loc[pairs["previous_sample_id"]]
    assert np.all(previous["t_start"].to_numpy() < pairs["t_start"].to_numpy())
    assert np.all(
        pairs["absolute_window_index"].to_numpy()
        - previous["absolute_window_index"].to_numpy()
        == 2
    )
    # A missing middle target must not turn non-contiguous supervised rows into lag-1 pairs.
    lag_one = make_lag_pairs(prepared, value_col="label_q5", lag=1)
    assert 104 not in set(lag_one.loc[lag_one["previous_sample_id"] == 102, "sample_id"])


def test_previous_label_excludes_first_window_of_every_record() -> None:
    prepared = prepare_temporal_frame(_temporal_source_frame())
    predictions = previous_label_predictions(prepared)
    first_ids = set(
        prepared.sort_values(
            ["source", "subject_id", "record_id", "t_start", "sample_id"],
            kind="mergesort",
        ).groupby(["source", "subject_id", "record_id"], sort=False).head(1)["sample_id"]
    )
    assert first_ids.isdisjoint(set(predictions["sample_id"]))
    assert len(predictions) == len(prepared) - len(first_ids)
    assert predictions["prediction_id"].is_unique
    assert np.array_equal(predictions["y_pred"], prepared.set_index("sample_id").loc[
        predictions["previous_sample_id"], "label_q5"
    ].to_numpy())


def test_runs_transitions_and_blocked_gap_are_finite() -> None:
    prepared = prepare_temporal_frame(_temporal_source_frame())
    prepared["outer_fold"] = np.where(prepared["subject_id"] == "s1", 1, 2)
    statistics, runs = calculate_temporal_statistics(prepared)
    assert statistics["sequence_definition"]["records_crossed"] is False
    assert statistics["sequence_definition"]["future_values_used"] is False
    assert statistics["label_q5"]["transitions"]["pairs"] == len(prepared) - 3
    assert runs["length_windows"].sum() == len(prepared)
    assert (runs["length_windows"] > 0).all()

    blocked = blocked_time_predictions(prepared, early_end=0.25, late_start=0.75)
    assert set(blocked["protocol"]) == {
        "blocked_time_early_adjacent",
        "blocked_time_late_adjacent",
        "blocked_time_cross_gap",
    }
    bridges = blocked.loc[blocked["protocol"] == "blocked_time_cross_gap"]
    assert len(bridges) == 3
    assert (bridges["gap_windows"] > 1).all()
    assert (bridges["gap_seconds"] > 10.0).all()


def test_temporal_functions_do_not_mutate_source_frame() -> None:
    source = _temporal_source_frame()
    original = source.copy(deep=True)
    prepared = prepare_temporal_frame(source)
    calculate_temporal_statistics(prepared)
    previous_label_predictions(prepared)
    pdt.assert_frame_equal(source, original)


def test_cli_plan_only_has_no_side_effects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "does-not-need-to-exist.parquet"
    output_dir = tmp_path / "outputs"
    temporal_report = tmp_path / "reports" / "temporal.md"
    diagnostic_report = tmp_path / "reports" / "diagnostic.md"
    summary_path = tmp_path / "reports" / "summary.json"
    spec_path = tmp_path / "audit.yaml"
    spec = {
        "audit": {
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "temporal_report": str(temporal_report),
            "diagnostic_report": str(diagnostic_report),
            "summary_path": str(summary_path),
        },
        "diagnostic_baselines": {"n_splits": 5},
    }
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    gitignore_before = gitignore.read_bytes()

    cli.main(["--temporal-target-audit", str(spec_path), "--plan-only"])
    output = capsys.readouterr().out
    assert "EEG/POW features: none" in output
    assert "Deep models trained: 0" in output
    assert "Writes performed: no" in output
    assert not output_dir.exists()
    assert not temporal_report.exists()
    assert not diagnostic_report.exists()
    assert not summary_path.exists()
    assert gitignore.read_bytes() == gitignore_before
