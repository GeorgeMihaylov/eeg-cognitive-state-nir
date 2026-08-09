from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bench.datasets.pm_union_raw_contract import (
    finalize_pm_union_composite,
    plan_pm_union_composite,
    pm_union_availability,
)
from bench.datasets.raw_eeg_window_dataset import (
    CANONICAL_EEG_CHANNELS,
    RawEEGWindowArrayView,
    infer_record_id,
)
from bench.datasets.raw_preprocessing import raw_preprocessing_hash
from bench.tasks.target_registry import PM_TARGET_COLUMNS


def _processed_fixture() -> pd.DataFrame:
    rows = []
    definitions = [
        ("S1", 1, 0.2, None, 0),
        ("S1", 1, None, 0.8, None),
        ("S2", 2, None, 0.6, None),
        ("S2", 2, None, None, None),
        ("S2", 2, 0.3, None, 1),
    ]
    for sample_id, (subject, day, attention, stress, label) in enumerate(definitions):
        catalog_row = {
            "source": "gpn_data",
            "subject_id": subject,
            "day": f"day{day}",
            "part": "",
            "datetime_from_name": f"2024.01.0{day}T12.00.00p03.00",
        }
        row = {
            **catalog_row,
            "record_id": infer_record_id(catalog_row),
            "t_start": float(sample_id * 10),
            "t_end": float(sample_id * 10 + 10),
            "label_q5": label,
            **{column: np.nan for column in PM_TARGET_COLUMNS},
        }
        row["target_attention"] = attention
        row["target_stress"] = stress
        rows.append(row)
    return pd.DataFrame(rows)


def _contract_fixture(tmp_path: Path) -> dict[str, Path | pd.DataFrame]:
    processed = _processed_fixture()
    processed_path = tmp_path / "processed.parquet"
    processed.to_parquet(processed_path, index=False)
    catalog_columns = [
        "source", "subject_id", "day", "part", "datetime_from_name"
    ]
    catalog = processed.loc[:, catalog_columns].drop_duplicates().copy()
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text("unused", encoding="utf-8")
    catalog["main_path"] = str(raw_path)
    catalog["main_rel_path"] = str(raw_path)
    catalog["header_row"] = 0
    catalog["separator"] = ","
    catalog["time_columns"] = json.dumps(["Timestamp"])
    catalog["eeg_columns"] = json.dumps(["EEG.AF3"])
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    semantic_hash = raw_preprocessing_hash(
        None, channels=CANONICAL_EEG_CHANNELS
    )
    old_cache = tmp_path / "historical.npy"
    np.save(old_cache, np.zeros((2, 14, 2560), dtype=np.float32))
    historical = pd.DataFrame(
        {
            "sample_id": [0, 4],
            "source": "gpn_data",
            "subject_id": ["S1", "S2"],
            "record_id": [processed.loc[0, "record_id"], processed.loc[4, "record_id"]],
            "record_group_id": [
                str(processed.loc[0, "record_id"]).split("__", 1)[1],
                str(processed.loc[4, "record_id"]).split("__", 1)[1],
            ],
            "t_start": [0.0, 40.0],
            "t_end": [10.0, 50.0],
            "label_q5": [0, 1],
            "outer_fold": [1, 2],
            "status": "ok",
            "rejection_reason": "",
            "cache_file": str(old_cache),
            "cache_offset": [0, 1],
            "n_channels": 14,
            "n_samples_expected": 2560,
            "sfreq_target": 256.0,
            "preprocessing_hash": semantic_hash,
            "preprocessing_variant": "raw",
        }
    )
    historical_path = tmp_path / "historical.parquet"
    historical.to_parquet(historical_path, index=False)
    logical = historical.loc[
        :, ["record_group_id", "record_id"]
    ].rename(columns={"record_id": "selected_record_id"})
    logical_path = tmp_path / "logical.parquet"
    logical.to_parquet(logical_path, index=False)
    audit_path = tmp_path / "schema.json"
    audit_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    return {
        "processed": processed,
        "processed_path": processed_path,
        "catalog_path": catalog_path,
        "historical": historical,
        "historical_path": historical_path,
        "logical_path": logical_path,
        "audit_path": audit_path,
        "semantic_hash": semantic_hash,
    }


def test_pm_union_membership_is_any_finite_pm_and_ignores_label_q5() -> None:
    frame = _processed_fixture()
    first = pm_union_availability(frame)
    frame["label_q5"] = [None, 4, 3, 2, None]
    second = pm_union_availability(frame)
    assert first.tolist() == [True, True, True, False, True]
    np.testing.assert_array_equal(first, second)


def test_pm_union_plan_reuses_historical_rows_and_fixed_contracts(tmp_path: Path) -> None:
    fixture = _contract_fixture(tmp_path)
    before = Path(fixture["historical_path"]).read_bytes()
    plan, delta, summary = plan_pm_union_composite(
        fixture["processed_path"],
        fixture["catalog_path"],
        fixture["historical_path"],
        fixture["logical_path"],
        audit_schema_path=fixture["audit_path"],
    )
    assert Path(fixture["historical_path"]).read_bytes() == before
    assert plan["sample_id"].tolist() == [0, 1, 2, 4]
    assert delta["sample_id"].tolist() == [1, 2]
    assert delta["label_q5"].isna().all()
    assert plan["sample_id"].is_unique
    assert summary["historical_rows_reused"] == 2
    assert summary["delta_candidate_rows"] == 2
    assert summary["delta_deduplicated_candidate_rows"] == 2
    assert summary["candidate_deduplicated_rows"] == 4
    assert summary["target_candidate_deduplicated_rows"] == {
        "target_attention": 2,
        "target_engagement": 0,
        "target_excitement": 0,
        "target_stress": 2,
        "target_relaxation": 0,
        "target_interest": 0,
        "target_focus": 0,
    }
    assert summary["subject_counts_by_outer_fold"] == {"1": 1, "2": 1}
    old = plan[plan["sample_id"].isin([0, 4])].reset_index(drop=True)
    pd.testing.assert_series_equal(
        old["cache_file"],
        fixture["historical"]["cache_file"],
        check_names=False,
    )
    assert old["cache_offset"].tolist() == [0, 1]
    assert plan.groupby("subject_id")["outer_fold"].nunique().eq(1).all()


def test_final_composite_reads_historical_and_separate_delta_shards(tmp_path: Path) -> None:
    fixture = _contract_fixture(tmp_path)
    _, delta, _ = plan_pm_union_composite(
        fixture["processed_path"],
        fixture["catalog_path"],
        fixture["historical_path"],
        fixture["logical_path"],
        audit_schema_path=fixture["audit_path"],
    )
    delta_cache_root = tmp_path / "pm_union_delta"
    delta_cache_root.mkdir()
    delta_cache = delta_cache_root / "delta.npy"
    np.save(delta_cache, np.ones((2, 14, 2560), dtype=np.float32))
    delta["status"] = "ok"
    delta["cache_file"] = str(delta_cache)
    delta["cache_offset"] = [0, 1]
    composite = finalize_pm_union_composite(
        fixture["historical"],
        delta,
        expected_preprocessing_hash=fixture["semantic_hash"],
    )
    view = RawEEGWindowArrayView(composite)
    assert view.shape == (4, 1, 14, 2560)
    assert float(view[0].mean()) == 0.0
    assert float(view[1].mean()) == 1.0
    assert composite["cache_file"].nunique() == 2


def test_relative_composite_cache_paths_resolve_from_runtime_root(tmp_path: Path) -> None:
    old = tmp_path / "cache" / "old.npy"
    delta = tmp_path / "delta" / "new.npy"
    old.parent.mkdir()
    delta.parent.mkdir()
    np.save(old, np.zeros((1, 2, 4), dtype=np.float32))
    np.save(delta, np.ones((1, 2, 4), dtype=np.float32))
    manifest = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "status": "ok",
            "cache_file": ["cache/old.npy", "delta/new.npy"],
            "cache_offset": [0, 0],
            "n_channels": 2,
            "n_samples_expected": 4,
        }
    )
    view = RawEEGWindowArrayView(manifest, cache_path_root=tmp_path)
    assert float(view[0].mean()) == 0.0
    assert float(view[1].mean()) == 1.0


def test_finalize_rejects_sample_overlap_and_preprocessing_mismatch(tmp_path: Path) -> None:
    fixture = _contract_fixture(tmp_path)
    historical = fixture["historical"]
    with np.testing.assert_raises_regex(ValueError, "overlap sample_id"):
        finalize_pm_union_composite(
            historical,
            historical.iloc[[0]],
            expected_preprocessing_hash=fixture["semantic_hash"],
        )
    bad = historical.iloc[[0]].copy()
    bad["sample_id"] = 99
    bad["preprocessing_hash"] = "different"
    with np.testing.assert_raises_regex(ValueError, "incompatible"):
        finalize_pm_union_composite(
            historical,
            bad,
            expected_preprocessing_hash=fixture["semantic_hash"],
        )
