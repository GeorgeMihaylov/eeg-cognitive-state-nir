"""Audit cross-source logical recordings and write the deduplication map."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets.logical_recordings import (  # noqa: E402
    build_deduplication_selection,
    build_logical_recording_map,
    ensure_record_group_ids,
)


def _schema_records(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as input_file:
        schema = json.load(input_file)
    return {
        str(record["record_id"]): record for record in schema.get("records", [])
    }


def _compare_cached_windows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    chunk_size: int = 32,
) -> dict[str, Any]:
    keys = ["t_start", "t_end"]
    left_ok = left.loc[left["status"].eq("ok")]
    right_ok = right.loc[right["status"].eq("ok")]
    pairs = left_ok.merge(
        right_ok,
        on=keys,
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    label_match = bool(
        len(pairs) and np.array_equal(
            pairs["label_q5_left"].to_numpy(),
            pairs["label_q5_right"].to_numpy(),
        )
    )
    result: dict[str, Any] = {
        "common_accepted_windows": int(len(pairs)),
        "accepted_windows_left": int(len(left_ok)),
        "accepted_windows_right": int(len(right_ok)),
        "labels_match_on_common_windows": label_match,
        "all_common_windows_exact": None,
        "all_common_windows_close": None,
        "max_abs_difference": None,
        "rmse": None,
    }
    if pairs.empty or not {"cache_file_left", "cache_file_right"}.issubset(pairs):
        return result
    left_paths = pairs["cache_file_left"].astype(str).unique()
    right_paths = pairs["cache_file_right"].astype(str).unique()
    if len(left_paths) != 1 or len(right_paths) != 1:
        return result
    left_array = np.load(left_paths[0], mmap_mode="r", allow_pickle=False)
    right_array = np.load(right_paths[0], mmap_mode="r", allow_pickle=False)
    exact = True
    close = True
    maximum = 0.0
    squared_sum = 0.0
    value_count = 0
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs.iloc[start:start + chunk_size]
        left_values = np.asarray(
            left_array[chunk["cache_offset_left"].to_numpy(dtype=np.int64)],
            dtype=np.float32,
        )
        right_values = np.asarray(
            right_array[chunk["cache_offset_right"].to_numpy(dtype=np.int64)],
            dtype=np.float32,
        )
        difference = left_values.astype(np.float64) - right_values.astype(np.float64)
        exact = exact and bool(np.array_equal(left_values, right_values))
        close = close and bool(np.allclose(left_values, right_values, rtol=1e-5, atol=1e-4))
        maximum = max(maximum, float(np.max(np.abs(difference))))
        squared_sum += float(np.square(difference).sum())
        value_count += int(difference.size)
    result.update({
        "all_common_windows_exact": exact,
        "all_common_windows_close": close,
        "max_abs_difference": maximum,
        "rmse": math.sqrt(squared_sum / value_count) if value_count else None,
    })
    return result


def _inner_group_audit(manifest: pd.DataFrame) -> dict[str, Any]:
    checks = []
    accepted = manifest.loc[manifest["status"].eq("ok")]
    for outer_fold in sorted(accepted["outer_fold"].astype(int).unique()):
        train = accepted.loc[accepted["outer_fold"].astype(int) != outer_fold]
        groups = train["record_group_id"].astype(str).to_numpy()
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=42
        )
        inner_train, inner_validation = next(splitter.split(train, train["label_q5"], groups))
        overlap = sorted(set(groups[inner_train]) & set(groups[inner_validation]))
        checks.append({
            "outer_fold": int(outer_fold),
            "inner_train_logical_records": int(len(set(groups[inner_train]))),
            "inner_validation_logical_records": int(len(set(groups[inner_validation]))),
            "logical_record_overlap": overlap,
        })
    return {
        "strategy": "GroupShuffleSplit grouped by record_group_id",
        "folds": checks,
        "all_disjoint": all(not item["logical_record_overlap"] for item in checks),
    }


def _format_list(values: Any) -> str:
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return "<br>".join(str(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/interim/raw_eeg_window_index_w10.parquet"
    )
    parser.add_argument(
        "--schema", default="data/interim/raw_eeg_schema.json"
    )
    parser.add_argument(
        "--output-map", default="data/interim/logical_recording_map.parquet"
    )
    parser.add_argument(
        "--report", default="reports/logical_recording_audit.md"
    )
    args = parser.parse_args()

    manifest = ensure_record_group_ids(pd.read_parquet(args.manifest))
    schema = _schema_records(Path(args.schema))
    selection = build_deduplication_selection(manifest, record_schema=schema)
    logical_map = build_logical_recording_map(manifest, record_schema=schema)
    comparison_by_group: dict[str, dict[str, Any]] = {}

    for group_id, group_rows in manifest.groupby("record_group_id", sort=True):
        record_ids = sorted(group_rows["record_id"].astype(str).unique())
        supervised_by_record = {
            record_id: int(len(group_rows.loc[group_rows["record_id"].eq(record_id)]))
            for record_id in record_ids
        }
        accepted_by_record = {
            record_id: int(group_rows.loc[
                group_rows["record_id"].eq(record_id), "status"
            ].eq("ok").sum())
            for record_id in record_ids
        }
        comparison: dict[str, Any] = {
            "record_ids": record_ids,
            "supervised_windows_by_record": supervised_by_record,
            "accepted_windows_by_record": accepted_by_record,
            "time_ranges_match": None,
            "signal_relationship": "single_source",
        }
        if len(record_ids) > 1:
            left_id, right_id = record_ids[:2]
            left_schema = schema[left_id]
            right_schema = schema[right_id]
            tolerance = max(
                1.0 / float(left_schema["sampling_rate_hz"]),
                1.0 / float(right_schema["sampling_rate_hz"]),
            )
            time_match = bool(
                abs(left_schema["timestamp_min"] - right_schema["timestamp_min"])
                <= tolerance
                and abs(left_schema["timestamp_max"] - right_schema["timestamp_max"])
                <= tolerance
            )
            comparison["time_ranges_match"] = time_match
            signal = _compare_cached_windows(
                group_rows.loc[group_rows["record_id"].eq(left_id)],
                group_rows.loc[group_rows["record_id"].eq(right_id)],
            )
            comparison["signal_comparison"] = signal
            complete_coverage = (
                signal["common_accepted_windows"]
                == min(signal["accepted_windows_left"], signal["accepted_windows_right"])
            )
            if time_match and complete_coverage and signal["all_common_windows_exact"]:
                relationship = "full_duplicate_export_exact_on_supervised_windows"
            elif time_match and complete_coverage and signal["all_common_windows_close"]:
                relationship = "same_recording_numeric_export_variant"
            elif signal["common_accepted_windows"] > 0:
                relationship = "partial_overlap_or_export_variant"
            else:
                relationship = "no_supervised_raw_overlap"
            comparison["signal_relationship"] = relationship
        comparison_by_group[str(group_id)] = comparison

    logical_map["supervised_windows_by_record"] = logical_map["record_group_id"].map(
        lambda value: json.dumps(
            comparison_by_group[str(value)]["supervised_windows_by_record"],
            sort_keys=True,
        )
    )
    logical_map["accepted_windows_by_record"] = logical_map["record_group_id"].map(
        lambda value: json.dumps(
            comparison_by_group[str(value)]["accepted_windows_by_record"],
            sort_keys=True,
        )
    )
    logical_map["time_ranges_match"] = logical_map["record_group_id"].map(
        lambda value: comparison_by_group[str(value)]["time_ranges_match"]
    )
    logical_map["signal_relationship"] = logical_map["record_group_id"].map(
        lambda value: comparison_by_group[str(value)]["signal_relationship"]
    )
    logical_map["signal_comparison"] = logical_map["record_group_id"].map(
        lambda value: json.dumps(
            comparison_by_group[str(value)].get("signal_comparison", {}),
            sort_keys=True,
        )
    )
    output_map = Path(args.output_map)
    output_map.parent.mkdir(parents=True, exist_ok=True)
    logical_map.to_parquet(output_map, index=False)

    cross_source = logical_map.loc[logical_map["present_in_both_sources"]]
    outer_fold_violations = logical_map.loc[
        logical_map["outer_folds"].map(len) != 1
    ]
    inner_audit = _inner_group_audit(manifest)
    label_match_count = 0
    time_match_count = 0
    exact_count = 0
    close_variant_count = 0
    for comparison in comparison_by_group.values():
        signal = comparison.get("signal_comparison")
        if not signal:
            continue
        time_match_count += int(bool(comparison["time_ranges_match"]))
        label_match_count += int(signal["labels_match_on_common_windows"])
        exact_count += int(bool(signal["all_common_windows_exact"]))
        close_variant_count += int(bool(signal["all_common_windows_close"]))

    table = logical_map[[
        "record_group_id", "source_record_ids", "sources", "subject_id",
        "start_datetimes", "duration_seconds", "supervised_windows_by_record",
        "accepted_windows_by_record", "label_distribution", "outer_folds",
        "signal_relationship", "selected_record_id",
    ]].copy()
    for column in (
        "source_record_ids", "sources", "start_datetimes", "duration_seconds",
        "outer_folds",
    ):
        table[column] = table[column].map(_format_list)
    report_lines = [
        "# Logical recording audit",
        "",
        "## Summary",
        "",
        f"- Source-specific records: **{manifest['record_id'].nunique()}**",
        f"- Logical recordings: **{manifest['record_group_id'].nunique()}**",
        f"- Present in both sources: **{len(cross_source)}**",
        f"- Source records removed by deterministic deduplication: "
        f"**{manifest['record_id'].nunique() - manifest['record_group_id'].nunique()}**",
        f"- Cross-source groups with matching labels on common accepted windows: "
        f"**{label_match_count}/{len(cross_source)}**",
        f"- Cross-source groups with matching raw timestamp ranges: "
        f"**{time_match_count}/{len(cross_source)}**",
        f"- Exact cached-signal duplicates on all common accepted windows: "
        f"**{exact_count}/{len(cross_source)}**",
        f"- Numerically close signal exports on all common accepted windows: "
        f"**{close_variant_count}/{len(cross_source)}**",
        f"- Logical groups spanning multiple outer folds: "
        f"**{len(outer_fold_violations)}**",
        f"- Simulated inner train/validation logical overlap: "
        f"**{not inner_audit['all_disjoint']}**",
        "",
        "`record_group_id` is the canonical `record_id` with only its source "
        "prefix removed. Every cross-source pair maps to one subject and one "
        "precomputed subject-level outer fold.",
        "",
        "## Deterministic selection rule",
        "",
        "Source records are ranked by accepted-window fraction (descending), raw "
        "EEG row count (descending), mean accepted-window missing fraction "
        "(ascending), fixed source priority (`gpn_data`, then `Old_EEG`), and "
        "lexical `record_id`. The exact inputs and reason are stored in the map.",
        "",
        "## Signal comparison method",
        "",
        "Time coverage is compared from the raw audit schema within one raw "
        "sample. Labels and float32 raw tensors are compared for every accepted "
        "10-second window sharing `(t_start, t_end)`. Therefore an 'exact' result "
        "is direct array equality over the supervised cached coverage; it is not a "
        "byte-level claim about differently compressed CSV files.",
        "",
        "## Inner validation audit",
        "",
        "```json",
        json.dumps(inner_audit, indent=2),
        "```",
        "",
        "## Per-logical-record table",
        "",
        table.to_markdown(index=False),
        "",
        "## Outputs",
        "",
        f"- Logical map: `{output_map.as_posix()}` (ignored by Git via `data/`)",
        f"- Selection candidates audited: {len(selection)}",
        "",
    ]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({
        "source_specific_records": int(manifest["record_id"].nunique()),
        "logical_recordings": int(manifest["record_group_id"].nunique()),
        "cross_source_logical_recordings": int(len(cross_source)),
        "exact_signal_duplicates": exact_count,
        "close_signal_variants": close_variant_count,
        "matching_time_ranges": time_match_count,
        "outer_fold_violations": int(len(outer_fold_violations)),
        "inner_validation_disjoint": bool(inner_audit["all_disjoint"]),
        "output_map": str(output_map),
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
