#!/usr/bin/env python3
"""Audit COG-BCI layouts and the explicit project Emotiv channel contract."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets.channel_contracts import (  # noqa: E402
    PROJECT_EMOTIV_CHANNEL_ORDER,
    apply_channel_policy,
    load_cog_bci_emotiv_mapping,
)
from bench.datasets.cog_bci_dataset import COGBCIDataset  # noqa: E402


def _write_json(path: Path, document: Any) -> None:
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _parse_list(value: Any) -> list[str]:
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a serialized list, got {value!r}")
    return [str(item) for item in parsed]


def _record_layout_tables(dataset: COGBCIDataset):
    records = dataset.records
    layout_counter = Counter(record.channel_layout_id for record in records)
    representative = {}
    for record in records:
        representative.setdefault(record.channel_layout_id, record)
    layout_rows = []
    for layout_id in sorted(representative):
        record = representative[layout_id]
        layout_rows.append({
            "channel_layout_id": layout_id,
            "records": int(layout_counter[layout_id]),
            "channel_count_total": record.channel_count_total,
            "channel_count_eeg": len(record.eeg_channel_names),
            "channel_count_auxiliary": len(record.auxiliary_channel_names),
            "has_cz": record.has_cz,
            "has_ecg1": record.has_ecg1,
            "channel_names": _json_list(record.channel_names_total),
            "eeg_channel_names": _json_list(record.eeg_channel_names),
            "auxiliary_channel_names": _json_list(
                record.auxiliary_channel_names
            ),
        })

    union = set().union(*(set(record.channel_names_total) for record in records))
    eeg_union = set().union(*(set(record.eeg_channel_names) for record in records))
    eeg_common = set.intersection(
        *(set(record.eeg_channel_names) for record in records)
    )
    first_eeg_order = records[0].eeg_channel_names
    common_order = tuple(name for name in first_eeg_order if name in eeg_common)
    union_order = tuple(
        dict.fromkeys(
            name for record in records for name in record.eeg_channel_names
        )
    )
    presence_rows = []
    for name in sorted(union, key=lambda value: value.casefold()):
        present = sum(name in record.channel_names_total for record in records)
        eeg_present = sum(name in record.eeg_channel_names for record in records)
        auxiliary_present = sum(
            name in record.auxiliary_channel_names for record in records
        )
        missing_position = sum(
            name in record.channels_without_scalp_position for record in records
        )
        presence_rows.append({
            "channel": name,
            "role": "eeg" if name in eeg_union else "auxiliary",
            "records_present": present,
            "records_absent": len(records) - present,
            "eeg_records": eeg_present,
            "auxiliary_records": auxiliary_present,
            "records_without_scalp_position": missing_position,
            "common_eeg": name in eeg_common,
        })
    common_document = {
        "schema_version": 1,
        "record_count": len(records),
        "layout_count": len(layout_counter),
        "common_eeg_channel_count": len(common_order),
        "common_eeg_channel_order": list(common_order),
        "eeg_union_channel_count": len(union_order),
        "eeg_union_channel_order": list(union_order),
        "partial_eeg_channels": [
            name for name in union_order if name not in eeg_common
        ],
        "auxiliary_channels": sorted(
            set().union(
                *(set(record.auxiliary_channel_names) for record in records)
            )
        ),
        "casefold_collisions": sorted({
            name
            for name in union
            if sum(other.casefold() == name.casefold() for other in union) > 1
        }),
        "duplicate_channel_name_records": sum(
            len(set(record.channel_names_total)) != len(record.channel_names_total)
            for record in records
        ),
    }
    return (
        pd.DataFrame(layout_rows),
        pd.DataFrame(presence_rows),
        common_document,
    )


def _project_contract_audit(
    catalog_path: Path,
    raw_cache_dir: Path,
) -> dict[str, Any]:
    catalog = pd.read_csv(catalog_path)
    source_rows = {}
    canonical = tuple(PROJECT_EMOTIV_CHANNEL_ORDER)
    for source, group in catalog.groupby("source", sort=True):
        layouts = [_parse_list(value) for value in group["eeg_columns"]]
        unique_layouts = sorted({tuple(layout) for layout in layouts})
        source_rows[str(source)] = {
            "records": len(group),
            "unique_eeg_prefixed_layouts": len(unique_layouts),
            "all_signal_channels_present": all(
                set(canonical).issubset(layout) for layout in layouts
            ),
            "signal_order_in_catalog_layout": [
                name for name in unique_layouts[0] if name in canonical
            ],
            "non_signal_eeg_prefixed_fields": [
                name for name in unique_layouts[0] if name not in canonical
            ],
        }

    shard_files = sorted(raw_cache_dir.glob("*.json"))
    shard_orders = []
    for path in shard_files:
        with path.open(encoding="utf-8") as input_file:
            shard_orders.append(tuple(json.load(input_file)["channels"]))
    unique_shard_orders = sorted(set(shard_orders))
    return {
        "schema_version": 1,
        "contract_name": "project_emotiv_raw_v1",
        "canonical_order": list(canonical),
        "canonical_order_source": (
            "bench/datasets/channel_contracts.py::"
            "PROJECT_EMOTIV_CHANNEL_ORDER"
        ),
        "catalog_path": catalog_path.as_posix(),
        "catalog_sources": source_rows,
        "catalog_sources_same_signal_order": len({
            tuple(value["signal_order_in_catalog_layout"])
            for value in source_rows.values()
        }) == 1,
        "raw_cache_metadata_dir": raw_cache_dir.as_posix(),
        "raw_cache_shards_checked": len(shard_files),
        "raw_cache_unique_channel_orders": [
            list(order) for order in unique_shard_orders
        ],
        "raw_cache_matches_contract": (
            bool(shard_files) and unique_shard_orders == [canonical]
        ),
        "npy_axis_contract": "[window, channel, time]",
        "channel_manifest_present": bool(shard_files),
        "observed_project_sensor_coordinates_available": False,
    }


def _coordinate_audit(dataset: COGBCIDataset) -> pd.DataFrame:
    import mne

    record = dataset.query(
        subject_ids=["sub-01"], task_families=["flanker"]
    )[0]
    raw = dataset.open_raw(
        record.record_id,
        preload=False,
        include_auxiliary_channels=True,
    )
    reference = mne.channels.make_standard_montage("standard_1020")
    reference_positions = reference.get_positions()["ch_pos"]
    rows = []
    for mapping in load_cog_bci_emotiv_mapping():
        source_index = raw.ch_names.index(mapping.cog_bci_channel)
        source = raw.info["chs"][source_index]["loc"][:3].astype(float)
        target = np.asarray(reference_positions[mapping.cog_bci_channel])
        source_available = bool(np.isfinite(source).all() and np.any(source))
        reference_available = bool(np.isfinite(target).all() and np.any(target))
        distance = (
            float(np.linalg.norm(source - target) * 1000.0)
            if source_available and reference_available
            else None
        )
        rows.append({
            "channel": mapping.emotiv_channel,
            "cog_bci_channel": mapping.cog_bci_channel,
            "cog_bci_x_m": source[0] if source_available else None,
            "cog_bci_y_m": source[1] if source_available else None,
            "cog_bci_z_m": source[2] if source_available else None,
            "reference": "MNE standard_1020",
            "reference_x_m": target[0] if reference_available else None,
            "reference_y_m": target[1] if reference_available else None,
            "reference_z_m": target[2] if reference_available else None,
            "distance_mm": distance,
            "coordinate_status": (
                "standard_reference_exact"
                if distance is not None and distance < 1e-6
                else "standard_reference_difference"
            ),
            "project_observed_coordinates_available": False,
        })
    return pd.DataFrame(rows)


def _policy_smoke(dataset: COGBCIDataset):
    smoke_rows = []
    errors = []
    selected_records = {
        "sub-01": dataset.query(
            subject_ids=["sub-01"], task_families=["flanker"]
        )[0],
        "sub-10": dataset.query(
            subject_ids=["sub-10"], task_families=["flanker"]
        )[0],
    }
    for subject_id, record in selected_records.items():
        original = dataset.open_raw(
            record.record_id,
            preload=False,
            include_auxiliary_channels=True,
        )
        original_names = tuple(original.ch_names)
        for policy_name in (
            "cog_bci_native",
            "cog_bci_common",
            "emotiv_common",
        ):
            try:
                policy = dataset.get_channel_policy(policy_name)
                result = apply_channel_policy(
                    original,
                    policy,
                    record_metadata=record,
                    copy=True,
                )
                smoke_rows.append({
                    "subject_id": subject_id,
                    "record_id": record.record_id,
                    "source_layout_id": record.channel_layout_id,
                    "physical_has_cz": record.has_cz,
                    "policy": policy_name,
                    "selected_channel_count": len(result.raw.ch_names),
                    "selected_channel_names": list(result.raw.ch_names),
                    "selected_has_cz": "Cz" in result.raw.ch_names,
                    "ecg1_excluded": "ECG1" not in result.raw.ch_names,
                    "source_preload": bool(original.preload),
                    "selected_preload": bool(result.raw.preload),
                    "source_unchanged": tuple(original.ch_names) == original_names,
                    "copy_created": result.raw is not original,
                    "sfreq_before": float(original.info["sfreq"]),
                    "sfreq_after": float(result.raw.info["sfreq"]),
                    "n_times_before": int(original.n_times),
                    "n_times_after": int(result.raw.n_times),
                    "provenance": dict(result.provenance),
                })
            except Exception as error:
                errors.append({
                    "subject_id": subject_id,
                    "record_id": record.record_id,
                    "policy": policy_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
    return smoke_rows, errors


def _mapping_frame() -> pd.DataFrame:
    return pd.DataFrame([
        mapping.to_dict() for mapping in load_cog_bci_emotiv_mapping()
    ])


def _runtime_report(summary: dict[str, Any]) -> str:
    return f"""# COG-BCI channel audit

Status: `diagnostic`

- records: {summary['record_count']}
- layouts: {summary['layout_count']}
- common EEG channels: {summary['common_eeg_channel_count']}
- EEG union channels: {summary['eeg_union_channel_count']}
- partial EEG channels: {', '.join(summary['partial_eeg_channels'])}
- project Emotiv channels: {summary['project_emotiv_channel_count']}
- mapping complete: {summary['mapping_complete']}
- mapping type: explicit namespace alias
- raw cache order verified: {summary['raw_cache_matches_contract']}
- policy smoke checks: {summary['policy_smoke_count']}
- policy smoke errors: {summary['error_count']}

No resampling, filtering, rereferencing, interpolation, windowing or training
was performed.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = COGBCIDataset({
        "data_path": args.data_root,
        "index_cache_path": args.index_cache,
        "use_index_cache": True,
    })
    layouts, presence, common = _record_layout_tables(dataset)
    project = _project_contract_audit(
        Path(args.catalog_path), Path(args.raw_cache_dir)
    )
    mapping = _mapping_frame()
    coordinates = _coordinate_audit(dataset)
    smoke_rows, errors = _policy_smoke(dataset)

    layouts.to_csv(output_dir / "cog_bci_channel_layouts.csv", index=False)
    presence.to_csv(output_dir / "cog_bci_channel_presence.csv", index=False)
    _write_json(output_dir / "cog_bci_common_channels.json", common)
    _write_json(output_dir / "project_emotiv_channel_contract.json", project)
    mapping.to_csv(output_dir / "cog_bci_emotiv_mapping.csv", index=False)
    coordinates.to_csv(output_dir / "coordinate_audit.csv", index=False)
    _write_json(output_dir / "channel_policy_smoke.json", smoke_rows)
    pd.DataFrame(
        errors,
        columns=["subject_id", "record_id", "policy", "error_type", "error"],
    ).to_csv(output_dir / "errors.csv", index=False)

    summary = {
        "schema_version": 1,
        "result_status": "diagnostic",
        "record_count": len(dataset.records),
        "subject_count": len({record.subject_id for record in dataset.records}),
        "session_count": len({record.session_id for record in dataset.records}),
        "layout_count": common["layout_count"],
        "common_eeg_channel_count": common["common_eeg_channel_count"],
        "common_eeg_channel_order": common["common_eeg_channel_order"],
        "eeg_union_channel_count": common["eeg_union_channel_count"],
        "partial_eeg_channels": common["partial_eeg_channels"],
        "project_emotiv_channel_count": len(PROJECT_EMOTIV_CHANNEL_ORDER),
        "project_emotiv_channel_order": list(PROJECT_EMOTIV_CHANNEL_ORDER),
        "mapping_complete": (
            len(mapping) == len(PROJECT_EMOTIV_CHANNEL_ORDER)
            and set(mapping["status"]) == {"explicit_alias_match"}
        ),
        "mapping_ambiguities": 0,
        "coordinate_reference": "MNE standard_1020",
        "project_observed_coordinates_available": False,
        "coordinate_max_distance_mm": float(
            coordinates["distance_mm"].max()
        ),
        "raw_cache_matches_contract": project["raw_cache_matches_contract"],
        "policy_smoke_count": len(smoke_rows),
        "error_count": len(errors),
        "forbidden_signal_operations_performed": False,
        "future_cz_interpolation_policy": "not_implemented",
    }
    _write_json(output_dir / "channel_audit_summary.json", summary)
    report = _runtime_report(summary)
    (output_dir / "channel_audit_report.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise RuntimeError(
            f"Channel policy smoke produced {len(errors)} error(s); "
            f"see {output_dir / 'errors.csv'}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="data/raw/cog_bci/extracted"
    )
    parser.add_argument(
        "--index-cache",
        default="benchmark_results/cog_bci_loader/record_index.json",
    )
    parser.add_argument(
        "--catalog-path", default="data/interim/emotiv_record_catalog.csv"
    )
    parser.add_argument(
        "--raw-cache-dir",
        default=(
            "data/interim/raw_eeg_cache_w10_v3/"
            "raw-2251ca950a467267"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/cog_bci_channel_audit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
