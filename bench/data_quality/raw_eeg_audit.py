"""Audit real raw EEG exports and write a machine-readable schema."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets.raw_eeg_window_dataset import (
    CANONICAL_EEG_CHANNELS,
    _parse_list,
    infer_record_id,
    resolve_raw_path,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def audit_record(row: pd.Series, repo_root: Path, chunksize: int) -> Dict[str, Any]:
    path = resolve_raw_path(row, repo_root=repo_root)
    channels = list(CANONICAL_EEG_CHANNELS)
    catalog_eeg = _parse_list(row.get("eeg_columns"))
    available = [channel for channel in channels if channel in catalog_eeg]
    missing = [channel for channel in channels if channel not in catalog_eeg]
    extra = sorted(set(catalog_eeg) - set(channels))
    usecols = ["Timestamp", *available]
    n_rows = 0
    finite_timestamp_rows = 0
    duplicate_timestamps = 0
    decreasing_timestamps = 0
    nonpositive_deltas = 0
    gap_count = 0
    irregular_count = 0
    delta_count = 0
    timestamp_min = float("inf")
    timestamp_max = float("-inf")
    previous_timestamp = None
    delta_samples: list[np.ndarray] = []
    amplitude_samples: list[np.ndarray] = []
    channel_missing = np.zeros(len(channels), dtype=np.int64)
    channel_min = np.full(len(channels), np.inf, dtype=np.float64)
    channel_max = np.full(len(channels), -np.inf, dtype=np.float64)
    nominal_delta = None
    nominal_sampling_rate = None

    reader = pd.read_csv(
        path,
        header=int(row.get("header_row", 0)),
        sep=str(row.get("separator", ",")),
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        numeric = chunk.apply(pd.to_numeric, errors="coerce")
        n_rows += len(numeric)
        timestamps = numeric["Timestamp"].to_numpy(dtype=np.float64)
        finite_time = timestamps[np.isfinite(timestamps)]
        finite_timestamp_rows += len(finite_time)
        if len(finite_time):
            timestamp_min = min(timestamp_min, float(finite_time.min()))
            timestamp_max = max(timestamp_max, float(finite_time.max()))
            if previous_timestamp is not None:
                finite_time = np.concatenate(([previous_timestamp], finite_time))
            deltas = np.diff(finite_time)
            previous_timestamp = float(finite_time[-1])
            duplicate_timestamps += int(np.sum(deltas == 0))
            decreasing_timestamps += int(np.sum(deltas < 0))
            nonpositive_deltas += int(np.sum(deltas <= 0))
            positive = deltas[deltas > 0]
            if len(positive) and nominal_delta is None:
                measured_rate = 1.0 / float(np.median(positive))
                nominal_sampling_rate = min(
                    (128.0, 256.0), key=lambda rate: abs(rate - measured_rate)
                )
                nominal_delta = 1.0 / nominal_sampling_rate
            if nominal_delta is not None:
                gap_count += int(np.sum(positive > 1.5 * nominal_delta))
                irregular_count += int(
                    np.sum(np.abs(positive - nominal_delta) > 0.25 * nominal_delta)
                )
            delta_count += len(positive)
            if len(positive):
                stride = max(1, len(positive) // 4096)
                delta_samples.append(positive[::stride][:4096])

        for channel_index, channel in enumerate(channels):
            if channel not in numeric:
                channel_missing[channel_index] += len(numeric)
                continue
            values = numeric[channel].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            channel_missing[channel_index] += len(values) - len(finite)
            if len(finite):
                channel_min[channel_index] = min(
                    channel_min[channel_index], float(finite.min())
                )
                channel_max[channel_index] = max(
                    channel_max[channel_index], float(finite.max())
                )
        if available:
            signal_rows = numeric[available].to_numpy(dtype=np.float64)
            stride = max(1, len(signal_rows) // 512)
            amplitude_samples.append(signal_rows[::stride][:512])

    sampled_deltas = np.concatenate(delta_samples) if delta_samples else np.array([])
    median_delta = float(np.median(sampled_deltas)) if len(sampled_deltas) else np.nan
    sampling_rate = 1.0 / median_delta if median_delta > 0 else np.nan
    sampled_amplitude = (
        np.concatenate(amplitude_samples, axis=0)
        if amplitude_samples
        else np.empty((0, len(available)))
    )
    finite_amplitude = sampled_amplitude[np.isfinite(sampled_amplitude)]
    amplitude_quantiles = (
        np.quantile(finite_amplitude, [0.01, 0.5, 0.99]).tolist()
        if len(finite_amplitude)
        else [np.nan, np.nan, np.nan]
    )
    record_id = infer_record_id(row)
    window_origin_abs = (
        (np.floor(timestamp_min / 10.0) + 0.5) * 10.0
        if np.isfinite(timestamp_min)
        else np.nan
    )
    return {
        "record_id": record_id,
        "source": str(row["source"]),
        "subject_id": str(row["subject_id"]),
        "raw_file_path": str(row.get("main_rel_path", path)),
        "compression": "bz2" if path.suffix.lower() == ".bz2" else "none",
        "file_size_bytes": int(path.stat().st_size),
        "header_row": int(row.get("header_row", 0)),
        "separator": str(row.get("separator", ",")),
        "time_columns": _parse_list(row.get("time_columns")),
        "canonical_channels_available": available,
        "canonical_channels_missing": missing,
        "extra_eeg_service_columns": extra,
        "n_rows": int(n_rows),
        "finite_timestamp_rows": int(finite_timestamp_rows),
        "timestamp_unit": "unix_seconds" if 1e9 < timestamp_min < 2e9 else "unknown",
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "duration_seconds": timestamp_max - timestamp_min,
        "window_origin_abs": window_origin_abs,
        "median_timestamp_delta_seconds": median_delta,
        "sampling_rate_hz": sampling_rate,
        "nominal_sampling_rate_hz": nominal_sampling_rate,
        "duplicate_timestamps": int(duplicate_timestamps),
        "decreasing_timestamps": int(decreasing_timestamps),
        "nonpositive_timestamp_deltas": int(nonpositive_deltas),
        "gap_count_gt_1_5_nominal": int(gap_count),
        "irregular_delta_count": int(irregular_count),
        "positive_delta_count": int(delta_count),
        "irregular_delta_fraction": irregular_count / delta_count if delta_count else np.nan,
        "delta_seconds_p01_p50_p99": (
            np.quantile(sampled_deltas, [0.01, 0.5, 0.99]).tolist()
            if len(sampled_deltas)
            else [np.nan, np.nan, np.nan]
        ),
        "channel_missing_counts": {
            channel: int(channel_missing[index])
            for index, channel in enumerate(channels)
        },
        "channel_min": {
            channel: channel_min[index]
            for index, channel in enumerate(channels)
        },
        "channel_max": {
            channel: channel_max[index]
            for index, channel in enumerate(channels)
        },
        "sampled_amplitude_p01_p50_p99": amplitude_quantiles,
    }


def summarize(records: list[Dict[str, Any]], catalog: pd.DataFrame) -> Dict[str, Any]:
    source_summaries: Dict[str, Any] = {}
    for source in sorted({record["source"] for record in records}):
        selected = [record for record in records if record["source"] == source]
        sfreq = np.asarray([record["sampling_rate_hz"] for record in selected])
        sampled_ranges = np.asarray(
            [record["sampled_amplitude_p01_p50_p99"] for record in selected],
            dtype=float,
        )
        source_summaries[source] = {
            "records": len(selected),
            "subjects": int(catalog.loc[catalog["source"] == source, "subject_id"].nunique()),
            "total_rows": int(sum(record["n_rows"] for record in selected)),
            "file_size_bytes": int(sum(record["file_size_bytes"] for record in selected)),
            "compression_counts": {
                compression: sum(record["compression"] == compression for record in selected)
                for compression in sorted({record["compression"] for record in selected})
            },
            "sampling_rate_hz_min_median_max": [
                float(np.nanmin(sfreq)), float(np.nanmedian(sfreq)), float(np.nanmax(sfreq))
            ],
            "nominal_sampling_rate_counts": {
                str(int(rate)): int(sum(record["nominal_sampling_rate_hz"] == rate for record in selected))
                for rate in (128.0, 256.0)
            },
            "records_with_duplicate_timestamps": int(sum(record["duplicate_timestamps"] > 0 for record in selected)),
            "records_with_decreasing_timestamps": int(sum(record["decreasing_timestamps"] > 0 for record in selected)),
            "records_with_channel_nans": int(sum(any(record["channel_missing_counts"].values()) for record in selected)),
            "gap_count_gt_1_5_nominal": int(sum(record["gap_count_gt_1_5_nominal"] for record in selected)),
            "sampled_amplitude_record_p01_median": float(np.nanmedian(sampled_ranges[:, 0])),
            "sampled_amplitude_record_p50_median": float(np.nanmedian(sampled_ranges[:, 1])),
            "sampled_amplitude_record_p99_median": float(np.nanmedian(sampled_ranges[:, 2])),
            "time_column_variants": {
                value: int(count)
                for value, count in catalog.loc[catalog["source"] == source, "time_columns"].value_counts().items()
            },
        }
    return source_summaries


def write_report(schema: Dict[str, Any], path: Path) -> None:
    sources = schema["sources"]
    lines = [
        "# Raw EEG audit",
        "",
        f"Audited records: **{schema['record_count']}**.",
        f"Canonical signal channels: **{len(schema['canonical_channels'])}**.",
        "",
        "## Canonical schema",
        "",
        "Channel order: `" + ", ".join(schema["canonical_channels"]) + "`.",
        "",
        "`Timestamp` is Unix time in seconds. Windows are aligned to the same absolute "
        "10-second bins used by the processed dataset. EEG service columns such as "
        "`EEG.Counter`, `EEG.Interpolated`, battery and marker fields are not model inputs.",
        "",
        "## Sources",
        "",
        "| source | records | rows | size GiB | compression | nominal rates | measured sfreq min/median/max | duplicate-ts records | channel-NaN records | gaps >1.5 samples | sampled amplitude p01/p50/p99 |",
        "|---|---:|---:|---:|---|---|---|---:|---:|---:|---|",
    ]
    for source, item in sources.items():
        sfreq = "/".join(f"{value:.3f}" for value in item["sampling_rate_hz_min_median_max"])
        amplitude = "/".join(
            f"{item[key]:.3f}"
            for key in (
                "sampled_amplitude_record_p01_median",
                "sampled_amplitude_record_p50_median",
                "sampled_amplitude_record_p99_median",
            )
        )
        compression = ", ".join(
            f"{key}:{value}" for key, value in item["compression_counts"].items()
        )
        nominal_rates = ", ".join(
            f"{key}Hz:{value}"
            for key, value in item["nominal_sampling_rate_counts"].items()
        )
        lines.append(
            f"| {source} | {item['records']} | {item['total_rows']} | "
            f"{item['file_size_bytes'] / 2**30:.2f} | {compression} | {nominal_rates} | {sfreq} | "
            f"{item['records_with_duplicate_timestamps']} | "
            f"{item['records_with_channel_nans']} | "
            f"{item['gap_count_gt_1_5_nominal']} | {amplitude} |"
        )
    lines.extend([
        "",
        "## Decisions",
        "",
        "- Use the intersection of the 14 named Emotiv signal channels in the fixed order above.",
        "- Preserve 256 Hz when the measured source rate is within 0.5%; the four measured 128 Hz exports are upsampled to the common 256 Hz target with polyphase resampling.",
        "- Select data by timestamps, collapse duplicate timestamps, regularize small jitter, and reject a window when more than 2% of an expected channel grid is absent.",
        "- Keep raw amplitudes in exported numeric units; physical units are not asserted because the exports do not provide a verified calibration field.",
        "- `Old_EEG` is uncompressed while `gpn_data` is mostly bzip2-compressed; this is an I/O difference, not a channel-schema difference.",
        "",
        "The machine-readable per-record measurements are in `data/interim/raw_eeg_schema.json`.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/interim/emotiv_record_catalog.csv")
    parser.add_argument("--schema", default="data/interim/raw_eeg_schema.json")
    parser.add_argument("--report", default="reports/raw_eeg_audit.md")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)
    if args.limit is not None:
        catalog = catalog.iloc[: args.limit].copy()
    records = []
    for index, row in catalog.iterrows():
        record = audit_record(row, Path(args.repo_root), args.chunksize)
        records.append(record)
        print(
            f"[{len(records):03d}/{len(catalog):03d}] {record['record_id']} "
            f"rows={record['n_rows']} sfreq={record['sampling_rate_hz']:.3f}",
            flush=True,
        )
    schema = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_path": str(args.catalog),
        "record_count": len(records),
        "canonical_channels": list(CANONICAL_EEG_CHANNELS),
        "timestamp_column": "Timestamp",
        "timestamp_unit": "unix_seconds",
        "sources": summarize(records, catalog),
        "records": records,
    }
    schema_path = Path(args.schema)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w", encoding="utf-8") as output_file:
        json.dump(schema, output_file, default=_json_value, indent=2)
    write_report(schema, Path(args.report))
    print(f"Schema: {schema_path}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
