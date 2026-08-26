"""Measure channel/source artifact distributions from cached raw EEG windows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets.raw_eeg_window_dataset import CANONICAL_EEG_CHANNELS  # noqa: E402


QUANTILES = (0.0, 0.01, 0.5, 0.99, 0.999, 1.0)


def _quantiles(values: list[np.ndarray]) -> dict[str, float]:
    if not values:
        return {}
    array = np.concatenate(values).astype(np.float64, copy=False)
    result = np.quantile(array, QUANTILES)
    return {
        f"p{quantile * 100:g}": float(value)
        for quantile, value in zip(QUANTILES, result)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/interim/raw_eeg_window_index_w10.parquet"
    )
    parser.add_argument(
        "--schema", default="data/interim/raw_eeg_schema.json"
    )
    parser.add_argument(
        "--report", default="reports/raw_eeg_artifact_audit.md"
    )
    parser.add_argument(
        "--stats", default="data/interim/raw_eeg_artifact_stats.json"
    )
    parser.add_argument("--amplitude-samples-per-record-channel", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    accepted = manifest.loc[manifest["status"].eq("ok")].copy()
    with open(args.schema, encoding="utf-8") as input_file:
        schema_document = json.load(input_file)
    schema = {
        str(record["record_id"]): record
        for record in schema_document.get("records", [])
    }
    rng = np.random.default_rng(42)
    accumulators: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "amplitude": [],
            "peak_to_peak": [],
            "variance": [],
            "flat_fraction": [],
            "sample_count": 0,
            "non_finite_count": 0,
            "clipped_count": 0,
            "window_count": 0,
        }
    )

    for record_id, rows in accepted.groupby("record_id", sort=True):
        cache_paths = rows["cache_file"].astype(str).unique()
        if len(cache_paths) != 1:
            raise ValueError(f"Record {record_id} has multiple cache shards")
        array = np.load(cache_paths[0], mmap_mode="r", allow_pickle=False)
        offsets = rows["cache_offset"].to_numpy(dtype=np.int64)
        source = str(rows["source"].iloc[0])
        record_schema = schema[str(record_id)]
        minima = np.asarray([
            record_schema["channel_min"][channel]
            for channel in CANONICAL_EEG_CHANNELS
        ], dtype=np.float32)
        maxima = np.asarray([
            record_schema["channel_max"][channel]
            for channel in CANONICAL_EEG_CHANNELS
        ], dtype=np.float32)
        sample_budget = int(args.amplitude_samples_per_record_channel)
        samples_seen = 0
        for start in range(0, len(offsets), args.chunk_size):
            values = np.asarray(
                array[offsets[start:start + args.chunk_size]], dtype=np.float32
            )
            finite = np.isfinite(values)
            peak_to_peak = np.ptp(values, axis=2)
            variance = np.var(values, axis=2)
            flat = np.mean(np.abs(np.diff(values, axis=2)) <= 1e-6, axis=2)
            clipped = np.isclose(
                values, minima[None, :, None], rtol=0, atol=1e-6
            ) | np.isclose(
                values, maxima[None, :, None], rtol=0, atol=1e-6
            )
            for channel_index, channel in enumerate(CANONICAL_EEG_CHANNELS):
                bucket = accumulators[(source, channel)]
                bucket["peak_to_peak"].append(peak_to_peak[:, channel_index])
                bucket["variance"].append(variance[:, channel_index])
                bucket["flat_fraction"].append(flat[:, channel_index])
                channel_values = values[:, channel_index, :].reshape(-1)
                remaining = max(sample_budget - samples_seen, 0)
                if remaining:
                    take = min(remaining, len(channel_values))
                    indices = rng.choice(len(channel_values), size=take, replace=False)
                    bucket["amplitude"].append(channel_values[indices])
                bucket["sample_count"] += int(channel_values.size)
                bucket["non_finite_count"] += int((~finite[:, channel_index]).sum())
                bucket["clipped_count"] += int(clipped[:, channel_index].sum())
                bucket["window_count"] += int(len(values))
            samples_seen += min(
                max(sample_budget - samples_seen, 0), values.shape[0] * values.shape[2]
            )

    rows = []
    for (source, channel), bucket in sorted(accumulators.items()):
        rows.append({
            "source": source,
            "channel": channel,
            "windows": int(bucket["window_count"]),
            "amplitude_quantiles": _quantiles(bucket["amplitude"]),
            "peak_to_peak_quantiles": _quantiles(bucket["peak_to_peak"]),
            "variance_quantiles": _quantiles(bucket["variance"]),
            "flat_fraction_quantiles": _quantiles(bucket["flat_fraction"]),
            "clipped_sample_percentage": (
                100.0 * bucket["clipped_count"] / bucket["sample_count"]
            ),
            "non_finite_sample_percentage": (
                100.0 * bucket["non_finite_count"] / bucket["sample_count"]
            ),
        })
    stats = {
        "manifest": args.manifest,
        "accepted_windows": int(len(accepted)),
        "source_specific_records": int(accepted["record_id"].nunique()),
        "method": {
            "amplitude_quantiles": (
                "deterministic sample of up to "
                f"{args.amplitude_samples_per_record_channel} values per record/channel"
            ),
            "window_distributions": "all accepted cached windows",
            "flat_sample": "absolute consecutive difference <= 1e-6",
            "clipped_sample": (
                "value equals the audited per-record raw channel minimum or maximum "
                "within absolute tolerance 1e-6"
            ),
        },
        "channels_by_source": rows,
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    table_rows = []
    for item in rows:
        table_rows.append({
            "source": item["source"],
            "channel": item["channel"],
            "amplitude p1 / p50 / p99": " / ".join(
                f"{item['amplitude_quantiles'][key]:.3f}"
                for key in ("p1", "p50", "p99")
            ),
            "peak-to-peak p50 / p99 / p99.9": " / ".join(
                f"{item['peak_to_peak_quantiles'][key]:.3f}"
                for key in ("p50", "p99", "p99.9")
            ),
            "variance p50 / p99 / p99.9": " / ".join(
                f"{item['variance_quantiles'][key]:.3f}"
                for key in ("p50", "p99", "p99.9")
            ),
            "flat p99 / p99.9 / max": " / ".join(
                f"{item['flat_fraction_quantiles'][key]:.6f}"
                for key in ("p99", "p99.9", "p100")
            ),
            "clipped %": f"{item['clipped_sample_percentage']:.8f}",
            "NaN/Inf %": f"{item['non_finite_sample_percentage']:.8f}",
        })
    table = pd.DataFrame(table_rows)
    maximum_flat_p999 = max(
        item["flat_fraction_quantiles"]["p99.9"] for item in rows
    )
    maximum_p2p_p999 = max(
        item["peak_to_peak_quantiles"]["p99.9"] for item in rows
    )
    report = [
        "# Raw EEG artifact audit",
        "",
        "## Scope and method",
        "",
        f"The audit covers all **{len(accepted):,}** accepted unfiltered cached "
        f"windows from **{accepted['record_id'].nunique()}** source-specific records. "
        "Peak-to-peak, variance and flatline distributions use every window. "
        f"Amplitude quantiles use a deterministic seed-42 sample of up to "
        f"{args.amplitude_samples_per_record_channel} values per record/channel.",
        "",
        "A flat sample is an absolute consecutive difference <= 1e-6. A clipped "
        "sample is exactly at the audited per-record raw channel minimum or maximum "
        "within 1e-6; this is an export-level saturation proxy, not a known ADC rail.",
        "",
        "## Channel/source distributions",
        "",
        table.to_markdown(index=False),
        "",
        "## Conservative threshold proposal",
        "",
        f"- Observed maximum source/channel p99.9 peak-to-peak: "
        f"**{maximum_p2p_p999:.3f}**.",
        f"- Observed maximum source/channel p99.9 flat fraction: "
        f"**{maximum_flat_p999:.6f}**.",
        "- Keep artifact rejection disabled for the controlled preprocessing ablation. "
        "This avoids changing two factors at once.",
        "- If a later QC experiment is registered, use the measured tail as the "
        "starting point: flag (do not automatically discard) windows above the "
        "per-source/channel p99.9 peak-to-peak or flat-fraction tail, inspect them, "
        "then freeze a threshold before evaluating outer test folds.",
        "- Do not set `max_abs_amplitude` on unreferenced raw values: the channels "
        "carry large DC offsets. Measure the bandpass+CAR distribution first if an "
        "absolute-amplitude rejection experiment is added.",
        "",
        "The three requested comparison runs therefore leave both artifact thresholds "
        "at `null` and `artifact_rejection.enabled: false`.",
        "",
    ]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "accepted_windows": int(len(accepted)),
        "groups": len(rows),
        "maximum_peak_to_peak_p99_9": maximum_p2p_p999,
        "maximum_flat_fraction_p99_9": maximum_flat_p999,
        "report": str(report_path),
        "stats": str(stats_path),
    }, indent=2))


if __name__ == "__main__":
    main()
