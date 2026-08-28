"""Read-only correctness and performance smoke for ``cogstate.features``."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.datasets.raw_eeg_window_dataset import RawEEGWindowArrayView
from cogstate.features import FeaturePipeline, FeaturePipelineConfig


REQUIRED_MANIFEST_COLUMNS = [
    "sample_id",
    "record_id",
    "record_group_id",
    "status",
    "cache_file",
    "cache_offset",
    "n_channels",
    "n_samples_expected",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("experiments/features/preliminary_model_zoo_features_v1.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--logical-recording-map", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root used to resolve relative cache_file paths from the manifest.",
    )
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument("--sample-entropy-windows", type=int, default=10)
    return parser


def _load_config(path: Path) -> FeaturePipelineConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return FeaturePipelineConfig.from_mapping(payload)


def _deduplicated_manifest(
    manifest_path: Path,
    logical_map_path: Path,
    *,
    max_windows: int,
) -> tuple[pd.DataFrame, int]:
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    manifest = pd.read_parquet(manifest_path, columns=REQUIRED_MANIFEST_COLUMNS)
    logical_map = pd.read_parquet(
        logical_map_path, columns=["record_group_id", "selected_record_id"]
    )
    if logical_map["record_group_id"].astype(str).duplicated().any():
        raise ValueError("logical recording map contains duplicate record_group_id")
    selected_ids = set(logical_map["selected_record_id"].astype(str))
    accepted = manifest.loc[
        manifest["status"].eq("ok")
        & manifest["record_id"].astype(str).isin(selected_ids)
    ].sort_values("sample_id", kind="stable")
    if accepted["sample_id"].duplicated().any():
        raise ValueError("deduplicated smoke manifest contains duplicate sample_id")
    total = len(accepted)
    if total == 0:
        raise ValueError("no accepted deduplicated windows were found")
    positions = np.linspace(
        0, total - 1, num=min(max_windows, total), dtype=np.int64
    )
    return accepted.iloc[positions].reset_index(drop=True), total


def _windows(frame: pd.DataFrame, data_root: Path) -> np.ndarray:
    view = RawEEGWindowArrayView(frame, cache_path_root=data_root)
    windows = np.stack([view[index][0].T for index in range(len(view))])
    if windows.shape[1:] != (2560, 14):
        raise ValueError(f"expected real windows [batch,2560,14], got {windows.shape}")
    if windows.dtype != np.float32 or not np.isfinite(windows).all():
        raise ValueError("real smoke windows must be finite float32")
    return windows


def _timed_pipeline(
    name: str,
    config: FeaturePipelineConfig,
    windows: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    pipeline = FeaturePipeline(config)
    start = time.perf_counter()
    matrix = pipeline.transform_batch(windows, chunk_size=8)
    elapsed = time.perf_counter() - start
    names = pipeline.feature_names(windows.shape[2])
    if matrix.shape != (len(windows), len(names)) or not np.isfinite(matrix).all():
        raise RuntimeError(f"{name} smoke output violates the feature contract")
    return (
        {
            "group": name,
            "windows": len(windows),
            "n_features": matrix.shape[1],
            "elapsed_seconds": elapsed,
            "seconds_per_window": elapsed / len(windows),
        },
        matrix,
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    base = _load_config(args.profile)
    frame, full_count = _deduplicated_manifest(
        args.manifest,
        args.logical_recording_map,
        max_windows=args.max_windows,
    )
    windows = _windows(frame, args.data_root)
    timings: list[dict[str, Any]] = []
    for name in ("spectral", "statistical", "entropy", "connectivity", "all"):
        enabled = name == "all"
        config = replace(
            base,
            include_spectral=enabled or name == "spectral",
            include_statistical=enabled or name == "statistical",
            include_entropy=enabled or name == "entropy",
            include_connectivity=enabled or name == "connectivity",
        )
        timing, _ = _timed_pipeline(name, config, windows)
        timings.append(timing)

    entropy_count = min(int(args.sample_entropy_windows), len(windows))
    if entropy_count <= 0:
        raise ValueError("sample_entropy_windows must be positive")
    entropy_on = replace(
        base,
        include_spectral=False,
        include_statistical=False,
        include_entropy=True,
        include_connectivity=False,
        entropy_config=replace(base.entropy_config, include_sample_entropy=True),
    )
    sample_timing, _ = _timed_pipeline(
        "entropy_sample_entropy_on", entropy_on, windows[:entropy_count]
    )
    timings.append(sample_timing)
    entropy_off = next(row for row in timings if row["group"] == "entropy")
    sample_timing["relative_cost_vs_entropy_without_sample"] = (
        sample_timing["seconds_per_window"] / entropy_off["seconds_per_window"]
    )

    pipeline = FeaturePipeline(base)
    specification = pipeline.feature_specification()
    return {
        "status": "passed",
        "target_free": True,
        "manifest": str(args.manifest),
        "logical_recording_map": str(args.logical_recording_map),
        "deduplicated_status_ok_windows": full_count,
        "sampled_windows": len(windows),
        "sample_ids": frame["sample_id"].astype(str).tolist(),
        "input_shape": list(windows.shape),
        "output_shape": [len(windows), len(pipeline.feature_names(14))],
        "feature_schema_version": specification["schema_version"],
        "feature_hash": pipeline.feature_hash(),
        "connectivity_pairs": len(specification["connectivity"]["channel_pairs"]),
        "timings": timings,
    }


def main() -> int:
    args = _parser().parse_args()
    print(json.dumps(run_smoke(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
