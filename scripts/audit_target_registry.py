"""Command-line entry point for the read-only benchmark target audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.analysis.target_registry_audit import run_target_registry_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit EEG benchmark target provenance without training models."
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/windowed_eeg_pm_dataset_w10.parquet",
    )
    parser.add_argument(
        "--validated-columns",
        default="data/interim/validated_columns.json",
    )
    parser.add_argument("--output-dir", default="reports/summary")
    parser.add_argument(
        "--report",
        default="reports/integration/full_target_registry_audit.md",
    )
    parser.add_argument(
        "--logical-map",
        default="data/interim/logical_recording_map.parquet",
    )
    parser.add_argument(
        "--raw-manifest",
        default="data/interim/raw_eeg_window_index_w10_raw_v3.parquet",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_target_registry_audit(
        args.dataset,
        args.validated_columns,
        args.output_dir,
        repo_root=REPO_ROOT,
        report_path=args.report,
        logical_map_path=args.logical_map,
        raw_manifest_path=args.raw_manifest,
    )
    print(f"status={result.status}")
    print(f"dataset_shape=({result.dataset_rows}, {result.dataset_columns})")
    print(f"feature_counts={dict(result.feature_counts)}")
    print(f"label_q5_bins={list(result.label_boundaries)}")
    for path in (*result.output_paths, result.report_path):
        print(path.resolve().relative_to(REPO_ROOT.resolve()).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
