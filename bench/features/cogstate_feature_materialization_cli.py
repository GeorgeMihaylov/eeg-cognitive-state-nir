"""CLI for the canonical target-free ``cogstate.features`` cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.features.cogstate_feature_cache import (
    RawEEGWindowArrayView,
    _window_batch,
    benchmark_worker_counts,
    build_canonical_feature_index,
    load_feature_profile,
    materialize_cogstate_features,
    plan_cogstate_feature_cache,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/raw_eeg_window_index_w10_pm_union_composite_v1.parquet"),
    )
    parser.add_argument(
        "--logical-map",
        type=Path,
        default=Path("data/interim/logical_recording_map.parquet"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("experiments/features/preliminary_model_zoo_features_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-workers", action="store_true")
    parser.add_argument("--benchmark-windows", type=int, default=32)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = build_parser().parse_args()
    manifest = _resolve(args.data_root, args.manifest)
    logical = _resolve(args.data_root, args.logical_map)
    if args.plan_only and args.benchmark_workers:
        raise ValueError("--plan-only and --benchmark-workers are mutually exclusive")
    if args.plan_only:
        plan = plan_cogstate_feature_cache(
            manifest_path=manifest,
            logical_recording_map_path=logical,
            cache_path_root=args.data_root,
            feature_profile_path=args.profile,
            output_dir=args.output_dir,
            max_rows=args.max_rows,
        )
        print(json.dumps(plan, indent=2))
        return 0
    if args.benchmark_workers:
        profile, _ = load_feature_profile(args.profile)
        index = build_canonical_feature_index(manifest, logical).head(args.benchmark_windows)
        view = RawEEGWindowArrayView(index, cache_path_root=args.data_root)
        windows = _window_batch(view, 0, len(index))
        rows = benchmark_worker_counts(profile, windows)
        print(json.dumps(rows, indent=2))
        return 0
    summary = materialize_cogstate_features(
        manifest_path=manifest,
        logical_recording_map_path=logical,
        cache_path_root=args.data_root,
        feature_profile_path=args.profile,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        workers=args.workers,
        resume=args.resume,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
