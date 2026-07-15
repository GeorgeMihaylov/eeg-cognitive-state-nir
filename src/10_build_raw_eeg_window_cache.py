"""Build the label_q5 raw EEG window index and reusable record shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets.raw_eeg_window_dataset import (  # noqa: E402
    build_raw_eeg_cache,
    build_raw_window_index,
)
from bench.datasets.raw_preprocessing import (  # noqa: E402
    normalize_raw_preprocessing,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed",
        default="data/processed/windowed_eeg_pm_dataset_w10.parquet",
    )
    parser.add_argument(
        "--catalog", default="data/interim/emotiv_record_catalog.csv"
    )
    parser.add_argument(
        "--audit-schema", default="data/interim/raw_eeg_schema.json"
    )
    parser.add_argument(
        "--output", default="data/interim/raw_eeg_window_index_w10.parquet"
    )
    parser.add_argument(
        "--stats", default="data/interim/raw_eeg_window_index_w10_stats.json"
    )
    parser.add_argument(
        "--cache-dir", default="data/interim/raw_eeg_cache_w10"
    )
    parser.add_argument("--target-sfreq", type=float, default=256.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.02)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--record-limit", type=int)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument(
        "--preprocessing-config",
        help=(
            "YAML containing raw_preprocessing (or the schema directly); cache "
            "shards are written to a hash-isolated subdirectory"
        ),
    )
    args = parser.parse_args()

    preprocessing_input = None
    if args.preprocessing_config:
        with open(args.preprocessing_config, encoding="utf-8") as input_file:
            preprocessing_document = yaml.safe_load(input_file) or {}
        preprocessing_input = preprocessing_document.get(
            "raw_preprocessing", preprocessing_document
        )
    raw_preprocessing = normalize_raw_preprocessing(
        preprocessing_input, default_resample_hz=args.target_sfreq
    )
    target_sfreq = float(raw_preprocessing["resample_hz"])

    index, matching = build_raw_window_index(
        args.processed,
        args.catalog,
        audit_schema_path=args.audit_schema,
        target_sfreq=target_sfreq,
        n_splits=args.n_splits,
    )
    cache_stats = None
    if not args.index_only:
        index, cache_stats = build_raw_eeg_cache(
            index,
            args.cache_dir,
            target_sfreq=target_sfreq,
            max_missing_fraction=args.max_missing_fraction,
            repo_root=REPO_ROOT,
            record_limit=args.record_limit,
            raw_preprocessing=raw_preprocessing,
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(output_path, index=False)
    stats = {
        "processed_path": args.processed,
        "catalog_path": args.catalog,
        "audit_schema_path": args.audit_schema,
        "output_path": str(output_path),
        "matching": matching,
        "cache": cache_stats,
        "raw_preprocessing": raw_preprocessing,
        "window_status_counts": {
            str(key): int(value)
            for key, value in index["status"].value_counts().items()
        },
        "rejection_reason_counts": {
            str(key): int(value)
            for key, value in index.loc[index["status"] != "ok", "rejection_reason"]
            .value_counts().items()
        },
        "accepted_records": int(
            index.loc[index["status"] == "ok", "record_id"].nunique()
        ),
        "accepted_subjects": int(
            index.loc[index["status"] == "ok", "subject_id"].nunique()
        ),
        "accepted_class_distribution": {
            str(int(key)): int(value)
            for key, value in index.loc[index["status"] == "ok", "label_q5"]
            .value_counts().sort_index().items()
        },
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as output_file:
        json.dump(stats, output_file, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
