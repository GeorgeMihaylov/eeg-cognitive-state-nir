"""CLI for preregistered LOW/HIGH weighted model personalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.pm_low_high_personalized_weighted_augmentation import (
    load_config,
    prepare_protocol,
    run_experiment,
    write_dry_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or explicitly execute frozen personalized weighted "
            "augmentation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/pm_diagnostics/"
            "pm_low_high_personalized_weighted_augmentation_v1.json"
        ),
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path(
            "F:/eeg-preproc-integration/benchmark_results/"
            "cogstate_features_emotiv14_canonical_v1"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute 570 fits after the mandatory dry-run (not the default).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only strictly validated complete per-run artifacts.",
    )
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error("--resume requires --run")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    config = load_config(config_path)
    context = prepare_protocol(
        config,
        root=ROOT,
        feature_cache_dir=args.feature_cache_dir,
        output_dir=output_dir,
    )
    result = write_dry_run(context)
    if args.run:
        result = run_experiment(context, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
