"""CLI for the preregistered seven-PM nested AutoML protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.pm_low_high_nested_automl import (
    freeze_selection,
    load_config,
    prepare_protocol,
    run_final_outer,
    run_inner_search,
    write_dry_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run or explicitly execute frozen nested AutoML stages."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/pm_diagnostics/pm_low_high_nested_automl_v1.json"
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
        "--run-search",
        action="store_true",
        help="Execute the 2730 inner fits; never implied by dry-run.",
    )
    parser.add_argument(
        "--freeze-selection",
        action="store_true",
        help="Freeze five selections from a complete candidate_scores.csv.",
    )
    parser.add_argument(
        "--run-final",
        action="store_true",
        help="Execute 35 selected outer fits after selection is frozen.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only strictly validated complete runtime artifacts.",
    )
    args = parser.parse_args()
    if args.resume and not (args.run_search or args.run_final):
        parser.error("--resume requires --run-search or --run-final")
    if args.run_final and args.run_search and not args.freeze_selection:
        parser.error(
            "Running search and final stage together requires --freeze-selection"
        )

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
    result: dict = write_dry_run(context)
    if args.run_search:
        result = run_inner_search(context, resume=args.resume)
    if args.freeze_selection:
        result = freeze_selection(context)
    if args.run_final:
        result = run_final_outer(context, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
