"""Preflight, smoke or execute the fixed five-lag LOW/HIGH experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.pm_eeg_absolute_lag import (
    load_config, prepare_protocol, run_experiment, write_dry_run,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path(
        "experiments/pm_diagnostics/pm_eeg_absolute_lag_v1.json"))
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_config(args.config if args.config.is_absolute() else root / args.config)
    output = args.output_dir
    if args.smoke and output is None:
        output = Path(config["output_dir"]) / "smoke"
    context = prepare_protocol(config, root=root, feature_cache_dir=args.feature_cache_dir,
                               output_dir=output, smoke=args.smoke)
    print(json.dumps(write_dry_run(context), indent=2))
    if not args.dry_run:
        print(json.dumps(run_experiment(context), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
