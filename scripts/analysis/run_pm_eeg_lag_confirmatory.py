"""Plan or execute the fixed previous-window EEG alignment experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.pm_eeg_lag_confirmatory import (
    load_config,
    prepare_protocol,
    run_experiment,
    write_dry_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/pm_diagnostics/pm_eeg_lag_confirmatory_v1.json"
        ),
    )
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", default=True)
    resume.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    context = prepare_protocol(
        config,
        root=root,
        feature_cache_dir=args.feature_cache_dir,
        output_dir=args.output_dir,
    )
    dry_summary = write_dry_run(context)
    if args.dry_run:
        print(json.dumps(dry_summary, indent=2, sort_keys=True))
        return 0
    status = run_experiment(context, resume=bool(args.resume))
    print(json.dumps({
        "experiment_id": config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "status": status,
        "output_dir": str(context.output_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
