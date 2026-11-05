"""Prepare H10-H120 manifests or execute the single bounded five-fit smoke."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.pm_temporal_predictability import load_config, prepare_protocol, write_dry_run, run_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("experiments/pm_diagnostics/pm_temporal_predictability_v1.json"))
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.root / args.config)
    context = prepare_protocol(config, root=args.root, feature_cache_dir=args.feature_cache_dir,
                               output_dir=args.output_dir, smoke=args.smoke)
    print(json.dumps(write_dry_run(context), indent=2))
    if args.smoke:
        print(json.dumps(run_smoke(context), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
