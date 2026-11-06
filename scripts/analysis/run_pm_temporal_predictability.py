"""Preflight or execute the explicit 175-fit matched history experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.pm_temporal_predictability import load_config, prepare_protocol, write_dry_run, run_smoke, run_full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("experiments/pm_diagnostics/pm_temporal_predictability_v1.json"))
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full-run", action="store_true", help="Exactly 175 ordered strict-common-support fits")
    mode.add_argument("--order-control", action="store_true", help="Exactly 25 shuffled H120 fits; reuse ordered reference")
    mode.add_argument("--order-control-dry-run", action="store_true", help="Preflight the 25-fit order control without training")
    args = parser.parse_args()
    config = load_config(args.root / args.config)
    if args.order_control or args.order_control_dry_run:
        from bench.experiments import pm_temporal_order_control as control
        context = control.prepare_protocol(config, root=args.root, feature_cache_dir=args.feature_cache_dir,
                                           output_dir=args.output_dir)
        print(json.dumps(control.write_dry_run(context), indent=2))
        if args.order_control:
            print(json.dumps(control.run(context), indent=2))
        return 0
    context = prepare_protocol(config, root=args.root, feature_cache_dir=args.feature_cache_dir,
                               output_dir=args.output_dir, smoke=args.smoke, full=not args.smoke)
    print(json.dumps(write_dry_run(context), indent=2))
    if args.smoke:
        print(json.dumps(run_smoke(context), indent=2))
    elif args.full_run:
        print(json.dumps(run_full(context), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
