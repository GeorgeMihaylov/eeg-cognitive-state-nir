"""Build, audit, smoke-test, or execute the CLARE/CL-Drive DANN protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.cross_dataset_clare_cldrive_dann import (
    execute,
    materialize_eeg_only_cache,
    plan_experiment,
    smoke_forward_backward,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-cache", action="store_true")
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required with --run; plan/cache/smoke never fit a model.",
    )
    args = parser.parse_args()
    if args.build_cache:
        result = materialize_eeg_only_cache(args.config)
    elif args.plan_only:
        result = plan_experiment(args.config)
    elif args.smoke:
        result = smoke_forward_backward(args.config)
    else:
        result = execute(
            args.config, resume=args.resume, confirm_training=args.confirm_training
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
