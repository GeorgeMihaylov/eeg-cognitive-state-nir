"""CLI for the canonical seven-PM engineered-feature baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.experiments.pm_all_targets_feature_baseline import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/pm_regression/pm_all_targets_feature_baseline.yaml"),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    result = run_baseline(
        args.config,
        plan_only=args.plan_only,
        smoke=args.smoke,
        resume=args.resume,
        max_runs=args.max_runs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
