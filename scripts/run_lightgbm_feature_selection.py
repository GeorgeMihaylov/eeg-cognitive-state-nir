from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.experiments.lightgbm_feature_selection import (
    finalize_existing_results,
    run_experiment,
    write_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(write_plan(args.config), indent=2, ensure_ascii=False, default=str))
        return
    if args.finalize_only:
        print(json.dumps(
            finalize_existing_results(args.config),
            indent=2,
            ensure_ascii=False,
            default=str,
        ))
        return
    if not args.smoke and not args.confirm_full:
        parser.error("full 140-run execution requires --confirm-full")
    results = run_experiment(
        args.config, smoke=args.smoke, resume=not args.no_resume
    )
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
