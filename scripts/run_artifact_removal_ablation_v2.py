from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.experiments.artifact_removal_ablation_v2 import (
    ARTIFACT_VARIANTS_V2,
    PM_METRICS,
    TASK_TYPES,
    aggregate_results,
    build_cache,
    run_experiment,
    write_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run target-independent artifact-removal ablation v2."
    )
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--build-preprocessing-cache", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--confirm-full", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--smoke-cache", action="store_true")
    parser.add_argument("--metric", action="append", choices=PM_METRICS)
    parser.add_argument("--variant", action="append", choices=ARTIFACT_VARIANTS_V2)
    parser.add_argument("--fold", action="append", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--task-type", action="append", choices=TASK_TYPES)
    args = parser.parse_args()
    if args.plan_only:
        result = write_plan(args.config)
    elif args.build_preprocessing_cache:
        result = build_cache(
            args.config, smoke=bool(args.smoke_cache), resume=bool(args.resume)
        )
    elif args.aggregate_only:
        result = aggregate_results(
            args.config, smoke=bool(args.smoke_cache), summaries=None
        )
    else:
        smoke = bool(args.smoke)
        if not smoke and not args.confirm_full:
            parser.error("Full 280-run execution requires --confirm-full")
        result = run_experiment(
            args.config,
            smoke=smoke,
            resume=bool(args.resume),
            metrics=set(args.metric or []) or None,
            variants=set(args.variant or []) or None,
            folds=set(args.fold or []) or None,
            task_types=set(args.task_type or []) or None,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()


