"""Deferred cheap-model EEG-feature check for experimental PM variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.analysis.pm_quality_downstream import plan_downstream, run_downstream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir")
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--outer-fold", type=int, action="append")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        result = plan_downstream(args.config, outer_folds=args.outer_fold)
    else:
        if not args.feature_cache_dir or not args.data_path:
            parser.error("--feature-cache-dir and --data-path are required unless --plan-only")
        result = run_downstream(
            args.config,
            feature_cache_dir=args.feature_cache_dir,
            data_path=args.data_path,
            output_dir=args.output_dir,
            outer_folds=args.outer_fold,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
