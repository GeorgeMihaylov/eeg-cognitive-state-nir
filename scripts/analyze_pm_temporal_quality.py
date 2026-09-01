"""CLI for the CPU-light PM temporal-quality experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.analysis.pm_temporal_quality import plan_analysis, run_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--reference-predictions")
    parser.add_argument("--raw-index-path")
    parser.add_argument("--raw-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        result = plan_analysis(args.config, output_dir=args.output_dir)
    else:
        result = run_analysis(
            args.config,
            data_path=args.data_path,
            reference_predictions=args.reference_predictions,
            raw_index_path=args.raw_index_path,
            raw_root=args.raw_root,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
