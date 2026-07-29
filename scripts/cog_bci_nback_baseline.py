#!/usr/bin/env python3
"""Run one configured COG-BCI N-Back diagnostic baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.experiments.cog_bci_nback_baseline import (  # noqa: E402
    BaselineRunOptions,
    COGBCINBackBaselineRunner,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    runner = COGBCINBackBaselineRunner(
        config,
        repository_root=REPOSITORY_ROOT,
        options=BaselineRunOptions(
            smoke=args.smoke,
            fold=args.fold,
            resume=args.resume,
        ),
    )
    summary = runner.run()
    if args.verbose:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"{summary['model']}: folds={summary['folds_completed']}, "
            f"records={summary['records_predicted']}, "
            f"seconds={summary['total_time_seconds']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
