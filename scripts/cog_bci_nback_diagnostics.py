#!/usr/bin/env python3
"""Run the COG-BCI N-Back weak-signal diagnostic workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.analysis.cog_bci_nback_diagnostics import (  # noqa: E402
    run_nback_signal_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    summary = run_nback_signal_diagnostics(
        config, repository_root=REPOSITORY_ROOT
    )
    if args.verbose:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        subject = summary["subject_disjoint"]["aggregate_metrics"]
        within = summary["within_subject"]["aggregate_metrics"]
        print(
            "COG-BCI N-Back diagnostics complete: "
            f"subject_models={len(subject)}, within_models={len(within)}, "
            f"seconds={summary['elapsed_seconds']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
