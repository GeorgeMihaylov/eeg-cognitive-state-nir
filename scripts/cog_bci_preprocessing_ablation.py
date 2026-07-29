#!/usr/bin/env python3
"""Run the preregistered COG-BCI N-Back preprocessing ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.analysis.cog_bci_preprocessing_ablation import (  # noqa: E402
    run_cog_bci_preprocessing_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    summary = run_cog_bci_preprocessing_ablation(
        config, repository_root=REPOSITORY_ROOT
    )
    if args.verbose:
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(
            f"selected={summary['selection']['selected_preprocessing']}, "
            f"decision={summary['decision']['recommendation']}, "
            f"seconds={summary['elapsed_seconds']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
