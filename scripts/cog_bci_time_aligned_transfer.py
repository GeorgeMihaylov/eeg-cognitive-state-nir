#!/usr/bin/env python3
"""Run the one-fold COG-BCI time-aligned contrastive transfer screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.experiments.cog_bci_time_aligned_transfer import (  # noqa: E402
    run_cog_bci_time_aligned_transfer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "experiments/cog_bci/"
            "time_aligned_eegnet_transfer_screening.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cog_bci_time_aligned_transfer(
        args.config, repository_root=REPOSITORY_ROOT
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
