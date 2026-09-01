"""Thin CLI for the COG-BCI 14/62-channel spectral benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.experiments.cog_bci_spectral_benchmark import (
    run_cog_bci_spectral_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the record-level COG-BCI N-Back spectral benchmark."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    run_cog_bci_spectral_benchmark(
        config,
        repository_root=REPOSITORY_ROOT,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
