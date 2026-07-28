#!/usr/bin/env python3
"""Materialize native COG-BCI target and subject-split manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.validation.cog_bci_protocol import (  # noqa: E402
    materialize_cog_bci_protocol,
)


def _load_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("COG-BCI protocol config must contain an object")
    allowed = {
        "dataset",
        "window_cache",
        "task_id",
        "target_name",
        "splitter",
        "inner_validation",
        "loso",
        "output_dir",
        "result_status",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"Unknown protocol config keys: {unknown}")
    return document


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    result = materialize_cog_bci_protocol(
        config, repository_root=REPO_ROOT
    )
    return result.protocol_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build COG-BCI native targets and leakage-safe splits."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    summary = run(args.config)
    if args.verbose:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"{summary['task_id']}: {summary['accepted_windows']} windows, "
            f"{summary['outer_folds']} outer folds, "
            f"{summary['loso_folds']} LOSO folds"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
