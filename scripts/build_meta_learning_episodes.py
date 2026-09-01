"""Materialize leakage-safe diagnostic meta-learning episodes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.meta import materialize_meta_learning_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/meta_learning/episode_infrastructure_smoke.json",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    root = REPOSITORY_ROOT
    config_path = root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = materialize_meta_learning_smoke(config, repository_root=root)
    print(json.dumps({
        "episodes": len(manifest.episodes),
        "errors": len(manifest.errors),
        "output_dir": config["output_dir"],
        "training_performed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
