"""Run the fold-1 ShallowConvNet PM preliminary streaming handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.experiments.preliminary_streaming_handoff import (  # noqa: E402
    run_preliminary_handoff,
)


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/targets/preliminary_streaming_handoff_shallow_fold1.yaml",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--target-id", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = _resolve_path(REPO_ROOT, args.config)
    manifest = run_preliminary_handoff(
        config_path,
        data_root=Path(args.data_root).resolve(),
        output_dir=(None if args.output_dir is None else Path(args.output_dir).resolve()),
        requested_target_ids=args.target_id,
        resume=args.resume,
    )
    print(json.dumps(manifest["status_counts"], indent=2))


if __name__ == "__main__":
    main()
