"""Run the approved CPU-only synthetic FOMAML diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.meta import run_fomaml_synthetic_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="experiments/meta_learning/fomaml_synthetic_smoke.json"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config = json.loads((REPOSITORY_ROOT / args.config).read_text(encoding="utf-8"))
    summary = run_fomaml_synthetic_smoke(config, repository_root=REPOSITORY_ROOT)
    if args.verbose:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
