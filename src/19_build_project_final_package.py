"""CLI for the deterministic final project consolidation package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.analysis.project_final_package import generate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate existing experiment artifacts without training."
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = generate(Path(args.repo_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
