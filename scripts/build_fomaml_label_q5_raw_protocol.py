"""Build the disabled raw-deduplicated FOMAML protocol and manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.experiments.fomaml_label_q5_raw_protocol import (  # noqa: E402
    build_fomaml_label_q5_raw_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/meta_learning/fomaml_label_q5_raw_protocol.json",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config = json.loads(
        (REPOSITORY_ROOT / args.config).read_text(encoding="utf-8")
    )
    result = build_fomaml_label_q5_raw_protocol(
        config, repository_root=REPOSITORY_ROOT
    )
    if args.verbose:
        print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
