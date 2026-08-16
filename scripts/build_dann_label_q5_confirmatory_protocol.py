"""Build the disabled confirmatory multi-fold DANN protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.experiments.dann_label_q5_confirmatory_protocol import (  # noqa: E402
    run_confirmatory_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "experiments/domain_adaptation/"
            "dann_label_q5_confirmatory_protocol.json"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    summary = run_confirmatory_protocol(
        arguments.config, repository_root=REPOSITORY_ROOT
    )
    if arguments.verbose:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
