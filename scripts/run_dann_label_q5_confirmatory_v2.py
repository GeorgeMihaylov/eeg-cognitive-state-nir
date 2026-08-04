"""Run or resume the preregistered DANN confirmatory-v2 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.experiments.dann_label_q5_confirmatory_v2 import (  # noqa: E402
    run_dann_label_q5_confirmatory_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "experiments/domain_adaptation/"
            "dann_label_q5_confirmatory_v2_execution.json"
        ),
    )
    parser.add_argument("--phase", choices=("all", "primary", "secondary"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-registry", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    summary = run_dann_label_q5_confirmatory_v2(
        arguments.config,
        repository_root=REPOSITORY_ROOT,
        phase=arguments.phase,
        resume=arguments.resume,
        verify_registry=arguments.verify_registry,
        aggregate_only=arguments.aggregate_only,
    )
    if arguments.verbose:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(summary.get("result_status", summary.get("phase", "registry_valid")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
