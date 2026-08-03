"""Run the preregistered task-8Shch raw-EEG DANN diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.experiments.dann_label_q5_raw_diagnostic import (  # noqa: E402
    run_dann_label_q5_raw_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/domain_adaptation/dann_label_q5_raw_diagnostic.json",
    )
    arguments = parser.parse_args()
    summary = run_dann_label_q5_raw_diagnostic(
        Path(arguments.config), repository_root=REPOSITORY_ROOT
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
