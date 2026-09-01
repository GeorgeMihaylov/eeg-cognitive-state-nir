"""CLI for the limited COG-BCI contrastive transfer screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bench.experiments.cog_bci_contrastive_transfer import (  # noqa: E402
    run_cog_bci_contrastive_transfer,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one COG-BCI contrastive EEGNet transfer screen."
    )
    parser.add_argument(
        "--config",
        default="experiments/cog_bci/contrastive_eegnet_transfer_screening.json",
    )
    args = parser.parse_args()
    config_path = REPOSITORY_ROOT / Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = run_cog_bci_contrastive_transfer(
        config, repository_root=REPOSITORY_ROOT
    )
    print(json.dumps(summary["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
