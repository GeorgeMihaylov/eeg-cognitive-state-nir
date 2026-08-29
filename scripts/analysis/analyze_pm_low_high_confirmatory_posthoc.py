"""CLI for the training-free LOW/HIGH confirmatory post-hoc audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.analysis.pm_low_high_confirmatory_posthoc import run_posthoc_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit completed LOW/HIGH predictions without training models."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(
            "reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = _ROOT / experiment_dir
    result = run_posthoc_analysis(
        experiment_dir,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
