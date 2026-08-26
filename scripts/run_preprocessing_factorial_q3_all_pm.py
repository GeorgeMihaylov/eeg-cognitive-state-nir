"""CLI entry point for the seven-PM preprocessing factorial experiment."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.preprocessing_factorial_q3_all_pm import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
