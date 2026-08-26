"""CLI for the reproducible colleague metrics package."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.analysis.colleague_metrics_package import main

if __name__ == "__main__":
    raise SystemExit(main())
