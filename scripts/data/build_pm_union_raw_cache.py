"""CLI for PM-union raw cache materialization."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.datasets.pm_union_raw_materialization import main

if __name__ == "__main__":
    main()
