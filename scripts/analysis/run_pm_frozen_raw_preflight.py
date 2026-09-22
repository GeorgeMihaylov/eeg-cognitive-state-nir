"""Read-only frozen feature-to-raw EEG contract preflight."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.analysis.pm_frozen_raw_preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
