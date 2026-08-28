"""Thin CLI for the seven-PM EEG/target lag sweep."""

import runpy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    runpy.run_module(
        "bench.analysis.pm_eeg_lag_sweep",
        run_name="__main__",
    )
