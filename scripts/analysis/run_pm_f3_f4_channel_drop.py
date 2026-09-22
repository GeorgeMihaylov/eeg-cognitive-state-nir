"""CLI for the fixed F3/F4 channel-drop ablation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments.pm_f3_f4_channel_drop import main

if __name__ == "__main__":
    raise SystemExit(main())
