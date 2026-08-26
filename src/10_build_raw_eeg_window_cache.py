"""Compatibility entry point for scripts.data.build_raw_eeg_window_cache."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.data.build_raw_eeg_window_cache import main

if __name__ == "__main__":
    main()
