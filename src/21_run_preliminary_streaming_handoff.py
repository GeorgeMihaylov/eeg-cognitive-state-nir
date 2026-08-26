"""Compatibility entry point for scripts.run_preliminary_streaming_handoff."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_preliminary_streaming_handoff import main

if __name__ == "__main__":
    main()
