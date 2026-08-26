"""CLI for Emotiv source-file inspection."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.data_quality.emotiv_file_inspection import main

if __name__ == "__main__":
    main()
