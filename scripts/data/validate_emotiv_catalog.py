"""CLI for Emotiv catalog and column quality control."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.data_quality.emotiv_catalog_validation import main

if __name__ == "__main__":
    main()
