"""Compatibility entry point for scripts.run_preprocessing_ablation."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_preprocessing_ablation import main

if __name__ == "__main__":
    main()
