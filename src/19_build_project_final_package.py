"""Compatibility entry point for scripts.analysis.build_project_final_package."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.analysis.build_project_final_package import main

if __name__ == "__main__":
    raise SystemExit(main())
