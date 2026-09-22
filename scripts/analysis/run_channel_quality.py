"""Thin entry point for the frozen label-free channel quality audit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.analysis.channel_quality import main

if __name__ == "__main__":
    raise SystemExit(main())
