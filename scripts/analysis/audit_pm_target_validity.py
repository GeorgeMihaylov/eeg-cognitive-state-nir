"""CLI for the training-free Emotiv PM target validity audit."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.analysis.pm_target_validity_audit import main


if __name__ == "__main__":
    main()
