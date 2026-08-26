"""Compatibility entry point; implementation moved to bench.analysis.experiment_summary."""

from importlib import import_module
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_implementation = import_module("bench.analysis.experiment_summary")
globals().update(
    {
        name: getattr(_implementation, name)
        for name in dir(_implementation)
        if not name.startswith("__")
    }
)

if __name__ == "__main__":
    _result = _implementation.main()
    if isinstance(_result, int):
        raise SystemExit(_result)
