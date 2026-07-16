"""Compatibility wrapper for the canonical preprocessing-matrix CLI."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli import main  # noqa: E402


def _canonical_cli_args(arguments: list[str]) -> list[str]:
    """Translate the former entry-point spelling without owning execution."""
    translated = list(arguments)
    requested_run = "--run" in translated
    if "--spec" in translated:
        translated[translated.index("--spec")] = "--experiment-matrix"
    translated = [value for value in translated if value != "--run"]
    if "--build-missing-caches" in translated and not requested_run:
        translated.append("--cache-only")
    return translated


if __name__ == "__main__":
    main(_canonical_cli_args(sys.argv[1:]))
