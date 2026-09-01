"""Portable deterministic paths for deeply nested runtime artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


# Leave headroom below the legacy Windows MAX_PATH limit (260 including the
# terminating null) for library-specific temporary suffixes and file handles.
PORTABLE_PATH_LIMIT = 240
DEFAULT_FILENAME_RESERVE = 64


def absolute_path_length(path: str | Path) -> int:
    """Return the character length of an absolute platform-native path."""

    return len(os.path.abspath(os.fspath(path)))


def portable_artifact_directory(
    root: str | Path,
    components: Sequence[Any],
    *,
    compact_namespace: str,
    path_limit: int = PORTABLE_PATH_LIMIT,
    filename_reserve: int = DEFAULT_FILENAME_RESERVE,
) -> Path:
    """Keep the logical hierarchy when safe, otherwise use a stable hash dir.

    The returned directory depends only on ``root`` and the ordered logical
    components.  Path placement is runtime-only and therefore does not alter
    benchmark, protocol, run-specification, or checkpoint identities.
    """

    root_path = Path(root)
    normalized = tuple(str(component) for component in components)
    if not normalized or any(not component for component in normalized):
        raise ValueError("artifact path components must be non-empty")
    if int(filename_reserve) <= 0:
        raise ValueError("filename_reserve must be positive")
    if int(path_limit) <= int(filename_reserve):
        raise ValueError("path_limit must exceed filename_reserve")

    candidate = root_path.joinpath(*normalized)
    probe_name = "x" * int(filename_reserve)
    if absolute_path_length(candidate / probe_name) <= int(path_limit):
        return candidate

    payload = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    compact = root_path / str(compact_namespace) / digest
    compact_length = absolute_path_length(compact / probe_name)
    if compact_length > int(path_limit):
        raise OSError(
            "Output root is too long for portable artifact paths: "
            f"root={root_path}, compact_length={compact_length}, "
            f"limit={path_limit}. Choose a shorter --output-dir."
        )
    return compact


__all__ = [
    "DEFAULT_FILENAME_RESERVE",
    "PORTABLE_PATH_LIMIT",
    "absolute_path_length",
    "portable_artifact_directory",
]
