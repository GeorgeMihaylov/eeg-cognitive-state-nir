from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "dist",
    "build",
}

DEFAULT_EXCLUDE_PREFIXES = {
    "data/external",
    "data/raw",
    "reports/runs",
    "reports/wearable_pm_alignment/runs",
}

DEFAULT_EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
    ".bak",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".parquet",
    ".pkl",
    ".npy",
    ".npz",
    ".fdt",
    ".edf",
    ".mat",
}


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def should_exclude(path: Path, root: Path, exclude_dirs: set[str], exclude_prefixes: set[str], exclude_suffixes: set[str]) -> bool:
    rel = rel_posix(path, root)

    if any(part in exclude_dirs for part in path.relative_to(root).parts):
        return True

    if any(rel == prefix or rel.startswith(prefix.rstrip("/") + "/") for prefix in exclude_prefixes):
        return True

    if path.is_file() and path.suffix.lower() in exclude_suffixes:
        return True

    return False


def build_tree(root: Path, exclude_dirs: set[str], exclude_prefixes: set[str], exclude_suffixes: set[str]) -> list[str]:
    lines: list[str] = []

    def walk(directory: Path, prefix: str = "") -> None:
        try:
            children = sorted(
                [p for p in directory.iterdir() if not should_exclude(p, root, exclude_dirs, exclude_prefixes, exclude_suffixes)],
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except PermissionError:
            lines.append(f"{prefix}[permission denied] {directory.name}/")
            return

        for idx, child in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if child.is_dir():
                lines.append(f"{prefix}{connector}{child.name}/")
                walk(child, next_prefix)
            else:
                size_kb = child.stat().st_size / 1024
                lines.append(f"{prefix}{connector}{child.name} ({size_kb:.1f} KB)")

    lines.append(f"{root.name}/")
    walk(root)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Export project directory tree to TXT.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root directory.")
    parser.add_argument("--output", type=Path, default=Path("project_structure.txt"), help="Output TXT file.")
    parser.add_argument("--include-data", action="store_true", help="Do not exclude data/raw and data/external.")
    parser.add_argument("--include-runs", action="store_true", help="Do not exclude reports/runs directories.")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()

    exclude_prefixes = set(DEFAULT_EXCLUDE_PREFIXES)

    if args.include_data:
        exclude_prefixes.discard("data/external")
        exclude_prefixes.discard("data/raw")

    if args.include_runs:
        exclude_prefixes.discard("reports/runs")
        exclude_prefixes.discard("reports/wearable_pm_alignment/runs")

    lines = build_tree(
        root=root,
        exclude_dirs=set(DEFAULT_EXCLUDE_DIRS),
        exclude_prefixes=exclude_prefixes,
        exclude_suffixes=set(DEFAULT_EXCLUDE_SUFFIXES),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 80)
    print("Project tree exported")
    print("=" * 80)
    print(f"Root:   {root}")
    print(f"Output: {output}")
    print(f"Lines:  {len(lines)}")


if __name__ == "__main__":
    main()