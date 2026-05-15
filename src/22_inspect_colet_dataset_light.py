#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lightweight COLET .mat inspector.

This version is designed for large MATLAB v7.3 / HDF5 files.
It does NOT recursively traverse the full file. It inspects only:
    - top-level keys
    - /Data group keys
    - limited children per group
    - shapes/dtypes
    - small previews only when safe

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\22_inspect_colet_dataset_light.py

Optional:
    D:\miniconda3\envs\eeg_nir\python.exe src\22_inspect_colet_dataset_light.py `
      --colet-root data\external\COLET `
      --max-children 80 `
      --max-depth 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import h5py
except Exception as exc:
    raise RuntimeError(
        "h5py is required for COLET .mat v7.3 files. Install with: "
        "D:\\miniconda3\\envs\\eeg_nir\\python.exe -m pip install h5py"
    ) from exc


@dataclass
class H5NodeInfo:
    version: str
    mat_path: str
    node_path: str
    depth: int
    node_type: str
    shape: str
    dtype: str
    n_children: Optional[int]
    child_names_preview: str
    attrs_json: str


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def find_mat_files(colet_root: Path) -> List[Path]:
    if not colet_root.exists():
        raise FileNotFoundError(f"COLET root does not exist: {colet_root}")

    mat_files = sorted(colet_root.glob("COLET_v*/*.mat"))
    if not mat_files and colet_root.name.startswith("COLET_v"):
        mat_files = sorted(colet_root.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under: {colet_root}")

    return mat_files


def safe_attrs(obj: Any) -> Dict[str, str]:
    attrs = {}
    try:
        for k, v in obj.attrs.items():
            try:
                attrs[str(k)] = str(v)
            except Exception:
                attrs[str(k)] = "<unserializable>"
    except Exception:
        pass
    return attrs


def node_info(version: str, mat_path: Path, node_path: str, obj: Any, depth: int, max_children: int) -> H5NodeInfo:
    node_type = type(obj).__name__
    shape = ""
    dtype = ""
    n_children = None
    child_names_preview = ""

    if hasattr(obj, "shape"):
        try:
            shape = str(tuple(obj.shape))
        except Exception:
            shape = "unknown"

    if hasattr(obj, "dtype"):
        try:
            dtype = str(obj.dtype)
        except Exception:
            dtype = "unknown"

    if isinstance(obj, h5py.Group):
        child_names = list(obj.keys())
        n_children = len(child_names)
        child_names_preview = ";".join(child_names[:max_children])

    return H5NodeInfo(
        version=version,
        mat_path=str(mat_path),
        node_path=node_path,
        depth=depth,
        node_type=node_type,
        shape=shape,
        dtype=dtype,
        n_children=n_children,
        child_names_preview=child_names_preview,
        attrs_json=json.dumps(safe_attrs(obj), ensure_ascii=False, sort_keys=True),
    )


def inspect_limited_group(
    version: str,
    mat_path: Path,
    h5: h5py.File,
    start_path: str,
    max_depth: int,
    max_children: int,
) -> List[H5NodeInfo]:
    rows: List[H5NodeInfo] = []

    def walk(path: str, depth: int) -> None:
        if depth > max_depth:
            return

        try:
            obj = h5[path]
        except Exception:
            return

        rows.append(node_info(version, mat_path, path, obj, depth, max_children))

        if not isinstance(obj, h5py.Group):
            return

        child_names = list(obj.keys())[:max_children]
        for child in child_names:
            child_path = path.rstrip("/") + "/" + child if path != "/" else "/" + child
            walk(child_path, depth + 1)

    walk(start_path, 0)
    return rows


def inspect_mat_file(mat_path: Path, max_depth: int, max_children: int) -> List[H5NodeInfo]:
    version = mat_path.parent.name
    rows: List[H5NodeInfo] = []

    with h5py.File(mat_path, "r") as h5:
        # Top-level nodes.
        rows.append(node_info(version, mat_path, "/", h5, 0, max_children))

        for key in list(h5.keys())[:max_children]:
            path = "/" + key
            obj = h5[path]
            rows.append(node_info(version, mat_path, path, obj, 1, max_children))

        # Main MATLAB struct group.
        if "Data" in h5:
            rows.extend(
                inspect_limited_group(
                    version=version,
                    mat_path=mat_path,
                    h5=h5,
                    start_path="/Data",
                    max_depth=max_depth,
                    max_children=max_children,
                )
            )

    # Remove duplicate rows while preserving order.
    seen = set()
    deduped = []
    for row in rows:
        key = (row.version, row.mat_path, row.node_path)
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


def read_readme(version_dir: Path, max_chars: int = 5000) -> str:
    path = version_dir / "readme.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def build_report(nodes_df: pd.DataFrame, readme_df: pd.DataFrame, output_dir: Path) -> str:
    lines: List[str] = []

    lines.append("# COLET lightweight HDF5 inspection report")
    lines.append("")
    lines.append("## MAT structure summary")
    lines.append("")
    show_cols = [
        "version",
        "node_path",
        "depth",
        "node_type",
        "shape",
        "dtype",
        "n_children",
        "child_names_preview",
    ]
    lines.append(nodes_df[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Version readme previews")
    lines.append("")
    for _, row in readme_df.iterrows():
        lines.append(f"### {row['version']}")
        lines.append("")
        lines.append("```text")
        lines.append(str(row["readme_preview"]))
        lines.append("```")
        lines.append("")

    lines.append("## What to inspect next")
    lines.append("")
    lines.append("If `/Data` contains MATLAB struct fields, identify fields that correspond to:")
    lines.append("")
    lines.append("```text")
    lines.append("subject / participant")
    lines.append("trial / task")
    lines.append("image / stimulus")
    lines.append("difficulty / workload")
    lines.append("gaze / fixation / pupil / eye movement features")
    lines.append("```")
    lines.append("")
    lines.append("Then implement `src/23_prepare_colet_dataset.py` to flatten the selected fields.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight COLET HDF5 .mat inspector.")
    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument("--colet-root", type=str, default="data/external/COLET", help="COLET root.")
    parser.add_argument("--output-dir", type=str, default="reports/wearable_pm_alignment")
    parser.add_argument("--max-depth", type=int, default=3, help="Max depth under /Data.")
    parser.add_argument("--max-children", type=int, default=80, help="Max children per group.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    colet_root = resolve_path(root, args.colet_root)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = find_mat_files(colet_root)

    print("=" * 80)
    print("COLET lightweight HDF5 inspection")
    print("=" * 80)
    print(f"COLET root: {colet_root}")
    print(f"MAT files: {[str(p) for p in mat_files]}")
    print(f"max_depth={args.max_depth} | max_children={args.max_children}")
    print("")

    all_rows: List[H5NodeInfo] = []
    readme_rows: List[Dict[str, str]] = []

    for mat_path in mat_files:
        version = mat_path.parent.name
        print(f"Inspecting {version}: {mat_path.name} ({mat_path.stat().st_size / (1024 ** 3):.2f} GB)")
        rows = inspect_mat_file(mat_path, args.max_depth, args.max_children)
        all_rows.extend(rows)
        readme_rows.append(
            {
                "version": version,
                "readme_path": str(mat_path.parent / "readme.txt"),
                "readme_preview": read_readme(mat_path.parent),
            }
        )

        # Print concise version summary.
        for row in rows:
            if row.node_path in ["/", "/Data"] or row.depth <= 2:
                print(
                    f"  {row.node_path} | type={row.node_type} | "
                    f"shape={row.shape} | dtype={row.dtype} | "
                    f"children={row.n_children} | preview={row.child_names_preview[:250]}"
                )
        print("")

    nodes_df = pd.DataFrame([asdict(r) for r in all_rows])
    readme_df = pd.DataFrame(readme_rows)

    nodes_csv = output_dir / "colet_h5_nodes_limited.csv"
    readme_csv = output_dir / "colet_readme_previews.csv"
    report_md = output_dir / "colet_h5_light_inspection_report.md"

    nodes_df.to_csv(nodes_csv, index=False, encoding="utf-8")
    readme_df.to_csv(readme_csv, index=False, encoding="utf-8")
    report_md.write_text(build_report(nodes_df, readme_df, output_dir), encoding="utf-8")

    print("=" * 80)
    print("Saved COLET lightweight inspection outputs")
    print("=" * 80)
    print(f"Nodes CSV: {nodes_csv}")
    print(f"Readme CSV: {readme_csv}")
    print(f"Report: {report_md}")
    print("")
    print("Top /Data children by version:")
    data_rows = nodes_df[nodes_df["node_path"] == "/Data"]
    if len(data_rows):
        print(data_rows[["version", "n_children", "child_names_preview"]].to_string(index=False))
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
