#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COLET task leaf probe.

Reads only this reference chain:
    /Data/task[subject_idx] -> task group -> annotation/blinks/gaze/pupil[task_idx]

It does not walk /#refs#, does not recursively inspect all subjects/tasks,
and does not load large gaze/pupil/blinks arrays.

Safe commands:
    D:\miniconda3\envs\eeg_nir\python.exe src\24_probe_colet_task_leafs.py `
      --version COLET_v3 `
      --subjects 1 `
      --tasks 1

    D:\miniconda3\envs\eeg_nir\python.exe src\24_probe_colet_task_leafs.py `
      --version COLET_v3 `
      --subjects 1 `
      --tasks 1 `
      --inspect-group-children
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import h5py
except Exception as exc:
    raise RuntimeError(
        "h5py is required. Install with: "
        "D:\\miniconda3\\envs\\eeg_nir\\python.exe -m pip install h5py"
    ) from exc


@dataclass
class LeafRow:
    version: str
    mat_path: str
    subject_index: int
    task_index: int
    field: str
    parent_dataset_path: str
    target_path: str
    node_type: str
    shape: str
    dtype: str
    n_children: Optional[int]
    child_preview: str
    attrs_json: str
    tiny_preview: str
    elapsed_sec: float


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def find_mat_path(colet_root: Path, version: str) -> Path:
    version_dir = colet_root / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Version directory not found: {version_dir}")
    mat_files = sorted(version_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {version_dir}")
    return mat_files[0]


def safe_shape(obj: Any) -> str:
    try:
        return str(tuple(obj.shape)) if hasattr(obj, "shape") else ""
    except Exception:
        return "unknown"


def safe_dtype(obj: Any) -> str:
    try:
        return str(obj.dtype) if hasattr(obj, "dtype") else ""
    except Exception:
        return "unknown"


def safe_attrs(obj: Any) -> str:
    out: Dict[str, str] = {}
    try:
        for k, v in obj.attrs.items():
            out[str(k)] = str(v)
    except Exception:
        pass
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


def child_preview(obj: Any, max_children: int) -> Tuple[Optional[int], str]:
    if not isinstance(obj, h5py.Group):
        return None, ""
    try:
        keys = list(obj.keys())
        return len(keys), ";".join(keys[:max_children])
    except Exception:
        return None, ""


def deref(h5: h5py.File, ref: Any) -> Optional[Any]:
    try:
        if isinstance(ref, h5py.Reference) and ref:
            return h5[ref]
    except Exception:
        return None
    return None


def decode_small_numeric_as_text(ds: h5py.Dataset, max_items: int = 200) -> str:
    try:
        size = int(np.prod(ds.shape)) if ds.shape else 1
        if size > max_items:
            return ""
        arr = np.asarray(ds[()])
        if "uint16" in str(ds.dtype) or "uint32" in str(ds.dtype):
            chars = []
            for x in arr.reshape(-1):
                ix = int(x)
                if ix == 0:
                    continue
                if 0 <= ix <= 0x10FFFF:
                    chars.append(chr(ix))
            text = "".join(chars)
            if text.strip():
                return text[:500]
        if size <= 40:
            return np.array2string(arr, threshold=40)[:500]
    except Exception:
        return ""
    return ""


def tiny_preview(obj: Any) -> str:
    if not isinstance(obj, h5py.Dataset):
        return ""
    return decode_small_numeric_as_text(obj)[:500]


def make_row(
    version: str,
    mat_path: Path,
    subject_index: int,
    task_index: int,
    field: str,
    parent_dataset_path: str,
    obj: Any,
    elapsed_sec: float,
    max_children: int,
) -> LeafRow:
    n_children, preview = child_preview(obj, max_children)
    return LeafRow(
        version=version,
        mat_path=str(mat_path),
        subject_index=subject_index,
        task_index=task_index,
        field=field,
        parent_dataset_path=parent_dataset_path,
        target_path=getattr(obj, "name", ""),
        node_type=type(obj).__name__,
        shape=safe_shape(obj),
        dtype=safe_dtype(obj),
        n_children=n_children,
        child_preview=preview,
        attrs_json=safe_attrs(obj),
        tiny_preview=tiny_preview(obj),
        elapsed_sec=round(elapsed_sec, 6),
    )


def inspect_task_leafs(
    h5: h5py.File,
    mat_path: Path,
    version: str,
    subjects: List[int],
    tasks: List[int],
    fields: List[str],
    max_children: int,
    inspect_group_children: bool,
) -> List[LeafRow]:
    rows: List[LeafRow] = []
    task_ds_path = "/Data/task"
    if task_ds_path not in h5:
        raise KeyError(f"{task_ds_path} not found")

    task_ds = h5[task_ds_path]
    n_subjects = int(task_ds.shape[0])

    for subject_index in subjects:
        if subject_index < 1 or subject_index > n_subjects:
            print(f"[WARN] subject {subject_index} outside 1..{n_subjects}")
            continue

        t0 = time.perf_counter()
        task_ref = task_ds[subject_index - 1, 0]
        task_group = deref(h5, task_ref)
        print(f"Subject {subject_index}: task group -> {getattr(task_group, 'name', None)} in {time.perf_counter() - t0:.6f}s")

        if task_group is None or not isinstance(task_group, h5py.Group):
            print(f"[WARN] task object for subject {subject_index} is not a group")
            continue

        print(f"  available fields: {list(task_group.keys())}")

        for field in fields:
            if field not in task_group:
                print(f"  [WARN] missing field: {field}")
                continue

            field_ds = task_group[field]
            print(f"  {field}: {getattr(field_ds, 'name', '')} shape={safe_shape(field_ds)} dtype={safe_dtype(field_ds)}")

            rows.append(make_row(version, mat_path, subject_index, -1, field, getattr(field_ds, "name", ""), field_ds, 0.0, max_children))

            if not isinstance(field_ds, h5py.Dataset) or field_ds.dtype != object:
                continue

            n_tasks = int(field_ds.shape[0]) if len(field_ds.shape) >= 1 else 0
            for task_index in tasks:
                if task_index < 1 or task_index > n_tasks:
                    print(f"    [WARN] task {task_index} outside 1..{n_tasks}")
                    continue

                t1 = time.perf_counter()
                ref = field_ds[task_index - 1, 0] if len(field_ds.shape) >= 2 else field_ds[task_index - 1]
                obj = deref(h5, ref)
                elapsed = time.perf_counter() - t1
                if obj is None:
                    print(f"    [WARN] deref failed for {field}[{task_index}]")
                    continue

                print(
                    f"    {field}[{task_index}] -> {getattr(obj, 'name', '')} | "
                    f"type={type(obj).__name__} shape={safe_shape(obj)} dtype={safe_dtype(obj)} elapsed={elapsed:.6f}s"
                )
                rows.append(make_row(version, mat_path, subject_index, task_index, field, getattr(field_ds, "name", ""), obj, elapsed, max_children))

                if inspect_group_children and isinstance(obj, h5py.Group):
                    child_names = list(obj.keys())[:max_children]
                    print(f"      child fields: {child_names}")
                    for child_name in child_names:
                        t2 = time.perf_counter()
                        child = obj[child_name]
                        child_elapsed = time.perf_counter() - t2
                        print(
                            f"      {child_name}: path={getattr(child, 'name', '')} "
                            f"type={type(child).__name__} shape={safe_shape(child)} dtype={safe_dtype(child)} "
                            f"preview={tiny_preview(child)[:120]}"
                        )
                        rows.append(make_row(version, mat_path, subject_index, task_index, f"{field}.{child_name}", getattr(obj, "name", ""), child, child_elapsed, max_children))
    return rows


def build_report(df: pd.DataFrame, mat_path: Path) -> str:
    lines = ["# COLET task leaf probe report", "", f"- MAT file: `{mat_path}`", "", "## Leaf metadata", ""]
    if len(df):
        show_cols = [
            "subject_index", "task_index", "field", "parent_dataset_path", "target_path",
            "node_type", "shape", "dtype", "n_children", "child_preview", "tiny_preview", "elapsed_sec"
        ]
        lines.append(df[show_cols].to_markdown(index=False))
    else:
        lines.append("No rows collected.")
    lines += ["", "## Next step", "", "Use this metadata to implement a selective extractor for annotation, gaze, pupil and blinks.", ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe COLET task leaf object references.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--colet-root", type=str, default="data/external/COLET")
    parser.add_argument("--version", type=str, default="COLET_v3")
    parser.add_argument("--output-dir", type=str, default="reports/wearable_pm_alignment")
    parser.add_argument("--subjects", type=str, default="1")
    parser.add_argument("--tasks", type=str, default="1")
    parser.add_argument("--fields", type=str, default="annotation,blinks,gaze,pupil")
    parser.add_argument("--max-children", type=int, default=40)
    parser.add_argument("--inspect-group-children", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    colet_root = resolve_path(root, args.colet_root)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_path = find_mat_path(colet_root, args.version)
    subjects = parse_int_list(args.subjects)
    tasks = parse_int_list(args.tasks)
    fields = [x.strip() for x in args.fields.split(",") if x.strip()]

    print("=" * 80)
    print("COLET task leaf probe")
    print("=" * 80)
    print(f"MAT path: {mat_path}")
    print(f"Size: {mat_path.stat().st_size / (1024 ** 3):.2f} GB")
    print(f"Subjects: {subjects}")
    print(f"Tasks: {tasks}")
    print(f"Fields: {fields}")
    print(f"Inspect group children: {args.inspect_group_children}")
    print("")

    t_open = time.perf_counter()
    with h5py.File(mat_path, "r") as h5:
        print(f"Opened file in {time.perf_counter() - t_open:.6f}s")
        rows = inspect_task_leafs(h5, mat_path, args.version, subjects, tasks, fields, args.max_children, args.inspect_group_children)

    df = pd.DataFrame([asdict(r) for r in rows])
    csv_path = output_dir / "colet_task_leaf_probe.csv"
    json_path = output_dir / "colet_task_leaf_probe.json"
    report_path = output_dir / "colet_task_leaf_probe_report.md"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(df, mat_path), encoding="utf-8")

    print("")
    print("=" * 80)
    print("Saved COLET task leaf probe outputs")
    print("=" * 80)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")
    print("")
    print("Rows:")
    if len(df):
        show_cols = ["subject_index", "task_index", "field", "target_path", "node_type", "shape", "dtype", "n_children", "child_preview", "tiny_preview"]
        print(df[show_cols].to_string(index=False))
    else:
        print("No rows.")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
