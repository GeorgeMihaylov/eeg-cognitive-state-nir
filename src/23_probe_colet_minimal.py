#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ultra-light COLET HDF5/MATLAB v7.3 probe.

This script intentionally does NOT:
    - recursively walk /#refs#
    - recursively walk task structs
    - read large numeric arrays
    - dereference nested object arrays deeply

It only:
    1. opens one COLET .mat file;
    2. prints /Data keys;
    3. reads the first N object references from /Data/subject_info and /Data/task;
    4. dereferences each selected object once;
    5. prints first-level field names / child metadata only;
    6. saves a small CSV/MD report.

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\23_probe_colet_minimal.py `
      --version COLET_v3 `
      --max-subjects 2 `
      --max-fields 40

If still slow:
    D:\miniconda3\envs\eeg_nir\python.exe src\23_probe_colet_minimal.py `
      --version COLET_v3 `
      --max-subjects 1 `
      --max-fields 15 `
      --datasets subject_info
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import h5py
except Exception as exc:
    raise RuntimeError(
        "h5py is required. Install with: "
        "D:\\miniconda3\\envs\\eeg_nir\\python.exe -m pip install h5py"
    ) from exc


@dataclass
class ProbeRow:
    version: str
    mat_path: str
    source_dataset: str
    subject_index: int
    ref_target_path: str
    field_name: str
    field_h5_path: str
    node_type: str
    shape: str
    dtype: str
    n_children: Optional[int]
    child_preview: str
    attrs_json: str
    elapsed_sec: float


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


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


def child_preview(obj: Any, max_fields: int) -> Tuple[Optional[int], str]:
    if not isinstance(obj, h5py.Group):
        return None, ""

    try:
        keys = list(obj.keys())
        return len(keys), ";".join(keys[:max_fields])
    except Exception:
        return None, ""


def metadata_row(
    version: str,
    mat_path: Path,
    source_dataset: str,
    subject_index: int,
    ref_target_path: str,
    field_name: str,
    obj: Any,
    elapsed_sec: float,
    max_fields: int,
) -> ProbeRow:
    n_children, preview = child_preview(obj, max_fields=max_fields)

    return ProbeRow(
        version=version,
        mat_path=str(mat_path),
        source_dataset=source_dataset,
        subject_index=subject_index,
        ref_target_path=ref_target_path,
        field_name=field_name,
        field_h5_path=getattr(obj, "name", ""),
        node_type=type(obj).__name__,
        shape=safe_shape(obj),
        dtype=safe_dtype(obj),
        n_children=n_children,
        child_preview=preview,
        attrs_json=safe_attrs(obj),
        elapsed_sec=round(elapsed_sec, 4),
    )


def deref_once(h5: h5py.File, ref: Any) -> Optional[Any]:
    try:
        if isinstance(ref, h5py.Reference) and ref:
            return h5[ref]
    except Exception:
        return None
    return None


def probe_dataset(
    h5: h5py.File,
    mat_path: Path,
    version: str,
    dataset_name: str,
    max_subjects: int,
    max_fields: int,
) -> List[ProbeRow]:
    rows: List[ProbeRow] = []
    ds_path = f"/Data/{dataset_name}"

    if ds_path not in h5:
        print(f"[WARN] Missing {ds_path}")
        return rows

    ds = h5[ds_path]
    print(f"{ds_path}: shape={safe_shape(ds)} dtype={safe_dtype(ds)}")

    rows.append(
        metadata_row(
            version=version,
            mat_path=mat_path,
            source_dataset=dataset_name,
            subject_index=-1,
            ref_target_path=ds_path,
            field_name="<dataset>",
            obj=ds,
            elapsed_sec=0.0,
            max_fields=max_fields,
        )
    )

    if not isinstance(ds, h5py.Dataset):
        return rows

    n_subjects = int(ds.shape[0]) if len(ds.shape) >= 1 else 0
    n = min(max_subjects, n_subjects)

    for i in range(n):
        t0 = time.perf_counter()

        try:
            ref = ds[i, 0] if len(ds.shape) >= 2 else ds[i]
        except Exception as exc:
            print(f"  [{dataset_name}][{i+1}] read ref failed: {exc}")
            continue

        obj = deref_once(h5, ref)
        elapsed = time.perf_counter() - t0

        if obj is None:
            print(f"  [{dataset_name}][{i+1}] deref failed")
            continue

        target_path = getattr(obj, "name", "")
        print(
            f"  [{dataset_name}][{i+1}] -> {target_path} | "
            f"type={type(obj).__name__} shape={safe_shape(obj)} dtype={safe_dtype(obj)} "
            f"deref_sec={elapsed:.4f}"
        )

        rows.append(
            metadata_row(
                version=version,
                mat_path=mat_path,
                source_dataset=dataset_name,
                subject_index=i + 1,
                ref_target_path=target_path,
                field_name="<target>",
                obj=obj,
                elapsed_sec=elapsed,
                max_fields=max_fields,
            )
        )

        if isinstance(obj, h5py.Group):
            try:
                field_names = list(obj.keys())[:max_fields]
            except Exception as exc:
                print(f"    cannot list fields: {exc}")
                continue

            print(f"    fields: {field_names}")

            for field in field_names:
                t1 = time.perf_counter()
                try:
                    child = obj[field]
                except Exception as exc:
                    print(f"    {field}: failed: {exc}")
                    continue

                child_elapsed = time.perf_counter() - t1
                n_children, preview = child_preview(child, max_fields=max_fields)

                print(
                    f"    {field}: path={getattr(child, 'name', '')} "
                    f"type={type(child).__name__} shape={safe_shape(child)} dtype={safe_dtype(child)} "
                    f"children={n_children} preview={preview[:200]}"
                )

                rows.append(
                    metadata_row(
                        version=version,
                        mat_path=mat_path,
                        source_dataset=dataset_name,
                        subject_index=i + 1,
                        ref_target_path=target_path,
                        field_name=field,
                        obj=child,
                        elapsed_sec=child_elapsed,
                        max_fields=max_fields,
                    )
                )

    return rows


def build_report(rows_df: pd.DataFrame, mat_path: Path) -> str:
    lines: List[str] = []
    lines.append("# COLET minimal probe report")
    lines.append("")
    lines.append(f"- MAT file: `{mat_path}`")
    lines.append("")
    lines.append("## Probe rows")
    lines.append("")
    if len(rows_df):
        show_cols = [
            "source_dataset",
            "subject_index",
            "ref_target_path",
            "field_name",
            "field_h5_path",
            "node_type",
            "shape",
            "dtype",
            "n_children",
            "child_preview",
            "elapsed_sec",
        ]
        lines.append(rows_df[show_cols].to_markdown(index=False))
    else:
        lines.append("No rows.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This report intentionally contains first-level metadata only. It is used to identify exact MATLAB/HDF5 field paths before implementing a selective reader.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal COLET HDF5 reference probe.")
    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument("--colet-root", type=str, default="data/external/COLET")
    parser.add_argument("--version", type=str, default="COLET_v3")
    parser.add_argument("--output-dir", type=str, default="reports/wearable_pm_alignment")
    parser.add_argument("--max-subjects", type=int, default=2)
    parser.add_argument("--max-fields", type=int, default=40)
    parser.add_argument(
        "--datasets",
        type=str,
        default="subject_info,task",
        help="Comma-separated /Data datasets to probe: subject_info,task",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    colet_root = resolve_path(root, args.colet_root)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_path = find_mat_path(colet_root, args.version)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    print("=" * 80)
    print("COLET minimal HDF5 probe")
    print("=" * 80)
    print(f"MAT path: {mat_path}")
    print(f"Size: {mat_path.stat().st_size / (1024 ** 3):.2f} GB")
    print(f"Datasets: {datasets}")
    print(f"Max subjects: {args.max_subjects}")
    print(f"Max fields: {args.max_fields}")
    print("")

    rows: List[ProbeRow] = []

    t_open = time.perf_counter()
    with h5py.File(mat_path, "r") as h5:
        print(f"Opened file in {time.perf_counter() - t_open:.4f} sec")
        print(f"Top keys: {list(h5.keys())}")
        if "Data" in h5:
            print(f"/Data keys: {list(h5['Data'].keys())}")
        print("")

        for dataset_name in datasets:
            rows.extend(
                probe_dataset(
                    h5=h5,
                    mat_path=mat_path,
                    version=args.version,
                    dataset_name=dataset_name,
                    max_subjects=args.max_subjects,
                    max_fields=args.max_fields,
                )
            )
            print("")

    rows_df = pd.DataFrame([asdict(r) for r in rows])

    csv_path = output_dir / "colet_minimal_probe.csv"
    json_path = output_dir / "colet_minimal_probe.json"
    report_path = output_dir / "colet_minimal_probe_report.md"

    rows_df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(rows_df, mat_path), encoding="utf-8")

    print("=" * 80)
    print("Saved COLET minimal probe outputs")
    print("=" * 80)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
