#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inspect WESAD dataset structure.

Purpose:
    1. Find all WESAD subject folders and *.pkl files.
    2. Inspect pickle structure safely.
    3. Inspect Empatica E4 CSV files.
    4. Summarize subject-level signal availability, lengths and labels.
    5. Save inventory reports before feature extraction.

Expected input structure:
    data/external/WESAD/WESAD/S2/S2.pkl
    data/external/WESAD/WESAD/S2/S2_E4_Data/ACC.csv
    data/external/WESAD/WESAD/S2/S2_E4_Data/BVP.csv
    ...

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\16_inspect_wesad_dataset.py
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


EXPECTED_E4_FILES = [
    "ACC.csv",
    "BVP.csv",
    "EDA.csv",
    "HR.csv",
    "IBI.csv",
    "TEMP.csv",
    "tags.csv",
    "info.txt",
]

WESAD_LABELS = {
    0: "not_defined_or_transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "unused_or_other",
    6: "unused_or_other",
    7: "unused_or_other",
}


@dataclass
class SubjectInventory:
    subject_id: str
    subject_dir: str
    pkl_path: str
    e4_dir: str
    has_pkl: bool
    has_e4_dir: bool
    e4_files_present: str
    e4_files_missing: str
    pkl_size_mb: float
    pickle_top_keys: str
    pickle_signal_keys: str
    pickle_chest_keys: str
    pickle_wrist_keys: str
    label_length: Optional[int]
    label_counts: str
    chest_signal_shapes: str
    wrist_signal_shapes: str
    e4_csv_summary: str
    quest_path: str
    readme_path: str
    respiban_path: str


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def find_wesad_base(wesad_root: Path) -> Path:
    """
    Accept both:
        data/external/WESAD
        data/external/WESAD/WESAD
    """
    if not wesad_root.exists():
        raise FileNotFoundError(f"WESAD root does not exist: {wesad_root}")

    direct_subjects = sorted(wesad_root.glob("S*/S*.pkl"))
    if direct_subjects:
        return wesad_root

    nested = wesad_root / "WESAD"
    nested_subjects = sorted(nested.glob("S*/S*.pkl"))
    if nested.exists() and nested_subjects:
        return nested

    recursive = sorted(wesad_root.rglob("S*/S*.pkl"))
    if recursive:
        return recursive[0].parent.parent

    raise FileNotFoundError(f"Could not find WESAD subject pickle files under: {wesad_root}")


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except TypeError:
            f.seek(0)
            return pickle.load(f)


def safe_shape(x: Any) -> str:
    try:
        return str(tuple(np.asarray(x).shape))
    except Exception:
        return "unknown"


def summarize_e4_csv(path: Path, max_preview_lines: int = 5) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "exists": path.exists(),
        "size_kb": None,
        "n_lines": None,
        "preview": None,
        "error": None,
    }

    if not path.exists():
        return info

    try:
        info["size_kb"] = round(path.stat().st_size / 1024, 2)
        n_lines = 0
        preview = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                n_lines += 1
                if i < max_preview_lines:
                    preview.append(line.strip())
        info["n_lines"] = n_lines
        info["preview"] = preview
    except Exception as exc:
        info["error"] = repr(exc)

    return info


def compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def inspect_subject(subject_dir: Path) -> SubjectInventory:
    subject_id = subject_dir.name
    pkl_path = subject_dir / f"{subject_id}.pkl"
    e4_dir = subject_dir / f"{subject_id}_E4_Data"

    quest_path = subject_dir / f"{subject_id}_quest.csv"
    readme_path = subject_dir / f"{subject_id}_readme.txt"
    respiban_path = subject_dir / f"{subject_id}_respiban.txt"

    present = []
    missing = []
    for filename in EXPECTED_E4_FILES:
        if (e4_dir / filename).exists():
            present.append(filename)
        else:
            missing.append(filename)

    pkl_size_mb = round(pkl_path.stat().st_size / (1024 * 1024), 3) if pkl_path.exists() else 0.0

    pickle_top_keys = []
    pickle_signal_keys = []
    pickle_chest_keys = []
    pickle_wrist_keys = []
    label_length: Optional[int] = None
    label_counts: Dict[str, int] = {}
    chest_signal_shapes: Dict[str, str] = {}
    wrist_signal_shapes: Dict[str, str] = {}

    if pkl_path.exists():
        try:
            data = load_pickle(pkl_path)
            if isinstance(data, dict):
                pickle_top_keys = list(map(str, data.keys()))

                signal = data.get("signal", {})
                if isinstance(signal, dict):
                    pickle_signal_keys = list(map(str, signal.keys()))

                    chest = signal.get("chest", {})
                    wrist = signal.get("wrist", {})

                    if isinstance(chest, dict):
                        pickle_chest_keys = list(map(str, chest.keys()))
                        chest_signal_shapes = {str(k): safe_shape(v) for k, v in chest.items()}

                    if isinstance(wrist, dict):
                        pickle_wrist_keys = list(map(str, wrist.keys()))
                        wrist_signal_shapes = {str(k): safe_shape(v) for k, v in wrist.items()}

                labels = data.get("label", None)
                if labels is not None:
                    labels_arr = np.asarray(labels).reshape(-1)
                    label_length = int(labels_arr.shape[0])
                    counts = Counter(labels_arr.astype(int).tolist())
                    label_counts = {
                        f"{int(k)}:{WESAD_LABELS.get(int(k), 'unknown')}": int(v)
                        for k, v in sorted(counts.items())
                    }
        except Exception as exc:
            pickle_top_keys = [f"ERROR: {repr(exc)}"]

    e4_summary = {}
    if e4_dir.exists():
        for filename in EXPECTED_E4_FILES:
            e4_summary[filename] = summarize_e4_csv(e4_dir / filename)

    return SubjectInventory(
        subject_id=subject_id,
        subject_dir=str(subject_dir),
        pkl_path=str(pkl_path),
        e4_dir=str(e4_dir),
        has_pkl=pkl_path.exists(),
        has_e4_dir=e4_dir.exists(),
        e4_files_present=";".join(present),
        e4_files_missing=";".join(missing),
        pkl_size_mb=pkl_size_mb,
        pickle_top_keys=";".join(pickle_top_keys),
        pickle_signal_keys=";".join(pickle_signal_keys),
        pickle_chest_keys=";".join(pickle_chest_keys),
        pickle_wrist_keys=";".join(pickle_wrist_keys),
        label_length=label_length,
        label_counts=compact_json(label_counts),
        chest_signal_shapes=compact_json(chest_signal_shapes),
        wrist_signal_shapes=compact_json(wrist_signal_shapes),
        e4_csv_summary=compact_json(e4_summary),
        quest_path=str(quest_path) if quest_path.exists() else "",
        readme_path=str(readme_path) if readme_path.exists() else "",
        respiban_path=str(respiban_path) if respiban_path.exists() else "",
    )


def build_markdown_report(
    inventory_df: pd.DataFrame,
    wesad_root: Path,
    wesad_base: Path,
    output_dir: Path,
) -> str:
    lines = []
    lines.append("# WESAD inspection report")
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    lines.append(f"- Requested WESAD root: `{wesad_root}`")
    lines.append(f"- Detected WESAD base: `{wesad_base}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append("")
    lines.append("## Dataset inventory")
    lines.append("")
    lines.append(f"- Subjects found: **{len(inventory_df)}**")
    lines.append(f"- Subjects with pickle: **{int(inventory_df['has_pkl'].sum())}**")
    lines.append(f"- Subjects with E4 directory: **{int(inventory_df['has_e4_dir'].sum())}**")
    lines.append("")

    show_cols = [
        "subject_id",
        "has_pkl",
        "has_e4_dir",
        "pkl_size_mb",
        "pickle_top_keys",
        "pickle_chest_keys",
        "pickle_wrist_keys",
        "label_length",
        "e4_files_missing",
    ]
    lines.append("## Subjects")
    lines.append("")
    lines.append(inventory_df[show_cols].to_markdown(index=False))
    lines.append("")

    label_cols = ["subject_id", "label_counts"]
    lines.append("## Label counts by subject")
    lines.append("")
    lines.append(inventory_df[label_cols].to_markdown(index=False))
    lines.append("")

    shape_cols = ["subject_id", "chest_signal_shapes", "wrist_signal_shapes"]
    lines.append("## Pickle signal shapes")
    lines.append("")
    lines.append(inventory_df[shape_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Expected WESAD labels")
    lines.append("")
    lines.append("| label | meaning |")
    lines.append("|---:|---|")
    for label, name in WESAD_LABELS.items():
        lines.append(f"| {label} | {name} |")
    lines.append("")

    lines.append("## Recommended next step")
    lines.append("")
    lines.append("Create a windowed wearable feature dataset from pickle files:")
    lines.append("")
    lines.append("```text")
    lines.append("src/17_prepare_wesad_windowed_dataset.py")
    lines.append("```")
    lines.append("")
    lines.append("Recommended first target:")
    lines.append("")
    lines.append("```text")
    lines.append("stress vs non-stress")
    lines.append("label 2 -> stress")
    lines.append("label 1/3/4 -> non-stress or separate multiclass classes")
    lines.append("ignore label 0 transient / undefined")
    lines.append("```")
    lines.append("")
    lines.append("Recommended first windowing:")
    lines.append("")
    lines.append("```text")
    lines.append("window_size_sec = 60")
    lines.append("step_size_sec = 10")
    lines.append("signals = wrist EDA, BVP, TEMP, ACC")
    lines.append("validation = GroupKFold by subject_id")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect WESAD dataset structure.")
    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument(
        "--wesad-root",
        type=str,
        default="data/external/WESAD",
        help="Path to WESAD root. Accepts data/external/WESAD or data/external/WESAD/WESAD.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/wearable_pm_alignment",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.root).resolve()
    wesad_root = resolve_path(project_root, args.wesad_root)
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wesad_base = find_wesad_base(wesad_root)
    subject_dirs = sorted(
        [p for p in wesad_base.glob("S*") if p.is_dir()],
        key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 10**9,
    )

    records = []
    for subject_dir in subject_dirs:
        pkl = subject_dir / f"{subject_dir.name}.pkl"
        if not pkl.exists():
            continue
        print(f"Inspecting {subject_dir.name}: {pkl}")
        records.append(inspect_subject(subject_dir))

    if not records:
        raise RuntimeError(f"No subject pickle files found in: {wesad_base}")

    inventory_df = pd.DataFrame([asdict(r) for r in records])

    inventory_csv = output_dir / "wesad_inventory.csv"
    inventory_json = output_dir / "wesad_inventory.json"
    report_md = output_dir / "wesad_inspection_report.md"

    inventory_df.to_csv(inventory_csv, index=False, encoding="utf-8")
    inventory_json.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_md.write_text(
        build_markdown_report(inventory_df, wesad_root, wesad_base, output_dir),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print("WESAD inspection completed")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"WESAD root: {wesad_root}")
    print(f"Detected WESAD base: {wesad_base}")
    print(f"Subjects inspected: {len(inventory_df)}")
    print("")
    print("Saved:")
    print(f"  {inventory_csv}")
    print(f"  {inventory_json}")
    print(f"  {report_md}")
    print("")
    print("Subject summary:")
    show_cols = [
        "subject_id",
        "has_pkl",
        "has_e4_dir",
        "pkl_size_mb",
        "pickle_wrist_keys",
        "label_length",
        "e4_files_missing",
    ]
    print(inventory_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
