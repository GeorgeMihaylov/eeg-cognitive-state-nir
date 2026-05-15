#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inspect COG-BCI dataset structure.

This script does NOT load EEG signals. It only scans files and checks the
BIDS-like folder structure.

Expected root:
    data/external/COG_BCI/raw/
        sub-01/ses-S1/behavioral/
        sub-01/ses-S1/chanlocs/
        sub-01/ses-S1/eeg/
        ...

Outputs:
    reports/cog_bci/cog_bci_file_inventory.csv
    reports/cog_bci/cog_bci_subject_session_summary.csv
    reports/cog_bci/cog_bci_task_file_summary.csv
    reports/cog_bci/cog_bci_missing_pairs.csv
    reports/cog_bci/cog_bci_nback_candidates.csv
    reports/cog_bci/cog_bci_matb_candidates.csv
    reports/cog_bci/cog_bci_inspection_report.md

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\27_inspect_cog_bci_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import pandas as pd


KNOWN_TASK_PATTERNS = [
    "Flanker",
    "MATBeasy",
    "MATBmed",
    "MATBdiff",
    "MATB-Easy",
    "MATB-Medium",
    "MATB-Difficult",
    "PVT",
    "Zero-Back",
    "One-Back",
    "Two-Back",
    "0-Back",
    "1-Back",
    "2-Back",
    "RS_Beg_EC",
    "RS_Beg_EO",
    "RS_End_EC",
    "RS_End_EO",
    "RS",
]


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def file_size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 3)
    except Exception:
        return 0.0


def detect_subject_session(path: Path) -> Tuple[str, str]:
    subject_id = ""
    session_id = ""
    for part in path.parts:
        if part.startswith("sub-"):
            subject_id = part
        if part.startswith("ses-"):
            session_id = part
    return subject_id, session_id


def detect_modality(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    suffix = path.suffix.lower()

    if "eeg" in parts or suffix in {".set", ".fdt"}:
        return "eeg"
    if "behavioral" in parts:
        return "behavioral"
    if "chanlocs" in parts:
        return "chanlocs"
    return "other"


def infer_task_name(path: Path) -> str:
    stem = path.stem
    low = stem.lower()

    for pattern in KNOWN_TASK_PATTERNS:
        if pattern.lower() in low:
            return pattern

    return stem


def build_file_inventory(raw_root: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue

        subject_id, session_id = detect_subject_session(path)
        rows.append(
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "modality": detect_modality(path),
                "task_name": infer_task_name(path),
                "suffix": path.suffix.lower(),
                "file_name": path.name,
                "relative_path": str(path.relative_to(raw_root)),
                "full_path": str(path),
                "size_mb": file_size_mb(path),
            }
        )

    return pd.DataFrame(rows)


def build_subject_session_summary(file_df: pd.DataFrame) -> pd.DataFrame:
    if file_df.empty:
        return pd.DataFrame()

    rows = []
    for (subject_id, session_id), g in file_df.groupby(["subject_id", "session_id"], dropna=False):
        eeg = g[g["modality"] == "eeg"]
        behavioral = g[g["modality"] == "behavioral"]
        chanlocs = g[g["modality"] == "chanlocs"]
        set_files = eeg[eeg["suffix"] == ".set"]
        fdt_files = eeg[eeg["suffix"] == ".fdt"]

        rows.append(
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "n_files": len(g),
                "n_set_files": len(set_files),
                "n_fdt_files": len(fdt_files),
                "n_behavioral_files": len(behavioral),
                "n_chanlocs_files": len(chanlocs),
                "eeg_size_mb": round(eeg["size_mb"].sum(), 3),
                "total_size_mb": round(g["size_mb"].sum(), 3),
                "tasks_set": ";".join(sorted(set(set_files["task_name"].astype(str)))),
                "tasks_behavioral": ";".join(sorted(set(behavioral["task_name"].astype(str)))),
                "has_eeg": len(eeg) > 0,
                "has_behavioral": len(behavioral) > 0,
                "has_chanlocs": len(chanlocs) > 0,
            }
        )

    return pd.DataFrame(rows).sort_values(["subject_id", "session_id"])


def build_task_file_summary(file_df: pd.DataFrame) -> pd.DataFrame:
    if file_df.empty:
        return pd.DataFrame()

    rows = []
    for (modality, task_name, suffix), g in file_df.groupby(["modality", "task_name", "suffix"], dropna=False):
        rows.append(
            {
                "modality": modality,
                "task_name": task_name,
                "suffix": suffix,
                "n_files": len(g),
                "n_subjects": g["subject_id"].nunique(),
                "n_subject_sessions": g[["subject_id", "session_id"]].drop_duplicates().shape[0],
                "total_size_mb": round(g["size_mb"].sum(), 3),
                "mean_size_mb": round(g["size_mb"].mean(), 3),
                "min_size_mb": round(g["size_mb"].min(), 3),
                "max_size_mb": round(g["size_mb"].max(), 3),
            }
        )

    return pd.DataFrame(rows).sort_values(["modality", "task_name", "suffix"])


def build_missing_pairs(file_df: pd.DataFrame) -> pd.DataFrame:
    if file_df.empty:
        return pd.DataFrame()

    eeg = file_df[file_df["modality"] == "eeg"]
    rows = []

    for (subject_id, session_id), g in eeg.groupby(["subject_id", "session_id"], dropna=False):
        set_files = g[g["suffix"] == ".set"].copy()
        fdt_files = g[g["suffix"] == ".fdt"].copy()

        set_stems = set(set_files["file_name"].str.replace(r"\.set$", "", regex=True))
        fdt_stems = set(fdt_files["file_name"].str.replace(r"\.fdt$", "", regex=True))

        for stem in sorted(set_stems - fdt_stems):
            row = set_files[set_files["file_name"] == f"{stem}.set"].iloc[0]
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "task_name": infer_task_name(Path(stem)),
                    "stem": stem,
                    "problem": "set_without_fdt",
                    "set_path": row["relative_path"],
                    "fdt_path": "",
                }
            )

        for stem in sorted(fdt_stems - set_stems):
            row = fdt_files[fdt_files["file_name"] == f"{stem}.fdt"].iloc[0]
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "task_name": infer_task_name(Path(stem)),
                    "stem": stem,
                    "problem": "fdt_without_set",
                    "set_path": "",
                    "fdt_path": row["relative_path"],
                }
            )

    return pd.DataFrame(rows)


def candidate_files(file_df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    if file_df.empty:
        return pd.DataFrame()
    df = file_df[
        (file_df["modality"] == "eeg") &
        (file_df["suffix"] == ".set") &
        (file_df["task_name"].str.lower().str.contains(keyword.lower(), na=False))
    ].copy()
    return df.sort_values(["subject_id", "session_id", "task_name"])


def build_report(
    raw_root: Path,
    file_df: pd.DataFrame,
    subject_session_df: pd.DataFrame,
    task_summary_df: pd.DataFrame,
    missing_pairs_df: pd.DataFrame,
    nback_df: pd.DataFrame,
    matb_df: pd.DataFrame,
) -> str:
    lines = []
    lines.append("# COG-BCI dataset inspection report")
    lines.append("")
    lines.append(f"- Raw root: `{raw_root}`")
    lines.append(f"- Files scanned: **{len(file_df)}**")
    lines.append("")

    if not file_df.empty:
        lines.append("## Global summary")
        lines.append("")
        lines.append(f"- Subjects detected: **{file_df['subject_id'].nunique()}**")
        lines.append(f"- Subject/session pairs: **{file_df[['subject_id', 'session_id']].drop_duplicates().shape[0]}**")
        lines.append(f"- Total scanned size: **{file_df['size_mb'].sum():.3f} MB**")
        lines.append("")
        lines.append("## Counts by modality")
        lines.append("")
        modality_counts = (
            file_df.groupby("modality")
            .agg(n_files=("file_name", "count"), total_size_mb=("size_mb", "sum"))
            .reset_index()
        )
        modality_counts["total_size_mb"] = modality_counts["total_size_mb"].round(3)
        lines.append(modality_counts.to_markdown(index=False))
        lines.append("")

    lines.append("## Subject-session summary")
    lines.append("")
    lines.append(subject_session_df.to_markdown(index=False) if not subject_session_df.empty else "No rows.")
    lines.append("")

    lines.append("## Task/file summary")
    lines.append("")
    lines.append(task_summary_df.to_markdown(index=False) if not task_summary_df.empty else "No rows.")
    lines.append("")

    lines.append("## Missing .set/.fdt pairs")
    lines.append("")
    lines.append(missing_pairs_df.to_markdown(index=False) if not missing_pairs_df.empty else "No missing pairs detected.")
    lines.append("")

    lines.append("## Candidate N-back files")
    lines.append("")
    show_cols = ["subject_id", "session_id", "task_name", "relative_path", "size_mb"]
    lines.append(nback_df[show_cols].to_markdown(index=False) if not nback_df.empty else "No N-back candidates detected.")
    lines.append("")

    lines.append("## Candidate MATB files")
    lines.append("")
    lines.append(matb_df[show_cols].to_markdown(index=False) if not matb_df.empty else "No MATB candidates detected.")
    lines.append("")

    lines.append("## Recommended next step")
    lines.append("")
    lines.append("Probe one `.set` file with MNE before building any dataset:")
    lines.append("")
    lines.append("```text")
    lines.append("src/28_probe_cog_bci_eeg_file.py")
    lines.append("```")
    lines.append("")
    lines.append("Recommended first baseline subset:")
    lines.append("")
    lines.append("```text")
    lines.append("subjects: sub-01, sub-02, sub-03")
    lines.append("session: ses-S1")
    lines.append("tasks: Zero-Back, One-Back, Two-Back")
    lines.append("target: workload_level = 0 / 1 / 2")
    lines.append("features: EEG bandpower")
    lines.append("validation: GroupKFold by subject_id")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect COG-BCI dataset structure.")
    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument("--raw-root", type=str, default="data/external/COG_BCI/raw")
    parser.add_argument("--output-dir", type=str, default="reports/cog_bci")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    raw_root = resolve_path(root, args.raw_root)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(f"COG-BCI raw root does not exist: {raw_root}")

    print("=" * 80)
    print("Inspect COG-BCI dataset")
    print("=" * 80)
    print(f"Raw root: {raw_root}")
    print(f"Output dir: {output_dir}")
    print("")

    file_df = build_file_inventory(raw_root)
    subject_session_df = build_subject_session_summary(file_df)
    task_summary_df = build_task_file_summary(file_df)
    missing_pairs_df = build_missing_pairs(file_df)
    nback_df = candidate_files(file_df, "back")
    matb_df = candidate_files(file_df, "matb")

    outputs = {
        "file_inventory": output_dir / "cog_bci_file_inventory.csv",
        "subject_session_summary": output_dir / "cog_bci_subject_session_summary.csv",
        "task_file_summary": output_dir / "cog_bci_task_file_summary.csv",
        "missing_pairs": output_dir / "cog_bci_missing_pairs.csv",
        "nback_candidates": output_dir / "cog_bci_nback_candidates.csv",
        "matb_candidates": output_dir / "cog_bci_matb_candidates.csv",
        "report": output_dir / "cog_bci_inspection_report.md",
        "source_files": output_dir / "source_files.json",
    }

    file_df.to_csv(outputs["file_inventory"], index=False, encoding="utf-8")
    subject_session_df.to_csv(outputs["subject_session_summary"], index=False, encoding="utf-8")
    task_summary_df.to_csv(outputs["task_file_summary"], index=False, encoding="utf-8")
    missing_pairs_df.to_csv(outputs["missing_pairs"], index=False, encoding="utf-8")
    nback_df.to_csv(outputs["nback_candidates"], index=False, encoding="utf-8")
    matb_df.to_csv(outputs["matb_candidates"], index=False, encoding="utf-8")

    outputs["report"].write_text(
        build_report(raw_root, file_df, subject_session_df, task_summary_df, missing_pairs_df, nback_df, matb_df),
        encoding="utf-8",
    )

    source_json = {
        "raw_root": str(raw_root),
        "outputs": {k: str(v) for k, v in outputs.items() if k != "source_files"},
    }
    outputs["source_files"].write_text(json.dumps(source_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("COG-BCI inspection completed")
    print("=" * 80)
    print(f"Files scanned: {len(file_df)}")
    if not file_df.empty:
        print(f"Subjects detected: {file_df['subject_id'].nunique()}")
        print(f"Subject/session pairs: {file_df[['subject_id', 'session_id']].drop_duplicates().shape[0]}")
        print(f"Total scanned size: {file_df['size_mb'].sum():.3f} MB")
    print("")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print("")
    print("Subject/session summary:")
    print(subject_session_df.to_string(index=False) if not subject_session_df.empty else "No rows.")
    print("")
    print("N-back candidates:")
    if not nback_df.empty:
        show_cols = ["subject_id", "session_id", "task_name", "relative_path", "size_mb"]
        print(nback_df[show_cols].to_string(index=False))
    else:
        print("No N-back candidates detected.")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
