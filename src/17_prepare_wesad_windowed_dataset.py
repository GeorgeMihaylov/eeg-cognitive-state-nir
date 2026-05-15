#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare WESAD windowed wearable feature dataset.

Input:
    data/external/WESAD/WESAD/S2/S2.pkl
    ...
    data/external/WESAD/WESAD/S17/S17.pkl

Output:
    data/processed/wesad_windowed_stress_dataset.parquet
    data/processed/wesad_windowed_stress_dataset.csv, optional
    reports/wearable_pm_alignment/wesad_windowed_stress_dataset_report.md

First target:
    binary stress classification:
        label 2 -> stress = 1
        labels 1, 3, 4 -> stress = 0
        labels 0, 5, 6, 7 -> ignored

Default windowing:
    window_size_sec = 60
    step_size_sec = 10

Signal sampling rates in WESAD pickle:
    label/chest timeline: 700 Hz
    wrist ACC: 32 Hz
    wrist BVP: 64 Hz
    wrist EDA: 4 Hz
    wrist TEMP: 4 Hz

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\17_prepare_wesad_windowed_dataset.py

Quick test:
    D:\miniconda3\envs\eeg_nir\python.exe src\17_prepare_wesad_windowed_dataset.py `
      --max-subjects 3 `
      --output-name wesad_windowed_stress_dataset_test
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LABEL_FS = 700.0

WESAD_LABELS = {
    0: "undefined_or_transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "unused_or_other",
    6: "unused_or_other",
    7: "unused_or_other",
}

WESAD_BINARY_TARGET = {
    1: 0,
    2: 1,
    3: 0,
    4: 0,
}

WESAD_MULTICLASS_TARGET = {
    1: 0,  # baseline
    2: 1,  # stress
    3: 2,  # amusement
    4: 3,  # meditation
}

SIGNAL_SPECS = {
    "ACC": {"fs": 32.0, "columns": ["x", "y", "z"]},
    "BVP": {"fs": 64.0, "columns": ["bvp"]},
    "EDA": {"fs": 4.0, "columns": ["eda"]},
    "TEMP": {"fs": 4.0, "columns": ["temp"]},
}


@dataclass
class SubjectWindowSummary:
    subject_id: str
    pkl_path: str
    duration_sec: float
    total_candidate_windows: int
    kept_windows: int
    skipped_short_signal: int
    skipped_low_valid_label_fraction: int
    skipped_ambiguous_label: int
    binary_label_counts: str
    multiclass_label_counts: str
    original_label_counts: str


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def find_wesad_base(wesad_root: Path) -> Path:
    if not wesad_root.exists():
        raise FileNotFoundError(f"WESAD root does not exist: {wesad_root}")

    if sorted(wesad_root.glob("S*/S*.pkl")):
        return wesad_root

    nested = wesad_root / "WESAD"
    if nested.exists() and sorted(nested.glob("S*/S*.pkl")):
        return nested

    recursive = sorted(wesad_root.rglob("S*/S*.pkl"))
    if recursive:
        return recursive[0].parent.parent

    raise FileNotFoundError(f"Could not find WESAD subject pickle files under: {wesad_root}")


def load_pickle(path: Path) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with path.open("rb") as f:
            try:
                return pickle.load(f, encoding="latin1")
            except TypeError:
                f.seek(0)
                return pickle.load(f)


def safe_array(x: Any, dtype: str = "float32") -> np.ndarray:
    arr = np.asarray(x)
    if dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def subject_sort_key(path: Path) -> int:
    name = path.parent.name if path.is_file() else path.name
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return 10**9


def summarize_numeric_vector(values: np.ndarray, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        stats = [
            "mean", "std", "min", "max", "median", "q25", "q75", "iqr",
            "range", "abs_mean", "rms", "energy", "slope", "diff_mean",
            "diff_std", "diff_abs_mean"
        ]
        for stat in stats:
            out[f"{prefix}_{stat}"] = np.nan
        return out

    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    diff = np.diff(arr) if arr.size > 1 else np.array([], dtype=np.float64)

    out[f"{prefix}_mean"] = float(np.mean(arr))
    out[f"{prefix}_std"] = float(np.std(arr))
    out[f"{prefix}_min"] = float(np.min(arr))
    out[f"{prefix}_max"] = float(np.max(arr))
    out[f"{prefix}_median"] = float(np.median(arr))
    out[f"{prefix}_q25"] = q25
    out[f"{prefix}_q75"] = q75
    out[f"{prefix}_iqr"] = q75 - q25
    out[f"{prefix}_range"] = float(np.max(arr) - np.min(arr))
    out[f"{prefix}_abs_mean"] = float(np.mean(np.abs(arr)))
    out[f"{prefix}_rms"] = float(np.sqrt(np.mean(arr ** 2)))
    out[f"{prefix}_energy"] = float(np.mean(arr ** 2))

    if arr.size > 2:
        x = np.arange(arr.size, dtype=np.float64)
        x = x - x.mean()
        denom = float(np.sum(x ** 2))
        if denom > 0:
            out[f"{prefix}_slope"] = float(np.sum(x * (arr - arr.mean())) / denom)
        else:
            out[f"{prefix}_slope"] = 0.0
    else:
        out[f"{prefix}_slope"] = 0.0

    if diff.size > 0:
        out[f"{prefix}_diff_mean"] = float(np.mean(diff))
        out[f"{prefix}_diff_std"] = float(np.std(diff))
        out[f"{prefix}_diff_abs_mean"] = float(np.mean(np.abs(diff)))
    else:
        out[f"{prefix}_diff_mean"] = 0.0
        out[f"{prefix}_diff_std"] = 0.0
        out[f"{prefix}_diff_abs_mean"] = 0.0

    return out


def count_simple_peaks(values: np.ndarray) -> Tuple[int, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0, 0.0

    threshold = float(np.mean(arr) + np.std(arr))
    mid = arr[1:-1]
    peaks = (mid > arr[:-2]) & (mid > arr[2:]) & (mid > threshold)
    n_peaks = int(np.sum(peaks))
    rate = float(n_peaks / arr.size)
    return n_peaks, rate


def extract_signal_features(
    signal_name: str,
    signal_arr: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> Tuple[Dict[str, float], bool]:
    spec = SIGNAL_SPECS[signal_name]
    fs = float(spec["fs"])
    cols = spec["columns"]

    start_idx = int(round(start_sec * fs))
    end_idx = int(round(end_sec * fs))

    if signal_arr.ndim == 1:
        signal_arr = signal_arr.reshape(-1, 1)

    if start_idx < 0 or end_idx > signal_arr.shape[0] or end_idx <= start_idx:
        return {}, False

    window = signal_arr[start_idx:end_idx]
    if window.shape[0] < max(3, int(0.5 * fs)):
        return {}, False

    out: Dict[str, float] = {}

    for i, col in enumerate(cols):
        if i >= window.shape[1]:
            continue
        prefix = f"{signal_name.lower()}_{col}"
        out.update(summarize_numeric_vector(window[:, i], prefix))

        if signal_name in {"EDA", "BVP"}:
            n_peaks, peak_rate = count_simple_peaks(window[:, i])
            out[f"{prefix}_simple_peak_count"] = float(n_peaks)
            out[f"{prefix}_simple_peak_rate"] = float(peak_rate)

    if signal_name == "ACC" and window.shape[1] >= 3:
        mag = np.sqrt(np.sum(window[:, :3].astype(np.float64) ** 2, axis=1))
        out.update(summarize_numeric_vector(mag, "acc_mag"))

    return out, True


def get_window_label(
    labels: np.ndarray,
    start_sec: float,
    end_sec: float,
    min_valid_label_fraction: float,
    min_majority_fraction: float,
) -> Tuple[Optional[int], Optional[int], Dict[str, float], str]:
    start_idx = int(round(start_sec * LABEL_FS))
    end_idx = int(round(end_sec * LABEL_FS))

    if start_idx < 0 or end_idx > labels.shape[0] or end_idx <= start_idx:
        return None, None, {}, "out_of_range"

    arr = labels[start_idx:end_idx].astype(int)
    counts = Counter(arr.tolist())

    valid_counts = {k: v for k, v in counts.items() if k in WESAD_BINARY_TARGET}
    valid_total = sum(valid_counts.values())
    total = int(arr.size)

    if total == 0:
        return None, None, {}, "empty"

    valid_fraction = valid_total / total
    if valid_fraction < min_valid_label_fraction:
        return None, None, {"valid_fraction": valid_fraction}, "low_valid_label_fraction"

    majority_label, majority_count = max(valid_counts.items(), key=lambda kv: kv[1])
    majority_fraction = majority_count / valid_total if valid_total else 0.0

    if majority_fraction < min_majority_fraction:
        return None, None, {
            "valid_fraction": valid_fraction,
            "majority_fraction": majority_fraction,
        }, "ambiguous_label"

    binary = WESAD_BINARY_TARGET.get(majority_label)
    multiclass = WESAD_MULTICLASS_TARGET.get(majority_label)

    meta = {
        "valid_fraction": float(valid_fraction),
        "majority_fraction": float(majority_fraction),
        "majority_original_label": float(majority_label),
        "majority_original_label_count": float(majority_count),
        "n_label_samples": float(total),
    }
    for label_id, count in sorted(counts.items()):
        label_name = WESAD_LABELS.get(int(label_id), "unknown")
        meta[f"label_count_{int(label_id)}_{label_name}"] = float(count)

    return binary, multiclass, meta, "ok"


def process_subject(
    pkl_path: Path,
    window_size_sec: float,
    step_size_sec: float,
    min_valid_label_fraction: float,
    min_majority_fraction: float,
) -> Tuple[List[Dict[str, float]], SubjectWindowSummary]:
    subject_id = pkl_path.parent.name
    data = load_pickle(pkl_path)

    labels = np.asarray(data["label"]).reshape(-1).astype(np.int16)
    signal = data.get("signal", {})
    wrist = signal.get("wrist", {})

    wrist_signals = {}
    for signal_name in SIGNAL_SPECS.keys():
        if signal_name not in wrist:
            raise KeyError(f"{subject_id}: wrist signal {signal_name} not found in pickle.")
        wrist_signals[signal_name] = safe_array(wrist[signal_name], dtype="float32")

    duration_sec = labels.shape[0] / LABEL_FS
    max_start = duration_sec - window_size_sec
    candidate_starts = np.arange(0.0, max_start + 1e-9, step_size_sec)

    rows: List[Dict[str, float]] = []
    skipped_short_signal = 0
    skipped_low_valid_label_fraction = 0
    skipped_ambiguous_label = 0

    binary_counter: Counter = Counter()
    multiclass_counter: Counter = Counter()

    original_label_counts_raw = Counter(labels.astype(int).tolist())
    original_label_counts = {
        f"{int(k)}:{WESAD_LABELS.get(int(k), 'unknown')}": int(v)
        for k, v in sorted(original_label_counts_raw.items())
    }

    for window_id, start_sec in enumerate(candidate_starts):
        end_sec = float(start_sec + window_size_sec)

        binary_label, multiclass_label, label_meta, label_status = get_window_label(
            labels=labels,
            start_sec=float(start_sec),
            end_sec=end_sec,
            min_valid_label_fraction=min_valid_label_fraction,
            min_majority_fraction=min_majority_fraction,
        )

        if label_status == "low_valid_label_fraction":
            skipped_low_valid_label_fraction += 1
            continue
        if label_status == "ambiguous_label":
            skipped_ambiguous_label += 1
            continue
        if label_status != "ok" or binary_label is None or multiclass_label is None:
            skipped_low_valid_label_fraction += 1
            continue

        features: Dict[str, float] = {}
        ok_all = True
        for signal_name, arr in wrist_signals.items():
            signal_features, ok = extract_signal_features(
                signal_name=signal_name,
                signal_arr=arr,
                start_sec=float(start_sec),
                end_sec=end_sec,
            )
            if not ok:
                ok_all = False
                break
            features.update(signal_features)

        if not ok_all:
            skipped_short_signal += 1
            continue

        row: Dict[str, float] = {
            "subject_id": subject_id,
            "subject_num": int(subject_id[1:]) if subject_id.startswith("S") and subject_id[1:].isdigit() else np.nan,
            "window_id": int(window_id),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "center_sec": float((start_sec + end_sec) / 2.0),
            "window_size_sec": float(window_size_sec),
            "step_size_sec": float(step_size_sec),
            "stress_binary": int(binary_label),
            "wesad_multiclass": int(multiclass_label),
            "wesad_label": int(label_meta.get("majority_original_label", -1)),
            "wesad_label_name": WESAD_LABELS.get(int(label_meta.get("majority_original_label", -1)), "unknown"),
        }
        row.update(label_meta)
        row.update(features)

        rows.append(row)
        binary_counter[int(binary_label)] += 1
        multiclass_counter[int(multiclass_label)] += 1

    summary = SubjectWindowSummary(
        subject_id=subject_id,
        pkl_path=str(pkl_path),
        duration_sec=float(duration_sec),
        total_candidate_windows=int(len(candidate_starts)),
        kept_windows=int(len(rows)),
        skipped_short_signal=int(skipped_short_signal),
        skipped_low_valid_label_fraction=int(skipped_low_valid_label_fraction),
        skipped_ambiguous_label=int(skipped_ambiguous_label),
        binary_label_counts=json.dumps(dict(binary_counter), sort_keys=True),
        multiclass_label_counts=json.dumps(dict(multiclass_counter), sort_keys=True),
        original_label_counts=json.dumps(original_label_counts, sort_keys=True),
    )

    return rows, summary


def build_report(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dataset_path: Path,
) -> str:
    lines = []
    lines.append("# WESAD windowed stress dataset report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(vars(args), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Output")
    lines.append("")
    lines.append(f"- Dataset: `{output_dataset_path}`")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append("")
    lines.append(f"- Rows: **{len(df)}**")
    lines.append(f"- Columns: **{df.shape[1]}**")
    lines.append(f"- Subjects: **{df['subject_id'].nunique() if len(df) else 0}**")
    feature_cols = [
        c for c in df.columns
        if c not in {
            "subject_id", "subject_num", "window_id", "start_sec", "end_sec", "center_sec",
            "window_size_sec", "step_size_sec", "stress_binary", "wesad_multiclass",
            "wesad_label", "wesad_label_name"
        } and not c.startswith("label_count_") and c not in {
            "valid_fraction", "majority_fraction", "majority_original_label",
            "majority_original_label_count", "n_label_samples"
        }
    ]
    lines.append(f"- Feature columns: **{len(feature_cols)}**")
    lines.append("")

    if len(df):
        lines.append("## Binary target distribution")
        lines.append("")
        target_counts = df["stress_binary"].value_counts().sort_index().reset_index()
        target_counts.columns = ["stress_binary", "n_windows"]
        target_counts["class_name"] = target_counts["stress_binary"].map({0: "non_stress", 1: "stress"})
        lines.append(target_counts[["stress_binary", "class_name", "n_windows"]].to_markdown(index=False))
        lines.append("")

        lines.append("## Original WESAD majority label distribution")
        lines.append("")
        label_counts = df["wesad_label_name"].value_counts().reset_index()
        label_counts.columns = ["wesad_label_name", "n_windows"]
        lines.append(label_counts.to_markdown(index=False))
        lines.append("")

    lines.append("## Subject summary")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Recommended next step")
    lines.append("")
    lines.append("Train first wearable stress baseline:")
    lines.append("")
    lines.append("```text")
    lines.append("src/18_train_wesad_stress_baseline.py")
    lines.append("```")
    lines.append("")
    lines.append("Recommended validation:")
    lines.append("")
    lines.append("```text")
    lines.append("GroupKFold by subject_id")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WESAD windowed stress dataset.")

    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument("--wesad-root", type=str, default="data/external/WESAD", help="WESAD root.")
    parser.add_argument("--output-name", type=str, default="wesad_windowed_stress_dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--report-dir", type=str, default="reports/wearable_pm_alignment")

    parser.add_argument("--window-size-sec", type=float, default=60.0)
    parser.add_argument("--step-size-sec", type=float, default=10.0)
    parser.add_argument("--min-valid-label-fraction", type=float, default=0.80)
    parser.add_argument("--min-majority-fraction", type=float, default=0.80)

    parser.add_argument("--max-subjects", type=int, default=0, help="0 means all subjects.")
    parser.add_argument("--save-csv", action="store_true", help="Also save CSV copy.")
    parser.add_argument("--no-parquet", action="store_true", help="Skip parquet output.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.root).resolve()
    wesad_root = resolve_path(project_root, args.wesad_root)
    output_dir = resolve_path(project_root, args.output_dir)
    report_dir = resolve_path(project_root, args.report_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    wesad_base = find_wesad_base(wesad_root)
    pkl_paths = sorted(wesad_base.glob("S*/S*.pkl"), key=lambda p: int(p.parent.name[1:]))

    if args.max_subjects and args.max_subjects > 0:
        pkl_paths = pkl_paths[: args.max_subjects]

    if not pkl_paths:
        raise RuntimeError(f"No WESAD pickle files found under: {wesad_base}")

    print("=" * 80)
    print("Prepare WESAD windowed stress dataset")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"WESAD base: {wesad_base}")
    print(f"Subjects: {len(pkl_paths)}")
    print(f"Window size: {args.window_size_sec} sec")
    print(f"Step size: {args.step_size_sec} sec")
    print("")

    all_rows: List[Dict[str, float]] = []
    summaries: List[SubjectWindowSummary] = []

    for pkl_path in pkl_paths:
        subject_id = pkl_path.parent.name
        print(f"Processing {subject_id}: {pkl_path}")
        rows, summary = process_subject(
            pkl_path=pkl_path,
            window_size_sec=args.window_size_sec,
            step_size_sec=args.step_size_sec,
            min_valid_label_fraction=args.min_valid_label_fraction,
            min_majority_fraction=args.min_majority_fraction,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"  candidate={summary.total_candidate_windows} "
            f"kept={summary.kept_windows} "
            f"binary_counts={summary.binary_label_counts}"
        )

    if not all_rows:
        raise RuntimeError("No windows were created. Check label filters and WESAD structure.")

    df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame([asdict(s) for s in summaries])

    # Sort metadata first, then feature columns.
    meta_cols = [
        "subject_id",
        "subject_num",
        "window_id",
        "start_sec",
        "end_sec",
        "center_sec",
        "window_size_sec",
        "step_size_sec",
        "stress_binary",
        "wesad_multiclass",
        "wesad_label",
        "wesad_label_name",
        "valid_fraction",
        "majority_fraction",
        "majority_original_label",
        "majority_original_label_count",
        "n_label_samples",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + sorted(other_cols)]

    output_parquet = output_dir / f"{args.output_name}.parquet"
    output_csv = output_dir / f"{args.output_name}.csv"
    summary_csv = report_dir / f"{args.output_name}_subject_summary.csv"
    report_md = report_dir / f"{args.output_name}_report.md"

    if not args.no_parquet:
        df.to_parquet(output_parquet, index=False)

    if args.save_csv:
        df.to_csv(output_csv, index=False, encoding="utf-8")

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    report_md.write_text(
        build_report(df, summary_df, args, output_parquet),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print("Saved WESAD windowed dataset")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print("Binary target distribution:")
    print(df["stress_binary"].value_counts().sort_index().to_string())
    print("")
    if not args.no_parquet:
        print(f"Dataset parquet: {output_parquet}")
    if args.save_csv:
        print(f"Dataset CSV: {output_csv}")
    print(f"Subject summary: {summary_csv}")
    print(f"Report: {report_md}")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
