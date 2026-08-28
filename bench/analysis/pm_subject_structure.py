"""Training-free subject-structure diagnostics for canonical PM targets."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from bench.analysis.pm_target_validity_cli import canonical_target_frame
from bench.tasks.target_registry import PM_METRICS
from bench.tasks.target_transforms import FoldLocalQuantileTargetTransform


def _entropy3(labels: np.ndarray) -> float:
    labels = labels[np.isfinite(labels)].astype(int)
    if not len(labels):
        return np.nan
    counts = np.bincount(labels, minlength=3).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(3.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-free subject-structure audit for Emotiv PM targets.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--audit-dir",
        default="reports/diagnostics/pm_target_validity_audit_full_v4",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/diagnostics/pm_subject_structure_v1",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(root / args.audit_dir / "pm_raw_record_audit.csv")
    transform_rows = []
    for (source, metric), group in raw.groupby(["source", "metric"], sort=True):
        for parameter in ["raw_to_scaled_slope", "raw_to_scaled_intercept"]:
            tmp = group[["subject_id", parameter]].copy()
            tmp[parameter] = pd.to_numeric(tmp[parameter], errors="coerce")
            tmp = tmp.dropna()
            if tmp.empty:
                continue
            subject = tmp.groupby("subject_id")[parameter].agg(["count", "mean", "std"]).reset_index()
            repeated = subject.loc[subject["count"] >= 2]
            within_var = float(np.nanmean(np.square(repeated["std"]))) if len(repeated) else np.nan
            between_var = float(np.nanvar(subject["mean"], ddof=1)) if len(subject) >= 2 else np.nan
            ratio = between_var / within_var if np.isfinite(within_var) and within_var > 0 else np.nan
            transform_rows.append({
                "source": source,
                "metric": metric,
                "parameter": parameter,
                "records": int(len(tmp)),
                "subjects": int(subject["subject_id"].nunique()),
                "subjects_with_repeats": int(len(repeated)),
                "subject_mean_median": float(subject["mean"].median()),
                "subject_mean_iqr": float(subject["mean"].quantile(.75) - subject["mean"].quantile(.25)),
                "between_subject_variance": between_var,
                "mean_within_subject_variance": within_var,
                "between_to_within_variance_ratio": ratio,
            })
    transform_summary = pd.DataFrame(transform_rows)
    transform_summary.to_csv(out / "raw_scaled_subject_variance.csv", index=False)

    df, diag = canonical_target_frame(
        root / "data/processed/windowed_eeg_pm_dataset_w10.parquet",
        root / "data/interim/logical_recording_map.parquet",
        root / "data/interim/raw_eeg_window_index_w10_raw_v3.parquet",
    )
    if len(df) != 30958:
        raise RuntimeError(f"canonical cohort mismatch: {len(df)}")

    groups = df["subject_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    subject_rows = []
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(df)), groups=groups), start=1):
        train_frame = df.iloc[train_idx]
        test_frame = df.iloc[test_idx]
        for metric in PM_METRICS:
            column = f"target_{metric}"
            y_train = pd.to_numeric(train_frame[column], errors="coerce").to_numpy(float)
            transform = FoldLocalQuantileTargetTransform(3, duplicates="drop").fit(y_train)
            manifest = transform.manifest()
            boundaries = manifest["boundaries"]
            test_values = pd.to_numeric(test_frame[column], errors="coerce").to_numpy(float)
            test_labels = transform.transform(test_values)
            temp = test_frame[["subject_id"]].copy()
            temp["value"] = test_values
            temp["label"] = test_labels
            per_subject = []
            for subject_id, sg in temp.groupby("subject_id", sort=True):
                labels = pd.to_numeric(sg["label"], errors="coerce").to_numpy(float)
                values = pd.to_numeric(sg["value"], errors="coerce").to_numpy(float)
                valid_labels = labels[np.isfinite(labels)].astype(int)
                if not len(valid_labels):
                    continue
                counts = np.bincount(valid_labels, minlength=3)
                props = counts / counts.sum()
                row = {
                    "fold": fold,
                    "metric": metric,
                    "subject_id": str(subject_id),
                    "n": int(counts.sum()),
                    "class0_fraction": float(props[0]),
                    "class1_fraction": float(props[1]),
                    "class2_fraction": float(props[2]),
                    "majority_fraction": float(props.max()),
                    "normalized_entropy": _entropy3(valid_labels.astype(float)),
                    "target_mean": float(np.nanmean(values)),
                    "target_std": float(np.nanstd(values)),
                    "boundary_1": float(boundaries[1]),
                    "boundary_2": float(boundaries[2]),
                }
                per_subject.append(row)
                subject_rows.append(row)
            if per_subject:
                ps = pd.DataFrame(per_subject)
                fold_rows.append({
                    "fold": fold,
                    "metric": metric,
                    "test_subjects": int(len(ps)),
                    "median_majority_fraction": float(ps["majority_fraction"].median()),
                    "mean_majority_fraction": float(ps["majority_fraction"].mean()),
                    "fraction_subjects_majority_ge_0_5": float((ps["majority_fraction"] >= .5).mean()),
                    "fraction_subjects_majority_ge_0_7": float((ps["majority_fraction"] >= .7).mean()),
                    "fraction_subjects_majority_ge_0_9": float((ps["majority_fraction"] >= .9).mean()),
                    "median_normalized_entropy": float(ps["normalized_entropy"].median()),
                    "mean_normalized_entropy": float(ps["normalized_entropy"].mean()),
                })

    subject_detail = pd.DataFrame(subject_rows)
    fold_summary = pd.DataFrame(fold_rows)
    metric_summary = (
        fold_summary.groupby("metric", sort=True)[
            [
                "median_majority_fraction",
                "mean_majority_fraction",
                "fraction_subjects_majority_ge_0_5",
                "fraction_subjects_majority_ge_0_7",
                "fraction_subjects_majority_ge_0_9",
                "median_normalized_entropy",
                "mean_normalized_entropy",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    subject_detail.to_csv(out / "q3_test_subject_detail.csv", index=False)
    fold_summary.to_csv(out / "q3_subject_structure_by_fold.csv", index=False)
    metric_summary.to_csv(out / "q3_subject_structure_summary.csv", index=False)

    print("cohort", diag)
    print("\nRAW->SCALED SUBJECT VARIANCE\n", transform_summary.to_string(index=False))
    print("\nQ3 SUBJECT STRUCTURE\n", metric_summary.to_string(index=False))


if __name__ == "__main__":
    main()
