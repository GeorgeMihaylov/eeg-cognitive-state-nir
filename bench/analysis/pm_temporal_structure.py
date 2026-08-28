"""Training-free temporal-structure diagnostics for canonical PM targets."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from bench.analysis.pm_target_validity_cli import canonical_target_frame
from bench.tasks.target_registry import PM_METRICS
from bench.tasks.target_transforms import FoldLocalQuantileTargetTransform

LAGS = (1, 2, 3, 6, 12)


def _sort_columns(frame: pd.DataFrame) -> list[str]:
    for candidate in ("t_start", "t_center", "sample_id"):
        if candidate in frame.columns:
            return ["record_id", candidate]
    return ["record_id"]


def _paired_lag(values: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= lag:
        return np.array([], dtype=float), np.array([], dtype=float)
    a = values[:-lag]
    b = values[lag:]
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-free temporal-structure audit for canonical PM targets.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="reports/diagnostics/pm_temporal_structure_v1",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df, diag = canonical_target_frame(
        root / "data/processed/windowed_eeg_pm_dataset_w10.parquet",
        root / "data/interim/logical_recording_map.parquet",
        root / "data/interim/raw_eeg_window_index_w10_raw_v3.parquet",
    )
    if len(df) != 30958:
        raise RuntimeError(f"canonical cohort mismatch: {len(df)}")
    if "record_id" not in df.columns:
        raise RuntimeError("canonical cohort has no record_id")

    df = df.sort_values(_sort_columns(df), kind="stable").reset_index(drop=True)

    continuous_rows = []
    for metric in PM_METRICS:
        column = f"target_{metric}"
        all_values = pd.to_numeric(df[column], errors="coerce")
        global_iqr = float(all_values.quantile(.75) - all_values.quantile(.25))
        record_stats = []
        for record_id, group in df.groupby("record_id", sort=False):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            if np.isfinite(values).sum() < 3:
                continue
            row = {"record_id": str(record_id), "n": int(np.isfinite(values).sum())}
            for lag in LAGS:
                a, b = _paired_lag(values, lag)
                if len(a) >= 3 and np.nanstd(a) > 0 and np.nanstd(b) > 0:
                    corr = float(np.corrcoef(a, b)[0, 1])
                else:
                    corr = np.nan
                row[f"rho_lag_{lag}"] = corr
                row[f"mae_lag_{lag}"] = float(np.mean(np.abs(b - a))) if len(a) else np.nan
            rho1 = row.get("rho_lag_1", np.nan)
            if np.isfinite(rho1) and -0.999 < rho1 < 0.999:
                row["ar1_effective_fraction"] = float((1.0 - rho1) / (1.0 + rho1))
            else:
                row["ar1_effective_fraction"] = np.nan
            record_stats.append(row)
        rs = pd.DataFrame(record_stats)
        summary = {
            "metric": metric,
            "records": int(len(rs)),
            "global_iqr": global_iqr,
        }
        for lag in LAGS:
            vals = pd.to_numeric(rs.get(f"rho_lag_{lag}"), errors="coerce").dropna()
            maes = pd.to_numeric(rs.get(f"mae_lag_{lag}"), errors="coerce").dropna()
            summary[f"rho_lag_{lag}_median"] = float(vals.median()) if len(vals) else np.nan
            summary[f"rho_lag_{lag}_q25"] = float(vals.quantile(.25)) if len(vals) else np.nan
            summary[f"rho_lag_{lag}_q75"] = float(vals.quantile(.75)) if len(vals) else np.nan
            summary[f"mae_lag_{lag}_median"] = float(maes.median()) if len(maes) else np.nan
            summary[f"mae_lag_{lag}_over_iqr"] = float(maes.median() / global_iqr) if len(maes) and global_iqr > 0 else np.nan
        eff = pd.to_numeric(rs.get("ar1_effective_fraction"), errors="coerce").dropna()
        summary["ar1_effective_fraction_median"] = float(eff.median()) if len(eff) else np.nan
        summary["ar1_effective_fraction_q25"] = float(eff.quantile(.25)) if len(eff) else np.nan
        summary["ar1_effective_fraction_q75"] = float(eff.quantile(.75)) if len(eff) else np.nan
        continuous_rows.append(summary)

    continuous_summary = pd.DataFrame(continuous_rows)
    continuous_summary.to_csv(out / "continuous_temporal_summary.csv", index=False)

    groups = df["subject_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    q3_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(df)), groups=groups), start=1):
        train_frame = df.iloc[train_idx]
        test_frame = df.iloc[test_idx].copy()
        for metric in PM_METRICS:
            column = f"target_{metric}"
            y_train = pd.to_numeric(train_frame[column], errors="coerce").to_numpy(float)
            transform = FoldLocalQuantileTargetTransform(3, duplicates="drop").fit(y_train)
            test_frame["_label"] = transform.transform(
                pd.to_numeric(test_frame[column], errors="coerce").to_numpy(float)
            )
            for lag in (1, 3, 6):
                same_total = 0
                pair_total = 0
                abs_delta_sum = 0.0
                for _, record in test_frame.groupby("record_id", sort=False):
                    labels = pd.to_numeric(record["_label"], errors="coerce").to_numpy(float)
                    a, b = _paired_lag(labels, lag)
                    if not len(a):
                        continue
                    pair_total += len(a)
                    same_total += int(np.sum(a == b))
                    abs_delta_sum += float(np.sum(np.abs(b - a)))
                q3_rows.append({
                    "fold": fold,
                    "metric": metric,
                    "lag_windows": lag,
                    "lag_seconds": lag * 10,
                    "pairs": int(pair_total),
                    "same_class_fraction": float(same_total / pair_total) if pair_total else np.nan,
                    "ordinal_mae": float(abs_delta_sum / pair_total) if pair_total else np.nan,
                })

    q3_detail = pd.DataFrame(q3_rows)
    q3_summary = (
        q3_detail.groupby(["metric", "lag_windows", "lag_seconds"], sort=True)[
            ["same_class_fraction", "ordinal_mae"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    q3_detail.to_csv(out / "q3_temporal_by_fold.csv", index=False)
    q3_summary.to_csv(out / "q3_temporal_summary.csv", index=False)

    print("cohort", diag)
    print("\nCONTINUOUS TEMPORAL STRUCTURE\n", continuous_summary.to_string(index=False))
    print("\nQ3 TEMPORAL PERSISTENCE\n", q3_summary.to_string(index=False))


if __name__ == "__main__":
    main()
