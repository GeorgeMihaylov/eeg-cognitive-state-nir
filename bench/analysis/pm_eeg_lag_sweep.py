"""Cheap subject-disjoint EEG-to-PM lag screen for all seven PM targets.

The diagnostic intentionally uses the existing EEG-only engineered columns in the
canonical processed table.  It is a timing screen, not a replacement for the
canonical 371-feature/model-zoo experiments.

Lag convention:
    lag_windows < 0: EEG feature window PRECEDES the PM target window.
    lag_windows = 0: contemporaneous EEG and PM target windows.
    lag_windows > 0: EEG feature window FOLLOWS the PM target window and is
                     non-causal/diagnostic only.

All candidate lags use the same matched target-window cohort.  Pairing is always
within record_id and never crosses recording boundaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from bench.analysis.pm_target_validity_cli import canonical_target_frame
from bench.datasets.base_eeg_data_loader import resolve_feature_columns
from bench.tasks.target_registry import PM_METRICS
from bench.tasks.target_transforms import FoldLocalQuantileTargetTransform

DEFAULT_LAGS = (-6, -3, -1, 0, 1, 3, 6)
WINDOW_SECONDS = 10.0


def _relative_grid_coordinate(
    frame: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, str]:
    """Convert a 10-second time column to an exact per-record integer grid.

    The historical builder stores ``t_center`` as 0, 10, 20, ... and ``t_start``
    as -5, 5, 15, ... relative to each recording.  Dividing ``t_start`` by ten
    and rounding is therefore invalid because NumPy's ties-to-even rounding can
    collapse adjacent half-integers.  Subtracting each record's own origin before
    division preserves the true 10-second spacing and any real missing windows.
    """
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise RuntimeError(f"{column} contains non-finite values")

    records = frame["record_id"].astype(str)
    origin = values.groupby(records, sort=False).transform("min")
    units = (values - origin) / WINDOW_SECONDS
    rounded = np.rint(units.to_numpy(dtype=float))
    residual = np.abs(units.to_numpy(dtype=float) - rounded)
    max_residual = float(np.max(residual)) if len(residual) else 0.0
    if max_residual > 1e-6:
        raise RuntimeError(
            f"{column} is not on a stable 10-second grid; max residual={max_residual:.6g}"
        )
    return rounded.astype(np.int64), f"per-record {column}/10s grid"


def _window_coordinate(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return an integer per-record window coordinate suitable for exact lag joins."""
    if "window_id" in frame.columns:
        values = pd.to_numeric(frame["window_id"], errors="coerce")
        if values.notna().all() and np.allclose(values.to_numpy(), np.round(values.to_numpy())):
            return np.round(values.to_numpy()).astype(np.int64), "window_id"

    if "t_center" in frame.columns:
        return _relative_grid_coordinate(frame, "t_center")
    if "t_start" in frame.columns:
        return _relative_grid_coordinate(frame, "t_start")
    if "t_end" in frame.columns:
        return _relative_grid_coordinate(frame, "t_end")
    raise RuntimeError("Need window_id, t_center, t_start, or t_end for lag pairing")


def _lag_positions(
    frame: pd.DataFrame,
    lags: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], np.ndarray, str]:
    coord, source = _window_coordinate(frame)
    record = frame["record_id"].astype(str).to_numpy()
    keys = list(zip(record.tolist(), coord.tolist()))
    if len(set(keys)) != len(keys):
        duplicated = pd.DataFrame({"record_id": record, "coord": coord}).loc[
            lambda x: x.duplicated(["record_id", "coord"], keep=False)
        ]
        preview = duplicated.head(12).to_dict(orient="records")
        raise RuntimeError(
            f"Duplicate (record_id, window) keys using {source}; examples={preview}"
        )
    lookup = {key: idx for idx, key in enumerate(keys)}

    positions: dict[int, np.ndarray] = {}
    for lag in lags:
        shifted = np.fromiter(
            (lookup.get((rec, int(win + lag)), -1) for rec, win in zip(record, coord)),
            dtype=np.int64,
            count=len(frame),
        )
        positions[int(lag)] = shifted

    common = np.ones(len(frame), dtype=bool)
    for shifted in positions.values():
        common &= shifted >= 0
    return positions, common, source


def _participant_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    participant_ids: np.ndarray,
) -> tuple[float, float, int]:
    f1_values: list[float] = []
    ba_values: list[float] = []
    for participant in np.unique(participant_ids):
        mask = participant_ids == participant
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) == 0:
            continue
        f1_values.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
        ba_values.append(float(balanced_accuracy_score(yt, yp)))
    return (
        float(np.mean(f1_values)) if f1_values else np.nan,
        float(np.mean(ba_values)) if ba_values else np.nan,
        len(f1_values),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seven-PM EEG/target lag sweep")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="reports/diagnostics/pm_eeg_lag_sweep_v1",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAGS),
        help="Feature-window offsets relative to the target window, in 10 s windows.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lags = tuple(dict.fromkeys(int(value) for value in args.lags))
    if 0 not in lags:
        raise ValueError("Lag sweep must include 0 for the matched reference")

    frame, diag = canonical_target_frame(
        root / "data/processed/windowed_eeg_pm_dataset_w10.parquet",
        root / "data/interim/logical_recording_map.parquet",
        root / "data/interim/raw_eeg_window_index_w10_raw_v3.parquet",
    )
    if len(frame) != 30_958:
        raise RuntimeError(f"canonical cohort mismatch: {len(frame)}")

    feature_columns = resolve_feature_columns(frame.columns.tolist(), "eeg")
    if not feature_columns:
        raise RuntimeError("No EEG.* engineered feature columns found in canonical table")

    lag_pos, common_mask, coordinate_source = _lag_positions(frame, lags)
    common_indices = np.flatnonzero(common_mask)
    if len(common_indices) == 0:
        raise RuntimeError("No target windows have EEG feature windows at all requested lags")

    subjects = frame["subject_id"].astype(str).to_numpy()
    groups = subjects.copy()
    splitter = GroupKFold(n_splits=5)
    fold_assignment = np.zeros(len(frame), dtype=np.int64)
    fold_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(frame)), groups=groups), start=1
    ):
        fold_assignment[test_idx] = fold
        fold_splits.append((train_idx, test_idx))

    rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for fold, (outer_train, outer_test) in enumerate(fold_splits, start=1):
        train_member = np.zeros(len(frame), dtype=bool)
        test_member = np.zeros(len(frame), dtype=bool)
        train_member[outer_train] = True
        test_member[outer_test] = True
        matched_train_targets = np.flatnonzero(common_mask & train_member)
        matched_test_targets = np.flatnonzero(common_mask & test_member)

        transforms: dict[str, FoldLocalQuantileTargetTransform] = {}
        labels: dict[str, np.ndarray] = {}
        for metric in PM_METRICS:
            column = f"target_{metric}"
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            train_values = values[outer_train]
            transform = FoldLocalQuantileTargetTransform(3, duplicates="drop").fit(train_values)
            transforms[metric] = transform
            labels[metric] = transform.transform(values)

        for lag in lags:
            feature_train_rows = lag_pos[lag][matched_train_targets]
            feature_test_rows = lag_pos[lag][matched_test_targets]
            if np.any(feature_train_rows < 0) or np.any(feature_test_rows < 0):
                raise RuntimeError("Internal matched-cohort error")

            x_train_raw = frame.iloc[feature_train_rows][feature_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            x_test_raw = frame.iloc[feature_test_rows][feature_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw))
            x_test = scaler.transform(imputer.transform(x_test_raw))

            for metric in PM_METRICS:
                y_all = labels[metric]
                y_train_all = y_all[matched_train_targets]
                y_test_all = y_all[matched_test_targets]
                train_valid = np.isfinite(y_train_all)
                test_valid = np.isfinite(y_test_all)
                y_train = y_train_all[train_valid].astype(int)
                y_test = y_test_all[test_valid].astype(int)
                if len(np.unique(y_train)) != 3:
                    raise RuntimeError(
                        f"fold {fold} metric {metric}: Q3 train labels do not contain 3 classes"
                    )

                model = RidgeClassifier(alpha=float(args.alpha), class_weight="balanced")
                model.fit(x_train[train_valid], y_train)
                prediction = model.predict(x_test[test_valid]).astype(int)
                participant_ids = subjects[matched_test_targets][test_valid]
                macro_f1, balanced_accuracy, n_participants = _participant_macro(
                    y_test, prediction, participant_ids
                )
                rows.append(
                    {
                        "fold": fold,
                        "metric": metric,
                        "lag_windows": int(lag),
                        "lag_seconds": int(lag * 10),
                        "causal": bool(lag <= 0),
                        "train_rows": int(train_valid.sum()),
                        "test_rows": int(test_valid.sum()),
                        "test_participants": int(n_participants),
                        "participant_macro_f1": macro_f1,
                        "participant_balanced_accuracy": balanced_accuracy,
                    }
                )

        count_rows.append(
            {
                "fold": fold,
                "outer_train_rows": int(len(outer_train)),
                "outer_test_rows": int(len(outer_test)),
                "matched_train_rows": int(len(matched_train_targets)),
                "matched_test_rows": int(len(matched_test_targets)),
                "matched_test_participants": int(
                    len(np.unique(subjects[matched_test_targets]))
                ),
            }
        )

    detail = pd.DataFrame(rows)
    counts = pd.DataFrame(count_rows)
    summary = (
        detail.groupby(["metric", "lag_windows", "lag_seconds", "causal"], sort=True)[
            ["participant_macro_f1", "participant_balanced_accuracy"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    zero = detail.loc[detail["lag_windows"].eq(0), [
        "fold",
        "metric",
        "participant_macro_f1",
        "participant_balanced_accuracy",
    ]].rename(
        columns={
            "participant_macro_f1": "f1_lag0",
            "participant_balanced_accuracy": "ba_lag0",
        }
    )
    delta = detail.merge(zero, on=["fold", "metric"], how="left", validate="many_to_one")
    delta["delta_f1_vs_lag0"] = delta["participant_macro_f1"] - delta["f1_lag0"]
    delta["delta_ba_vs_lag0"] = delta["participant_balanced_accuracy"] - delta["ba_lag0"]
    delta_summary = (
        delta.groupby(["metric", "lag_windows", "lag_seconds", "causal"], sort=True)[
            ["delta_f1_vs_lag0", "delta_ba_vs_lag0"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    pooled = (
        delta.groupby(["lag_windows", "lag_seconds", "causal"], sort=True)[
            ["delta_f1_vs_lag0", "delta_ba_vs_lag0"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    detail.to_csv(out / "pm_eeg_lag_by_fold.csv", index=False)
    summary.to_csv(out / "pm_eeg_lag_summary.csv", index=False)
    delta.to_csv(out / "pm_eeg_lag_delta_by_fold.csv", index=False)
    delta_summary.to_csv(out / "pm_eeg_lag_delta_summary.csv", index=False)
    pooled.to_csv(out / "pm_eeg_lag_pooled_delta.csv", index=False)
    counts.to_csv(out / "pm_eeg_lag_matched_counts.csv", index=False)

    print("cohort", diag)
    print(
        "lag_contract",
        {
            "lags_windows": lags,
            "lags_seconds": tuple(lag * 10 for lag in lags),
            "negative_lag": "EEG precedes PM target",
            "positive_lag": "EEG follows PM target; non-causal diagnostic",
            "matched_target_rows": int(len(common_indices)),
            "canonical_rows": int(len(frame)),
            "eeg_feature_count": int(len(feature_columns)),
            "window_coordinate": coordinate_source,
            "model": "RidgeClassifier",
            "alpha": float(args.alpha),
        },
    )
    print("\nMATCHED COUNTS\n", counts.to_string(index=False))
    print("\nLAG SWEEP SUMMARY\n", summary.to_string(index=False))
    print("\nDELTA VS LAG0\n", delta_summary.to_string(index=False))
    print("\nPOOLED DELTA VS LAG0\n", pooled.to_string(index=False))


if __name__ == "__main__":
    main()
