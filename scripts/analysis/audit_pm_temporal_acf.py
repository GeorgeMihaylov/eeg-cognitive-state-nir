"""Feature-trajectory autocorrelation audit for Temporal Predictability Audit.

No predictive model is fitted. Pearson autocorrelation is computed only
between exact record-relative EEG feature windows; records are never joined.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.analysis.pm_eeg_lag_sweep import _lag_positions
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low


LAGS = (-1, -2, -3, -6, -12)
MIN_PAIRS_PER_RECORD = 20


def featurewise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise Pearson r for columns, allowing non-finite cells."""
    finite = np.isfinite(x) & np.isfinite(y)
    n = finite.sum(axis=0)

    x0 = np.where(finite, x, 0.0).astype(np.float64, copy=False)
    y0 = np.where(finite, y, 0.0).astype(np.float64, copy=False)

    safe_n = np.maximum(n, 1)
    mx = x0.sum(axis=0) / safe_n
    my = y0.sum(axis=0) / safe_n

    dx = np.where(finite, x - mx, 0.0)
    dy = np.where(finite, y - my, 0.0)

    numerator = np.sum(dx * dy, axis=0)
    denominator = np.sqrt(
        np.sum(dx * dx, axis=0) * np.sum(dy * dy, axis=0)
    )

    result = np.full(x.shape[1], np.nan, dtype=np.float64)
    valid = (
        (n >= MIN_PAIRS_PER_RECORD)
        & np.isfinite(denominator)
        & (denominator > 0)
    )
    result[valid] = numerator[valid] / denominator[valid]
    return result


def q(values: np.ndarray, value: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, value)) if len(values) else np.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/pm_diagnostics/"
            "pm_low_high_q3_extremes_confirmatory_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/diagnostics/"
            "pm_temporal_predictability_audit_v1"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    cache = args.feature_cache_dir.resolve()
    config_path = (
        args.config if args.config.is_absolute()
        else root / args.config
    )
    output = (
        args.output_dir if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    output.mkdir(parents=True, exist_ok=True)

    cfg = low.load_config(config_path)

    context = low.prepare_protocol(
        cfg,
        root=root,
        feature_cache_dir=cache,
        output_dir=(
            root / "benchmark_results" / "_temporal_acf_protocol"
        ),
    )

    full = context.full.reset_index(drop=True)
    matrix = np.asarray(context.matrix, dtype=np.float32)
    feature_names = list(context.feature_names)

    if matrix.shape != (len(full), len(feature_names)):
        raise RuntimeError("Feature matrix/index shape mismatch")

    if len(feature_names) != 371:
        raise RuntimeError(
            f"Expected 371 canonical features, got {len(feature_names)}"
        )

    families = np.asarray(
        [str(name).split("__", 1)[0] for name in feature_names],
        dtype=object,
    )
    family_names = sorted(set(families.tolist()))

    expected_families = {
        "spectral",
        "statistical",
        "entropy",
        "connectivity",
    }
    if set(family_names) != expected_families:
        raise RuntimeError(
            f"Unexpected feature families: {family_names}"
        )

    # Force real t_start coordinates and exact same-record pairing.
    pair_frame = full[["record_id", "t_start"]].copy()
    positions, _, coordinate_source = _lag_positions(
        pair_frame,
        LAGS,
    )

    if "t_start" not in coordinate_source:
        raise RuntimeError(
            f"Unexpected coordinate source: {coordinate_source}"
        )

    record_rows = []
    feature_rows = []
    feature_corr_by_lag: dict[int, list[np.ndarray]] = {
        lag: [] for lag in LAGS
    }

    for lag in LAGS:
        pos = positions[lag]
        valid = pos >= 0

        current_idx = np.flatnonzero(valid)
        previous_idx = pos[valid]

        current_meta = full.iloc[current_idx].reset_index(drop=True)
        previous_meta = full.iloc[previous_idx].reset_index(drop=True)

        if not current_meta["record_id"].astype(str).equals(
            previous_meta["record_id"].astype(str)
        ):
            raise RuntimeError(f"lag {lag}: crossed record_id")

        if not current_meta["subject_id"].astype(str).equals(
            previous_meta["subject_id"].astype(str)
        ):
            raise RuntimeError(f"lag {lag}: crossed subject_id")

        delta = (
            pd.to_numeric(
                previous_meta["t_start"], errors="coerce"
            ).to_numpy()
            - pd.to_numeric(
                current_meta["t_start"], errors="coerce"
            ).to_numpy()
        )

        expected = lag * 10.0
        if not np.allclose(delta, expected, atol=1e-6, rtol=0):
            raise RuntimeError(
                f"lag {lag}: timestamp mismatch"
            )

        current_records = (
            current_meta["record_id"].astype(str).to_numpy()
        )

        record_corrs_for_lag = []

        for record_id in np.unique(current_records):
            mask = current_records == record_id
            n_pairs = int(mask.sum())

            if n_pairs < MIN_PAIRS_PER_RECORD:
                continue

            x = matrix[current_idx[mask]]
            y = matrix[previous_idx[mask]]

            corr = featurewise_corr(x, y)

            if not np.isfinite(corr).any():
                continue

            record_corrs_for_lag.append(corr)

            subject_ids = current_meta.loc[
                mask, "subject_id"
            ].astype(str).unique()

            if len(subject_ids) != 1:
                raise RuntimeError(
                    f"record {record_id} spans subjects"
                )

            groups = [("all", np.ones(len(feature_names), dtype=bool))]
            groups.extend(
                (family, families == family)
                for family in family_names
            )

            for family, feature_mask in groups:
                values = corr[feature_mask]
                values = values[np.isfinite(values)]

                if not len(values):
                    continue

                record_rows.append(
                    {
                        "record_id": record_id,
                        "subject_id": subject_ids[0],
                        "lag_windows": abs(lag),
                        "lag_seconds": abs(lag) * 10,
                        "feature_family": family,
                        "n_pairs": n_pairs,
                        "n_valid_features": int(len(values)),
                        "median_r": float(np.median(values)),
                        "q25_r": q(values, 0.25),
                        "q75_r": q(values, 0.75),
                        "fraction_positive": float(
                            np.mean(values > 0)
                        ),
                    }
                )

        if not record_corrs_for_lag:
            raise RuntimeError(
                f"No eligible records for lag {lag}"
            )

        stacked = np.vstack(record_corrs_for_lag)
        feature_corr_by_lag[lag] = record_corrs_for_lag

        for feature_index, feature_name in enumerate(feature_names):
            values = stacked[:, feature_index]
            values = values[np.isfinite(values)]

            feature_rows.append(
                {
                    "feature": feature_name,
                    "feature_family": families[feature_index],
                    "lag_windows": abs(lag),
                    "lag_seconds": abs(lag) * 10,
                    "n_records": int(len(values)),
                    "median_r": (
                        float(np.median(values))
                        if len(values) else np.nan
                    ),
                    "q25_r": q(values, 0.25),
                    "q75_r": q(values, 0.75),
                    "fraction_positive": (
                        float(np.mean(values > 0))
                        if len(values) else np.nan
                    ),
                }
            )

    record_df = pd.DataFrame(record_rows)
    feature_df = pd.DataFrame(feature_rows)

    summary_rows = []

    for (lag_seconds, family), group in record_df.groupby(
        ["lag_seconds", "feature_family"],
        sort=True,
    ):
        values = group["median_r"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        summary_rows.append(
            {
                "lag_seconds": int(lag_seconds),
                "feature_family": family,
                "n_records": int(len(values)),
                "median_record_r": float(np.median(values)),
                "q25_record_r": q(values, 0.25),
                "q75_record_r": q(values, 0.75),
                "fraction_records_positive": float(
                    np.mean(values > 0)
                ),
                "median_pairs_per_record": float(
                    group["n_pairs"].median()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["feature_family", "lag_seconds"]
    )

    record_path = output / "acf_by_record_family.csv"
    feature_path = output / "acf_by_feature.csv"
    summary_path = output / "acf_summary.csv"

    record_df.to_csv(record_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=== TEMPORAL FEATURE ACF AUDIT ===")
    print("coordinate_source:", coordinate_source)
    print("feature matrix:", matrix.shape)
    print("eligible record minimum pairs:", MIN_PAIRS_PER_RECORD)
    print()

    pivot = summary_df.pivot(
        index="feature_family",
        columns="lag_seconds",
        values="median_record_r",
    )

    print("=== MEDIAN RECORD-LEVEL AUTOCORRELATION ===")
    print(pivot.round(4).to_string())
    print()

    print("=== IQR BY FAMILY/LAG ===")
    print(
        summary_df[
            [
                "feature_family",
                "lag_seconds",
                "n_records",
                "median_record_r",
                "q25_record_r",
                "q75_record_r",
            ]
        ].round(4).to_string(index=False)
    )
    print()

    print("Outputs:")
    print(summary_path)
    print(record_path)
    print(feature_path)
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
