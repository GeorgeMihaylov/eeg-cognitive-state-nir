"""Mutual-information temporal dependence audit with within-record surrogates.

Feature trajectories are discretized independently within each EEG record.
Observed delayed normalized mutual information is compared with row-permuted
within-record surrogate pairings. No predictive model is fitted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.analysis.pm_eeg_lag_sweep import _lag_positions
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low


LAGS = (-1, -2, -3, -6, -12)
N_BINS = 5
MIN_PAIRS_PER_RECORD = 50
N_SURROGATES = 10
SEED = 42


def quantile_discretize(
    matrix: np.ndarray,
    full: pd.DataFrame,
) -> np.ndarray:
    """Discretize every feature independently within each record."""
    result = np.full(matrix.shape, -1, dtype=np.int8)

    for _, indices in full.groupby("record_id", sort=False).groups.items():
        indices = np.asarray(list(indices), dtype=np.int64)
        block = matrix[indices]

        for feature_index in range(block.shape[1]):
            values = block[:, feature_index]
            finite = np.isfinite(values)
            n = int(finite.sum())

            if n < 2:
                continue

            ranks = rankdata(
                values[finite],
                method="average",
            )

            bins = np.floor(
                (ranks - 1.0) * N_BINS / n
            ).astype(np.int16)

            bins = np.clip(
                bins,
                0,
                N_BINS - 1,
            )

            result[
                indices[finite],
                feature_index,
            ] = bins.astype(np.int8)

    return result


def permute_valid_pairs(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute y independently per feature over the observed valid-pair set.

    This preserves the exact number of valid x/y pairs and both discretized
    marginal distributions used by the observed MI estimate.
    """
    if x.shape != y.shape:
        raise ValueError("Surrogate inputs have different shapes")

    result = np.full_like(y, -1)
    valid = (x >= 0) & (y >= 0)

    for feature_index in range(y.shape[1]):
        indices = np.flatnonzero(valid[:, feature_index])

        if not len(indices):
            continue

        shuffled_indices = rng.permutation(indices)

        result[
            indices,
            feature_index,
        ] = y[
            shuffled_indices,
            feature_index,
        ]

    return result


def columnwise_nanmedian(
    values: np.ndarray,
) -> np.ndarray:
    """Column-wise NaN median without warnings for all-NaN columns."""
    result = np.full(
        values.shape[1],
        np.nan,
        dtype=np.float64,
    )

    usable = np.isfinite(values).any(axis=0)

    if usable.any():
        result[usable] = np.nanmedian(
            values[:, usable],
            axis=0,
        )

    return result


def normalized_mi(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Symmetric MI/sqrt(Hx*Hy) for all feature columns."""
    if x.shape != y.shape:
        raise ValueError("MI inputs have different shapes")

    n_features = x.shape[1]
    invalid_code = N_BINS * N_BINS
    stride = invalid_code + 1

    valid = (x >= 0) & (y >= 0)

    pair_code = np.where(
        valid,
        x.astype(np.int16) * N_BINS + y.astype(np.int16),
        invalid_code,
    ).astype(np.int64)

    offsets = (
        np.arange(n_features, dtype=np.int64)[None, :]
        * stride
    )

    encoded = pair_code + offsets

    counts = np.bincount(
        encoded.ravel(),
        minlength=n_features * stride,
    ).reshape(n_features, stride)

    joint = counts[:, :invalid_code].reshape(
        n_features,
        N_BINS,
        N_BINS,
    ).astype(np.float64)

    n = joint.sum(axis=(1, 2))

    pxy = np.divide(
        joint,
        n[:, None, None],
        out=np.zeros_like(joint),
        where=n[:, None, None] > 0,
    )

    px = pxy.sum(axis=2)
    py = pxy.sum(axis=1)

    expected = px[:, :, None] * py[:, None, :]

    mi_terms = np.zeros_like(pxy)
    positive = (pxy > 0) & (expected > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        mi_terms[positive] = (
            pxy[positive]
            * np.log(
                pxy[positive]
                / expected[positive]
            )
        )

    mi = mi_terms.sum(axis=(1, 2))

    with np.errstate(divide="ignore", invalid="ignore"):
        hx_terms = np.where(
            px > 0,
            px * np.log(px),
            0.0,
        )
        hy_terms = np.where(
            py > 0,
            py * np.log(py),
            0.0,
        )

    hx = -hx_terms.sum(axis=1)
    hy = -hy_terms.sum(axis=1)

    denominator = np.sqrt(hx * hy)

    result = np.full(n_features, np.nan, dtype=np.float64)

    usable = (
        (n >= MIN_PAIRS_PER_RECORD)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    result[usable] = mi[usable] / denominator[usable]

    return result


def quantile(values: np.ndarray, p: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not len(values):
        return np.nan

    return float(np.quantile(values, p))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        required=True,
    )
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
        args.config
        if args.config.is_absolute()
        else root / args.config
    )

    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = low.load_config(config_path)

    context = low.prepare_protocol(
        cfg,
        root=root,
        feature_cache_dir=cache,
        output_dir=(
            root
            / "benchmark_results"
            / "_temporal_mi_protocol"
        ),
    )

    full = context.full.reset_index(drop=True)
    matrix = np.asarray(
        context.matrix,
        dtype=np.float32,
    )
    feature_names = list(context.feature_names)

    if matrix.shape != (
        len(full),
        len(feature_names),
    ):
        raise RuntimeError(
            "Feature matrix/index shape mismatch"
        )

    if len(feature_names) != 371:
        raise RuntimeError(
            f"Expected 371 features, got {len(feature_names)}"
        )

    families = np.asarray(
        [
            str(name).split("__", 1)[0]
            for name in feature_names
        ],
        dtype=object,
    )

    family_names = sorted(
        set(families.tolist())
    )

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

    print("Discretizing features within records...")

    discrete = quantile_discretize(
        matrix,
        full,
    )

    pair_frame = full[
        ["record_id", "t_start"]
    ].copy()

    positions, _, coordinate_source = _lag_positions(
        pair_frame,
        LAGS,
    )

    if "t_start" not in coordinate_source:
        raise RuntimeError(
            f"Unexpected coordinate source: {coordinate_source}"
        )

    rng = np.random.default_rng(SEED)

    record_rows = []
    feature_rows = []

    for lag in LAGS:
        pos = positions[lag]
        valid = pos >= 0

        current_idx = np.flatnonzero(valid)
        previous_idx = pos[valid]

        current_meta = full.iloc[
            current_idx
        ].reset_index(drop=True)

        previous_meta = full.iloc[
            previous_idx
        ].reset_index(drop=True)

        for column in (
            "record_id",
            "subject_id",
            "outer_fold",
        ):
            if not current_meta[
                column
            ].astype(str).equals(
                previous_meta[column].astype(str)
            ):
                raise RuntimeError(
                    f"lag {lag}: crossed {column}"
                )

        delta = (
            pd.to_numeric(
                previous_meta["t_start"],
                errors="coerce",
            ).to_numpy()
            - pd.to_numeric(
                current_meta["t_start"],
                errors="coerce",
            ).to_numpy()
        )

        if not np.allclose(
            delta,
            lag * 10.0,
            atol=1e-6,
            rtol=0,
        ):
            raise RuntimeError(
                f"lag {lag}: timestamp mismatch"
            )

        record_values = (
            current_meta["record_id"]
            .astype(str)
            .to_numpy()
        )

        observed_vectors = []
        surrogate_vectors = []
        excess_vectors = []

        for record_id in np.unique(record_values):
            record_mask = (
                record_values == record_id
            )

            n_pairs = int(
                record_mask.sum()
            )

            if n_pairs < MIN_PAIRS_PER_RECORD:
                continue

            x = discrete[
                current_idx[record_mask]
            ]

            y = discrete[
                previous_idx[record_mask]
            ]

            observed = normalized_mi(
                x,
                y,
            )

            surrogate_runs = []

            for _ in range(N_SURROGATES):
                surrogate_y = permute_valid_pairs(
                    x,
                    y,
                    rng,
                )

                surrogate_runs.append(
                    normalized_mi(
                        x,
                        surrogate_y,
                    )
                )

            surrogate_stack = np.vstack(
                surrogate_runs
            )

            surrogate_median = columnwise_nanmedian(
                surrogate_stack
            )

            excess = (
                observed
                - surrogate_median
            )

            observed_vectors.append(
                observed
            )
            surrogate_vectors.append(
                surrogate_median
            )
            excess_vectors.append(
                excess
            )

            subject_ids = (
                current_meta.loc[
                    record_mask,
                    "subject_id",
                ]
                .astype(str)
                .unique()
            )

            if len(subject_ids) != 1:
                raise RuntimeError(
                    f"record {record_id} spans subjects"
                )

            groups = [
                (
                    "all",
                    np.ones(
                        len(feature_names),
                        dtype=bool,
                    ),
                )
            ]

            groups.extend(
                (
                    family,
                    families == family,
                )
                for family in family_names
            )

            for family, feature_mask in groups:
                obs = observed[
                    feature_mask
                ]
                sur = surrogate_median[
                    feature_mask
                ]
                exc = excess[
                    feature_mask
                ]

                usable = (
                    np.isfinite(obs)
                    & np.isfinite(sur)
                    & np.isfinite(exc)
                )

                if not usable.any():
                    continue

                obs = obs[usable]
                sur = sur[usable]
                exc = exc[usable]

                record_rows.append(
                    {
                        "record_id": record_id,
                        "subject_id": subject_ids[0],
                        "lag_windows": abs(lag),
                        "lag_seconds": abs(lag) * 10,
                        "feature_family": family,
                        "n_pairs": n_pairs,
                        "n_valid_features": int(
                            len(obs)
                        ),
                        "median_observed_nmi": float(
                            np.median(obs)
                        ),
                        "median_surrogate_nmi": float(
                            np.median(sur)
                        ),
                        "median_excess_nmi": float(
                            np.median(exc)
                        ),
                        "fraction_excess_positive": float(
                            np.mean(exc > 0)
                        ),
                    }
                )

        if not observed_vectors:
            raise RuntimeError(
                f"No eligible records for lag {lag}"
            )

        observed_stack = np.vstack(
            observed_vectors
        )
        surrogate_stack = np.vstack(
            surrogate_vectors
        )
        excess_stack = np.vstack(
            excess_vectors
        )

        for feature_index, feature_name in enumerate(
            feature_names
        ):
            obs = observed_stack[
                :, feature_index
            ]
            sur = surrogate_stack[
                :, feature_index
            ]
            exc = excess_stack[
                :, feature_index
            ]

            usable = (
                np.isfinite(obs)
                & np.isfinite(sur)
                & np.isfinite(exc)
            )

            if not usable.any():
                continue

            feature_rows.append(
                {
                    "feature": feature_name,
                    "feature_family": families[
                        feature_index
                    ],
                    "lag_windows": abs(lag),
                    "lag_seconds": abs(lag) * 10,
                    "n_records": int(
                        usable.sum()
                    ),
                    "median_observed_nmi": float(
                        np.median(obs[usable])
                    ),
                    "median_surrogate_nmi": float(
                        np.median(sur[usable])
                    ),
                    "median_excess_nmi": float(
                        np.median(exc[usable])
                    ),
                    "q25_excess_nmi": quantile(
                        exc[usable],
                        0.25,
                    ),
                    "q75_excess_nmi": quantile(
                        exc[usable],
                        0.75,
                    ),
                    "fraction_records_excess_positive": float(
                        np.mean(
                            exc[usable] > 0
                        )
                    ),
                }
            )

    record_df = pd.DataFrame(
        record_rows
    )
    feature_df = pd.DataFrame(
        feature_rows
    )

    summary_rows = []

    for (
        lag_seconds,
        family,
    ), group in record_df.groupby(
        [
            "lag_seconds",
            "feature_family",
        ],
        sort=True,
    ):
        obs = group[
            "median_observed_nmi"
        ].to_numpy(dtype=float)

        sur = group[
            "median_surrogate_nmi"
        ].to_numpy(dtype=float)

        exc = group[
            "median_excess_nmi"
        ].to_numpy(dtype=float)

        usable = (
            np.isfinite(obs)
            & np.isfinite(sur)
            & np.isfinite(exc)
        )

        obs = obs[usable]
        sur = sur[usable]
        exc = exc[usable]

        summary_rows.append(
            {
                "lag_seconds": int(
                    lag_seconds
                ),
                "feature_family": family,
                "n_records": int(
                    len(exc)
                ),
                "median_observed_nmi": float(
                    np.median(obs)
                ),
                "median_surrogate_nmi": float(
                    np.median(sur)
                ),
                "median_excess_nmi": float(
                    np.median(exc)
                ),
                "q25_excess_nmi": quantile(
                    exc,
                    0.25,
                ),
                "q75_excess_nmi": quantile(
                    exc,
                    0.75,
                ),
                "fraction_records_excess_positive": float(
                    np.mean(exc > 0)
                ),
                "median_pairs_per_record": float(
                    group[
                        "n_pairs"
                    ].median()
                ),
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(
            [
                "feature_family",
                "lag_seconds",
            ]
        )
    )

    summary_path = (
        output / "mi_surrogate_summary.csv"
    )
    record_path = (
        output / "mi_surrogate_by_record_family.csv"
    )
    feature_path = (
        output / "mi_surrogate_by_feature.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )
    record_df.to_csv(
        record_path,
        index=False,
    )
    feature_df.to_csv(
        feature_path,
        index=False,
    )

    print()
    print(
        "=== TEMPORAL FEATURE MI + SURROGATE AUDIT ==="
    )
    print(
        "coordinate_source:",
        coordinate_source,
    )
    print(
        "feature matrix:",
        matrix.shape,
    )
    print(
        "quantile bins:",
        N_BINS,
    )
    print(
        "surrogates:",
        N_SURROGATES,
    )
    print(
        "minimum pairs:",
        MIN_PAIRS_PER_RECORD,
    )
    print()

    excess_pivot = summary_df.pivot(
        index="feature_family",
        columns="lag_seconds",
        values="median_excess_nmi",
    )

    print(
        "=== MEDIAN RECORD-LEVEL EXCESS NMI ==="
    )
    print(
        excess_pivot
        .round(4)
        .to_string()
    )
    print()

    observed_pivot = summary_df.pivot(
        index="feature_family",
        columns="lag_seconds",
        values="median_observed_nmi",
    )

    print(
        "=== MEDIAN RECORD-LEVEL OBSERVED NMI ==="
    )
    print(
        observed_pivot
        .round(4)
        .to_string()
    )
    print()

    surrogate_pivot = summary_df.pivot(
        index="feature_family",
        columns="lag_seconds",
        values="median_surrogate_nmi",
    )

    print(
        "=== MEDIAN RECORD-LEVEL SURROGATE NMI ==="
    )
    print(
        surrogate_pivot
        .round(4)
        .to_string()
    )
    print()

    print(
        "=== EXCESS NMI DETAILS ==="
    )
    print(
        summary_df[
            [
                "feature_family",
                "lag_seconds",
                "n_records",
                "median_excess_nmi",
                "q25_excess_nmi",
                "q75_excess_nmi",
                "fraction_records_excess_positive",
            ]
        ]
        .round(4)
        .to_string(index=False)
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
