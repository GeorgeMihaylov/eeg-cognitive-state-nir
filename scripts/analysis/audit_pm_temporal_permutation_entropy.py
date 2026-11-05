"""Permutation-entropy audit of canonical EEG feature trajectories.

The analysis uses only the longest exact 10 s contiguous segment of each
record. Permutation entropy is computed on feature trajectories, not raw EEG.
Temporal row permutations within the same segment provide the surrogate null.
No predictive model is fitted.
"""
from __future__ import annotations

import argparse
import math
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.experiments import pm_low_high_q3_extremes_confirmatory as low
from bench.analysis.temporal_feature_diagnostics import contiguous_segments, permute_finite_columns


ORDERS = (3, 4, 5)
MIN_SEGMENT_WINDOWS = 100
MIN_PATTERNS = 50
N_SURROGATES = 10
SEED = 42


def permutation_entropy_columns(
    matrix: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized permutation entropy independently for every feature.

    Stable ordering is used for ties. The same tie rule is applied to
    observed and surrogate trajectories.
    """
    if matrix.ndim != 2:
        raise ValueError("Expected [time, features] matrix")

    if order not in ORDERS:
        raise ValueError("Supported PE orders: 3, 4, 5")
    n_time, n_features = matrix.shape

    entropy = np.full(
        n_features,
        np.nan,
        dtype=np.float64,
    )
    n_patterns = np.zeros(
        n_features,
        dtype=np.int64,
    )
    tie_fraction = np.full(
        n_features,
        np.nan,
        dtype=np.float64,
    )

    if n_time < order:
        return entropy, n_patterns, tie_fraction

    powers = (
        order ** np.arange(order, dtype=np.int64)
    )

    normalization = math.log(
        math.factorial(order)
    )

    for feature_index in range(n_features):
        values = matrix[:, feature_index]

        windows = np.lib.stride_tricks.sliding_window_view(
            values,
            order,
        )

        finite = np.isfinite(windows).all(axis=1)

        windows = windows[finite]

        n_patterns[feature_index] = len(windows)
        if len(windows) < max(MIN_PATTERNS, 5 * math.factorial(order)):
            continue

        sorted_values = np.sort(
            windows,
            axis=1,
        )

        tied = np.any(
            np.isclose(
                np.diff(
                    sorted_values,
                    axis=1,
                ),
                0.0,
                atol=0.0,
                rtol=0.0,
            ),
            axis=1,
        )

        tie_fraction[feature_index] = float(
            np.mean(tied)
        )

        orderings = np.argsort(
            windows,
            axis=1,
            kind="stable",
        )

        codes = (
            orderings.astype(np.int64)
            * powers[None, :]
        ).sum(axis=1)

        _, counts = np.unique(
            codes,
            return_counts=True,
        )

        probabilities = (
            counts.astype(np.float64)
            / counts.sum()
        )

        h = -np.sum(
            probabilities
            * np.log(probabilities)
        )

        entropy[feature_index] = (
            float(h / normalization)
            if normalization > 0
            else np.nan
        )

        n_patterns[feature_index] = int(
            len(windows)
        )

    return entropy, n_patterns, tie_fraction


def safe_nanmedian_columns(
    matrix: np.ndarray,
) -> np.ndarray:
    result = np.full(
        matrix.shape[1],
        np.nan,
        dtype=np.float64,
    )

    usable = np.isfinite(matrix).any(axis=0)

    if usable.any():
        result[usable] = np.nanmedian(
            matrix[:, usable],
            axis=0,
        )

    return result


def q(
    values: np.ndarray,
    probability: float,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]

    if not len(values):
        return np.nan

    return float(
        np.quantile(
            values,
            probability,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

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
            "pm_temporal_predictability_audit_v1/pe_final_v1"
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

    cfg = low.load_config(
        config_path
    )

    context = low.prepare_protocol(
        cfg,
        root=root,
        feature_cache_dir=cache,
        output_dir=(
            root
            / "benchmark_results"
            / "_temporal_pe_protocol"
        ),
    )

    full = (
        context.full
        .reset_index(drop=True)
        .copy()
    )

    matrix = np.asarray(
        context.matrix,
        dtype=np.float32,
    )

    feature_names = list(
        context.feature_names
    )

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

    expected_families = {
        "spectral",
        "statistical",
        "entropy",
        "connectivity",
    }

    if set(families.tolist()) != expected_families:
        raise RuntimeError(
            "Unexpected feature families"
        )

    # Select longest exact-contiguous segment independently per record.
    selected = []

    for record_id, group in full.groupby(
        "record_id",
        sort=False,
    ):
        segments = contiguous_segments(
            group
        )

        longest = max(
            segments,
            key=len,
        )

        if len(longest) < MIN_SEGMENT_WINDOWS:
            continue

        subject_ids = (
            full.loc[
                longest,
                "subject_id",
            ]
            .astype(str)
            .unique()
        )

        if len(subject_ids) != 1:
            raise RuntimeError(
                f"record {record_id} spans subjects"
            )

        selected.append(
            {
                "record_id": str(record_id),
                "subject_id": subject_ids[0],
                "indices": longest,
                "segment_windows": int(
                    len(longest)
                ),
            }
        )

    if not selected:
        raise RuntimeError(
            "No eligible contiguous segments"
        )

    protocol = {
        "experiment_type": "diagnostic", "schema_version": "pe-final-v1",
        "orders": list(ORDERS), "delay_windows": 1,
        "minimum_segment_windows": MIN_SEGMENT_WINDOWS,
        "minimum_patterns": {str(m): max(MIN_PATTERNS, 5 * math.factorial(m)) for m in ORDERS},
        "length_rule": "heuristic >=5*m! valid patterns, at least 50; finite-sample bias remains",
        "normalization": "H/log(m!)", "encoding": "base-m digits of stable argsort; injective",
        "ties": "stable temporal-index order; tie fractions reported",
        "segments": "longest exact-contiguous per record; earliest breaks length ties",
        "surrogates": N_SURROGATES, "seed": SEED,
        "surrogate_rule": "within segment; finite mask fixed",
        "aggregation": "median features within record/family, then equal-record median",
        "reference_protocol_hash": context.protocol["protocol_hash"],
        "feature_values_sha256": hashlib.sha256(memoryview(matrix).cast("B")).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "insufficient_rule": "NaN PE when valid pattern count below minimum",
    }
    low._atomic_json(output / "permutation_entropy_protocol.json", protocol)

    rng = np.random.default_rng(
        SEED
    )

    record_rows = []
    feature_rows = []

    for order in ORDERS:
        observed_vectors = []
        surrogate_vectors = []
        structure_vectors = []

        for item in selected:
            indices = item["indices"]
            block = matrix[indices]

            observed, n_patterns, tie_fraction = (
                permutation_entropy_columns(
                    block,
                    order,
                )
            )

            surrogate_runs = []

            for _ in range(N_SURROGATES):
                shuffled = permute_finite_columns(block, rng)

                surrogate_pe, _, _ = (
                    permutation_entropy_columns(
                        shuffled,
                        order,
                    )
                )

                surrogate_runs.append(
                    surrogate_pe
                )

            surrogate_stack = np.vstack(
                surrogate_runs
            )

            surrogate_median = (
                safe_nanmedian_columns(
                    surrogate_stack
                )
            )

            # Positive means the observed temporal ordering is more
            # structured / less entropic than shuffled time.
            structure = (
                surrogate_median
                - observed
            )

            observed_vectors.append(
                observed
            )
            surrogate_vectors.append(
                surrogate_median
            )
            structure_vectors.append(
                structure
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
                for family in sorted(
                    expected_families
                )
            )

            for family, feature_mask in groups:
                usable = (
                    feature_mask
                    & np.isfinite(observed)
                    & np.isfinite(surrogate_median)
                    & np.isfinite(structure)
                )

                if not usable.any():
                    continue

                obs = observed[usable]
                sur = surrogate_median[usable]
                st = structure[usable]
                ties = tie_fraction[usable]
                patterns = n_patterns[usable]

                record_rows.append(
                    {
                        "record_id": item[
                            "record_id"
                        ],
                        "subject_id": item[
                            "subject_id"
                        ],
                        "segment_windows": item[
                            "segment_windows"
                        ],
                        "order": order,
                        "feature_family": family,
                        "n_valid_features": int(
                            usable.sum()
                        ),
                        "median_observed_pe": float(
                            np.median(obs)
                        ),
                        "median_surrogate_pe": float(
                            np.median(sur)
                        ),
                        "median_structure_delta": float(
                            np.median(st)
                        ),
                        "fraction_structure_positive": float(
                            np.mean(st > 0)
                        ),
                        "median_tie_fraction": float(
                            np.nanmedian(ties)
                        ),
                        "median_n_patterns": float(
                            np.median(patterns)
                        ),
                    }
                )

        observed_stack = np.vstack(
            observed_vectors
        )
        surrogate_stack = np.vstack(
            surrogate_vectors
        )
        structure_stack = np.vstack(
            structure_vectors
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
            st = structure_stack[
                :, feature_index
            ]

            usable = (
                np.isfinite(obs)
                & np.isfinite(sur)
                & np.isfinite(st)
            )

            if not usable.any():
                continue

            feature_rows.append(
                {
                    "feature": feature_name,
                    "feature_family": families[
                        feature_index
                    ],
                    "order": order,
                    "n_records": int(
                        usable.sum()
                    ),
                    "median_observed_pe": float(
                        np.median(obs[usable])
                    ),
                    "median_surrogate_pe": float(
                        np.median(sur[usable])
                    ),
                    "median_structure_delta": float(
                        np.median(st[usable])
                    ),
                    "q25_structure_delta": q(
                        st[usable],
                        0.25,
                    ),
                    "q75_structure_delta": q(
                        st[usable],
                        0.75,
                    ),
                    "fraction_records_structure_positive": float(
                        np.mean(
                            st[usable] > 0
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
        order,
        family,
    ), group in record_df.groupby(
        [
            "order",
            "feature_family",
        ],
        sort=True,
    ):
        observed = group[
            "median_observed_pe"
        ].to_numpy(dtype=float)

        surrogate = group[
            "median_surrogate_pe"
        ].to_numpy(dtype=float)

        structure = group[
            "median_structure_delta"
        ].to_numpy(dtype=float)

        usable = (
            np.isfinite(observed)
            & np.isfinite(surrogate)
            & np.isfinite(structure)
        )

        observed = observed[usable]
        surrogate = surrogate[usable]
        structure = structure[usable]

        summary_rows.append(
            {
                "order": int(order),
                "feature_family": family,
                "n_records": int(
                    len(structure)
                ),
                "median_observed_pe": float(
                    np.median(observed)
                ),
                "median_surrogate_pe": float(
                    np.median(surrogate)
                ),
                "median_structure_delta": float(
                    np.median(structure)
                ),
                "q25_structure_delta": q(
                    structure,
                    0.25,
                ),
                "q75_structure_delta": q(
                    structure,
                    0.75,
                ),
                "fraction_records_structure_positive": float(
                    np.mean(
                        structure > 0
                    )
                ),
                "median_segment_windows": float(
                    group[
                        "segment_windows"
                    ].median()
                ),
                "median_tie_fraction": float(
                    group[
                        "median_tie_fraction"
                    ].median()
                ),
            }
        )

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "feature_family",
                "order",
            ]
        )
    )

    segment_df = pd.DataFrame(
        [
            {
                "record_id": item["record_id"],
                "subject_id": item["subject_id"],
                "segment_windows": item[
                    "segment_windows"
                ],
                "segment_seconds": item[
                    "segment_windows"
                ] * 10,
            }
            for item in selected
        ]
    )

    summary_path = (
        output
        / "permutation_entropy_summary.csv"
    )

    record_path = (
        output
        / "permutation_entropy_by_record_family.csv"
    )

    feature_path = (
        output
        / "permutation_entropy_by_feature.csv"
    )

    segment_path = (
        output
        / "permutation_entropy_segments.csv"
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

    segment_df.to_csv(
        segment_path,
        index=False,
    )

    print(
        "=== TEMPORAL PERMUTATION ENTROPY AUDIT ==="
    )

    print(
        "eligible records:",
        len(selected),
    )

    print(
        "minimum segment windows:",
        MIN_SEGMENT_WINDOWS,
    )

    print(
        "surrogates:",
        N_SURROGATES,
    )

    print()

    print(
        "=== MEDIAN NORMALIZED OBSERVED PE ==="
    )

    print(
        summary_df.pivot(
            index="feature_family",
            columns="order",
            values="median_observed_pe",
        )
        .round(4)
        .to_string()
    )

    print()

    print(
        "=== MEDIAN SURROGATE PE ==="
    )

    print(
        summary_df.pivot(
            index="feature_family",
            columns="order",
            values="median_surrogate_pe",
        )
        .round(4)
        .to_string()
    )

    print()

    print(
        "=== MEDIAN TEMPORAL STRUCTURE DELTA "
        "(surrogate - observed) ==="
    )

    print(
        summary_df.pivot(
            index="feature_family",
            columns="order",
            values="median_structure_delta",
        )
        .round(4)
        .to_string()
    )

    print()

    print(
        "=== STRUCTURE DETAILS ==="
    )

    print(
        summary_df[
            [
                "feature_family",
                "order",
                "n_records",
                "median_structure_delta",
                "q25_structure_delta",
                "q75_structure_delta",
                "fraction_records_structure_positive",
                "median_tie_fraction",
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
    print(segment_path)
    print("All checks passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
