"""Audit temporal-history support before predictive EEG history experiments.

No models are fitted. The audit reuses the canonical LOW/HIGH protocol and
feature cache, then checks availability of exact 10 s EEG history blocks for
H10, H20, H30, H60 and H120.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.analysis.pm_eeg_lag_sweep import _lag_positions
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low


HORIZONS = {
    "H10": (-1,),
    "H20": (-2, -1),
    "H30": (-3, -2, -1),
    "H60": tuple(range(-6, 0)),
    "H120": tuple(range(-12, 0)),
}
HORIZON_ORDER = tuple(HORIZONS)
ALL_OFFSETS = tuple(range(-12, 0))


def _retention(n: int, base: int) -> float:
    return float(n / base) if base else float("nan")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


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
            "reports/diagnostics/pm_temporal_predictability_audit_v1"
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
    output.mkdir(parents=True, exist_ok=True)

    config = low.load_config(config_path)

    # Use a runtime-only location for artifacts produced by the reused
    # reference protocol. Scientific support outputs go to output_dir.
    runtime_dir = root / "benchmark_results" / "_temporal_support_protocol"

    context = low.prepare_protocol(
        config,
        root=root,
        feature_cache_dir=cache,
        output_dir=runtime_dir,
    )
    full = context.full.reset_index(drop=True).copy()

    required = {
        "sample_id",
        "subject_id",
        "record_id",
        "outer_fold",
        "t_start",
        "t_end",
        *(f"target_{pm}" for pm in low.PM_NAMES),
    }
    missing = sorted(required.difference(full.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if full["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id values")

    t_start = pd.to_numeric(full["t_start"], errors="coerce").to_numpy()
    t_end = pd.to_numeric(full["t_end"], errors="coerce").to_numpy()

    if not np.isfinite(t_start).all() or not np.isfinite(t_end).all():
        raise RuntimeError("Non-finite temporal bounds")

    duration = t_end - t_start
    if not np.allclose(duration, 10.0, atol=1e-6, rtol=0):
        raise RuntimeError("Expected exact 10 s canonical windows")

    # Force real record-relative time coordinates. Do not allow window_id
    # to become the pairing source.
    pair_frame = full[["record_id", "t_start"]].copy()

    positions, common_h120, coordinate_source = _lag_positions(
        pair_frame,
        ALL_OFFSETS,
    )

    if "t_start" not in coordinate_source:
        raise RuntimeError(
            f"Unexpected temporal coordinate source: {coordinate_source}"
        )

    # Validate every available history pair against real timestamps and
    # grouping identities.
    for lag in ALL_OFFSETS:
        pos = positions[lag]
        valid = pos >= 0

        target_idx = np.flatnonzero(valid)
        history_idx = pos[valid]

        target = full.iloc[target_idx].reset_index(drop=True)
        history = full.iloc[history_idx].reset_index(drop=True)

        for column in ("record_id", "subject_id", "outer_fold"):
            left = target[column].astype(str).reset_index(drop=True)
            right = history[column].astype(str).reset_index(drop=True)
            if not left.equals(right):
                raise RuntimeError(f"lag {lag}: crossed {column}")

        actual_delta = (
            pd.to_numeric(history["t_start"], errors="coerce").to_numpy()
            - pd.to_numeric(target["t_start"], errors="coerce").to_numpy()
        )
        expected_delta = lag * 10.0

        if not np.allclose(
            actual_delta,
            expected_delta,
            atol=1e-6,
            rtol=0,
        ):
            raise RuntimeError(
                f"lag {lag}: expected {expected_delta}s timestamp delta"
            )

    support_masks: dict[str, np.ndarray] = {}

    for horizon, required_lags in HORIZONS.items():
        mask = np.ones(len(full), dtype=bool)
        for lag in required_lags:
            mask &= positions[lag] >= 0
        support_masks[horizon] = mask

    # Longer histories must always be subsets of shorter histories.
    for shorter, longer in zip(
        HORIZON_ORDER[:-1],
        HORIZON_ORDER[1:],
    ):
        if np.any(support_masks[longer] & ~support_masks[shorter]):
            raise RuntimeError(
                f"{longer} support is not a subset of {shorter}"
            )

    if not np.array_equal(common_h120, support_masks["H120"]):
        raise RuntimeError("H120 common-mask identity check failed")

    # Global temporal support.
    global_rows = []
    global_h10 = int(support_masks["H10"].sum())

    for horizon in HORIZON_ORDER:
        mask = support_masks[horizon]
        n = int(mask.sum())
        global_rows.append(
            {
                "horizon": horizon,
                "history_seconds": int(horizon[1:]),
                "required_blocks": len(HORIZONS[horizon]),
                "n_windows": n,
                "retention_vs_h10": _retention(n, global_h10),
                "n_subjects": int(
                    full.loc[mask, "subject_id"].nunique()
                ),
                "n_records": int(
                    full.loc[mask, "record_id"].nunique()
                ),
            }
        )

    global_df = pd.DataFrame(global_rows)

    # Target-specific complete-case support.
    target_rows = []

    for pm in low.PM_NAMES:
        target_col = f"target_{pm}"
        valid_target = np.isfinite(
            pd.to_numeric(
                full[target_col],
                errors="coerce",
            ).to_numpy()
        )

        h10_base = int(
            (valid_target & support_masks["H10"]).sum()
        )

        for horizon in HORIZON_ORDER:
            mask = valid_target & support_masks[horizon]
            n = int(mask.sum())

            target_rows.append(
                {
                    "pm": pm,
                    "target_id": target_col,
                    "horizon": horizon,
                    "history_seconds": int(horizon[1:]),
                    "required_blocks": len(HORIZONS[horizon]),
                    "n_windows": n,
                    "retention_vs_h10": _retention(n, h10_base),
                    "n_subjects": int(
                        full.loc[mask, "subject_id"].nunique()
                    ),
                    "n_records": int(
                        full.loc[mask, "record_id"].nunique()
                    ),
                    "strict_common_support": horizon == "H120",
                }
            )

    target_df = pd.DataFrame(target_rows)

    # Fold/partition support. Thresholds and labels are deliberately not
    # recomputed here; this is only a temporal availability audit.
    fold_rows = []

    folds = sorted(
        int(value) for value in full["outer_fold"].unique()
    )
    fold_values = full["outer_fold"].to_numpy()

    for pm in low.PM_NAMES:
        target_col = f"target_{pm}"
        valid_target = np.isfinite(
            pd.to_numeric(
                full[target_col],
                errors="coerce",
            ).to_numpy()
        )

        for fold in folds:
            for partition in ("train", "test"):
                partition_mask = (
                    fold_values != fold
                    if partition == "train"
                    else fold_values == fold
                )

                h10_base = int(
                    (
                        valid_target
                        & partition_mask
                        & support_masks["H10"]
                    ).sum()
                )

                for horizon in HORIZON_ORDER:
                    mask = (
                        valid_target
                        & partition_mask
                        & support_masks[horizon]
                    )
                    n = int(mask.sum())

                    fold_rows.append(
                        {
                            "pm": pm,
                            "target_id": target_col,
                            "outer_fold": fold,
                            "partition": partition,
                            "horizon": horizon,
                            "history_seconds": int(horizon[1:]),
                            "n_windows": n,
                            "retention_vs_h10": _retention(
                                n,
                                h10_base,
                            ),
                            "n_subjects": int(
                                full.loc[
                                    mask, "subject_id"
                                ].nunique()
                            ),
                            "n_records": int(
                                full.loc[
                                    mask, "record_id"
                                ].nunique()
                            ),
                        }
                    )

    fold_df = pd.DataFrame(fold_rows)

    global_path = output / "history_support_global.csv"
    target_path = output / "history_support_audit.csv"
    fold_path = output / "history_support_by_fold_target.csv"
    summary_path = output / "history_support_summary.json"
    readme_path = output / "README.md"

    global_df.to_csv(global_path, index=False)
    target_df.to_csv(target_path, index=False)
    fold_df.to_csv(fold_path, index=False)

    h120_target = target_df[
        target_df["horizon"] == "H120"
    ].copy()

    test_h120 = fold_df[
        (fold_df["partition"] == "test")
        & (fold_df["horizon"] == "H120")
    ]

    summary = {
        "schema_version": "pm-temporal-history-support-v1",
        "coordinate_source": coordinate_source,
        "canonical_rows": int(len(full)),
        "canonical_subjects": int(full["subject_id"].nunique()),
        "canonical_records": int(full["record_id"].nunique()),
        "feature_count": int(config["feature_cache_identity"]["n_features"]),
        "history_horizons_seconds": [10, 20, 30, 60, 120],
        "primary_support_mode": "strict_common_support_h120",
        "secondary_support_mode": "horizon_specific_support",
        "h120_global_windows": int(
            support_masks["H120"].sum()
        ),
        "h120_global_retention_vs_h10": float(
            global_df.loc[
                global_df["horizon"] == "H120",
                "retention_vs_h10",
            ].iloc[0]
        ),
        "h120_global_subjects": int(
            full.loc[
                support_masks["H120"],
                "subject_id",
            ].nunique()
        ),
        "h120_global_records": int(
            full.loc[
                support_masks["H120"],
                "record_id",
            ].nunique()
        ),
        "minimum_test_fold_h120_retention": float(
            test_h120["retention_vs_h10"].min()
        ),
        "target_h120_windows": {
            row.pm: int(row.n_windows)
            for row in h120_target.itertuples()
        },
        "target_h120_retention_vs_h10": {
            row.pm: float(row.retention_vs_h10)
            for row in h120_target.itertuples()
        },
        "model_fits": 0,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme_lines = [
        "# Temporal Predictability Audit - history support",
        "",
        "This stage audits exact EEG history availability before any "
        "predictive model fitting.",
        "",
        "## Contract",
        "",
        "- Canonical feature cache: 371 float32 EEG features.",
        "- Canonical block: exact 10 s EEG window.",
        "- History uses real record-relative `t_start`.",
        "- No previous-available substitution is allowed.",
        "- History never crosses subject, record or outer-fold identity.",
        "- Horizons: H10, H20, H30, H60 and H120.",
        "- Primary comparison support: `strict_common_support_h120`.",
        "- Secondary sensitivity support: `horizon_specific_support`.",
        "- LOW/HIGH thresholds are not refit in this support audit.",
        "- No models are fitted.",
        "",
        "## Global support",
        "",
        *_markdown_table(
            global_df[
                [
                    "horizon",
                    "required_blocks",
                    "n_windows",
                    "retention_vs_h10",
                    "n_subjects",
                    "n_records",
                ]
            ]
        ),
        "",
        "## Key result",
        "",
        f"H120 retains "
        f"{summary['h120_global_retention_vs_h10']:.2%} "
        f"of H10 support "
        f"({summary['h120_global_windows']} windows), with "
        f"{summary['h120_global_subjects']} subjects and "
        f"{summary['h120_global_records']} records.",
        "",
        f"The minimum H120 retention across target/test-fold "
        f"combinations is "
        f"{summary['minimum_test_fold_h120_retention']:.2%}.",
        "",
        "The retained support is sufficient to use H120 as the "
        "matched support for the primary H10-H120 comparison.",
        "",
    ]

    readme_path.write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )

    print("=== TEMPORAL HISTORY SUPPORT AUDIT ===")
    print(global_df.to_string(index=False))
    print()
    print("H120 target support:")
    print(
        h120_target[
            [
                "pm",
                "n_windows",
                "retention_vs_h10",
                "n_subjects",
                "n_records",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Minimum test-fold H120 retention:",
        f"{summary['minimum_test_fold_h120_retention']:.4f}",
    )
    print("Outputs:", output)
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
