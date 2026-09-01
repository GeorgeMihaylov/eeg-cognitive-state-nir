from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


ROOT = Path(
    "benchmark_results/cross_dataset_clare_cldrive_dann_v1"
)

PROTOCOL_HASH = (
    "df0470b0a578b05a58414b3253f7845061513c539fe3a073f0917abf26e63698"
)

MODES = [
    "target_only",
    "source_only",
    "dann",
]

DIRECTION_NAMES = {
    "cl_drive_to_clare": "CL-Drive -> CLARE",
    "clare_to_cl_drive": "CLARE -> CL-Drive",
}


# ============================================================
# Utilities
# ============================================================

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)

    running = 0.0

    for rank, index in enumerate(order):
        value = (m - rank) * p[index]
        running = max(running, value)
        adjusted_sorted[rank] = min(running, 1.0)

    adjusted = np.empty(m, dtype=float)

    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]

    return adjusted


def bootstrap_mean_ci(
    values,
    seed=42,
    n_boot=10000,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_boot, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def participant_balanced_accuracy(
    y_true,
    y_pred,
):
    """
    Balanced Accuracy is calculated over classes
    actually present in y_true for a participant.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        return float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )


def participant_metrics(predictions):
    """
    dataset_participant_id is the statistical unit.

    cross_dataset_person_key is a leakage-control identity,
    not the grouping key for target participant metrics.
    """

    rows = []

    for participant_id, part in predictions.groupby(
        "dataset_participant_id",
        sort=True,
    ):
        y_true = part[
            "y_true"
        ].to_numpy(dtype=int)

        y_pred = part[
            "y_pred"
        ].to_numpy(dtype=int)

        rows.append(
            {
                # Unified name used by the remainder of
                # this analysis script.
                "participant_id": str(
                    participant_id
                ),
                "windows": int(len(part)),
                "true_classes": int(
                    len(np.unique(y_true))
                ),
                "predicted_classes": int(
                    len(np.unique(y_pred))
                ),
                "accuracy": float(
                    accuracy_score(
                        y_true,
                        y_pred,
                    )
                ),
                "balanced_accuracy":
                    participant_balanced_accuracy(
                        y_true,
                        y_pred,
                    ),
                "macro_f1": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=[0, 1, 2],
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 1. Load 30 completed runs
# ============================================================

summary_paths = sorted(
    ROOT.glob(
        "runs/*/run_summary.json"
    )
)

if len(summary_paths) != 30:
    raise RuntimeError(
        "Expected 30 run summaries, "
        f"got {len(summary_paths)}"
    )


run_rows = []
participant_frames = []


for summary_path in summary_paths:

    payload = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("status") != "complete":
        raise RuntimeError(
            "Incomplete run: "
            + summary_path.parent.name
        )

    if (
        payload.get("protocol_hash")
        != PROTOCOL_HASH
    ):
        raise RuntimeError(
            "Protocol hash mismatch: "
            + summary_path.parent.name
        )

    direction = str(
        payload["direction"]
    )

    fold = int(
        payload["fold"]
    )

    mode = str(
        payload["mode"]
    )

    if mode not in MODES:
        raise RuntimeError(
            f"Unknown mode: {mode}"
        )

    metrics = payload["metrics"]

    row = {
        "run_id": payload["run_id"],
        "direction": direction,
        "direction_name":
            DIRECTION_NAMES.get(
                direction,
                direction,
            ),
        "fold": fold,
        "mode": mode,
        "epochs_trained": int(
            payload["epochs_trained"]
        ),
        "training_time_seconds": float(
            payload[
                "training_time_seconds"
            ]
        ),
        "test_sample_ids_hash":
            payload[
                "test_sample_ids_hash"
            ],
        "participants": int(
            metrics["participants"]
        ),
        "windows": int(
            metrics["windows"]
        ),
    }

    for level in [
        "participant_macro",
        "pooled_window",
    ]:
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]:
            row[
                f"{level}_{metric}"
            ] = float(
                metrics[level][metric]
            )

    if mode == "dann":
        if (
            payload.get(
                "target_adaptation_task_labels_accessible"
            )
            is not False
        ):
            raise RuntimeError(
                "DANN target-adaptation "
                "labels were accessible: "
                + payload["run_id"]
            )

    run_rows.append(row)

    # --------------------------------------------------------
    # Reconstruct participant-level metrics from predictions
    # --------------------------------------------------------

    pred_path = (
        summary_path.parent
        / "predictions.parquet"
    )

    if not pred_path.exists():
        raise RuntimeError(
            f"Missing predictions: {pred_path}"
        )

    pred = pd.read_parquet(
        pred_path
    )

    required = {
        "dataset_participant_id",
        "y_true",
        "y_pred",
        "sample_id",
    }

    missing = (
        required
        - set(pred.columns)
    )

    if missing:
        raise RuntimeError(
            f"{payload['run_id']} "
            f"predictions missing "
            f"{sorted(missing)}; "
            f"columns={list(pred.columns)}"
        )

    pm = participant_metrics(pred)

    pm.insert(
        0,
        "mode",
        mode,
    )

    pm.insert(
        0,
        "fold",
        fold,
    )

    pm.insert(
        0,
        "direction",
        direction,
    )

    participant_frames.append(
        pm
    )


runs = pd.DataFrame(
    run_rows
)

participants = pd.concat(
    participant_frames,
    ignore_index=True,
)


# ============================================================
# 2. Pairing audit
# ============================================================

pairing = (
    runs
    .groupby(
        ["direction", "fold"]
    )
    .agg(
        modes=("mode", "nunique"),
        hashes=(
            "test_sample_ids_hash",
            "nunique",
        ),
        participants=(
            "participants",
            "nunique",
        ),
        windows=(
            "windows",
            "nunique",
        ),
    )
    .reset_index()
)


if not (
    pairing["modes"] == 3
).all():
    raise RuntimeError(
        "Not every direction/fold "
        "has all 3 modes"
    )


if not (
    pairing["hashes"] == 1
).all():
    raise RuntimeError(
        "Test sample IDs differ "
        "between paired modes"
    )


if not (
    pairing["participants"] == 1
).all():
    raise RuntimeError(
        "Participant counts differ "
        "between paired modes"
    )


if not (
    pairing["windows"] == 1
).all():
    raise RuntimeError(
        "Window counts differ "
        "between paired modes"
    )


for (
    direction,
    fold,
), part in participants.groupby(
    ["direction", "fold"]
):

    sets = {
        mode: set(
            part.loc[
                part["mode"].eq(mode),
                "participant_id",
            ]
        )
        for mode in MODES
    }

    if not (
        sets["target_only"]
        == sets["source_only"]
        == sets["dann"]
    ):
        raise RuntimeError(
            "Participant pairing failed for "
            f"{direction} fold {fold}"
        )


print("=" * 110)
print("PAIRING AUDIT")
print("=" * 110)
print("30 complete runs: yes")
print("same test hash across modes: yes")
print("same participants across modes: yes")
print(
    "DANN target adaptation labels "
    "inaccessible: yes"
)


# ============================================================
# 3. Verify reconstructed participant metrics
# ============================================================

verification_rows = []


for _, run in runs.iterrows():

    part = participants.loc[
        participants[
            "direction"
        ].eq(
            run["direction"]
        )
        & participants[
            "fold"
        ].eq(
            run["fold"]
        )
        & participants[
            "mode"
        ].eq(
            run["mode"]
        )
    ]

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]:

        reconstructed = float(
            part[metric].mean()
        )

        stored = float(
            run[
                f"participant_macro_{metric}"
            ]
        )

        verification_rows.append(
            {
                "run_id":
                    run["run_id"],
                "metric":
                    metric,
                "stored":
                    stored,
                "reconstructed":
                    reconstructed,
                "abs_difference":
                    abs(
                        stored
                        - reconstructed
                    ),
            }
        )


verification = pd.DataFrame(
    verification_rows
)


print()
print("=" * 110)
print(
    "PARTICIPANT METRIC RECONSTRUCTION"
)
print("=" * 110)

max_difference = float(
    verification[
        "abs_difference"
    ].max()
)

print(
    "maximum absolute difference:",
    max_difference,
)

if max_difference > 1e-8:
    print(
        "WARNING: reconstructed "
        "participant metrics differ "
        "from stored results."
    )
else:
    print(
        "reconstruction matches "
        "stored metrics"
    )


# ============================================================
# 4. Participant-macro summary
#
# Each target participant occurs in test exactly once per
# direction. Therefore this gives equal participant weight.
# ============================================================

direction_rows = []


for (
    direction,
    mode,
), part in participants.groupby(
    ["direction", "mode"],
    sort=True,
):

    row = {
        "direction":
            direction,
        "direction_name":
            DIRECTION_NAMES.get(
                direction,
                direction,
            ),
        "mode":
            mode,
        "participants":
            int(
                part[
                    "participant_id"
                ].nunique()
            ),
    }

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]:

        values = (
            part[metric]
            .to_numpy(dtype=float)
        )

        lo, hi = bootstrap_mean_ci(
            values,
            seed=42,
        )

        row[metric] = float(
            np.mean(values)
        )

        row[
            f"{metric}_std"
        ] = float(
            np.std(
                values,
                ddof=1,
            )
        )

        row[
            f"{metric}_ci95_low"
        ] = lo

        row[
            f"{metric}_ci95_high"
        ] = hi

    direction_rows.append(row)


direction_summary = (
    pd.DataFrame(
        direction_rows
    )
)


# ============================================================
# 5. Paired participant-level effects
# ============================================================

wide = participants.pivot(
    index=[
        "direction",
        "fold",
        "participant_id",
    ],
    columns="mode",
    values=[
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ],
)


effect_rows = []


CONTRASTS = {
    # Negative value = cross-dataset degradation
    "transfer_gap": (
        "source_only",
        "target_only",
    ),

    # Positive value = DANN improvement
    "adaptation_effect": (
        "dann",
        "source_only",
    ),
}


for direction in sorted(
    participants[
        "direction"
    ].unique()
):

    direction_wide = (
        wide.loc[direction]
    )

    for metric in [
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "weighted_f1",
    ]:

        for (
            contrast,
            (
                first_mode,
                second_mode,
            ),
        ) in CONTRASTS.items():

            first = direction_wide[
                (
                    metric,
                    first_mode,
                )
            ].to_numpy(
                dtype=float
            )

            second = direction_wide[
                (
                    metric,
                    second_mode,
                )
            ].to_numpy(
                dtype=float
            )

            delta = (
                first
                - second
            )

            nonzero = delta[
                ~np.isclose(
                    delta,
                    0.0,
                )
            ]

            if len(nonzero) == 0:
                statistic = 0.0
                p_value = 1.0

            else:
                result = wilcoxon(
                    first,
                    second,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )

                statistic = float(
                    result.statistic
                )

                p_value = float(
                    result.pvalue
                )

            lo, hi = bootstrap_mean_ci(
                delta,
                seed=42,
            )

            effect_rows.append(
                {
                    "direction":
                        direction,
                    "direction_name":
                        DIRECTION_NAMES.get(
                            direction,
                            direction,
                        ),
                    "contrast":
                        contrast,
                    "metric":
                        metric,
                    "participants":
                        len(delta),
                    "mean_delta":
                        float(
                            np.mean(
                                delta
                            )
                        ),
                    "median_delta":
                        float(
                            np.median(
                                delta
                            )
                        ),
                    "std_delta":
                        float(
                            np.std(
                                delta,
                                ddof=1,
                            )
                        ),
                    "mean_delta_ci95_low":
                        lo,
                    "mean_delta_ci95_high":
                        hi,
                    "wins":
                        int(
                            np.sum(
                                delta
                                > 1e-12
                            )
                        ),
                    "ties":
                        int(
                            np.sum(
                                np.abs(
                                    delta
                                )
                                <= 1e-12
                            )
                        ),
                    "losses":
                        int(
                            np.sum(
                                delta
                                < -1e-12
                            )
                        ),
                    "wilcoxon_statistic":
                        statistic,
                    "p_value":
                        p_value,
                }
            )


effects = pd.DataFrame(
    effect_rows
)


# ============================================================
# 6. Holm correction
#
# Separate families:
# - transfer_gap
# - adaptation_effect
#
# Each contains:
# 2 directions x 2 primary metrics.
# ============================================================

effects[
    "holm_family"
] = "descriptive"

effects[
    "p_holm"
] = np.nan


for contrast in [
    "transfer_gap",
    "adaptation_effect",
]:

    mask = (
        effects[
            "contrast"
        ].eq(
            contrast
        )
        & effects[
            "metric"
        ].isin(
            [
                "macro_f1",
                "balanced_accuracy",
            ]
        )
    )

    p_values = (
        effects.loc[
            mask,
            "p_value",
        ]
        .to_numpy(
            dtype=float
        )
    )

    effects.loc[
        mask,
        "p_holm",
    ] = holm_adjust(
        p_values
    )

    effects.loc[
        mask,
        "holm_family",
    ] = (
        contrast
        + "_2directions_x_2primarymetrics"
    )


# ============================================================
# 7. Report-ready table
# ============================================================

report_rows = []


for direction in DIRECTION_NAMES:

    mode_rows = (
        direction_summary
        .loc[
            direction_summary[
                "direction"
            ].eq(
                direction
            )
        ]
        .set_index(
            "mode"
        )
    )

    if len(mode_rows) != 3:
        continue

    report_rows.append(
        {
            "direction":
                DIRECTION_NAMES[
                    direction
                ],

            "participants":
                int(
                    mode_rows.loc[
                        "source_only",
                        "participants",
                    ]
                ),

            "target_only_macro_f1":
                mode_rows.loc[
                    "target_only",
                    "macro_f1",
                ],

            "source_only_macro_f1":
                mode_rows.loc[
                    "source_only",
                    "macro_f1",
                ],

            "dann_macro_f1":
                mode_rows.loc[
                    "dann",
                    "macro_f1",
                ],

            "transfer_delta_macro_f1":
                (
                    mode_rows.loc[
                        "source_only",
                        "macro_f1",
                    ]
                    - mode_rows.loc[
                        "target_only",
                        "macro_f1",
                    ]
                ),

            "dann_delta_macro_f1":
                (
                    mode_rows.loc[
                        "dann",
                        "macro_f1",
                    ]
                    - mode_rows.loc[
                        "source_only",
                        "macro_f1",
                    ]
                ),

            "target_only_balanced_accuracy":
                mode_rows.loc[
                    "target_only",
                    "balanced_accuracy",
                ],

            "source_only_balanced_accuracy":
                mode_rows.loc[
                    "source_only",
                    "balanced_accuracy",
                ],

            "dann_balanced_accuracy":
                mode_rows.loc[
                    "dann",
                    "balanced_accuracy",
                ],

            "transfer_delta_balanced_accuracy":
                (
                    mode_rows.loc[
                        "source_only",
                        "balanced_accuracy",
                    ]
                    - mode_rows.loc[
                        "target_only",
                        "balanced_accuracy",
                    ]
                ),

            "dann_delta_balanced_accuracy":
                (
                    mode_rows.loc[
                        "dann",
                        "balanced_accuracy",
                    ]
                    - mode_rows.loc[
                        "source_only",
                        "balanced_accuracy",
                    ]
                ),
        }
    )


report_table = pd.DataFrame(
    report_rows
)


# ============================================================
# 8. Save artifacts
# ============================================================

runs.to_csv(
    ROOT
    / "analysis_run_results.csv",
    index=False,
)

participants.to_csv(
    ROOT
    / "analysis_participant_metrics.csv",
    index=False,
)

verification.to_csv(
    ROOT
    / "analysis_participant_metric_verification.csv",
    index=False,
)

direction_summary.to_csv(
    ROOT
    / "analysis_direction_mode_summary.csv",
    index=False,
)

effects.to_csv(
    ROOT
    / "analysis_paired_effects.csv",
    index=False,
)

report_table.to_csv(
    ROOT
    / "analysis_report_table.csv",
    index=False,
)


# ============================================================
# 9. Print report-ready results
# ============================================================

print()
print("=" * 110)
print(
    "DIRECTION x MODE — PARTICIPANT-MACRO"
)
print("=" * 110)

print(
    direction_summary[
        [
            "direction_name",
            "mode",
            "participants",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("REPORT TABLE")
print("=" * 110)

print(
    report_table
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print(
    "PRIMARY PAIRED STATISTICS"
)
print("=" * 110)

primary = effects.loc[
    effects[
        "metric"
    ].isin(
        [
            "macro_f1",
            "balanced_accuracy",
        ]
    ),
    [
        "direction_name",
        "contrast",
        "metric",
        "participants",
        "mean_delta",
        "median_delta",
        "wins",
        "ties",
        "losses",
        "p_value",
        "p_holm",
    ],
]

print(
    primary
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("TRAINING")
print("=" * 110)

training_summary = (
    runs
    .groupby(
        [
            "direction_name",
            "mode",
        ]
    )
    .agg(
        runs=(
            "run_id",
            "size",
        ),
        epochs_mean=(
            "epochs_trained",
            "mean",
        ),
        epochs_min=(
            "epochs_trained",
            "min",
        ),
        epochs_max=(
            "epochs_trained",
            "max",
        ),
        training_seconds=(
            "training_time_seconds",
            "sum",
        ),
    )
    .reset_index()
)

print(
    training_summary
    .round(3)
    .to_string(
        index=False
    )
)


print()
print(
    "Analysis artifacts saved to:"
)
print(ROOT)

print()
print("=" * 110)
print("ANALYSIS COMPLETE")
print("=" * 110)
