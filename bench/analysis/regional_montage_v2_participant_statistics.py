from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    REPO_ROOT
    / "benchmark_results"
    / "xgboost_regional_montage_transfer_v2"
)

OUT = ROOT / "participant_statistics"
OUT.mkdir(parents=True, exist_ok=True)

PARTICIPANTS_PATH = ROOT / "full_participant_metrics.csv"
PREDICTIONS_PATH = ROOT / "full_predictions.parquet"
SUMMARY_PATH = ROOT / "full_summary.json"

BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 2026

PROFILES = [
    "full_14",
    "reduced_12",
    "regional_10",
    "coverage_8",
    "coverage_6",
]

PRIMARY_PROFILES = [
    "regional_10",
    "coverage_8",
    "coverage_6",
]

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
]

PM_ORDER = [
    "attention",
    "engagement",
    "excitement",
    "stress",
    "relaxation",
    "interest",
    "focus",
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def percentile_bootstrap_mean_ci(values, seed):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    means = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float64)

    # Participant is the bootstrap unit.
    for i in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(x, size=len(x), replace=True)
        means[i] = sample.mean()

    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def matched_rank_biserial(values):
    """
    Matched-pairs rank-biserial correlation.

    Zeros are excluded, matching Wilcoxon zero_method='wilcox'.
    Positive value means target profile > full_14.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x != 0]

    if len(x) == 0:
        return 0.0

    ranks = rankdata(np.abs(x), method="average")

    w_plus = float(ranks[x > 0].sum())
    w_minus = float(ranks[x < 0].sum())

    denom = w_plus + w_minus

    if denom == 0:
        return 0.0

    return (w_plus - w_minus) / denom


def wilcoxon_two_sided(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    if np.all(x == 0):
        return 1.0

    result = wilcoxon(
        x,
        zero_method="wilcox",
        alternative="two-sided",
        correction=False,
        method="auto",
    )

    return float(result.pvalue)


def holm_adjust(p_values):
    """
    Holm step-down adjustment.
    """
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(len(p), np.nan, dtype=np.float64)

    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out

    order = valid[np.argsort(p[valid])]
    m = len(order)

    previous = 0.0

    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p[idx]
        adjusted = max(adjusted, previous)
        adjusted = min(adjusted, 1.0)

        out[idx] = adjusted
        previous = adjusted

    return out


def participant_level_table(df, metric):
    """
    Average available PMs within participant first.
    """
    return (
        df.groupby(
            ["subject_id", "profile"],
            as_index=False,
            sort=True,
        )[metric]
        .mean()
    )


def paired_delta(participant_table, target_profile, metric):
    base = (
        participant_table[
            participant_table["profile"] == "full_14"
        ][["subject_id", metric]]
        .rename(columns={metric: "baseline"})
    )

    target = (
        participant_table[
            participant_table["profile"] == target_profile
        ][["subject_id", metric]]
        .rename(columns={metric: "target"})
    )

    paired = base.merge(
        target,
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )

    paired["delta"] = paired["target"] - paired["baseline"]

    return paired


def statistics_row(paired, profile, metric, seed):
    delta = paired["delta"].to_numpy(dtype=np.float64)

    ci_lo, ci_hi = percentile_bootstrap_mean_ci(
        delta,
        seed=seed,
    )

    return {
        "profile": profile,
        "metric": metric,
        "n_participants": int(len(delta)),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "bootstrap_mean_ci_low": ci_lo,
        "bootstrap_mean_ci_high": ci_hi,
        "wilcoxon_p_raw": wilcoxon_two_sided(delta),
        "rank_biserial": float(matched_rank_biserial(delta)),
        "positive_count": int(np.sum(delta > 0)),
        "negative_count": int(np.sum(delta < 0)),
        "zero_count": int(np.sum(delta == 0)),
        "positive_fraction": float(np.mean(delta > 0)),
    }


# ------------------------------------------------------------
# Load and audit
# ------------------------------------------------------------

participants = pd.read_csv(PARTICIPANTS_PATH)
predictions = pd.read_parquet(PREDICTIONS_PATH)
summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

print("=== INPUT AUDIT ===")
print("participant rows:", len(participants))
print("prediction rows:", len(predictions))
print("subjects:", participants["subject_id"].nunique())
print("PM:", sorted(participants["pm"].unique()))
print("profiles:", sorted(participants["profile"].unique()))

assert set(participants["pm"]) == set(PM_ORDER)
assert set(participants["profile"]) == set(PROFILES)

if summary["completed_xgboost_trainings"] != 35:
    raise RuntimeError("Expected 35 completed XGBoost trainings")

if summary["completed_prediction_evaluations"] != 175:
    raise RuntimeError("Expected 175 evaluations")

if not summary["same_booster_all_profiles"]:
    raise RuntimeError("Booster identity invariant failed")

if not summary["exact_profile_sample_identity"]:
    raise RuntimeError("Profile sample identity invariant failed")


# ------------------------------------------------------------
# Participant × PM coverage
# ------------------------------------------------------------

coverage = (
    participants.groupby(
        ["subject_id", "profile"],
        as_index=False,
    )["pm"]
    .nunique()
    .rename(columns={"pm": "available_pm_count"})
)

base_coverage = coverage[
    coverage["profile"] == "full_14"
].copy()

complete_subjects = set(
    base_coverage.loc[
        base_coverage["available_pm_count"] == 7,
        "subject_id",
    ]
)

print("subjects with all 7 PM:", len(complete_subjects))


# ------------------------------------------------------------
# Primary participant-level inference
# 3 profiles × 3 metrics = 9 tests, one Holm family.
# ------------------------------------------------------------

primary_rows = []
paired_delta_frames = []

test_seed = BOOTSTRAP_SEED

for profile in PRIMARY_PROFILES:
    for metric in METRICS:
        table = participant_level_table(participants, metric)
        paired = paired_delta(table, profile, metric)

        temp = paired.copy()
        temp["profile"] = profile
        temp["metric"] = metric
        paired_delta_frames.append(temp)

        primary_rows.append(
            statistics_row(
                paired,
                profile,
                metric,
                seed=test_seed,
            )
        )

        test_seed += 1

primary = pd.DataFrame(primary_rows)

primary["wilcoxon_p_holm"] = holm_adjust(
    primary["wilcoxon_p_raw"].to_numpy()
)

primary["holm_significant_0_05"] = (
    primary["wilcoxon_p_holm"] < 0.05
)

primary.to_csv(
    OUT / "primary_statistics.csv",
    index=False,
)

pd.concat(
    paired_delta_frames,
    ignore_index=True,
).to_csv(
    OUT / "participant_deltas.csv",
    index=False,
)


# ------------------------------------------------------------
# Secondary descriptive comparison: reduced_12 vs full_14
# Not included in the primary Holm family.
# ------------------------------------------------------------

secondary_rows = []

for i, metric in enumerate(METRICS):
    table = participant_level_table(participants, metric)
    paired = paired_delta(table, "reduced_12", metric)

    secondary_rows.append(
        statistics_row(
            paired,
            "reduced_12",
            metric,
            seed=3000 + i,
        )
    )

secondary = pd.DataFrame(secondary_rows)

secondary.to_csv(
    OUT / "reduced_12_secondary_statistics.csv",
    index=False,
)


# ------------------------------------------------------------
# Complete-7-PM sensitivity analysis
# ------------------------------------------------------------

complete = participants[
    participants["subject_id"].isin(complete_subjects)
].copy()

complete_rows = []

for profile in PRIMARY_PROFILES:
    for metric in METRICS:
        table = participant_level_table(complete, metric)
        paired = paired_delta(table, profile, metric)

        complete_rows.append(
            statistics_row(
                paired,
                profile,
                metric,
                seed=4000 + len(complete_rows),
            )
        )

complete_stats = pd.DataFrame(complete_rows)

complete_stats["wilcoxon_p_holm"] = holm_adjust(
    complete_stats["wilcoxon_p_raw"].to_numpy()
)

complete_stats.to_csv(
    OUT / "complete_7pm_sensitivity.csv",
    index=False,
)


# ------------------------------------------------------------
# Per-fold descriptive effects
# ------------------------------------------------------------

fold_rows = []

for fold, part in participants.groupby("outer_fold"):
    for profile in PROFILES[1:]:
        for metric in METRICS:
            subject_table = participant_level_table(part, metric)
            paired = paired_delta(subject_table, profile, metric)

            fold_rows.append({
                "outer_fold": int(fold),
                "profile": profile,
                "metric": metric,
                "n_participants": len(paired),
                "mean_delta": float(paired["delta"].mean()),
                "median_delta": float(paired["delta"].median()),
            })

pd.DataFrame(fold_rows).to_csv(
    OUT / "per_fold_descriptive_deltas.csv",
    index=False,
)


# ------------------------------------------------------------
# Missing-class audit from exact predictions
# ------------------------------------------------------------

class_rows = []

full_pred = predictions[
    predictions["profile"] == "full_14"
]

for (subject_id, pm), part in full_pred.groupby(
    ["subject_id", "pm"],
    sort=True,
):
    classes = sorted(part["y_true"].unique().tolist())

    class_rows.append({
        "subject_id": subject_id,
        "pm": pm,
        "n_samples": len(part),
        "n_true_classes": len(classes),
        "true_classes": ",".join(map(str, classes)),
    })

class_coverage = pd.DataFrame(class_rows)

class_coverage.to_csv(
    OUT / "true_class_coverage.csv",
    index=False,
)

class_counts = (
    class_coverage["n_true_classes"]
    .value_counts()
    .sort_index()
)

print("\n=== TRUE-CLASS COVERAGE ===")
for n_classes, count in class_counts.items():
    print(f"{n_classes} true classes: {count}")


# ------------------------------------------------------------
# Fixed-label Balanced Accuracy sensitivity
#
# BA = mean recall across the predefined classes 0,1,2,
# including zero recall when a class is absent in y_true.
# ------------------------------------------------------------

fixed_ba_rows = []

for (profile, subject_id, pm), part in predictions.groupby(
    ["profile", "subject_id", "pm"],
    sort=True,
):
    fixed_ba = recall_score(
        part["y_true"],
        part["y_pred"],
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    fixed_ba_rows.append({
        "profile": profile,
        "subject_id": subject_id,
        "pm": pm,
        "fixed_label_balanced_accuracy": float(fixed_ba),
    })

fixed_ba_pm = pd.DataFrame(fixed_ba_rows)

fixed_ba_pm.to_csv(
    OUT / "fixed_label_balanced_accuracy_participant_pm.csv",
    index=False,
)

fixed_ba_subject = (
    fixed_ba_pm.groupby(
        ["subject_id", "profile"],
        as_index=False,
    )["fixed_label_balanced_accuracy"]
    .mean()
)

fixed_ba_stats_rows = []

for profile in PRIMARY_PROFILES:
    base = (
        fixed_ba_subject[
            fixed_ba_subject["profile"] == "full_14"
        ][["subject_id", "fixed_label_balanced_accuracy"]]
        .rename(columns={
            "fixed_label_balanced_accuracy": "baseline"
        })
    )

    target = (
        fixed_ba_subject[
            fixed_ba_subject["profile"] == profile
        ][["subject_id", "fixed_label_balanced_accuracy"]]
        .rename(columns={
            "fixed_label_balanced_accuracy": "target"
        })
    )

    paired = base.merge(
        target,
        on="subject_id",
        validate="one_to_one",
    )

    paired["delta"] = paired["target"] - paired["baseline"]

    fixed_ba_stats_rows.append(
        statistics_row(
            paired,
            profile,
            "fixed_label_balanced_accuracy",
            seed=5000 + len(fixed_ba_stats_rows),
        )
    )

fixed_ba_stats = pd.DataFrame(fixed_ba_stats_rows)

fixed_ba_stats["wilcoxon_p_holm"] = holm_adjust(
    fixed_ba_stats["wilcoxon_p_raw"].to_numpy()
)

fixed_ba_stats.to_csv(
    OUT / "fixed_label_balanced_accuracy_statistics.csv",
    index=False,
)


# ------------------------------------------------------------
# Final compact summary
# ------------------------------------------------------------

summary_out = {
    "status": "participant_statistics_complete",
    "source_result_status": summary["result_status"],
    "protocol_hash": summary.get("protocol_hash"),
    "plan_hash": summary.get("plan_hash"),
    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "inferential_unit": "participant",
    "pm_aggregation": "mean across available PM within participant",
    "primary_profiles": PRIMARY_PROFILES,
    "primary_metrics": METRICS,
    "primary_holm_family_size": 9,
    "wilcoxon": {
        "alternative": "two-sided",
        "zero_method": "wilcox",
    },
    "subjects": int(participants["subject_id"].nunique()),
    "complete_7pm_subjects": int(len(complete_subjects)),
    "true_class_coverage": {
        str(int(k)): int(v)
        for k, v in class_counts.items()
    },
}

(OUT / "statistics_summary.json").write_text(
    json.dumps(
        summary_out,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

show_cols = [
    "profile",
    "metric",
    "n_participants",
    "mean_delta",
    "median_delta",
    "bootstrap_mean_ci_low",
    "bootstrap_mean_ci_high",
    "wilcoxon_p_raw",
    "wilcoxon_p_holm",
    "rank_biserial",
    "positive_count",
    "negative_count",
]

print("\n=== PRIMARY STATISTICS ===")
print(
    primary[show_cols]
    .to_string(index=False)
)

print("\n=== REDUCED_12 SECONDARY ===")
print(
    secondary[
        [
            "metric",
            "n_participants",
            "mean_delta",
            "bootstrap_mean_ci_low",
            "bootstrap_mean_ci_high",
            "wilcoxon_p_raw",
            "rank_biserial",
        ]
    ].to_string(index=False)
)

print("\n=== COMPLETE-7PM SENSITIVITY ===")
print(
    complete_stats[show_cols]
    .to_string(index=False)
)

print("\n=== FIXED-LABEL BA SENSITIVITY ===")
print(
    fixed_ba_stats[
        [
            "profile",
            "n_participants",
            "mean_delta",
            "median_delta",
            "bootstrap_mean_ci_low",
            "bootstrap_mean_ci_high",
            "wilcoxon_p_raw",
            "wilcoxon_p_holm",
            "rank_biserial",
        ]
    ].to_string(index=False)
)

print("\nOutput:", OUT)
print("No model fitting was performed.")
