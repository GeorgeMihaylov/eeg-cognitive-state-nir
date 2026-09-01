from __future__ import annotations

import numpy as np
import pandas as pd

from bench.analysis.ordinal_transformer_multiseed_statistics import (
    average_subject_metrics_across_seeds,
    build_multiseed_hypotheses,
    seed_consistency_table,
)
from bench.analysis.ordinal_transformer_statistics import SUBJECT_METRICS


def _subject_seed_metrics() -> pd.DataFrame:
    rows = []
    for method in ("categorical", "coral", "corn"):
        for group in ("eeg_only", "eeg_pow"):
            for seed in (7, 42, 123):
                for subject in range(53):
                    base = 0.35 + subject / 1000 + seed / 100000
                    ordinal_gain = 0.02 if method == "coral" else 0.01 if method == "corn" else 0.0
                    row = {
                        "run_key": f"{method}_{group}",
                        "method": method,
                        "feature_group": group,
                        "seed": seed,
                        "subject_id": f"s-{subject:02d}",
                        "fold": subject % 5 + 1,
                        "source_membership": "synthetic",
                    }
                    row.update({metric: base for metric in SUBJECT_METRICS})
                    row["ordinal_mae"] = 0.8 - ordinal_gain + subject / 10000
                    row["severe_error_rate"] = 0.3 - ordinal_gain / 2 + subject / 10000
                    row["balanced_accuracy"] = base - ordinal_gain / 4
                    row["macro_f1"] = base - ordinal_gain / 4
                    rows.append(row)
    return pd.DataFrame(rows)


def test_subject_seed_rows_are_averaged_before_inference() -> None:
    subject_seed = _subject_seed_metrics()
    averaged = average_subject_metrics_across_seeds(subject_seed)
    assert len(subject_seed) == 954
    assert len(averaged) == 318
    assert averaged["seeds_averaged"].eq(3).all()
    assert averaged.groupby(["run_key", "subject_id"]).size().eq(1).all()


def test_holm_families_keep_primary_and_secondary_separate() -> None:
    averaged = average_subject_metrics_across_seeds(_subject_seed_metrics())
    primary, secondary = build_multiseed_hypotheses(
        averaged, n_resamples=200, random_state=42
    )
    assert len(primary) == 8
    assert len(secondary) == 24
    assert {row["family"] for row in primary} == {
        "primary_eeg_only", "primary_eeg_pow"
    }
    assert {row["family"] for row in secondary} == {
        "secondary_eeg_only", "secondary_eeg_pow"
    }
    assert all("holm_adjusted_p_value" in row for row in primary + secondary)
    assert all(row["n_valid_pairs"] == 53 for row in primary + secondary)


def test_seed_consistency_uses_oriented_improvement() -> None:
    rows = seed_consistency_table(_subject_seed_metrics())
    coral_mae = next(
        row for row in rows
        if row["candidate"] == "coral"
        and row["feature_group"] == "eeg_pow"
        and row["metric"] == "ordinal_mae"
    )
    assert coral_mae["positive_seeds"] == 3
    assert coral_mae["direction_label"] == "positive_in_3_of_3"
    assert np.isclose(coral_mae["seed_7_mean_improvement"], 0.02)
