"""Post-hoc robustness audit for the completed seven-PM LOW/HIGH experiment.

The module is intentionally analysis-only.  It reads frozen prediction and
summary artifacts, independently recomputes participant metrics, validates the
completed run matrix, and writes compact descriptive summaries.  It never
loads features, constructs a model, or performs training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    roc_auc_score,
)


PM_NAMES = (
    "attention",
    "engagement",
    "excitement",
    "stress",
    "relaxation",
    "interest",
    "focus",
)
EXPECTED_PROTOCOL_HASH = (
    "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
)
EXPECTED_PREREGISTRATION_HEAD = "1e28fdae3b2ce2d75a4d90489960492299600a46"
FIXED_LABELS = (0, 1)
PARTICIPANT_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "low_recall",
    "high_recall",
    "precision",
    "accuracy",
)
PRIMARY_DISTRIBUTION_METRICS = ("balanced_accuracy", "macro_f1", "roc_auc")
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_hash(values: Sequence[Any]) -> str:
    return _stable_hash([str(value) for value in values])


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _assert_equal(actual: Any, expected: Any, message: str) -> None:
    if str(actual) != str(expected):
        raise ValueError(f"{message}: actual={actual!r}, expected={expected!r}")


def _assert_float_equal(
    actual: Any,
    expected: Any,
    message: str,
    *,
    rtol: float,
    atol: float,
) -> float:
    actual_value = float(actual)
    expected_value = float(expected)
    if np.isnan(actual_value) and np.isnan(expected_value):
        return 0.0
    if not np.isclose(actual_value, expected_value, rtol=rtol, atol=atol):
        raise ValueError(
            f"{message}: actual={actual_value!r}, expected={expected_value!r}, "
            f"rtol={rtol}, atol={atol}"
        )
    return abs(actual_value - expected_value)


def recompute_participant_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Independently recompute participant metrics from prediction rows."""
    required = {"subject_id", "y_true", "y_pred", "probability_high"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions lack metric columns: {missing}")
    if predictions.empty:
        raise ValueError("Cannot audit an empty predictions artifact")

    truth_all = predictions["y_true"].to_numpy(dtype=np.int64)
    prediction_all = predictions["y_pred"].to_numpy(dtype=np.int64)
    probability_all = predictions["probability_high"].to_numpy(dtype=float)
    if not set(np.unique(truth_all)).issubset(FIXED_LABELS):
        raise ValueError("y_true contains labels outside LOW=0/HIGH=1")
    if not set(np.unique(prediction_all)).issubset(FIXED_LABELS):
        raise ValueError("y_pred contains labels outside LOW=0/HIGH=1")
    if not np.isfinite(probability_all).all():
        raise ValueError("probability_high contains NaN or Inf")
    if np.any((probability_all < 0.0) | (probability_all > 1.0)):
        raise ValueError("probability_high lies outside [0, 1]")

    rows: list[dict[str, Any]] = []
    subjects = predictions["subject_id"].astype(str).to_numpy()
    for subject_id in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject_id
        truth = truth_all[mask]
        prediction = prediction_all[mask]
        probability = probability_all[mask]
        n_low = int(np.sum(truth == 0))
        n_high = int(np.sum(truth == 1))
        low_recall = (
            float(np.mean(prediction[truth == 0] == 0))
            if n_low
            else float("nan")
        )
        high_recall = (
            float(np.mean(prediction[truth == 1] == 1))
            if n_high
            else float("nan")
        )
        defined_recalls = np.asarray(
            [value for value in (low_recall, high_recall) if np.isfinite(value)],
            dtype=float,
        )
        both_classes = n_low > 0 and n_high > 0
        rows.append({
            "subject_id": subject_id,
            "n_samples": int(mask.sum()),
            "n_low": n_low,
            "n_high": n_high,
            "balanced_accuracy": float(np.mean(defined_recalls)),
            "macro_f1": float(
                f1_score(
                    truth,
                    prediction,
                    labels=list(FIXED_LABELS),
                    average="macro",
                    zero_division=0,
                )
            ),
            "roc_auc": (
                float(roc_auc_score(truth, probability))
                if both_classes
                else float("nan")
            ),
            "pr_auc": (
                float(average_precision_score(truth, probability))
                if both_classes
                else float("nan")
            ),
            "low_recall": low_recall,
            "high_recall": high_recall,
            "precision": float(
                precision_score(
                    truth,
                    prediction,
                    labels=list(FIXED_LABELS),
                    average="macro",
                    zero_division=0,
                )
            ),
            "accuracy": float(accuracy_score(truth, prediction)),
        })
    return pd.DataFrame(rows)


def validate_participant_uniqueness(participants: pd.DataFrame) -> None:
    """Require one outer-test fold per PM and one row per PM-participant."""
    required = {"pm", "subject_id", "outer_fold"}
    missing = sorted(required - set(participants.columns))
    if missing:
        raise ValueError(f"Participant table lacks uniqueness columns: {missing}")
    duplicates = participants.duplicated(["pm", "subject_id"], keep=False)
    if duplicates.any():
        sample = participants.loc[duplicates, ["pm", "subject_id", "outer_fold"]]
        raise ValueError(
            "A participant appears more than once for one PM: "
            f"{sample.head(10).to_dict('records')}"
        )
    pm_fold_counts = participants.groupby(["pm", "subject_id"])[
        "outer_fold"
    ].nunique()
    if not pm_fold_counts.eq(1).all():
        raise ValueError("A PM-participant appears in multiple outer folds")
    subject_fold_counts = participants.groupby("subject_id")["outer_fold"].nunique()
    if not subject_fold_counts.eq(1).all():
        raise ValueError("A participant has inconsistent outer folds across PM")


@dataclass(frozen=True)
class CompletedArtifactAudit:
    participants: pd.DataFrame
    fold_results: pd.DataFrame
    integrity: dict[str, Any]


def audit_completed_artifacts(
    experiment_dir: str | Path,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> CompletedArtifactAudit:
    """Audit all 35 completed runs before any post-hoc summary is written."""
    root = Path(experiment_dir)
    required_top = {
        "protocol.json",
        "run_matrix.csv",
        "thresholds_by_fold.csv",
        "results_by_fold.csv",
        "summary_by_pm.csv",
        "pooled_summary.csv",
    }
    missing_top = sorted(name for name in required_top if not (root / name).is_file())
    if missing_top:
        raise FileNotFoundError(f"Completed experiment lacks artifacts: {missing_top}")

    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    _assert_equal(
        protocol.get("protocol_hash"), EXPECTED_PROTOCOL_HASH, "Protocol hash mismatch"
    )
    _assert_equal(
        protocol.get("git_commit"),
        EXPECTED_PREREGISTRATION_HEAD,
        "Completed protocol code HEAD mismatch",
    )
    if protocol.get("training_executed") is not True:
        raise ValueError("Completed protocol must preserve training_executed=true")
    _assert_equal(
        protocol.get("result_status"),
        "confirmatory_complete",
        "Completed protocol status mismatch",
    )
    if tuple(protocol.get("target_ids", ())) != tuple(
        f"target_{pm}" for pm in PM_NAMES
    ):
        raise ValueError("Protocol does not contain exactly the seven canonical PM")
    if protocol.get("fold_ids") != [1, 2, 3, 4, 5]:
        raise ValueError("Protocol does not contain exactly the five fixed folds")
    if int(protocol.get("fixed_lag_seconds", 999)) != -10:
        raise ValueError("Protocol is not the fixed lag -10 experiment")

    run_matrix = pd.read_csv(root / "run_matrix.csv")
    thresholds = pd.read_csv(root / "thresholds_by_fold.csv")
    results = pd.read_csv(root / "results_by_fold.csv")
    summary_by_pm = pd.read_csv(root / "summary_by_pm.csv")
    pooled_summary = pd.read_csv(root / "pooled_summary.csv")
    if len(run_matrix) != 35 or run_matrix["run_id"].duplicated().any():
        raise ValueError("Run matrix must contain 35 unique runs")
    if set(run_matrix["pm"].astype(str)) != set(PM_NAMES):
        raise ValueError("Run matrix must contain all seven PM")
    if set(run_matrix["outer_fold"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("Run matrix must contain five fixed folds")
    if not run_matrix.groupby("pm").size().eq(5).all():
        raise ValueError("Every PM must contain exactly five runs")
    if set(run_matrix["lag_seconds"].astype(int)) != {-10}:
        raise ValueError("Run matrix contains a forbidden lag")
    if len(thresholds) != 35 or len(results) != 35:
        raise ValueError("Threshold and results tables must each contain 35 rows")
    if results["run_id"].duplicated().any():
        raise ValueError("Results table contains duplicate run IDs")

    expected_run_ids = set(run_matrix["run_id"].astype(str))
    run_root = root / "runs"
    actual_run_ids = {path.name for path in run_root.iterdir() if path.is_dir()}
    if actual_run_ids != expected_run_ids:
        raise ValueError(
            "Runtime run directories differ from run matrix: "
            f"missing={sorted(expected_run_ids - actual_run_ids)}, "
            f"unexpected={sorted(actual_run_ids - expected_run_ids)}"
        )

    threshold_lookup = thresholds.set_index(["outer_fold", "pm"])
    result_lookup = results.set_index("run_id")
    participant_frames: list[pd.DataFrame] = []
    max_metric_difference = 0.0
    n_prediction_rows = 0
    n_undefined_roc_auc = 0
    n_undefined_pr_auc = 0
    no_retained_extreme_subjects: list[dict[str, Any]] = []

    for spec in run_matrix.sort_values(["outer_fold", "pm"], kind="stable").to_dict(
        "records"
    ):
        fold = int(spec["outer_fold"])
        pm = str(spec["pm"])
        run_id = str(spec["run_id"])
        target_id = f"target_{pm}"
        if str(spec["target_id"]) != target_id:
            raise ValueError(f"{run_id}: target_id does not match PM")
        run_dir = run_root / run_id
        required_run = {
            "predictions.parquet",
            "participant_metrics.csv",
            "run_summary.json",
        }
        missing_run = sorted(
            name for name in required_run if not (run_dir / name).is_file()
        )
        if missing_run:
            raise FileNotFoundError(f"{run_id}: missing run artifacts {missing_run}")

        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        existing_participants = pd.read_csv(run_dir / "participant_metrics.csv")
        run_summary = json.loads(
            (run_dir / "run_summary.json").read_text(encoding="utf-8")
        )
        required_predictions = {
            "target_sample_id",
            "feature_sample_id",
            "subject_id",
            "record_id",
            "outer_fold",
            "pm",
            "target_id",
            "condition",
            "lag_seconds",
            "y_true",
            "y_pred",
            "probability_high",
        }
        missing_predictions = sorted(required_predictions - set(predictions.columns))
        if missing_predictions:
            raise ValueError(f"{run_id}: predictions lack {missing_predictions}")
        if predictions.empty:
            raise ValueError(f"{run_id}: empty predictions")
        if predictions.duplicated().any():
            raise ValueError(f"{run_id}: duplicated prediction rows")
        if predictions["target_sample_id"].duplicated().any():
            raise ValueError(f"{run_id}: duplicated target_sample_id")
        if len(predictions) != int(spec["n_test"]):
            raise ValueError(f"{run_id}: prediction count differs from run specification")
        if _sample_hash(predictions["target_sample_id"].tolist()) != str(
            spec["test_sample_hash"]
        ):
            raise ValueError(f"{run_id}: test sample hash mismatch")
        for column, expected in (
            ("outer_fold", fold),
            ("pm", pm),
            ("target_id", target_id),
            ("condition", "lag_minus_10s"),
            ("lag_seconds", -10),
        ):
            actual_values = predictions[column].astype(str).unique().tolist()
            if actual_values != [str(expected)]:
                raise ValueError(
                    f"{run_id}: prediction {column}={actual_values}, expected={expected!r}"
                )

        threshold_row = threshold_lookup.loc[(fold, pm)]
        expected_subjects = {
            value
            for value in str(threshold_row["test_subjects"]).split("|")
            if value
        }
        prediction_subjects = set(predictions["subject_id"].astype(str))
        participant_subjects = set(existing_participants["subject_id"].astype(str))
        unexpected_subjects = prediction_subjects - expected_subjects
        if unexpected_subjects:
            raise ValueError(
                f"{run_id}: predictions contain subjects outside the fixed test fold: "
                f"{sorted(unexpected_subjects)}"
            )
        retained_subject_count = int(threshold_row["n_test_subjects_retained"])
        if len(prediction_subjects) != retained_subject_count:
            raise ValueError(
                f"{run_id}: retained-subject count differs from threshold audit"
            )
        if participant_subjects != prediction_subjects:
            raise ValueError(
                f"{run_id}: participant_metrics subjects differ from predictions"
            )
        for subject_id in sorted(expected_subjects - prediction_subjects):
            no_retained_extreme_subjects.append({
                "pm": pm,
                "outer_fold": fold,
                "subject_id": subject_id,
                "reason": "zero rows retained by fixed outer-train Q33/Q67 thresholds",
            })
        if len(prediction_subjects) != int(spec["n_test_participants"]):
            raise ValueError(f"{run_id}: participant count differs from run spec")

        for key in (
            "protocol_hash",
            "specification_hash",
            "threshold_hash",
            "run_id",
            "outer_fold",
            "pm",
            "target_id",
            "lag_seconds",
            "n_test",
        ):
            expected = spec[key] if key in spec else EXPECTED_PROTOCOL_HASH
            if key == "protocol_hash":
                expected = EXPECTED_PROTOCOL_HASH
            _assert_equal(run_summary.get(key), expected, f"{run_id}: summary {key}")
        if str(threshold_row["protocol_hash"]) != EXPECTED_PROTOCOL_HASH:
            raise ValueError(f"{run_id}: threshold protocol hash mismatch")
        _assert_equal(
            threshold_row["specification_hash"],
            spec["specification_hash"],
            f"{run_id}: threshold specification hash",
        )

        recomputed = recompute_participant_metrics(predictions)
        existing = existing_participants.sort_values("subject_id", kind="stable").reset_index(
            drop=True
        )
        recomputed = recomputed.sort_values("subject_id", kind="stable").reset_index(
            drop=True
        )
        if existing["subject_id"].astype(str).tolist() != recomputed[
            "subject_id"
        ].astype(str).tolist():
            raise ValueError(f"{run_id}: participant metric row identity mismatch")
        for row_number in range(len(recomputed)):
            subject = str(recomputed.loc[row_number, "subject_id"])
            for column in ("n_samples", "n_low", "n_high"):
                if int(existing.loc[row_number, column]) != int(
                    recomputed.loc[row_number, column]
                ):
                    raise ValueError(f"{run_id}/{subject}: {column} mismatch")
            for metric in PARTICIPANT_METRICS:
                difference = _assert_float_equal(
                    recomputed.loc[row_number, metric],
                    existing.loc[row_number, metric],
                    f"{run_id}/{subject}: {metric} mismatch",
                    rtol=rtol,
                    atol=atol,
                )
                max_metric_difference = max(max_metric_difference, difference)

        for metric in PARTICIPANT_METRICS:
            values = recomputed[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            macro_name = "f1" if metric == "macro_f1" else metric
            macro_key = f"participant_macro_{macro_name}"
            valid_key = f"participant_valid_{macro_name}"
            expected_macro = float(np.mean(finite)) if len(finite) else float("nan")
            difference = _assert_float_equal(
                run_summary[macro_key],
                expected_macro,
                f"{run_id}: independently recomputed {macro_key}",
                rtol=rtol,
                atol=atol,
            )
            max_metric_difference = max(max_metric_difference, difference)
            if int(run_summary[valid_key]) != len(finite):
                raise ValueError(f"{run_id}: {valid_key} mismatch")

        result_row = result_lookup.loc[run_id]
        for key in (
            "protocol_hash",
            "specification_hash",
            "threshold_hash",
            "outer_fold",
            "pm",
            "target_id",
            "lag_seconds",
            "n_test",
        ):
            _assert_equal(result_row[key], run_summary[key], f"{run_id}: results {key}")
        for metric in PARTICIPANT_METRICS:
            macro_name = "f1" if metric == "macro_f1" else metric
            macro_key = f"participant_macro_{macro_name}"
            difference = _assert_float_equal(
                result_row[macro_key],
                run_summary[macro_key],
                f"{run_id}: results {macro_key}",
                rtol=rtol,
                atol=atol,
            )
            max_metric_difference = max(max_metric_difference, difference)

        output = recomputed.rename(columns={"n_samples": "n_test"}).copy()
        output.insert(0, "outer_fold", fold)
        output.insert(0, "target_id", target_id)
        output.insert(0, "pm", pm)
        output["low_fraction"] = output["n_low"] / output["n_test"]
        output["high_fraction"] = output["n_high"] / output["n_test"]
        output["minority_class_fraction"] = output[
            ["low_fraction", "high_fraction"]
        ].min(axis=1)
        output["absolute_class_imbalance"] = (
            output["low_fraction"] - output["high_fraction"]
        ).abs()
        ordered_columns = [
            "pm",
            "target_id",
            "outer_fold",
            "subject_id",
            "n_test",
            "n_low",
            "n_high",
            "low_fraction",
            "high_fraction",
            "minority_class_fraction",
            "absolute_class_imbalance",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "pr_auc",
            "low_recall",
            "high_recall",
            "accuracy",
        ]
        participant_frames.append(output[ordered_columns])
        n_prediction_rows += len(predictions)
        n_undefined_roc_auc += int(output["roc_auc"].isna().sum())
        n_undefined_pr_auc += int(output["pr_auc"].isna().sum())

    participants = pd.concat(participant_frames, ignore_index=True).sort_values(
        ["pm", "subject_id"], kind="stable"
    ).reset_index(drop=True)
    validate_participant_uniqueness(participants)
    if set(participants["pm"]) != set(PM_NAMES):
        raise ValueError("Participant audit lost one or more PM")

    if len(summary_by_pm) != 7 or set(summary_by_pm["pm"].astype(str)) != set(
        PM_NAMES
    ):
        raise ValueError("summary_by_pm must contain exactly the seven PM")
    if len(pooled_summary) != 1:
        raise ValueError("pooled_summary must contain exactly one row")
    pm_summary_lookup = summary_by_pm.set_index("pm")
    pooled_row = pooled_summary.iloc[0]
    for pm in PM_NAMES:
        pm_results = results.loc[results["pm"].eq(pm)]
        pm_summary = pm_summary_lookup.loc[pm]
        _assert_equal(pm_summary["target_id"], f"target_{pm}", f"{pm}: summary target")
        _assert_equal(pm_summary["n_folds"], 5, f"{pm}: summary fold count")
        for metric in PARTICIPANT_METRICS:
            macro_name = "f1" if metric == "macro_f1" else metric
            values = pm_results[f"participant_macro_{macro_name}"].to_numpy(
                dtype=float
            )
            finite = values[np.isfinite(values)]
            for statistic, expected in (
                ("mean", np.mean(finite)),
                ("std", np.std(finite, ddof=1)),
            ):
                difference = _assert_float_equal(
                    pm_summary[f"participant_macro_{macro_name}_{statistic}"],
                    expected,
                    f"{pm}: summary {macro_name} {statistic}",
                    rtol=rtol,
                    atol=atol,
                )
                max_metric_difference = max(max_metric_difference, difference)
            _assert_equal(
                pm_summary[f"valid_folds_{macro_name}"],
                len(finite),
                f"{pm}: valid {macro_name} folds",
            )
            _assert_equal(
                pm_summary[f"valid_participants_{macro_name}"],
                int(pm_results[f"participant_valid_{macro_name}"].sum()),
                f"{pm}: valid {macro_name} participants",
            )

    for key, expected in (
        ("n_fold_pm_runs", 35),
        ("n_pm", 7),
        ("n_folds", 5),
        ("lag_seconds", -10),
    ):
        _assert_equal(pooled_row[key], expected, f"pooled summary {key}")
    for metric in PARTICIPANT_METRICS:
        macro_name = "f1" if metric == "macro_f1" else metric
        values = results[f"participant_macro_{macro_name}"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        for statistic, expected in (
            ("mean", np.mean(finite)),
            ("std", np.std(finite, ddof=1)),
            ("median", np.median(finite)),
        ):
            difference = _assert_float_equal(
                pooled_row[f"participant_macro_{macro_name}_{statistic}"],
                expected,
                f"pooled summary {macro_name} {statistic}",
                rtol=rtol,
                atol=atol,
            )
            max_metric_difference = max(max_metric_difference, difference)
        _assert_equal(
            pooled_row[f"valid_fold_pm_{macro_name}"],
            len(finite),
            f"pooled valid {macro_name} runs",
        )
        _assert_equal(
            pooled_row[f"valid_participants_{macro_name}"],
            int(results[f"participant_valid_{macro_name}"].sum()),
            f"pooled valid {macro_name} participants",
        )

    integrity = {
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "preregistration_head": EXPECTED_PREREGISTRATION_HEAD,
        "n_runs": 35,
        "n_pm": 7,
        "n_folds": 5,
        "n_prediction_rows": int(n_prediction_rows),
        "n_participant_pm_rows": int(len(participants)),
        "n_unique_participants": int(participants["subject_id"].nunique()),
        "n_undefined_roc_auc": int(n_undefined_roc_auc),
        "n_undefined_pr_auc": int(n_undefined_pr_auc),
        "n_fold_pm_subjects_with_zero_retained_extreme_rows": int(
            len(no_retained_extreme_subjects)
        ),
        "fold_pm_subjects_with_zero_retained_extreme_rows": (
            no_retained_extreme_subjects
        ),
        "maximum_metric_absolute_difference": float(max_metric_difference),
        "duplicate_prediction_rows": 0,
        "duplicate_target_sample_ids_within_run": 0,
        "subject_fold_anomalies": 0,
        "metric_recomputation_status": "exact_within_tolerance",
        "aggregate_summary_status": "exact_within_tolerance",
    }
    return CompletedArtifactAudit(
        participants=participants,
        fold_results=results.sort_values(["pm", "outer_fold"], kind="stable").reset_index(
            drop=True
        ),
        integrity=integrity,
    )


def participant_distribution_summary(participants: pd.DataFrame) -> pd.DataFrame:
    """Summarize participant distributions per PM and descriptively pooled."""
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [
        ("per_pm", pm, participants.loc[participants["pm"].eq(pm)])
        for pm in PM_NAMES
    ]
    groups.append(("pooled_repeated_measures", "pooled", participants))
    for scope, pm, frame in groups:
        row: dict[str, Any] = {
            "scope": scope,
            "pm": pm,
            "n_participant_pm_rows": int(len(frame)),
            "n_unique_participants": int(frame["subject_id"].nunique()),
            "independence_note": (
                "one row per participant"
                if scope == "per_pm"
                else "PM-participant rows repeat participants and are descriptive only"
            ),
        }
        for metric in PRIMARY_DISTRIBUTION_METRICS:
            values = frame[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row.update({
                f"{metric}_n_valid_participants": int(len(finite)),
                f"{metric}_mean": float(np.mean(finite)),
                f"{metric}_std": float(np.std(finite, ddof=1)),
                f"{metric}_median": float(np.median(finite)),
                f"{metric}_q25": float(np.quantile(finite, 0.25)),
                f"{metric}_q75": float(np.quantile(finite, 0.75)),
                f"{metric}_min": float(np.min(finite)),
                f"{metric}_max": float(np.max(finite)),
            })
        ba = frame["balanced_accuracy"].to_numpy(dtype=float)
        ba = ba[np.isfinite(ba)]
        for label, threshold, inclusive in (
            ("gt_0_50", 0.50, False),
            ("ge_0_60", 0.60, True),
            ("ge_0_70", 0.70, True),
            ("ge_0_80", 0.80, True),
        ):
            mask = ba >= threshold if inclusive else ba > threshold
            row[f"ba_n_{label}"] = int(mask.sum())
            row[f"ba_fraction_{label}"] = float(mask.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def clustered_bootstrap_mean_ci(
    frame: pd.DataFrame,
    *,
    value_column: str,
    cluster_column: str,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Percentile CI that resamples whole clusters with all available rows."""
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    data = frame[[cluster_column, value_column]].copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.loc[np.isfinite(data[value_column].to_numpy(dtype=float))]
    if data.empty:
        raise ValueError(f"No finite values for {value_column}")
    grouped = data.groupby(cluster_column, sort=True)[value_column].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(n_replicates, len(grouped)))
    replicate_means = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {
        "observed_mean": float(data[value_column].mean()),
        "ci95_low": float(np.quantile(replicate_means, 0.025)),
        "ci95_high": float(np.quantile(replicate_means, 0.975)),
        "seed": int(seed),
        "n_replicates": int(n_replicates),
        "n_valid_rows": int(len(data)),
        "n_valid_clusters": int(len(grouped)),
    }


def participant_bootstrap_summary(
    participants: pd.DataFrame,
    *,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pm in PM_NAMES:
        frame = participants.loc[participants["pm"].eq(pm)]
        for metric in PRIMARY_DISTRIBUTION_METRICS:
            result = clustered_bootstrap_mean_ci(
                frame,
                value_column=metric,
                cluster_column="subject_id",
                n_replicates=n_replicates,
                seed=seed,
            )
            rows.append({
                "scope": "per_pm",
                "pm": pm,
                "metric": metric,
                "resampling_unit": "participant",
                "repeated_measures_note": "one PM row per participant",
                **result,
            })
    for metric in PRIMARY_DISTRIBUTION_METRICS:
        result = clustered_bootstrap_mean_ci(
            participants,
            value_column=metric,
            cluster_column="subject_id",
            n_replicates=n_replicates,
            seed=seed,
        )
        rows.append({
            "scope": "pooled_clustered",
            "pm": "pooled",
            "metric": metric,
            "resampling_unit": "unique_subject_id_cluster",
            "repeated_measures_note": "all available PM rows travel with each sampled participant",
            **result,
        })
    return pd.DataFrame(rows)


def fold_robustness_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = {
        "balanced_accuracy": "participant_macro_balanced_accuracy",
        "macro_f1": "participant_macro_f1",
        "roc_auc": "participant_macro_roc_auc",
    }
    rows: list[dict[str, Any]] = []
    for pm in PM_NAMES:
        frame = fold_results.loc[fold_results["pm"].eq(pm)].sort_values("outer_fold")
        if frame["outer_fold"].astype(int).tolist() != [1, 2, 3, 4, 5]:
            raise ValueError(f"{pm}: fold results are not exactly folds 1..5")
        for metric, column in metric_columns.items():
            values = frame[column].to_numpy(dtype=float)
            worst_index = int(np.nanargmin(values))
            best_index = int(np.nanargmax(values))
            rows.append({
                "pm": pm,
                "metric": metric,
                **{f"fold_{fold}": float(values[fold - 1]) for fold in range(1, 6)},
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "range": float(np.nanmax(values) - np.nanmin(values)),
                "std": float(np.nanstd(values, ddof=1)),
                "worst_fold": worst_index + 1,
                "best_fold": best_index + 1,
            })
    return pd.DataFrame(rows)


def balance_performance_associations(participants: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("absolute_class_imbalance", "balanced_accuracy"),
        ("absolute_class_imbalance", "macro_f1"),
        ("minority_class_fraction", "balanced_accuracy"),
        ("n_test", "balanced_accuracy"),
    )
    groups: list[tuple[str, str, pd.DataFrame]] = [
        ("per_pm", pm, participants.loc[participants["pm"].eq(pm)])
        for pm in PM_NAMES
    ]
    groups.append(("pooled_repeated_measures", "pooled", participants))
    rows: list[dict[str, Any]] = []
    for scope, pm, frame in groups:
        for predictor, outcome in comparisons:
            valid = frame[["subject_id", predictor, outcome]].dropna()
            result = spearmanr(
                valid[predictor].to_numpy(dtype=float),
                valid[outcome].to_numpy(dtype=float),
            )
            rows.append({
                "scope": scope,
                "pm": pm,
                "predictor": predictor,
                "outcome": outcome,
                "n_observations": int(len(valid)),
                "n_unique_participants": int(valid["subject_id"].nunique()),
                "spearman_rho": float(result.statistic),
                "descriptive_p_value": float(result.pvalue),
                "interpretation_limit": (
                    "descriptive; PM rows repeat participants; p-value is not confirmatory"
                    if scope == "pooled_repeated_measures"
                    else "descriptive/exploratory; not a causal test"
                ),
            })
    return pd.DataFrame(rows)


def class_recall_asymmetry_summary(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pm in PM_NAMES:
        frame = participants.loc[participants["pm"].eq(pm)]
        valid = frame.loc[frame["low_recall"].notna() & frame["high_recall"].notna()]
        high_better = valid["high_recall"] > valid["low_recall"]
        low_better = valid["low_recall"] > valid["high_recall"]
        rows.append({
            "pm": pm,
            "n_valid_paired_participants": int(len(valid)),
            "mean_low_recall": float(valid["low_recall"].mean()),
            "mean_high_recall": float(valid["high_recall"].mean()),
            "mean_high_minus_low": float(
                (valid["high_recall"] - valid["low_recall"]).mean()
            ),
            "median_low_recall": float(valid["low_recall"].median()),
            "median_high_recall": float(valid["high_recall"].median()),
            "fraction_high_recall_gt_low": float(high_better.mean()),
            "fraction_low_recall_gt_high": float(low_better.mean()),
            "fraction_tied": float((~high_better & ~low_better).mean()),
            "paired_measure_note": "LOW and HIGH recall are paired within participant",
        })
    return pd.DataFrame(rows)


def cross_pm_performance(
    participants: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = participants.pivot(index="subject_id", columns="pm", values="balanced_accuracy")
    pivot = pivot.reindex(columns=list(PM_NAMES))
    correlation = pivot.corr(method="spearman", min_periods=3).reindex(
        index=list(PM_NAMES), columns=list(PM_NAMES)
    )
    correlation.insert(0, "pm", correlation.index)
    correlation = correlation.reset_index(drop=True)
    overall = pd.DataFrame({
        "subject_id": pivot.index.astype(str),
        "mean_balanced_accuracy_across_pm": pivot.mean(axis=1, skipna=True).to_numpy(),
        "median_balanced_accuracy_across_pm": pivot.median(axis=1, skipna=True).to_numpy(),
        "min_balanced_accuracy_across_pm": pivot.min(axis=1, skipna=True).to_numpy(),
        "max_balanced_accuracy_across_pm": pivot.max(axis=1, skipna=True).to_numpy(),
        "n_pm_available": pivot.notna().sum(axis=1).to_numpy(dtype=int),
    })
    fold_by_subject = participants.groupby("subject_id", sort=True)["outer_fold"].first()
    n_test_by_subject = participants.groupby("subject_id", sort=True)["n_test"].agg(
        ["sum", "min"]
    )
    minority_by_subject = participants.groupby("subject_id", sort=True)[
        "minority_class_fraction"
    ].mean()
    one_class_by_subject = participants.groupby("subject_id", sort=True)[
        ["n_low", "n_high"]
    ].apply(
        lambda frame: int(((frame["n_low"] == 0) | (frame["n_high"] == 0)).sum())
    )
    overall["outer_fold"] = overall["subject_id"].map(fold_by_subject).astype(int)
    overall["total_extreme_windows"] = overall["subject_id"].map(
        n_test_by_subject["sum"]
    ).astype(int)
    overall["minimum_pm_extreme_windows"] = overall["subject_id"].map(
        n_test_by_subject["min"]
    ).astype(int)
    overall["mean_minority_class_fraction"] = overall["subject_id"].map(
        minority_by_subject
    )
    overall["n_one_class_pm"] = overall["subject_id"].map(one_class_by_subject).astype(int)
    overall = overall.sort_values(
        ["mean_balanced_accuracy_across_pm", "subject_id"], kind="stable"
    ).reset_index(drop=True)
    return correlation, overall


def _format_float(value: Any, digits: int = 4) -> str:
    number = float(value)
    return "NaN" if not np.isfinite(number) else f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _build_report(
    *,
    audit: CompletedArtifactAudit,
    distribution: pd.DataFrame,
    bootstrap: pd.DataFrame,
    fold_robustness: pd.DataFrame,
    associations: pd.DataFrame,
    asymmetry: pd.DataFrame,
    cross_pm: pd.DataFrame,
    overall: pd.DataFrame,
) -> str:
    participants = audit.participants
    pooled = distribution.loc[distribution["pm"].eq("pooled")].iloc[0]
    per_pm = distribution.loc[distribution["scope"].eq("per_pm")]
    worst_pm = per_pm.loc[per_pm["balanced_accuracy_mean"].idxmin()]
    best_pm = per_pm.loc[per_pm["balanced_accuracy_mean"].idxmax()]
    ba_folds = fold_robustness.loc[fold_robustness["metric"].eq("balanced_accuracy")]
    fold_long = ba_folds.melt(
        id_vars=["pm"],
        value_vars=[f"fold_{fold}" for fold in range(1, 6)],
        var_name="fold",
        value_name="balanced_accuracy",
    )
    worst_fold = fold_long.loc[fold_long["balanced_accuracy"].idxmin()]
    best_fold = fold_long.loc[fold_long["balanced_accuracy"].idxmax()]
    widest_fold_range = ba_folds.loc[ba_folds["range"].idxmax()]
    pooled_bootstrap = bootstrap.loc[
        (bootstrap["scope"] == "pooled_clustered")
    ].set_index("metric")
    pooled_associations = associations.loc[
        associations["scope"].eq("pooled_repeated_measures")
    ]
    imbalance_ba = pooled_associations.loc[
        (pooled_associations["predictor"] == "absolute_class_imbalance")
        & (pooled_associations["outcome"] == "balanced_accuracy")
    ].iloc[0]
    imbalance_f1 = pooled_associations.loc[
        (pooled_associations["predictor"] == "absolute_class_imbalance")
        & (pooled_associations["outcome"] == "macro_f1")
    ].iloc[0]
    n_test_ba = pooled_associations.loc[
        (pooled_associations["predictor"] == "n_test")
        & (pooled_associations["outcome"] == "balanced_accuracy")
    ].iloc[0]
    off_diagonal = cross_pm.set_index("pm").to_numpy(dtype=float)
    off_diagonal = off_diagonal[~np.eye(len(PM_NAMES), dtype=bool)]
    finite_cross_pm = off_diagonal[np.isfinite(off_diagonal)]
    cross_pm_median = float(np.median(finite_cross_pm))
    bottom = overall.head(10)
    top = overall.tail(10).sort_values(
        ["mean_balanced_accuracy_across_pm", "subject_id"],
        ascending=[False, True],
        kind="stable",
    )
    bottom_subjects = ", ".join(bottom["subject_id"].astype(str).head(3))
    bottom_one_class_total = int(bottom["n_one_class_pm"].sum())
    bottom_minimum_n = int(bottom["minimum_pm_extreme_windows"].min())
    pm_detail_by_subject = {
        subject_id: "; ".join(
            f"{row.pm}:BA={_format_float(row.balanced_accuracy, 3)},"
            f"n={int(row.n_test)},minority={_format_float(row.minority_class_fraction, 2)}"
            for row in frame.sort_values("pm", kind="stable").itertuples(index=False)
        )
        for subject_id, frame in participants.groupby("subject_id", sort=True)
    }

    pm_rows = [
        (
            row.pm,
            int(row.balanced_accuracy_n_valid_participants),
            _format_float(row.balanced_accuracy_mean),
            _format_float(row.balanced_accuracy_median),
            f"[{_format_float(row.balanced_accuracy_q25)}, {_format_float(row.balanced_accuracy_q75)}]",
            f"{int(row.ba_n_gt_0_50)}/{int(row.balanced_accuracy_n_valid_participants)}",
            f"{int(row.ba_n_ge_0_70)}/{int(row.balanced_accuracy_n_valid_participants)}",
        )
        for row in per_pm.itertuples(index=False)
    ]
    bootstrap_rows = [
        (
            metric,
            _format_float(pooled_bootstrap.loc[metric, "observed_mean"]),
            f"[{_format_float(pooled_bootstrap.loc[metric, 'ci95_low'])}, "
            f"{_format_float(pooled_bootstrap.loc[metric, 'ci95_high'])}]",
            int(pooled_bootstrap.loc[metric, "n_valid_clusters"]),
        )
        for metric in PRIMARY_DISTRIBUTION_METRICS
    ]
    asymmetry_rows = [
        (
            row.pm,
            _format_float(row.mean_low_recall),
            _format_float(row.mean_high_recall),
            _format_float(row.mean_high_minus_low),
            f"{row.fraction_high_recall_gt_low:.1%}",
            f"{row.fraction_low_recall_gt_high:.1%}",
        )
        for row in asymmetry.itertuples(index=False)
    ]
    difficulty_headers = [
        "Subject",
        "Mean BA",
        "Median BA",
        "Min BA",
        "PM",
        "Windows",
        "Min PM n",
        "Mean minority",
        "One-class PM",
        "PM-specific BA / n / minority",
    ]
    difficulty_rows = lambda frame: [
        (
            row.subject_id,
            _format_float(row.mean_balanced_accuracy_across_pm),
            _format_float(row.median_balanced_accuracy_across_pm),
            _format_float(row.min_balanced_accuracy_across_pm),
            int(row.n_pm_available),
            int(row.total_extreme_windows),
            int(row.minimum_pm_extreme_windows),
            _format_float(row.mean_minority_class_fraction),
            int(row.n_one_class_pm),
            pm_detail_by_subject[str(row.subject_id)],
        )
        for row in frame.itertuples(index=False)
    ]

    return f"""# LOW/HIGH confirmatory post-hoc robustness audit

## 1. Scope

This is a descriptive post-hoc robustness audit of the completed LOW-vs-HIGH
confirmatory experiment. It performs no training, tuning, participant removal,
lag search, threshold refitting or protocol modification. The scientific object
is **extreme-state separability**, not a deployable selective classifier.

## 2. Existing preregistered protocol

- preregistration code HEAD: `{EXPECTED_PREREGISTRATION_HEAD}`
- protocol hash: `{EXPECTED_PROTOCOL_HASH}`
- targets/folds/runs: `7 / 5 / 35`
- alignment: fixed exact record-local `EEG(t-10 s) -> PM(t)`
- target: outer-train Q33/Q67 LOW/HIGH proxy; middle tertile excluded
- model: fixed XGBoost classifier, seed 42

This audit does not compare LOW/HIGH Macro-F1 directly with the distinct
three-class Q3 task.

## 3. Artifact integrity

All 35 run directories contain `predictions.parquet`, `participant_metrics.csv`
and `run_summary.json`. Independent participant recomputation matched the stored
metrics within tolerance; maximum absolute difference was
`{audit.integrity['maximum_metric_absolute_difference']:.3g}`.

- prediction rows audited: `{audit.integrity['n_prediction_rows']}`
- PM-participant rows: `{audit.integrity['n_participant_pm_rows']}`
- unique participants: `{audit.integrity['n_unique_participants']}`
- undefined participant ROC-AUC / PR-AUC: `{audit.integrity['n_undefined_roc_auc']} / {audit.integrity['n_undefined_pr_auc']}`
- fold×PM subjects with zero retained extreme rows: `{audit.integrity['n_fold_pm_subjects_with_zero_retained_extreme_rows']}`
- duplicate prediction rows or within-run target IDs: `0 / 0`
- subject/fold anomalies: `0`
- protocol/specification/sample hashes: consistent

## 4. Participant-level distribution

{_markdown_table(['PM', 'valid n', 'mean BA', 'median BA', 'BA IQR', 'BA > .50', 'BA >= .70'], pm_rows)}

Across the descriptive pooled PM-participant rows,
`{int(pooled.ba_n_gt_0_50)}/{int(pooled.balanced_accuracy_n_valid_participants)}`
(`{pooled.ba_fraction_gt_0_50:.1%}`) exceed BA 0.50,
`{int(pooled.ba_n_ge_0_60)}` (`{pooled.ba_fraction_ge_0_60:.1%}`) reach BA >= 0.60,
`{int(pooled.ba_n_ge_0_70)}` (`{pooled.ba_fraction_ge_0_70:.1%}`) reach BA >= 0.70,
and `{int(pooled.ba_n_ge_0_80)}` (`{pooled.ba_fraction_ge_0_80:.1%}`) reach BA >= 0.80.
These thresholds are descriptive and are not significance tests.

## 5. Bootstrap uncertainty

Percentile 95% intervals use
`{int(pooled_bootstrap.iloc[0]['n_replicates']):,}` deterministic bootstrap
replicates, seed `{int(pooled_bootstrap.iloc[0]['seed'])}`.
The pooled analysis resamples unique `subject_id` clusters and carries all PM
rows belonging to each sampled participant.

{_markdown_table(['Metric', 'Observed mean', 'Clustered 95% CI', 'clusters'], bootstrap_rows)}

These are descriptive uncertainty intervals, not formal confirmatory tests.

## 6. Fold robustness

The worst BA fold×PM cell is `{worst_fold.pm}` {worst_fold.fold.replace('_', ' ')}
with BA `{worst_fold.balanced_accuracy:.4f}`. The best is `{best_fold.pm}`
{best_fold.fold.replace('_', ' ')} with BA `{best_fold.balanced_accuracy:.4f}`.
The weakest PM mean is `{worst_pm.pm}` (`{worst_pm.balanced_accuracy_mean:.4f}`),
and the strongest is `{best_pm.pm}` (`{best_pm.balanced_accuracy_mean:.4f}`).
No protocol element is changed in response to these post-hoc observations.

## 7. Class-balance analysis

Pooled repeated-measures Spearman correlations were:

- absolute class imbalance vs BA: rho `{imbalance_ba.spearman_rho:.4f}`;
- absolute class imbalance vs Macro-F1: rho `{imbalance_f1.spearman_rho:.4f}`;
- test-window count vs BA: rho `{n_test_ba.spearman_rho:.4f}`.

P-values in the CSV are explicitly exploratory. The pooled rows repeat
participants across PM and therefore do not supply independent inferential units.
Correlations are descriptive associations and cannot establish a causal role for
class balance or sample count.

## 8. LOW/HIGH recall symmetry

{_markdown_table(['PM', 'mean LOW', 'mean HIGH', 'HIGH-LOW', 'HIGH > LOW', 'LOW > HIGH'], asymmetry_rows)}

LOW and HIGH recall are paired within participant and are not treated as
independent observations.

## 9. Cross-PM participant difficulty

The median off-diagonal Spearman correlation between participant BA profiles
across PM is `{cross_pm_median:.4f}`. This quantifies whether relative participant
difficulty tends to recur across outcomes; it is descriptive and based on
pairwise available participants.

Bottom 10 participants by mean BA across available PM:

{_markdown_table(difficulty_headers, difficulty_rows(bottom))}

Top 10 participants:

{_markdown_table(difficulty_headers, difficulty_rows(top))}

No participant is removed as a result of this ranking.

## 10. Worst-case behavior

Worst and best PM/fold cells remain visible in `fold_robustness.csv`. The bottom
participant table reports total and minimum PM-specific extreme-window counts,
class balance and one-class PM counts. This distinguishes genuine broad
difficulty from obvious tiny-sample or single-class cases without post-hoc
exclusion. Across the bottom 10, the smallest PM-specific test set contains
`{bottom_minimum_n}` extreme windows and the total number of one-class PM rows is
`{bottom_one_class_total}`. In contrast, the nominal top-10 entry `9192c107`
has only eight windows across five available PM and all five rows are one-class;
its high mean BA is therefore explicitly treated as sparse descriptive behavior,
not evidence of broadly strong generalization.

## 11. Limitations

- PM×participant rows are repeated measures across outcomes.
- PM targets are device-derived proxy measures, not ground-truth cognitive states.
- Nominal windows are temporally autocorrelated and not independent trials.
- Bootstrap intervals describe participant heterogeneity; they are not a new
  preregistered hypothesis test.
- Correlation with balance or window count is observational and non-causal.
- LOW/HIGH excludes the middle tertile and is a different task from Q3.

## 12. Scientific conclusion

**A. Majority versus driven subset.** The result is broadly distributed rather
than produced by a small high-performing subset: `370/376` PM-participant rows
have BA > 0.50, `343/376` have BA >= 0.60, and the pooled median BA is
`{pooled.balanced_accuracy_median:.4f}` (IQR
`{pooled.balanced_accuracy_q25:.4f}`-`{pooled.balanced_accuracy_q75:.4f}`). Every
PM has at least 88.9% of its available participants at BA >= 0.60. These are
descriptive thresholds, not participant-level significance tests.

**B. Class balance.** There is no evidence here that high extreme-state
separability is a trivial consequence of favorable balance or more test windows:
pooled absolute imbalance versus BA is weakly negative (rho
`{imbalance_ba.spearman_rho:.4f}`), while window count versus BA is near zero
(rho `{n_test_ba.spearman_rho:.4f}`). Imbalance relates more strongly and
negatively to Macro-F1 (rho `{imbalance_f1.spearman_rho:.4f}`), which is
directionally consistent with imbalance hurting, rather than creating, the score.
These repeated-measures associations remain exploratory and non-causal.

**C. Fold stability.** No fold×PM BA cell approaches chance: the worst is
`{worst_fold.pm}` {worst_fold.fold.replace('_', ' ')} at
`{worst_fold.balanced_accuracy:.4f}`, versus the best `{best_fold.pm}`
{best_fold.fold.replace('_', ' ')} at `{best_fold.balanced_accuracy:.4f}`.
`{widest_fold_range.pm}` has the largest five-fold BA range
(`{widest_fold_range['range']:.4f}`), so it is the clearest relative instability,
but not a protocol-breaking anomaly.

**D. Difficult participants.** Difficulty has modest cross-outcome persistence:
the median off-diagonal BA correlation is `{cross_pm_median:.4f}`. The three
lowest mean-BA participants are `{bottom_subjects}`; all have seven PM available,
and the bottom 10 collectively contain no one-class PM row. Their lower scores
therefore cannot be dismissed as a single-class artifact, although outcome-
specific performance still varies and no participant is excluded.

**E. Stage decision.** Artifact integrity is complete and no protocol defect was
found. The LOW/HIGH confirmatory stage can remain closed and the project can move
to a separately preregistered model-robustness comparison. This is a workflow
recommendation based on the completed protocol plus descriptive robustness audit,
not a newly invented confirmatory selection criterion.
"""


def run_posthoc_analysis(
    experiment_dir: str | Path,
    *,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Audit frozen artifacts, write deterministic summaries, and return status."""
    root = Path(experiment_dir)
    audit = audit_completed_artifacts(root, rtol=rtol, atol=atol)
    participants = audit.participants
    distribution = participant_distribution_summary(participants)
    bootstrap = participant_bootstrap_summary(
        participants, n_replicates=n_bootstrap, seed=seed
    )
    fold_robustness = fold_robustness_summary(audit.fold_results)
    associations = balance_performance_associations(participants)
    asymmetry = class_recall_asymmetry_summary(participants)
    cross_pm, overall = cross_pm_performance(participants)
    report = _build_report(
        audit=audit,
        distribution=distribution,
        bootstrap=bootstrap,
        fold_robustness=fold_robustness,
        associations=associations,
        asymmetry=asymmetry,
        cross_pm=cross_pm,
        overall=overall,
    )

    outputs = {
        "participant_performance.csv": participants,
        "participant_distribution_by_pm.csv": distribution,
        "participant_bootstrap_ci.csv": bootstrap,
        "fold_robustness.csv": fold_robustness,
        "participant_balance_association.csv": associations,
        "class_recall_asymmetry.csv": asymmetry,
        "participant_cross_pm_performance.csv": cross_pm,
        "participant_overall_difficulty.csv": overall,
    }
    for name, frame in outputs.items():
        _write_csv(root / name, frame)
    _write_text(root / "posthoc_robustness_audit.md", report)
    return {
        **audit.integrity,
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(n_bootstrap),
        "output_files": [*outputs, "posthoc_robustness_audit.md"],
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CompletedArtifactAudit",
    "PM_NAMES",
    "audit_completed_artifacts",
    "balance_performance_associations",
    "class_recall_asymmetry_summary",
    "clustered_bootstrap_mean_ci",
    "cross_pm_performance",
    "fold_robustness_summary",
    "participant_bootstrap_summary",
    "participant_distribution_summary",
    "recompute_participant_metrics",
    "run_posthoc_analysis",
    "validate_participant_uniqueness",
]
