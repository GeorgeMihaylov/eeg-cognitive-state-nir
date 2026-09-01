"""Subject-level analysis of the finalized auxiliary-CORN selection policy.

The analysis consumes completed prediction artifacts only.  One paired observation
per subject is used for inference after averaging repeated initialization seeds.
The finalized policy is compared with the paired categorical Transformer and the
pure CORN Transformer on exactly aligned outer-test sequences.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from bench.analysis.ordinal_transformer_statistics import (
    FEATURE_GROUPS,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    SUBJECT_METRICS,
    calculate_prediction_metrics,
    categorical_expected_rank,
    paired_metric_comparison,
)
from bench.analysis.paired_statistics import apply_holm_by_family
from bench.validation.metrics import MetricsCalculator


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("categorical", "corn", "policy")
SEEDS = (7, 42, 123)
IDENTITY_COLUMNS = (
    "sequence_id", "fold", "subject_id", "record_id", "source", "y_true"
)
PRIMARY_METRICS = ("ordinal_mae", "severe_error_rate", "balanced_accuracy")
SECONDARY_METRICS = (
    "macro_f1",
    "quadratic_weighted_kappa",
    "adjacent_accuracy",
    "expected_rank_mae",
    "expected_rank_spearman",
)
HEADLINE_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "severe_error_rate",
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_file(run_directory: Path) -> Path:
    paths = list(run_directory.glob("**/group_kfold_subject/predictions.parquet"))
    if len(paths) != 1:
        raise ValueError(
            f"Expected one unified prediction artifact in {run_directory}, found {len(paths)}"
        )
    return paths[0]


def _canonical_identity(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing identity columns: {missing}")
    if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
        raise ValueError("sequence_id must be complete and unique")
    selected = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
    selected["sequence_id"] = selected["sequence_id"].astype(str)
    selected["fold"] = pd.to_numeric(selected["fold"], errors="raise").astype(int)
    selected["subject_id"] = selected["subject_id"].astype(str)
    selected["record_id"] = selected["record_id"].astype(str)
    selected["source"] = selected["source"].astype(str)
    selected["y_true"] = pd.to_numeric(selected["y_true"], errors="raise").astype(int)
    return selected.sort_values("sequence_id", kind="mergesort").reset_index(drop=True)


def require_three_way_alignment(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    expected = set(METHODS)
    if set(frames) != expected:
        raise ValueError(f"Expected methods {sorted(expected)}, observed {sorted(frames)}")
    reference = _canonical_identity(frames["categorical"])
    comparisons: list[dict[str, Any]] = []
    for method in ("corn", "policy"):
        candidate = _canonical_identity(frames[method])
        count_mismatch = len(reference) != len(candidate)
        mismatches: dict[str, int] = {}
        if count_mismatch:
            mismatches = {column: max(len(reference), len(candidate)) for column in IDENTITY_COLUMNS}
        else:
            for column in IDENTITY_COLUMNS:
                mismatches[column] = int((reference[column] != candidate[column]).sum())
        if count_mismatch or any(mismatches.values()):
            raise ValueError(
                f"Three-way outer alignment failed for {method}: "
                f"count_mismatch={count_mismatch}, mismatches={mismatches}"
            )
        comparisons.append({
            "reference": "categorical",
            "candidate": method,
            "rows": int(len(candidate)),
            "mismatches": mismatches,
            "exact_match": True,
        })
    return {
        "exact_match": True,
        "rows": int(len(reference)),
        "subjects": int(reference["subject_id"].nunique()),
        "folds": int(reference["fold"].nunique()),
        "comparisons": comparisons,
    }


def _expected_rank(method: str, frame: pd.DataFrame) -> np.ndarray:
    if method == "categorical":
        return categorical_expected_rank(frame)
    if method == "corn":
        if "expected_rank" not in frame.columns:
            raise ValueError("Pure CORN predictions require expected_rank")
        return frame["expected_rank"].to_numpy(dtype=float)
    if method == "policy":
        if "categorical_expected_rank" in frame.columns:
            return frame["categorical_expected_rank"].to_numpy(dtype=float)
        return categorical_expected_rank(frame)
    raise ValueError(f"Unknown method: {method}")


def calculate_policy_subject_metrics(
    by_seed: Mapping[int, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, frames in sorted(by_seed.items()):
        for run_key, frame in sorted(frames.items()):
            method, feature_group = run_key.split("_", 1)
            frame = frame.copy()
            frame["_analysis_expected_rank"] = _expected_rank(method, frame)
            for subject_id, group in frame.groupby("subject_id", sort=True):
                folds = group["fold"].unique()
                if len(folds) != 1:
                    raise ValueError("A subject appears in more than one outer fold")
                metrics = calculate_prediction_metrics(
                    group,
                    expected_rank=group["_analysis_expected_rank"].to_numpy(dtype=float),
                )
                row: dict[str, Any] = {
                    "run_key": run_key,
                    "method": method,
                    "feature_group": feature_group,
                    "seed": int(seed),
                    "subject_id": str(subject_id),
                    "fold": int(folds[0]),
                    "source_membership": "+".join(sorted(group["source"].astype(str).unique())),
                    "n_sequences": int(len(group)),
                    **metrics,
                }
                if method == "policy":
                    branch = group.get("policy_branch", pd.Series("joint_selected", index=group.index))
                    aux_available = group.get("aux_available", pd.Series(False, index=group.index)).astype(bool)
                    row.update({
                        "joint_sequence_fraction": float((branch == "joint_selected").mean()),
                        "fallback_sequence_fraction": float((branch == "categorical_fallback").mean()),
                        "auxiliary_coverage_fraction": float(aux_available.mean()),
                        "categorical_aux_disagreement_rate": np.nan,
                        "selected_auxiliary_weight_mean": np.nan,
                    })
                    if bool(aux_available.any()) and "aux_ordinal_prediction" in group.columns:
                        available = group.loc[aux_available]
                        row["categorical_aux_disagreement_rate"] = float(
                            (available["y_pred"].to_numpy(dtype=int)
                             != available["aux_ordinal_prediction"].to_numpy(dtype=int)).mean()
                        )
                    if "selected_auxiliary_weight" in group.columns:
                        values = pd.to_numeric(
                            group.loc[aux_available, "selected_auxiliary_weight"],
                            errors="coerce",
                        ).dropna()
                        if len(values):
                            row["selected_auxiliary_weight_mean"] = float(values.mean())
                rows.append(row)
    result = pd.DataFrame(rows)
    expected_rows = len(METHODS) * len(FEATURE_GROUPS) * len(SEEDS) * 53
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} subject-seed rows, observed {len(result)}")
    return result


def average_subject_metrics_across_seeds(subject_seed: pd.DataFrame) -> pd.DataFrame:
    counts = subject_seed.groupby(["run_key", "subject_id"], sort=True)["seed"].nunique()
    if not bool((counts == len(SEEDS)).all()):
        raise ValueError("Every method/group/subject must contain exactly three seeds")
    numeric = [
        column for column in SUBJECT_METRICS + (
            "n_sequences",
            "joint_sequence_fraction",
            "fallback_sequence_fraction",
            "auxiliary_coverage_fraction",
            "categorical_aux_disagreement_rate",
            "selected_auxiliary_weight_mean",
        )
        if column in subject_seed.columns
    ]
    rows: list[dict[str, Any]] = []
    for (run_key, subject_id), group in subject_seed.groupby(
        ["run_key", "subject_id"], sort=True
    ):
        methods = group["method"].unique()
        feature_groups = group["feature_group"].unique()
        folds = group["fold"].unique()
        sources = group["source_membership"].unique()
        if not (len(methods) == len(feature_groups) == len(folds) == len(sources) == 1):
            raise ValueError("Subject identity metadata changed between seeds")
        row = {
            "run_key": run_key,
            "method": methods[0],
            "feature_group": feature_groups[0],
            "subject_id": subject_id,
            "fold": int(folds[0]),
            "source_membership": sources[0],
            "seeds_averaged": len(SEEDS),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.mean(skipna=True)) if values.notna().any() else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    expected_rows = len(METHODS) * len(FEATURE_GROUPS) * 53
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} averaged rows, observed {len(result)}")
    return result


def _paired_tables(
    averaged: pd.DataFrame,
    *,
    bootstrap_samples: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    for feature_group in FEATURE_GROUPS:
        policy = f"policy_{feature_group}"
        categorical = f"categorical_{feature_group}"
        corn = f"corn_{feature_group}"
        for metric in PRIMARY_METRICS:
            primary.append(paired_metric_comparison(
                averaged,
                candidate_key=policy,
                reference_key=categorical,
                metric=metric,
                family=f"policy_vs_categorical_primary_{feature_group}",
                hypothesis_tier="primary",
                n_resamples=bootstrap_samples,
                random_state=random_state,
            ))
        for reference_key, label in ((categorical, "categorical"), (corn, "corn")):
            metrics = SECONDARY_METRICS if label == "categorical" else HEADLINE_METRICS
            for metric in metrics:
                secondary.append(paired_metric_comparison(
                    averaged,
                    candidate_key=policy,
                    reference_key=reference_key,
                    metric=metric,
                    family=f"policy_vs_{label}_{feature_group}",
                    hypothesis_tier="secondary",
                    n_resamples=bootstrap_samples,
                    random_state=random_state,
                ))
    return apply_holm_by_family(primary), apply_holm_by_family(secondary)


def _seed_consistency(subject_seed: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_group in FEATURE_GROUPS:
        for reference in ("categorical", "corn"):
            for metric in HEADLINE_METRICS:
                row: dict[str, Any] = {
                    "feature_group": feature_group,
                    "candidate": "policy",
                    "reference": reference,
                    "metric": metric,
                }
                effects: list[float] = []
                for seed in SEEDS:
                    subset = subject_seed[subject_seed["seed"] == seed]
                    candidate = subset[subset["run_key"] == f"policy_{feature_group}"].sort_values("subject_id")
                    baseline = subset[subset["run_key"] == f"{reference}_{feature_group}"].sort_values("subject_id")
                    if not candidate["subject_id"].reset_index(drop=True).equals(
                        baseline["subject_id"].reset_index(drop=True)
                    ):
                        raise ValueError("Seed-level subject pairs differ")
                    raw = candidate[metric].to_numpy(float) - baseline[metric].to_numpy(float)
                    improvement = raw if metric in HIGHER_IS_BETTER else -raw
                    effect = float(np.nanmean(improvement))
                    effects.append(effect)
                    row[f"seed_{seed}_mean_improvement"] = effect
                row.update({
                    "positive_seeds": int(np.count_nonzero(np.asarray(effects) > 0)),
                    "negative_seeds": int(np.count_nonzero(np.asarray(effects) < 0)),
                    "minimum_seed_effect": float(np.min(effects)),
                    "maximum_seed_effect": float(np.max(effects)),
                    "between_seed_standard_deviation": float(np.std(effects, ddof=0)),
                    "direction_label": (
                        "consistent_positive" if all(value > 0 for value in effects)
                        else "consistent_nonpositive" if all(value <= 0 for value in effects)
                        else "changes_sign"
                    ),
                })
                rows.append(row)
    return rows


def _subject_effects(averaged: pd.DataFrame) -> pd.DataFrame:
    indexed = averaged.set_index(["run_key", "subject_id"])
    rows: list[dict[str, Any]] = []
    for feature_group in FEATURE_GROUPS:
        reference_key = f"categorical_{feature_group}"
        policy_key = f"policy_{feature_group}"
        subjects = sorted(averaged.loc[averaged["run_key"] == reference_key, "subject_id"])
        baseline_ba = np.asarray([
            indexed.loc[(reference_key, subject), "balanced_accuracy"] for subject in subjects
        ], dtype=float)
        q25 = float(np.quantile(baseline_ba, 0.25))
        for subject in subjects:
            reference = indexed.loc[(reference_key, subject)]
            policy = indexed.loc[(policy_key, subject)]
            ba = float(policy["balanced_accuracy"] - reference["balanced_accuracy"])
            mae = float(reference["ordinal_mae"] - policy["ordinal_mae"])
            severe = float(reference["severe_error_rate"] - policy["severe_error_rate"])
            rows.append({
                "feature_group": feature_group,
                "subject_id": subject,
                "fold": int(reference["fold"]),
                "source_membership": reference["source_membership"],
                "baseline_balanced_accuracy": float(reference["balanced_accuracy"]),
                "hard_subject": bool(float(reference["balanced_accuracy"]) <= q25),
                "balanced_accuracy_improvement": ba,
                "ordinal_mae_improvement": mae,
                "severe_error_improvement": severe,
                "ordinal_metrics_both_improved": bool(mae > 0 and severe > 0),
                "ordinal_gain_with_ba_loss": bool((mae > 0 or severe > 0) and ba < 0),
                "joint_sequence_fraction": float(policy.get("joint_sequence_fraction", np.nan)),
                "fallback_sequence_fraction": float(policy.get("fallback_sequence_fraction", np.nan)),
                "auxiliary_coverage_fraction": float(policy.get("auxiliary_coverage_fraction", np.nan)),
                "categorical_aux_disagreement_rate": float(policy.get("categorical_aux_disagreement_rate", np.nan)),
            })
    return pd.DataFrame(rows)


def _hard_subject_summary(effects: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_group, group in effects.groupby("feature_group", sort=True):
        for label, subset in (
            ("hard_quartile", group[group["hard_subject"]]),
            ("remaining_subjects", group[~group["hard_subject"]]),
        ):
            rows.append({
                "feature_group": feature_group,
                "difficulty_group": label,
                "subjects": int(len(subset)),
                "mean_balanced_accuracy_improvement": float(subset["balanced_accuracy_improvement"].mean()),
                "mean_ordinal_mae_improvement": float(subset["ordinal_mae_improvement"].mean()),
                "mean_severe_error_improvement": float(subset["severe_error_improvement"].mean()),
                "fraction_ordinal_metrics_both_improved": float(subset["ordinal_metrics_both_improved"].mean()),
                "fraction_ordinal_gain_with_ba_loss": float(subset["ordinal_gain_with_ba_loss"].mean()),
            })
    return rows


def _safe_spearman(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return {"status": "undefined", "n_subjects": int(len(frame)), "rho": np.nan, "p_value": np.nan}
    result = spearmanr(frame["x"].to_numpy(float), frame["y"].to_numpy(float))
    return {
        "status": "completed",
        "n_subjects": int(len(frame)),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _disagreement_analysis(effects: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_group, group in effects.groupby("feature_group", sort=True):
        for outcome in (
            "balanced_accuracy_improvement",
            "ordinal_mae_improvement",
            "severe_error_improvement",
        ):
            rows.append({
                "feature_group": feature_group,
                "signal": "categorical_aux_disagreement_rate",
                "outcome": outcome,
                **_safe_spearman(group["categorical_aux_disagreement_rate"], group[outcome]),
            })
        for outcome in (
            "balanced_accuracy_improvement",
            "ordinal_mae_improvement",
            "severe_error_improvement",
        ):
            rows.append({
                "feature_group": feature_group,
                "signal": "auxiliary_coverage_fraction",
                "outcome": outcome,
                **_safe_spearman(group["auxiliary_coverage_fraction"], group[outcome]),
            })
    return rows


def _decision(primary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {
        (row["feature_group"], row["metric"]): row
        for row in primary
    }
    groups: dict[str, Any] = {}
    for feature_group in FEATURE_GROUPS:
        mae = rows[(feature_group, "ordinal_mae")]
        severe = rows[(feature_group, "severe_error_rate")]
        ba = rows[(feature_group, "balanced_accuracy")]
        ordinal_supported = bool(
            (mae["mean_improvement"] > 0 and mae["bootstrap_ci_low"] > 0)
            or (severe["mean_improvement"] > 0 and severe["bootstrap_ci_low"] > 0)
        )
        ba_noninferior_descriptive = bool(ba["raw_mean_delta"] >= -0.01)
        groups[feature_group] = {
            "ordinal_supported": ordinal_supported,
            "balanced_accuracy_mean_delta": ba["raw_mean_delta"],
            "balanced_accuracy_guard": -0.01,
            "balanced_accuracy_noninferior_descriptive": ba_noninferior_descriptive,
            "classification": (
                "supported_with_ba_guard" if ordinal_supported and ba_noninferior_descriptive
                else "ordinal_gain_ba_tradeoff" if ordinal_supported
                else "not_supported"
            ),
        }
    return {
        "primary_feature_group": "eeg_pow",
        "control_feature_group": "eeg_only",
        "groups": groups,
        "selected_decision": groups["eeg_pow"]["classification"],
        "caution": (
            "The BA guard is reported descriptively on outer-test subject means; "
            "the actual lambda/fallback decisions used inner validation only."
        ),
    }


class AuxiliaryCornPolicyStatistics:
    def __init__(
        self, config_path: str | Path, *, output_dir: str | Path | None = None
    ) -> None:
        self.config_path = _repo_path(config_path)
        self.document = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        analysis = self.document["analysis"]
        if analysis.get("type") != "auxiliary_corn_policy_statistics":
            raise ValueError("Expected auxiliary_corn_policy_statistics analysis")
        self.ordinal_summary_path = _repo_path(analysis["ordinal_run_summary"])
        self.policy_summary_path = _repo_path(analysis["policy_summary"])
        self.output_dir = _repo_path(output_dir or analysis["output_dir"])
        self.report_path = _repo_path(analysis["report_path"])
        self.summary_path = _repo_path(analysis["summary_path"])
        self.decision_report_path = _repo_path(analysis["decision_report_path"])

    def _run_index(self) -> list[dict[str, Any]]:
        rows = _load_json(self.ordinal_summary_path)["run_index"]
        selected = [row for row in rows if row["method"] in {"categorical", "corn"}]
        keys = {(row["method"], row["feature_group"], int(row["seed"])) for row in selected}
        expected = {
            (method, feature_group, seed)
            for method in ("categorical", "corn")
            for feature_group in FEATURE_GROUPS
            for seed in SEEDS
        }
        if keys != expected or len(selected) != len(expected):
            raise ValueError("Ordinal run index must resolve exactly 12 categorical/CORN runs")
        return selected

    def _policy_input_path(self) -> Path:
        summary = _load_json(self.policy_summary_path)
        if summary.get("status") != "completed" or not summary.get("ready_for_subject_level_analysis"):
            raise ValueError("Finalized policy is not ready for subject-level analysis")
        return _repo_path(summary["artifacts"]["subject_level_analysis_input"])

    def plan(self) -> dict[str, Any]:
        rows = self._run_index()
        policy_path = self._policy_input_path()
        return {
            "valid": True,
            "analysis_unit": "subject_id after averaging three repeated seeds",
            "independent_subjects": int(self.document["expected"]["subjects"]),
            "seeds": list(SEEDS),
            "reference_runs": rows,
            "policy_input": _display(policy_path),
            "comparisons": [
                "policy vs categorical (primary)",
                "policy vs pure CORN (secondary)",
            ],
            "bootstrap_samples": int(self.document["analysis"]["bootstrap_samples"]),
            "output_dir": _display(self.output_dir),
        }

    @staticmethod
    def render_plan(plan: Mapping[str, Any]) -> str:
        return "\n".join([
            "Auxiliary-CORN finalized-policy subject analysis plan",
            f"Analysis unit: {plan['analysis_unit']}",
            f"Independent subjects: {plan['independent_subjects']}; seeds: {plan['seeds']}",
            f"Reference runs: {len(plan['reference_runs'])}; policy input: {plan['policy_input']}",
            f"Comparisons: {plan['comparisons']}",
            f"Bootstrap samples: {plan['bootstrap_samples']}",
            "No model fitting or hyperparameter selection is performed.",
        ])

    def execute(self) -> dict[str, Any]:
        expected = self.document["expected"]
        expected_sequences = int(expected["sequences"])
        expected_subjects = int(expected["subjects"])
        policy = pd.read_parquet(self._policy_input_path())
        run_index = self._run_index()
        by_seed: dict[int, dict[str, pd.DataFrame]] = {seed: {} for seed in SEEDS}
        for row in run_index:
            seed = int(row["seed"])
            key = f"{row['method']}_{row['feature_group']}"
            by_seed[seed][key] = pd.read_parquet(
                _prediction_file(_repo_path(row["run_directory"]))
            )
        for seed in SEEDS:
            for feature_group in FEATURE_GROUPS:
                subset = policy.loc[
                    (policy["seed"].astype(int) == seed)
                    & (policy["feature_group"].astype(str) == feature_group)
                ].copy()
                if len(subset) != expected_sequences:
                    raise ValueError(
                        f"Policy {feature_group} seed={seed} has {len(subset)} rows, "
                        f"expected {expected_sequences}"
                    )
                by_seed[seed][f"policy_{feature_group}"] = subset

        alignment: dict[str, Any] = {}
        for seed in SEEDS:
            for feature_group in FEATURE_GROUPS:
                frames = {
                    method: by_seed[seed][f"{method}_{feature_group}"]
                    for method in METHODS
                }
                audit = require_three_way_alignment(frames)
                if audit["rows"] != expected_sequences or audit["subjects"] != expected_subjects:
                    raise ValueError("Aligned prediction dimensions changed")
                alignment[f"{feature_group}_seed_{seed}"] = audit

        subject_seed = calculate_policy_subject_metrics(by_seed)
        averaged = average_subject_metrics_across_seeds(subject_seed)
        bootstrap_samples = int(self.document["analysis"]["bootstrap_samples"])
        random_state = int(self.document["analysis"]["random_state"])
        primary, secondary = _paired_tables(
            averaged,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        )
        consistency = _seed_consistency(subject_seed)
        effects = _subject_effects(averaged)
        hard = _hard_subject_summary(effects)
        disagreement = _disagreement_analysis(effects)
        decision = _decision(primary)

        aggregate_metrics: list[dict[str, Any]] = []
        class_metrics: list[dict[str, Any]] = []
        for seed, frames in sorted(by_seed.items()):
            for run_key, frame in sorted(frames.items()):
                method, feature_group = run_key.split("_", 1)
                expected_rank = _expected_rank(method, frame)
                aggregate_metrics.append({
                    "run_key": run_key,
                    "method": method,
                    "feature_group": feature_group,
                    "seed": seed,
                    **calculate_prediction_metrics(frame, expected_rank=expected_rank),
                })
                for class_row in MetricsCalculator.calculate_class_metrics(
                    frame["y_true"].to_numpy(dtype=int),
                    frame["y_pred"].to_numpy(dtype=int),
                    labels=np.arange(5),
                ):
                    class_metrics.append({
                        "run_key": run_key,
                        "method": method,
                        "feature_group": feature_group,
                        "seed": seed,
                        **class_row,
                    })

        policy_summary = _load_json(self.policy_summary_path)
        fallback_units = [
            {
                "selection_id": item["selection_id"],
                "feature_group": item["feature_group"],
                "seed": int(item["seed"]),
                "outer_fold": int(item["outer_fold"]),
                "fallback_reason": item["fallback_reason"],
                "outer_test_rows": int(item["outer_test_rows"]),
            }
            for item in policy_summary["outcomes"]
            if item["policy_branch"] == "categorical_fallback"
        ]
        source_path = _repo_path(expected["source_parquet"])
        source_hash = _sha256(source_path)
        if source_hash != expected["source_parquet_sha256"]:
            raise ValueError("Source Parquet SHA-256 changed")

        summary = {
            "schema_version": "auxiliary-corn-policy-statistics-v1",
            "status": "completed",
            "analysis_unit": "subject_id",
            "independent_subjects": expected_subjects,
            "seeds": list(SEEDS),
            "subject_seed_rows": int(len(subject_seed)),
            "averaged_subject_rows": int(len(averaged)),
            "alignment": {"all_exact": True, "checks": alignment},
            "policy_composition": {
                "joint_units": int(policy_summary["selection_units_joint"]),
                "fallback_units": int(policy_summary["selection_units_fallback"]),
                "selected_lambda_counts": policy_summary["selected_lambda_counts"],
                "fallback_details": fallback_units,
            },
            "aggregate_metrics_by_seed": aggregate_metrics,
            "subject_level_multiseed_means": {
                run_key: {
                    metric: float(group[metric].mean(skipna=True))
                    for metric in SUBJECT_METRICS
                }
                for run_key, group in averaged.groupby("run_key", sort=True)
            },
            "primary_hypotheses": primary,
            "secondary_hypotheses": secondary,
            "seed_consistency": consistency,
            "hard_subject_analysis": hard,
            "disagreement_analysis": disagreement,
            "class_metrics_by_seed": class_metrics,
            "decision": decision,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": random_state,
            "source_parquet_sha256": source_hash,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        subject_seed.to_parquet(self.output_dir / "subject_seed_metrics.parquet", index=False)
        averaged.to_parquet(self.output_dir / "subject_multiseed_metrics.parquet", index=False)
        pd.DataFrame(primary + secondary).to_parquet(
            self.output_dir / "paired_comparisons.parquet", index=False
        )
        pd.DataFrame(consistency).to_parquet(
            self.output_dir / "seed_consistency.parquet", index=False
        )
        effects.to_parquet(self.output_dir / "subject_effects.parquet", index=False)
        pd.DataFrame(fallback_units).to_csv(self.output_dir / "fallback_units.csv", index=False)
        _write_json(self.output_dir / "disagreement_analysis.json", disagreement)
        _write_json(self.output_dir / "decision.json", decision)
        _write_json(self.summary_path, summary)

        report = [
            "# Finalized auxiliary-CORN policy: subject-level analysis", "",
            "Inference uses one paired observation per subject after averaging seeds 7, 42, and 123. All comparisons use exactly aligned outer-test sequences.", "",
            "## Policy composition", "",
            f"- Joint auxiliary-CORN units: {policy_summary['selection_units_joint']}.",
            f"- Categorical fallback units: {policy_summary['selection_units_fallback']}.",
            f"- Selected lambda counts: {policy_summary['selected_lambda_counts']}.", "",
            "## Primary policy-versus-categorical hypotheses", "",
            "| Group | Metric | Reference mean | Policy mean | Mean improvement | 95% bootstrap CI | Holm p | Improved/degraded/tied |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
        for row in primary:
            report.append(
                f"| {row['feature_group']} | {row['metric']} | {row['reference_mean']:.5f} | "
                f"{row['candidate_mean']:.5f} | {row['mean_improvement']:.5f} | "
                f"[{row['bootstrap_ci_low']:.5f}, {row['bootstrap_ci_high']:.5f}] | "
                f"{row['holm_adjusted_p_value']:.5g} | "
                f"{row['subjects_improved']}/{row['subjects_degraded']}/{row['ties']} |"
            )
        report.extend([
            "", "## Secondary paired comparisons", "",
            "| Group | Candidate | Reference | Metric | Mean improvement | 95% bootstrap CI | Holm p |",
            "| --- | --- | --- | --- | ---: | --- | ---: |",
        ])
        for row in secondary:
            report.append(
                f"| {row['feature_group']} | policy | {row['reference'].split('_', 1)[0]} | "
                f"{row['metric']} | {row['mean_improvement']:.5f} | "
                f"[{row['bootstrap_ci_low']:.5f}, {row['bootstrap_ci_high']:.5f}] | "
                f"{row['holm_adjusted_p_value']:.5g} |"
            )
        report.extend([
            "", "## Aggregate outer-test metrics by seed", "",
            "| Method/group | Seed | Balanced accuracy | Macro F1 | QWK | Ordinal MAE | Severe error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in aggregate_metrics:
            report.append(
                f"| {row['run_key']} | {row['seed']} | {row['balanced_accuracy']:.4f} | "
                f"{row['macro_f1']:.4f} | {row['quadratic_weighted_kappa']:.4f} | "
                f"{row['ordinal_mae']:.4f} | {row['severe_error_rate']:.4f} |"
            )
        report.extend([
            "", "## Hard-subject analysis", "",
            "Hard subjects are the lowest quartile by categorical balanced accuracy within each feature group.", "",
            "| Group | Subset | Subjects | BA improvement | Ordinal-MAE improvement | Severe-error improvement | Both ordinal metrics improved |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in hard:
            report.append(
                f"| {row['feature_group']} | {row['difficulty_group']} | {row['subjects']} | "
                f"{row['mean_balanced_accuracy_improvement']:.5f} | "
                f"{row['mean_ordinal_mae_improvement']:.5f} | "
                f"{row['mean_severe_error_improvement']:.5f} | "
                f"{row['fraction_ordinal_metrics_both_improved']:.3f} |"
            )
        report.extend([
            "", "## Seed consistency", "",
            "| Group | Reference | Metric | Seed 7 | Seed 42 | Seed 123 | Positive seeds | Direction |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in consistency:
            report.append(
                f"| {row['feature_group']} | {row['reference']} | {row['metric']} | "
                f"{row['seed_7_mean_improvement']:.5f} | "
                f"{row['seed_42_mean_improvement']:.5f} | "
                f"{row['seed_123_mean_improvement']:.5f} | "
                f"{row['positive_seeds']}/3 | {row['direction_label']} |"
            )
        report.extend([
            "", "## Disagreement and policy-gain associations", "",
            "These correlations are descriptive and use subject-level seed-averaged values.", "",
            "| Group | Signal | Outcome | Subjects | Spearman rho | p-value | Status |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ])
        for row in disagreement:
            rho = "NA" if row['rho'] is None or not np.isfinite(row['rho']) else f"{row['rho']:.4f}"
            p_value = "NA" if row['p_value'] is None or not np.isfinite(row['p_value']) else f"{row['p_value']:.5g}"
            report.append(
                f"| {row['feature_group']} | {row['signal']} | {row['outcome']} | "
                f"{row['n_subjects']} | {rho} | {p_value} | {row['status']} |"
            )
        report.extend([
            "", "## Fallback units", "",
            "| Selection unit | Group | Seed | Fold | Outer rows |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for row in fallback_units:
            report.append(
                f"| {row['selection_id']} | {row['feature_group']} | {row['seed']} | "
                f"{row['outer_fold']} | {row['outer_test_rows']} |"
            )
        report.extend([
            "", "## Decision", "",
            f"Primary-feature-group classification: **{decision['selected_decision']}**.",
            "The categorical fallback is a post-execution protocol amendment and is reported explicitly. No outer-test result was used for lambda or branch selection.", "",
            "## Limitations", "",
            "Seeds are repeated model initializations, not independent subjects. The fallback policy was added after the protective aborts and must not be presented as preregistered. Disagreement associations are descriptive and do not establish a calibrated decision rule for new users.",
        ])
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

        decision_report = [
            "# Finalized auxiliary-CORN policy decision", "",
            f"Primary feature group: **eeg_pow**.",
            f"Decision: **{decision['selected_decision']}**.", "",
            "The decision is based on subject-level paired outer-test analysis after averaging three seeds. Hyperparameter and fallback branch decisions remain inner-validation-only.",
        ]
        self.decision_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.decision_report_path.write_text("\n".join(decision_report) + "\n", encoding="utf-8")

        return {
            "status": "completed",
            "config": _display(self.config_path),
            "summary": _display(self.summary_path),
            "report": _display(self.report_path),
            "decision_report": _display(self.decision_report_path),
            "output_directory": _display(self.output_dir),
            "subject_seed_rows": int(len(subject_seed)),
            "averaged_subject_rows": int(len(averaged)),
            "alignment_exact": True,
            "ready_for_article_reporting": True,
        }
