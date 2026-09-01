"""Subject-level repeated-seed analysis for ordinal Transformer heads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from bench.analysis.ordinal_transformer_statistics import (
    FEATURE_GROUPS,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    SUBJECT_METRICS,
    build_subject_effect_types,
    calculate_prediction_metrics,
    calculate_subject_metrics,
    categorical_expected_rank,
    hard_subject_summary,
    paired_metric_comparison,
    require_six_way_alignment,
)
from bench.analysis.paired_statistics import apply_holm_by_family
from bench.validation.metrics import MetricsCalculator


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("categorical", "coral", "corn")
SEEDS = (7, 42, 123)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


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
        raise ValueError(f"Expected one unified prediction artifact in {run_directory}")
    return paths[0]


def average_subject_metrics_across_seeds(subject_seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated initializations before any paired inference."""
    expected_rows = len(METHODS) * len(FEATURE_GROUPS) * len(SEEDS) * 53
    if len(subject_seed_metrics) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} subject x seed rows, observed {len(subject_seed_metrics)}"
        )
    counts = subject_seed_metrics.groupby(
        ["run_key", "subject_id"], sort=True
    )["seed"].nunique()
    if not bool((counts == len(SEEDS)).all()):
        raise ValueError("Every method/group/subject must contain exactly three seeds")
    numeric = [metric for metric in SUBJECT_METRICS if metric in subject_seed_metrics]
    rows: list[dict[str, Any]] = []
    for (run_key, subject_id), group in subject_seed_metrics.groupby(
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
        row.update({metric: float(group[metric].mean(skipna=True)) for metric in numeric})
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(METHODS) * len(FEATURE_GROUPS) * 53:
        raise ValueError("Averaging did not produce one row per subject/method/group")
    return result


def build_multiseed_hypotheses(
    averaged: pd.DataFrame, *, n_resamples: int, random_state: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    for group in FEATURE_GROUPS:
        for method in ("coral", "corn"):
            for metric in PRIMARY_METRICS:
                primary.append(paired_metric_comparison(
                    averaged,
                    candidate_key=f"{method}_{group}",
                    reference_key=f"categorical_{group}",
                    metric=metric,
                    family=f"primary_{group}",
                    hypothesis_tier="primary",
                    n_resamples=n_resamples,
                    random_state=random_state,
                ))
            for metric in SECONDARY_METRICS:
                secondary.append(paired_metric_comparison(
                    averaged,
                    candidate_key=f"{method}_{group}",
                    reference_key=f"categorical_{group}",
                    metric=metric,
                    family=f"secondary_{group}",
                    hypothesis_tier="secondary",
                    n_resamples=n_resamples,
                    random_state=random_state,
                ))
    return apply_holm_by_family(primary), apply_holm_by_family(secondary)


def seed_consistency_table(subject_seed_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in FEATURE_GROUPS:
        for method in ("coral", "corn"):
            for metric in PRIMARY_METRICS + SECONDARY_METRICS:
                effects: list[float] = []
                row: dict[str, Any] = {
                    "feature_group": group,
                    "candidate": method,
                    "reference": "categorical",
                    "metric": metric,
                }
                for seed in SEEDS:
                    subset = subject_seed_metrics[subject_seed_metrics["seed"] == seed]
                    candidate = subset[subset["run_key"] == f"{method}_{group}"].sort_values("subject_id")
                    reference = subset[subset["run_key"] == f"categorical_{group}"].sort_values("subject_id")
                    if not candidate["subject_id"].reset_index(drop=True).equals(
                        reference["subject_id"].reset_index(drop=True)
                    ):
                        raise ValueError("Seed-level subject pairs differ")
                    raw = candidate[metric].to_numpy(float) - reference[metric].to_numpy(float)
                    improvement = raw if metric in HIGHER_IS_BETTER else -raw
                    effect = float(np.nanmean(improvement))
                    effects.append(effect)
                    row[f"seed_{seed}_mean_improvement"] = effect
                positives = int(np.count_nonzero(np.asarray(effects) > 0))
                negatives = int(np.count_nonzero(np.asarray(effects) < 0))
                row.update({
                    "positive_seeds": positives,
                    "minimum_seed_effect": float(np.min(effects)),
                    "maximum_seed_effect": float(np.max(effects)),
                    "between_seed_standard_deviation": float(np.std(effects, ddof=0)),
                    "direction_label": (
                        f"positive_in_{positives}_of_3"
                        if not (positives and negatives)
                        else "changes_sign"
                    ),
                })
                rows.append(row)
    return rows


def tradeoff_table(averaged: pd.DataFrame) -> list[dict[str, Any]]:
    indexed = averaged.set_index(["run_key", "subject_id"])
    rows: list[dict[str, Any]] = []
    for group in FEATURE_GROUPS:
        subjects = sorted(averaged.loc[
            averaged["run_key"] == f"categorical_{group}", "subject_id"
        ])
        for method in ("coral", "corn"):
            for quality in ("balanced_accuracy", "macro_f1"):
                counts = {
                    "ordinal_improved_quality_improved": 0,
                    "ordinal_improved_quality_degraded": 0,
                    "ordinal_degraded_quality_improved": 0,
                    "both_degraded": 0,
                    "ties": 0,
                }
                for subject in subjects:
                    ref = indexed.loc[(f"categorical_{group}", subject)]
                    cand = indexed.loc[(f"{method}_{group}", subject)]
                    ordinal = float(ref["ordinal_mae"] - cand["ordinal_mae"])
                    quality_delta = float(cand[quality] - ref[quality])
                    if ordinal > 0 and quality_delta > 0:
                        key = "ordinal_improved_quality_improved"
                    elif ordinal > 0 and quality_delta < 0:
                        key = "ordinal_improved_quality_degraded"
                    elif ordinal < 0 and quality_delta > 0:
                        key = "ordinal_degraded_quality_improved"
                    elif ordinal < 0 and quality_delta < 0:
                        key = "both_degraded"
                    else:
                        key = "ties"
                    counts[key] += 1
                rows.append({
                    "feature_group": group,
                    "candidate": method,
                    "quality_metric": quality,
                    **counts,
                })
    return rows


def select_multiseed_decision(
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    consistency: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    consistency_index = {
        (row["feature_group"], row["candidate"], row["metric"]): row
        for row in consistency
    }
    qualified: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []
    for group in ("eeg_pow", "eeg_only"):
        for method in ("coral", "corn"):
            primary_rows = [
                row for row in primary
                if row["feature_group"] == group and row["candidate"] == f"{method}_{group}"
            ]
            supported = [
                row for row in primary_rows
                if row["holm_adjusted_p_value"] < 0.05
                and row["bootstrap_ci_low"] > 0
                and consistency_index[(group, method, row["metric"])]["positive_seeds"] >= 2
            ]
            quality_rows = [
                row for row in secondary
                if row["feature_group"] == group
                and row["candidate"] == f"{method}_{group}"
                and row["metric"] in {"balanced_accuracy", "macro_f1"}
            ]
            confirmed_quality_loss = any(
                row["holm_adjusted_p_value"] < 0.05 and row["bootstrap_ci_high"] < 0
                for row in quality_rows
            )
            candidate = {
                "feature_group": group,
                "head": method,
                "supported_primary_metrics": [row["metric"] for row in supported],
                "confirmed_quality_loss": confirmed_quality_loss,
                "score": float(sum(row["mean_improvement"] for row in supported)),
            }
            if supported and not confirmed_quality_loss:
                qualified.append(candidate)
            elif supported and confirmed_quality_loss:
                auxiliary.append(candidate)
    if qualified:
        selected = sorted(qualified, key=lambda row: (row["score"], row["feature_group"] == "eeg_pow"), reverse=True)[0]
        decision = "A"
        next_experiment = "Confirm the selected pure ordinal head on an external or nested-validation cohort."
    elif auxiliary:
        selected = sorted(auxiliary, key=lambda row: row["score"], reverse=True)[0]
        decision = "B"
        next_experiment = "Categorical cross-entropy with an auxiliary ordinal loss."
    else:
        selected = None
        decision = "C"
        next_experiment = "Subject-risk/worst-subject optimization or joint categorical-continuous modelling."
    evidence_against: list[str] = []
    if selected is not None:
        selected_group = selected["feature_group"]
        selected_head = selected["head"]
        for metric in PRIMARY_METRICS:
            row = consistency_index[(selected_group, selected_head, metric)]
            if row["positive_seeds"] < 3:
                evidence_against.append(
                    f"{metric} changes sign across seeds ({row['positive_seeds']}/3 positive)."
                )
        for row in secondary:
            if (
                row["feature_group"] == selected_group
                and row["candidate"] == f"{selected_head}_{selected_group}"
                and row["metric"] in {"balanced_accuracy", "macro_f1"}
                and row.get("mean_improvement") is not None
                and float(row["mean_improvement"]) < 0
            ):
                evidence_against.append(
                    f"Mean {row['metric']} change is negative ({row['mean_improvement']:+.5f}) "
                    "although it is not Holm-confirmed."
                )
    return {
        "selected_decision": decision,
        "selected_head": None if selected is None else selected["head"],
        "primary_feature_group": "eeg_pow",
        "control_feature_group": "eeg_only",
        "evidence_for": qualified or auxiliary,
        "evidence_against": (
            evidence_against if selected is not None else [
                "No head satisfied adjusted-p, bootstrap-CI, and two-of-three-seed criteria without a confirmed categorical-quality cost."
            ]
        ),
        "remaining_uncertainty": [
            "Only three initialization seeds were evaluated.",
            "All inference is internal to the same 53 subjects and one benchmark dataset.",
        ],
        "next_experiment": next_experiment,
        "runs_not_recommended": [
            "Additional pure ordinal-head seeds before the selected next experiment",
            "New preprocessing variants as a response to this head comparison",
        ],
    }


class OrdinalTransformerMultiseedStatistics:
    def __init__(
        self, config_path: str | Path, *, output_dir: str | Path | None = None
    ) -> None:
        self.config_path = _repo_path(config_path)
        self.document = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        analysis = self.document["analysis"]
        if analysis.get("type") != "ordinal_transformer_multiseed_statistics":
            raise ValueError("Expected ordinal_transformer_multiseed_statistics analysis")
        self.run_summary_path = _repo_path(analysis["run_summary"])
        self.output_dir = _repo_path(output_dir or analysis["output_dir"])
        self.report_path = _repo_path(analysis["report_path"])
        self.summary_path = _repo_path(analysis["summary_path"])
        self.decision_path = _repo_path(analysis["decision_report_path"])

    def _run_index(self) -> list[dict[str, Any]]:
        payload = json.loads(self.run_summary_path.read_text(encoding="utf-8"))
        rows = payload["run_index"]
        keys = {(row["method"], row["feature_group"], int(row["seed"])) for row in rows}
        expected = {
            (method, group, seed)
            for method in METHODS for group in FEATURE_GROUPS for seed in SEEDS
        }
        if keys != expected or len(rows) != len(expected):
            raise ValueError("Run index must contain exactly 18 method/group/seed runs")
        return rows

    def plan(self) -> dict[str, Any]:
        rows = self._run_index()
        return {
            "valid": True,
            "analysis_unit": "subject_id after averaging three repeated seeds",
            "independent_subjects": 53,
            "seeds": list(SEEDS),
            "runs": rows,
            "primary_families": [f"primary_{group}" for group in FEATURE_GROUPS],
            "secondary_families": [f"secondary_{group}" for group in FEATURE_GROUPS],
            "bootstrap_samples": int(self.document["analysis"]["bootstrap_samples"]),
            "output_dir": _display(self.output_dir),
        }

    @staticmethod
    def render_plan(plan: Mapping[str, Any]) -> str:
        lines = [
            "Ordinal Transformer multiseed statistical plan",
            f"Analysis unit: {plan['analysis_unit']}",
            f"Independent subjects: {plan['independent_subjects']}; seeds: {plan['seeds']}",
            f"Runs: {len(plan['runs'])}; bootstrap samples: {plan['bootstrap_samples']}",
            f"Primary families: {plan['primary_families']}",
            f"Secondary families: {plan['secondary_families']}",
            "Plan-only writes no statistical artifacts.",
        ]
        return "\n".join(lines)

    def execute(self) -> dict[str, Any]:
        run_rows = self._run_index()
        by_seed: dict[int, dict[str, pd.DataFrame]] = {seed: {} for seed in SEEDS}
        for row in run_rows:
            key = f"{row['method']}_{row['feature_group']}"
            by_seed[int(row["seed"])][key] = pd.read_parquet(
                _prediction_file(_repo_path(row["run_directory"]))
            )
        alignment = {
            str(seed): require_six_way_alignment(frames)
            for seed, frames in by_seed.items()
        }
        reference = by_seed[42]["categorical_eeg_only"]
        between_seed: dict[str, Any] = {}
        for seed, frames in by_seed.items():
            for key, frame in frames.items():
                audit = require_six_way_alignment({
                    method_key: (
                        reference if method_key == "categorical_eeg_only" else
                        frame if method_key == key else by_seed[42][method_key]
                    )
                    for method_key in (
                        "categorical_eeg_only", "coral_eeg_only", "corn_eeg_only",
                        "categorical_eeg_pow", "coral_eeg_pow", "corn_eeg_pow",
                    )
                })
                between_seed[f"{key}_seed{seed}"] = audit["exact_match"]
        subject_frames: list[pd.DataFrame] = []
        for seed, frames in by_seed.items():
            values = calculate_subject_metrics(frames)
            values["seed"] = seed
            subject_frames.append(values)
        subject_seed = pd.concat(subject_frames, ignore_index=True)
        averaged = average_subject_metrics_across_seeds(subject_seed)
        aggregate_metrics: list[dict[str, Any]] = []
        source_metrics: list[dict[str, Any]] = []
        class_metrics: list[dict[str, Any]] = []
        for seed, frames in by_seed.items():
            for run_key, frame in sorted(frames.items()):
                method, feature_group = run_key.split("_", 1)
                expected_rank = (
                    categorical_expected_rank(frame)
                    if method == "categorical"
                    else frame["expected_rank"].to_numpy(dtype=float)
                )
                aggregate_metrics.append({
                    "run_key": run_key,
                    "method": method,
                    "feature_group": feature_group,
                    "seed": seed,
                    **calculate_prediction_metrics(frame, expected_rank=expected_rank),
                })
                for source, group_frame in frame.groupby("source", sort=True):
                    source_expected = (
                        categorical_expected_rank(group_frame)
                        if method == "categorical"
                        else group_frame["expected_rank"].to_numpy(dtype=float)
                    )
                    source_metrics.append({
                        "run_key": run_key,
                        "method": method,
                        "feature_group": feature_group,
                        "seed": seed,
                        "source": str(source),
                        "subjects": int(group_frame["subject_id"].nunique()),
                        **calculate_prediction_metrics(
                            group_frame, expected_rank=source_expected
                        ),
                    })
                for row in MetricsCalculator.calculate_class_metrics(
                    frame["y_true"].to_numpy(dtype=int),
                    frame["y_pred"].to_numpy(dtype=int),
                    labels=np.arange(5),
                ):
                    class_metrics.append({
                        "run_key": run_key,
                        "method": method,
                        "feature_group": feature_group,
                        "seed": seed,
                        **row,
                    })
        config = self.document["analysis"]
        primary, secondary = build_multiseed_hypotheses(
            averaged,
            n_resamples=int(config["bootstrap_samples"]),
            random_state=int(config["random_state"]),
        )
        consistency = seed_consistency_table(subject_seed)
        effects = build_subject_effect_types(averaged)
        hard = hard_subject_summary(effects)
        tradeoffs = tradeoff_table(averaged)
        decision = select_multiseed_decision(primary, secondary, consistency)
        source_path = _repo_path(self.document["expected"]["source_parquet"])
        source_hash = _sha256(source_path)
        if source_hash != self.document["expected"]["source_parquet_sha256"]:
            raise ValueError("Source Parquet SHA-256 changed")
        summary = {
            "schema_version": "ordinal-transformer-multiseed-statistics-v1",
            "analysis_unit": "subject_id",
            "independent_subjects": 53,
            "repeated_measurement": "model initialization seed, averaged within subject",
            "seeds": list(SEEDS),
            "subject_seed_rows": int(len(subject_seed)),
            "averaged_subject_rows": int(len(averaged)),
            "bootstrap_samples": int(config["bootstrap_samples"]),
            "bootstrap_seed": int(config["random_state"]),
            "alignment_within_seed": alignment,
            "alignment_between_seeds": {"all_exact": all(between_seed.values()), "checks": between_seed},
            "aggregate_metrics_by_seed": aggregate_metrics,
            "subject_level_multiseed_means": {
                run_key: {
                    metric: float(group[metric].mean(skipna=True))
                    for metric in SUBJECT_METRICS
                }
                for run_key, group in averaged.groupby("run_key", sort=True)
            },
            "source_metrics_by_seed": source_metrics,
            "class_metrics_by_seed": class_metrics,
            "primary_hypotheses": primary,
            "secondary_hypotheses": secondary,
            "seed_consistency": consistency,
            "subject_heterogeneity": {
                "effect_type_counts": effects.groupby(
                    ["feature_group", "candidate", "effect_type"], sort=True
                ).size().rename("subjects").reset_index().to_dict(orient="records"),
                "hard_subjects": hard,
            },
            "categorical_quality_tradeoffs": tradeoffs,
            "decision": decision,
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
        effects.to_parquet(self.output_dir / "subject_effect_types.parquet", index=False)
        _write_json(self.output_dir / "decision.json", decision)
        _write_json(self.summary_path, summary)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        report = [
            "# Ordinal Transformer multiseed statistics", "",
            "The inferential unit is one subject. Seeds are repeated initializations and were averaged within each of 53 subjects before paired inference.", "",
            "## Primary hypotheses", "",
            "| Group | Head | Metric | Mean improvement | 95% bootstrap CI | Holm p | Improved/degraded/tied | Positive seeds |",
            "| --- | --- | --- | ---: | --- | ---: | --- | ---: |",
        ]
        lookup = {(row["feature_group"], row["candidate"], row["metric"]): row for row in consistency}
        for row in primary:
            method = row["candidate"].split("_", 1)[0]
            seed_row = lookup[(row["feature_group"], method, row["metric"])]
            report.append(
                f"| {row['feature_group']} | {method} | {row['metric']} | {row['mean_improvement']:.5f} | "
                f"[{row['bootstrap_ci_low']:.5f}, {row['bootstrap_ci_high']:.5f}] | "
                f"{row['holm_adjusted_p_value']:.5g} | {row['subjects_improved']}/{row['subjects_degraded']}/{row['ties']} | "
                f"{seed_row['positive_seeds']}/3 |"
            )
        report.extend([
            "", "## Aggregate metrics by seed", "",
            "| Method/group | Seed | Balanced accuracy | Macro F1 | QWK | Ordinal MAE | Severe error |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in aggregate_metrics:
            report.append(
                f"| {row['run_key']} | {row['seed']} | {row['balanced_accuracy']:.4f} | "
                f"{row['macro_f1']:.4f} | {row['quadratic_weighted_kappa']:.4f} | "
                f"{row['ordinal_mae']:.4f} | {row['severe_error_rate']:.4f} |"
            )
        report.extend([
            "", "## Secondary hypotheses", "",
            "| Group | Head | Metric | Mean improvement | 95% bootstrap CI | Holm p |", "| --- | --- | --- | ---: | --- | ---: |",
        ])
        for row in secondary:
            report.append(
                f"| {row['feature_group']} | {row['candidate'].split('_', 1)[0]} | {row['metric']} | "
                f"{row['mean_improvement']:.5f} | [{row['bootstrap_ci_low']:.5f}, {row['bootstrap_ci_high']:.5f}] | "
                f"{row['holm_adjusted_p_value']:.5g} |"
            )
        report.extend([
            "", "## Seed consistency", "",
            "| Group | Head | Metric | Seed 7 | Seed 42 | Seed 123 | Positive seeds | Label |", "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in consistency:
            report.append(
                f"| {row['feature_group']} | {row['candidate']} | {row['metric']} | "
                f"{row['seed_7_mean_improvement']:.5f} | {row['seed_42_mean_improvement']:.5f} | "
                f"{row['seed_123_mean_improvement']:.5f} | {row['positive_seeds']}/3 | {row['direction_label']} |"
            )
        report.extend([
            "", "## Subject heterogeneity and hard subjects", "",
            "Primary comparison rows include the 10th/25th/50th/75th/90th percentiles, worst- and best-quartile means, and improved/degraded/tied subject counts. Hard-subject summaries use the lowest categorical baseline quartile within each feature group.", "",
            "| Group | Candidate | Difficulty quartile | Subjects | Ordinal-MAE improvement | Severe-error improvement | Fraction improved |", "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for row in hard:
            report.append(
                f"| {row['feature_group']} | {row['candidate']} | {row['difficulty_group']} | "
                f"{row['subjects']} | {row['mean_ordinal_mae_improvement']:.5f} | "
                f"{row['mean_severe_error_improvement']:.5f} | {row['fraction_ordinal_mae_improved']:.3f} |"
            )
        report.extend([
            "", "## BA and macro-F1 trade-offs", "",
            "| Group | Head | Quality metric | Ordinal+quality improved | Ordinal improved/quality degraded | Ordinal degraded/quality improved | Both degraded | Ties |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in tradeoffs:
            report.append(
                f"| {row['feature_group']} | {row['candidate']} | {row['quality_metric']} | "
                f"{row['ordinal_improved_quality_improved']} | {row['ordinal_improved_quality_degraded']} | "
                f"{row['ordinal_degraded_quality_improved']} | {row['both_degraded']} | {row['ties']} |"
            )
        report.extend([
            "", "## Feature-group interpretation", "",
            "EEG+POW is the primary feature group and EEG-only is the control. Effects are reported separately; a benefit confined to EEG-only is not interpreted as a universal ordinal-head advantage.", "",
            "## Source- and class-level results", "",
            "Source-level and per-class metrics for all 18 runs are stored in the JSON summary. They are descriptive because sources and classes are not independent inferential units.", "",
            "## Limitations", "",
            "Seeds are not independent people; source/fold views are descriptive; three seeds do not characterize the full initialization distribution.",
        ])
        self.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        decision_lines = [
            "# Ordinal Transformer multiseed decision", "",
            f"Selected decision: **{decision['selected_decision']}**.",
            f"Selected head: **{decision['selected_head'] or 'none'}**.",
            f"Primary feature group: {decision['primary_feature_group']}; control: {decision['control_feature_group']}.",
            "", "## Evidence for", "",
            *(
                [
                    f"- {row['head']} on {row['feature_group']}: supported primary metrics "
                    f"{', '.join(row['supported_primary_metrics'])}; confirmed BA/F1 loss={row['confirmed_quality_loss']}."
                    for row in decision["evidence_for"]
                ] or ["- No candidate met the complete selection rule."]
            ),
            "", "## Evidence against", "",
            *([f"- {value}" for value in decision["evidence_against"]] or ["- No criterion-level evidence against the selected head."]),
            "", "## Remaining uncertainty", "",
            *[f"- {value}" for value in decision["remaining_uncertainty"]],
            "", "## Next experiment", "",
            decision["next_experiment"], "",
            "Runs not recommended:",
            *[f"- {value}" for value in decision["runs_not_recommended"]],
        ]
        self.decision_path.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
        return {
            "status": "completed",
            "decision": decision,
            "subject_seed_rows": len(subject_seed),
            "averaged_subject_rows": len(averaged),
            "artifacts": {
                "output_dir": _display(self.output_dir),
                "summary": _display(self.summary_path),
                "report": _display(self.report_path),
                "decision": _display(self.decision_path),
            },
        }


__all__ = [
    "OrdinalTransformerMultiseedStatistics",
    "average_subject_metrics_across_seeds",
    "build_multiseed_hypotheses",
    "seed_consistency_table",
    "select_multiseed_decision",
    "tradeoff_table",
]
