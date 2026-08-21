"""Final participant-level analysis of robust-shrinkage personalization.

This module is deliberately read-only with respect to completed execution
artifacts.  It reads the five locked outer folds, validates their provenance,
and writes a new statistical analysis directory.  It never imports, builds,
fits, or mutates a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from bench.analysis.paired_statistics import (
    apply_holm_by_family,
    paired_subject_statistics,
)


PROTOCOL_HASH = "b3fb15a1f69cba5c171c59fd68c0d842d69d6d0f223a49094626cd2ff96ea5c8"
PLAN_HASH = "63e8631ff2a19ccb5f6dde7e823e610657c268ab3b2e842d1b9f58483d25eb4d"
SCHEMA_VERSION = "xgboost-robust-shrinkage-final-analysis-v1"
PM_NAMES = (
    "attention", "engagement", "excitement", "stress",
    "relaxation", "interest", "focus",
)
PRIMARY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
SECONDARY_METRICS = (*PRIMARY_METRICS, "weighted_f1")
EXPECTED_ALPHAS = {1: 0.5, 2: 0.5, 3: 0.25, 4: 0.25, 5: 0.5}
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2026
IDENTITY_COLUMNS = (
    "subject_id", "pm", "outer_fold", "calibration_sample_hash",
    "evaluation_sample_hash", "q3_transform_hash",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _assert_hashes(document: Mapping[str, Any], *, label: str) -> None:
    if document.get("protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError(f"{label} protocol hash mismatch")
    if document.get("plan_hash") != PLAN_HASH:
        raise RuntimeError(f"{label} plan hash mismatch")


def summarize_delta(
    values: Iterable[float],
    *,
    metric: str,
    comparison: str,
    family: str,
    cohort: str,
    pm: str = "all_pm",
) -> dict[str, Any]:
    """Summarize one participant-level delta with the locked inference rule."""
    array = np.asarray(list(values), dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite participant deltas for {comparison}/{metric}")
    statistics = paired_subject_statistics(
        array, np.zeros_like(array), n_resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=0.95, random_state=BOOTSTRAP_SEED,
    )
    return {
        "family": family,
        "comparison": comparison,
        "cohort": cohort,
        "pm": pm,
        "metric": metric,
        "n_participants": int(statistics["n_subjects"]),
        "mean_delta": float(statistics["mean_difference"]),
        "median_delta": float(statistics["median_difference"]),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci95_low": float(statistics["ci_low"]),
        "bootstrap_ci95_high": float(statistics["ci_high"]),
        "wilcoxon_zero_method": "wilcox",
        "wilcoxon_alternative": "two-sided",
        "wilcoxon_status": statistics["wilcoxon_status"],
        "wilcoxon_statistic": statistics["wilcoxon_statistic"],
        "wilcoxon_p_value": statistics["wilcoxon_p_value"],
        "rank_biserial": float(statistics["rank_biserial"]),
        "participants_positive": int(statistics["subjects_improved"]),
        "participants_negative": int(statistics["subjects_degraded"]),
        "participants_tied": int(statistics["ties"]),
        "positive_fraction": float(statistics["fraction_improved"]),
    }


def participant_macro_deltas(participant_pm: pd.DataFrame) -> pd.DataFrame:
    """Give every participant one equal-weight mean over available PMs."""
    required = {"subject_id", "outer_fold", "pm"} | {
        f"delta_{metric}" for metric in SECONDARY_METRICS
    }
    missing = sorted(required - set(participant_pm.columns))
    if missing:
        raise ValueError(f"Participant results are missing columns: {missing}")
    if participant_pm.duplicated(["subject_id", "pm"]).any():
        raise ValueError("Participant/PM rows must be unique")
    fold_counts = participant_pm.groupby("subject_id")["outer_fold"].nunique()
    if int(fold_counts.max()) != 1:
        raise RuntimeError("A participant appears in more than one outer fold")
    aggregation = {
        f"delta_{metric}": (f"delta_{metric}", "mean")
        for metric in SECONDARY_METRICS
    }
    result = participant_pm.groupby(
        ["subject_id", "outer_fold"], sort=True, as_index=False,
    ).agg(**aggregation, pm_count=("pm", "nunique"))
    return result.sort_values("subject_id", kind="mergesort").reset_index(drop=True)


def fixed_label_balanced_accuracy(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute three-label BA and audit missing-class warning conditions."""
    required = {
        "outer_fold", "pm", "subject_id", "sample_id", "y_true",
        "zero_shot_y_pred", "adapted_y_pred",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")
    if predictions.duplicated(["pm", "sample_id"]).any():
        raise RuntimeError("Duplicate PM/sample_id predictions")
    rows: list[dict[str, Any]] = []
    for (fold, pm, subject), group in predictions.groupby(
        ["outer_fold", "pm", "subject_id"], sort=True,
    ):
        truth = group["y_true"].to_numpy(dtype=int)
        zero = group["zero_shot_y_pred"].to_numpy(dtype=int)
        adapted = group["adapted_y_pred"].to_numpy(dtype=int)
        true_classes = sorted(set(truth.tolist()))
        zero_extra = sorted(set(zero.tolist()) - set(true_classes))
        adapted_extra = sorted(set(adapted.tolist()) - set(true_classes))
        zero_ba = float(recall_score(
            truth, zero, labels=[0, 1, 2], average="macro", zero_division=0,
        ))
        adapted_ba = float(recall_score(
            truth, adapted, labels=[0, 1, 2], average="macro", zero_division=0,
        ))
        rows.append({
            "outer_fold": int(fold), "pm": str(pm), "subject_id": str(subject),
            "evaluation_windows": int(len(group)),
            "true_class_count": int(len(true_classes)),
            "true_classes": "|".join(map(str, true_classes)),
            "zero_shot_predicted_absent_classes": "|".join(map(str, zero_extra)),
            "adapted_predicted_absent_classes": "|".join(map(str, adapted_extra)),
            "zero_shot_warning_condition": bool(zero_extra),
            "adapted_warning_condition": bool(adapted_extra),
            "any_warning_condition": bool(zero_extra or adapted_extra),
            "zero_shot_balanced_accuracy_fixed_labels": zero_ba,
            "adapted_balanced_accuracy_fixed_labels": adapted_ba,
            "delta_balanced_accuracy_fixed_labels": adapted_ba - zero_ba,
        })
    audit = pd.DataFrame(rows)
    participant = audit.groupby(
        ["subject_id", "outer_fold"], sort=True, as_index=False,
    ).agg(
        delta_balanced_accuracy_fixed_labels=(
            "delta_balanced_accuracy_fixed_labels", "mean"
        ),
        pm_count=("pm", "nunique"),
    )
    return audit, participant


def validate_cross_method_identity(
    robust: pd.DataFrame,
    prior: pd.DataFrame,
    *,
    prior_method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require exact identity equality before any paired effect comparison."""
    robust_identity = robust[list(IDENTITY_COLUMNS)].copy()
    prior_identity = prior[list(IDENTITY_COLUMNS)].copy()
    keys = ["subject_id", "pm", "outer_fold"]
    if robust_identity.duplicated(keys).any() or prior_identity.duplicated(keys).any():
        raise RuntimeError(f"Duplicate identity rows for {prior_method}")
    joined = robust_identity.merge(
        prior_identity, on=keys, how="outer", suffixes=("_robust", "_prior"),
        indicator=True, validate="one_to_one",
    )
    hash_fields = (
        "calibration_sample_hash", "evaluation_sample_hash", "q3_transform_hash",
    )
    equality = {
        field: int((
            joined[f"{field}_robust"].astype(str)
            == joined[f"{field}_prior"].astype(str)
        ).sum())
        for field in hash_fields
    }
    exact = (
        len(joined) == len(robust) == len(prior)
        and joined["_merge"].eq("both").all()
        and all(count == len(joined) for count in equality.values())
    )
    audit = {
        "prior_method": prior_method,
        "robust_rows": int(len(robust)), "prior_rows": int(len(prior)),
        "joined_rows": int(len(joined)),
        "both_rows": int(joined["_merge"].eq("both").sum()),
        "hash_equal_rows": equality,
        "identity_exact": bool(exact),
    }
    return joined, audit


class FinalAnalysis:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        experiment_root: str | Path,
        output_dir: str | Path,
        xgboost_prior_results: str | Path,
        shallow_prior_results: str | Path,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.experiment_root = Path(experiment_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.xgboost_prior_results = Path(xgboost_prior_results).resolve()
        self.shallow_prior_results = Path(shallow_prior_results).resolve()
        if self.output_dir == self.experiment_root or self.output_dir.parent != self.experiment_root:
            raise ValueError("output_dir must be a new direct child of experiment_root")
        self._inputs: dict[Path, str] = {}

    def _track(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self._inputs.setdefault(resolved, sha256_file(resolved))

    def _json(self, path: Path, label: str) -> dict[str, Any]:
        self._track(path)
        return _read_json(path, label)

    def _csv(self, path: Path) -> pd.DataFrame:
        self._track(path)
        return pd.read_csv(path)

    def _parquet(self, path: Path) -> pd.DataFrame:
        self._track(path)
        return pd.read_parquet(path)

    def load_and_validate(self) -> dict[str, Any]:
        root_manifest = self._json(
            self.experiment_root / "protocol_manifest.json", "protocol manifest"
        )
        _assert_hashes(root_manifest, label="root manifest")
        if root_manifest.get("inner_model_units") != 140 or root_manifest.get("outer_evaluation_units") != 35:
            raise RuntimeError("Root plan counts differ from 140 inner / 35 outer")
        participant_frames: list[pd.DataFrame] = []
        prediction_frames: list[pd.DataFrame] = []
        candidate_frames: list[pd.DataFrame] = []
        fold_rows: list[dict[str, Any]] = []
        inner_specifications: set[str] = set()
        inner_checkpoint_paths: set[Path] = set()
        base_identities: set[str] = set()
        base_checkpoint_paths: set[Path] = set()
        decision_alphas: dict[int, float] = {}
        fold_audits: dict[str, Any] = {}

        for fold in range(1, 6):
            directory = self.experiment_root / f"fold_{fold:02d}"
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing outer fold directory: {directory}")
            decision = self._json(directory / "selection_decision.json", "selection decision")
            inner_audit = self._json(directory / "inner_leakage_audit.json", "inner audit")
            outer_audit = self._json(directory / "outer_leakage_audit.json", "outer audit")
            summary = self._json(directory / "fold_summary.json", "fold summary")
            for label, document in (
                ("selection decision", decision), ("fold summary", summary),
            ):
                _assert_hashes(document, label=f"fold {fold} {label}")
            selected = float(decision["selected_alpha"])
            if selected != EXPECTED_ALPHAS[fold] or float(summary["selected_alpha"]) != selected:
                raise RuntimeError(f"Fold {fold} selected alpha mismatch")
            decision_alphas[fold] = selected
            if decision.get("outer_test_opened") is not False:
                raise RuntimeError(f"Fold {fold} selection decision was not locked")
            inner_checks = {
                "inner_model_units": int(inner_audit.get("inner_model_units", -1)) == 28,
                "inner_subject_overlap_zero": int(inner_audit.get("max_inner_train_pseudo_subject_overlap", -1)) == 0,
                "outer_sample_overlap_zero": int(inner_audit.get("max_outer_test_sample_overlap", -1)) == 0,
                "outer_subject_overlap_zero": int(inner_audit.get("max_outer_test_subject_overlap", -1)) == 0,
                "calibration_evaluation_overlap_zero": int(inner_audit.get("max_calibration_evaluation_overlap", -1)) == 0,
                "real_outer_fold_absent": not bool(inner_audit.get("real_outer_fold_present_in_any_inner_bundle", True)),
                "chronological": bool(inner_audit.get("all_calibration_before_evaluation", False)),
            }
            outer_checks = {
                "outer_subject_overlap_zero": int(outer_audit.get("outer_subject_overlap", -1)) == 0,
                "calibration_evaluation_overlap_zero": int(outer_audit.get("calibration_evaluation_overlap_max", -1)) == 0,
                "chronological": bool(outer_audit.get("all_calibration_before_evaluation", False)),
                "opened_after_selection": bool(outer_audit.get("outer_test_opened_after_selection_decision", False)),
            }
            if not all(inner_checks.values()) or not all(outer_checks.values()):
                raise RuntimeError(f"Fold {fold} leakage audit failed")

            manifests = sorted((directory / "inner_models").rglob("manifest.json"))
            if len(manifests) != 28:
                raise RuntimeError(f"Fold {fold} has {len(manifests)} inner manifests")
            for manifest_path in manifests:
                manifest = self._json(manifest_path, "inner model manifest")
                _assert_hashes(manifest, label="inner model manifest")
                if manifest.get("status") != "complete":
                    raise RuntimeError(f"Incomplete inner model: {manifest_path}")
                specification = str(manifest.get("specification_hash", ""))
                if not specification or specification in inner_specifications:
                    raise RuntimeError("Inner specification hashes are missing or duplicated")
                inner_specifications.add(specification)
                checkpoint = manifest_path.parent / "xgboost_base.ubj"
                self._track(checkpoint)
                if sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
                    raise RuntimeError(f"Inner checkpoint hash mismatch: {checkpoint}")
                inner_checkpoint_paths.add(checkpoint.resolve())

            bases = outer_audit.get("source_bases", {})
            if set(bases) != set(PM_NAMES):
                raise RuntimeError(f"Fold {fold} does not contain seven source bases")
            for pm, base in bases.items():
                identity = str(base.get("base_checkpoint_identity_hash", ""))
                if not identity or identity in base_identities:
                    raise RuntimeError("Outer base identities are missing or duplicated")
                base_identities.add(identity)
                checkpoint = Path(base["base_checkpoint"]).resolve()
                manifest_path = Path(base["base_directory"]).resolve() / "base_checkpoint_manifest.json"
                self._track(checkpoint)
                self._track(manifest_path)
                if sha256_file(checkpoint) != base.get("base_checkpoint_sha256"):
                    raise RuntimeError(f"Outer base checkpoint changed: {checkpoint}")
                if sha256_file(checkpoint) != base.get("checkpoint_sha256_after"):
                    raise RuntimeError(f"Outer base after-hash mismatch: {checkpoint}")
                if sha256_file(manifest_path) != base.get("base_manifest_sha256"):
                    raise RuntimeError(f"Outer base manifest changed: {manifest_path}")
                if base.get("booster_hash_before") != base.get("booster_hash_after"):
                    raise RuntimeError(f"Outer booster changed for fold {fold}/{pm}")
                if not bool(base.get("source_base_unchanged")):
                    raise RuntimeError(f"Outer base audit failed for fold {fold}/{pm}")
                base_checkpoint_paths.add(checkpoint)

            participant = self._csv(directory / "outer_participant_results.csv")
            participant = participant.rename(columns={"target_transform_hash": "q3_transform_hash"})
            predictions = self._parquet(directory / "outer_predictions.parquet")
            candidates = self._csv(directory / "candidate_summary.csv")
            participant_frames.append(participant)
            prediction_frames.append(predictions)
            candidates.insert(0, "outer_fold", fold)
            candidates["selected_alpha"] = selected
            candidate_frames.append(candidates)
            fold_audits[str(fold)] = {"inner": inner_checks, "outer": outer_checks}

        participant_pm = pd.concat(participant_frames, ignore_index=True)
        predictions = pd.concat(prediction_frames, ignore_index=True)
        candidates = pd.concat(candidate_frames, ignore_index=True)
        if participant_pm.duplicated(["subject_id", "pm"]).any():
            raise RuntimeError("Duplicate participant/PM result rows")
        if len(participant_pm) != 373 or participant_pm["subject_id"].nunique() != 54:
            raise RuntimeError("Expected 373 participant/PM rows and 54 participants")
        coverage = participant_pm.groupby("subject_id")["pm"].nunique()
        if int(coverage.eq(7).sum()) != 53 or sorted(coverage[coverage.ne(7)].tolist()) != [2]:
            raise RuntimeError("Complete-seven-PM cohort differs from 53 + one partial")
        if len(inner_specifications) != 140 or len(inner_checkpoint_paths) != 140:
            raise RuntimeError("Expected 140 unique inner model units")
        if len(base_identities) != 35 or len(base_checkpoint_paths) != 35:
            raise RuntimeError("Expected 35 unique final outer base identities")

        return {
            "root_manifest": root_manifest,
            "participant_pm": participant_pm,
            "predictions": predictions,
            "candidates": candidates,
            "decision_alphas": decision_alphas,
            "fold_audits": fold_audits,
            "inner_specifications": inner_specifications,
            "base_identities": base_identities,
        }

    @staticmethod
    def _statistics_rows(
        frame: pd.DataFrame, *, metrics: Sequence[str], comparison: str,
        family: str, cohort: str,
    ) -> list[dict[str, Any]]:
        return [summarize_delta(
            frame[f"delta_{metric}"], metric=metric, comparison=comparison,
            family=family, cohort=cohort,
        ) for metric in metrics]

    def _prior(self, path: Path, *, mode: str, model: str) -> pd.DataFrame:
        frame = self._csv(path)
        selected = frame.loc[
            frame["task_type"].eq("classification")
            & frame["model"].eq(model)
            & frame["mode"].eq(mode)
            & np.isclose(frame["budget_fraction"], 0.2)
            & frame["status"].eq("completed")
        ].copy()
        if len(selected) != 373:
            raise RuntimeError(f"Expected 373 completed rows for {model}/{mode}")
        return selected

    def run(self) -> dict[str, Any]:
        if self.output_dir.exists():
            raise FileExistsError(f"Final analysis output already exists: {self.output_dir}")
        data = self.load_and_validate()
        participant_pm = data["participant_pm"]
        participant = participant_macro_deltas(participant_pm)
        if len(participant) != 54:
            raise RuntimeError("Primary participant cohort must contain 54 rows")
        complete = participant.loc[participant["pm_count"].eq(7)].copy()
        if len(complete) != 53:
            raise RuntimeError("Complete-seven-PM sensitivity cohort must contain 53 rows")

        primary_rows = self._statistics_rows(
            participant, metrics=PRIMARY_METRICS,
            comparison="robust_shrinkage_selected_vs_zero_shot",
            family="primary_three_metrics", cohort="all_54_participants",
        )
        primary = pd.DataFrame(apply_holm_by_family(primary_rows))
        primary["significant_holm_0_05"] = primary["holm_adjusted_p_value"] < 0.05

        complete_rows = self._statistics_rows(
            complete, metrics=PRIMARY_METRICS,
            comparison="robust_shrinkage_selected_vs_zero_shot",
            family="complete_seven_pm_three_metrics",
            cohort="complete_seven_pm_53_participants",
        )
        complete_statistics = pd.DataFrame(apply_holm_by_family(complete_rows))
        complete_statistics["significant_holm_0_05"] = complete_statistics["holm_adjusted_p_value"] < 0.05

        per_pm_rows: list[dict[str, Any]] = []
        for pm in PM_NAMES:
            pm_frame = participant_pm.loc[participant_pm["pm"].eq(pm)]
            for metric in PRIMARY_METRICS:
                per_pm_rows.append(summarize_delta(
                    pm_frame[f"delta_{metric}"], metric=metric,
                    comparison="robust_shrinkage_selected_vs_zero_shot",
                    family="per_pm_exploratory_21_tests", cohort="available_participants",
                    pm=pm,
                ))
        per_pm = pd.DataFrame(apply_holm_by_family(per_pm_rows))
        per_pm["analysis_role"] = "exploratory"
        per_pm["significant_holm_0_05"] = per_pm["holm_adjusted_p_value"] < 0.05

        class_audit, fixed_participant = fixed_label_balanced_accuracy(data["predictions"])
        fixed_rows = [summarize_delta(
            fixed_participant["delta_balanced_accuracy_fixed_labels"],
            metric="balanced_accuracy_fixed_labels", comparison="robust_shrinkage_selected_vs_zero_shot",
            family="fixed_label_ba_sensitivity", cohort="all_54_participants",
        )]
        fixed_complete = fixed_participant.loc[fixed_participant["pm_count"].eq(7)]
        fixed_rows.append(summarize_delta(
            fixed_complete["delta_balanced_accuracy_fixed_labels"],
            metric="balanced_accuracy_fixed_labels", comparison="robust_shrinkage_selected_vs_zero_shot",
            family="fixed_label_ba_complete_sensitivity", cohort="complete_seven_pm_53_participants",
        ))
        fixed_statistics = pd.DataFrame(fixed_rows)

        fold_rows: list[dict[str, Any]] = []
        for fold in range(1, 6):
            group = participant.loc[participant["outer_fold"].eq(fold)]
            fold_rows.append({
                "fold": fold, "selected_alpha": data["decision_alphas"][fold],
                "n_participants": int(len(group)),
                **{f"delta_{metric}": float(group[f"delta_{metric}"].mean()) for metric in PRIMARY_METRICS},
                **{f"participant_positive_fraction_{metric}": float((group[f"delta_{metric}"] > 0).mean()) for metric in PRIMARY_METRICS},
                "negative_macro_f1_fold": bool(group["delta_macro_f1"].mean() < 0),
            })
        fold_summary = pd.DataFrame(fold_rows)

        candidates = data["candidates"].copy()
        candidates["candidate_rank"] = np.nan
        # The explicit loop avoids relying on pandas group-apply index behavior
        # across supported pandas releases.
        for fold, indices in candidates.groupby("outer_fold", sort=True).groups.items():
            ordered = candidates.loc[indices].sort_values(
                ["macro_f1", "balanced_accuracy", "alpha"],
                ascending=[False, False, True], kind="mergesort",
            )
            candidates.loc[ordered.index, "candidate_rank"] = np.arange(1, len(ordered) + 1)
        frequencies = pd.Series(data["decision_alphas"]).value_counts()
        candidates["selected"] = np.isclose(candidates["alpha"], candidates["selected_alpha"])
        candidates["selection_frequency_across_folds"] = candidates["alpha"].map(frequencies).fillna(0).astype(int)
        fold_effects = fold_summary.rename(columns={"fold": "outer_fold"})
        candidates = candidates.merge(
            fold_effects[["outer_fold", *[f"delta_{metric}" for metric in PRIMARY_METRICS]]],
            on="outer_fold", how="left", validate="many_to_one",
        )

        robust_identity = participant_pm.copy()
        xgb_prior = self._prior(self.xgboost_prior_results, mode="margin_head", model="xgboost")
        shallow_prior = self._prior(self.shallow_prior_results, mode="full_model", model="torch_shallow_convnet")
        comparisons: list[dict[str, Any]] = []
        identity_audits: dict[str, Any] = {}
        for name, prior in (
            ("xgboost_margin_head_20pct", xgb_prior),
            ("shallowconvnet_full_model_20pct", shallow_prior),
        ):
            _, identity_audit = validate_cross_method_identity(
                robust_identity, prior, prior_method=name,
            )
            identity_audits[name] = identity_audit
            if not identity_audit["identity_exact"]:
                continue
            joined = robust_identity.merge(
                prior, on=["subject_id", "pm", "outer_fold"], how="inner",
                suffixes=("_robust", "_prior"), validate="one_to_one",
            )
            for metric in PRIMARY_METRICS:
                joined[f"effect_difference_{metric}"] = (
                    joined[f"delta_{metric}_robust"] - joined[f"delta_{metric}_prior"]
                )
            subject_effects = joined.groupby("subject_id", sort=True, as_index=False).agg(
                **{f"effect_difference_{metric}": (f"effect_difference_{metric}", "mean") for metric in PRIMARY_METRICS},
                pm_count=("pm", "nunique"),
            )
            for metric in PRIMARY_METRICS:
                row = summarize_delta(
                    subject_effects[f"effect_difference_{metric}"], metric=metric,
                    comparison=f"robust_shrinkage_delta_minus_{name}_delta",
                    family="cross_method_secondary_six_tests", cohort="identity_matched_54_participants",
                )
                row["identity_exact"] = True
                comparisons.append(row)
        cross_method = pd.DataFrame(apply_holm_by_family(comparisons)) if comparisons else pd.DataFrame()
        if not cross_method.empty:
            cross_method["analysis_role"] = "secondary"

        before_hashes = dict(self._inputs)
        self.output_dir.mkdir(parents=False, exist_ok=False)
        primary.to_csv(self.output_dir / "primary_participant_statistics.csv", index=False)
        participant.to_csv(self.output_dir / "participant_macro_deltas.csv", index=False)
        complete_statistics.to_csv(self.output_dir / "complete_seven_pm_sensitivity.csv", index=False)
        per_pm.to_csv(self.output_dir / "per_pm_statistics.csv", index=False)
        fold_summary.to_csv(self.output_dir / "fold_summary.csv", index=False)
        candidates.sort_values(["outer_fold", "alpha"]).to_csv(
            self.output_dir / "alpha_selection_summary.csv", index=False
        )
        class_audit.to_csv(self.output_dir / "class_coverage_audit.csv", index=False)
        fixed_statistics.to_csv(
            self.output_dir / "balanced_accuracy_fixed_labels_sensitivity.csv", index=False
        )
        if not cross_method.empty:
            cross_method.to_csv(self.output_dir / "cross_method_comparison.csv", index=False)

        after_hashes = {path: sha256_file(path) for path in before_hashes}
        unchanged = all(before_hashes[path] == after_hashes[path] for path in before_hashes)
        if not unchanged:
            raise RuntimeError("A completed source artifact changed during analysis")
        input_artifacts = [
            {
                "path": _relative(path, self.repo_root),
                "sha256_before": before_hashes[path],
                "sha256_after": after_hashes[path],
                "unchanged": before_hashes[path] == after_hashes[path],
            }
            for path in sorted(before_hashes, key=lambda item: str(item).lower())
        ]
        observed_class_counts = class_audit["true_class_count"].value_counts()
        class_counts = {
            str(class_count): int(observed_class_counts.get(class_count, 0))
            for class_count in (1, 2, 3)
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": "seven_pm_xgboost_robust_shrinkage_personalization_v1",
            "result_status": "confirmatory_complete",
            "protocol_hash": PROTOCOL_HASH, "plan_hash": PLAN_HASH,
            "inferential_unit": "participant",
            "primary_participants": 54,
            "complete_seven_pm_participants": 53,
            "participant_pm_rows": int(len(participant_pm)),
            "selected_alphas": {str(key): value for key, value in data["decision_alphas"].items()},
            "primary_statistics": primary.to_dict("records"),
            "complete_seven_pm_sensitivity": complete_statistics.to_dict("records"),
            "fixed_label_balanced_accuracy_sensitivity": fixed_statistics.to_dict("records"),
            "class_coverage": class_counts,
            "warning_condition_sets": {
                "zero_shot": int(class_audit["zero_shot_warning_condition"].sum()),
                "adapted": int(class_audit["adapted_warning_condition"].sum()),
                "either": int(class_audit["any_warning_condition"].sum()),
            },
            "cross_method_identity": identity_audits,
            "cross_method_comparison": cross_method.to_dict("records"),
            "fold_5_macro_f1_negative": bool(fold_summary.loc[fold_summary["fold"].eq(5), "negative_macro_f1_fold"].iloc[0]),
            "training_executed": False,
            "existing_runtime_artifacts_modified": False,
        }
        _write_json(self.output_dir / "final_summary.json", summary)
        audit = {
            "schema_version": SCHEMA_VERSION,
            "result_status": "confirmatory_complete",
            "protocol_hash": PROTOCOL_HASH, "plan_hash": PLAN_HASH,
            "folds_present": [1, 2, 3, 4, 5],
            "fold_audits": data["fold_audits"],
            "inner_xgboost_units": len(data["inner_specifications"]),
            "unique_outer_base_identities": len(data["base_identities"]),
            "all_input_artifacts_unchanged": unchanged,
            "input_artifacts": input_artifacts,
            "identity_audits": identity_audits,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "ci": "percentile_95"},
            "wilcoxon": {"alternative": "two-sided", "zero_method": "wilcox"},
            "multiple_testing": {
                "primary": "Holm-Bonferroni across 3 primary metrics",
                "per_pm_exploratory": "Holm-Bonferroni across 7 PM x 3 metrics",
                "cross_method_secondary": "Holm-Bonferroni across 2 methods x 3 metrics",
            },
            "training_executed": False,
        }
        _write_json(self.output_dir / "final_audit_manifest.json", audit)
        readme = self._readme(summary, primary, per_pm, fold_summary, fixed_statistics, cross_method)
        (self.output_dir / "README.md").write_text(readme, encoding="utf-8")
        return summary

    @staticmethod
    def _readme(
        summary: Mapping[str, Any], primary: pd.DataFrame, per_pm: pd.DataFrame,
        folds: pd.DataFrame, fixed: pd.DataFrame, cross: pd.DataFrame,
    ) -> str:
        lines = [
            "# XGBoost robust-shrinkage personalization: final analysis",
            "",
            "Status: `confirmatory_complete`. Inferential unit: participant. No model training was executed.",
            "",
            f"Protocol: `{PROTOCOL_HASH}`. Plan: `{PLAN_HASH}`.",
            "",
            "## Primary participant-level inference",
            "",
            "| metric | n | mean delta | median delta | bootstrap 95% CI | Wilcoxon p | Holm p | rank-biserial | positive fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in primary.to_dict("records"):
            lines.append(
                f"| {row['metric']} | {row['n_participants']} | {row['mean_delta']:.6f} | "
                f"{row['median_delta']:.6f} | [{row['bootstrap_ci95_low']:.6f}, {row['bootstrap_ci95_high']:.6f}] | "
                f"{row['wilcoxon_p_value']:.6g} | {row['holm_adjusted_p_value']:.6g} | "
                f"{row['rank_biserial']:.6f} | {row['positive_fraction']:.3f} |"
            )
        lines += ["", "## Fold robustness", "", folds.to_markdown(index=False), "",
                  "Fold 5 is explicitly retained as a negative outer-fold result; no alpha was changed after outer evaluation.",
                  "", "## Missing-class sensitivity", "",
                  "True-class coverage across 373 participant×PM evaluation sets: 1 class = 0,",
                  "2 classes = 6, 3 classes = 367. Six sets satisfy the sklearn warning",
                  "condition because at least one prediction contains a class absent from",
                  "`y_true`.", "", fixed.to_markdown(index=False), "",
                  "## Per-PM exploratory Macro F1", ""]
        lines.append(per_pm.loc[per_pm["metric"].eq("macro_f1")].to_markdown(index=False))
        lines += ["", "## Secondary cross-method comparison", ""]
        lines.append("Not performed because identity equality failed." if cross.empty else cross.to_markdown(index=False))
        lines += ["", "All source artifacts were SHA-256 checked before and after analysis and remained unchanged.", ""]
        return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        default="benchmark_results/xgboost_robust_shrinkage_personalization_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/xgboost_robust_shrinkage_personalization_v1/final_analysis",
    )
    parser.add_argument(
        "--xgboost-prior-results",
        default="benchmark_results/personalization_calibration_xgboost_v1/participant_results.csv",
    )
    parser.add_argument(
        "--shallow-prior-results",
        default=("benchmark_results/personalization_calibration_v1_classification/"
                 "execution_scopes/model_torch_shallow_convnet/participant_results.csv"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    resolve = lambda value: (repo_root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    analysis = FinalAnalysis(
        repo_root=repo_root,
        experiment_root=resolve(args.experiment_root),
        output_dir=resolve(args.output_dir),
        xgboost_prior_results=resolve(args.xgboost_prior_results),
        shallow_prior_results=resolve(args.shallow_prior_results),
    )
    result = analysis.run()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FinalAnalysis", "fixed_label_balanced_accuracy", "participant_macro_deltas",
    "summarize_delta", "validate_cross_method_identity",
]
