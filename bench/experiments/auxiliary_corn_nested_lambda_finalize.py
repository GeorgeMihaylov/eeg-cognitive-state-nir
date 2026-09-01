"""Finalize nested auxiliary-CORN selection with a categorical safe fallback.

The original nested experiment intentionally aborts a selection unit when none of
its joint candidates satisfies the validation balanced-accuracy guard.  This
module preserves that audit and materializes a complete deployable policy by
using the already trained paired categorical baseline for such units.  No model
is fitted and no outer-test result participates in the fallback decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from bench.experiments.auxiliary_corn_nested_lambda import (
    AUXILIARY_WEIGHTS,
    OUTER_ALIGNMENT_COLUMNS,
    _lambda_token,
)
from bench.experiments.ordinal_transformer import (
    _relative_path,
    _repo_path,
    _write_json,
    stable_frame_sha256,
)
from bench.validation.metrics import MetricsCalculator


_SELECTION_RE = re.compile(
    r"^(?P<feature_group>eeg_pow|eeg_only)_seed(?P<seed>\d+)_fold(?P<fold>\d+)$"
)
_REQUIRED_SOURCE_STATUSES = {"completed", "aborted_no_eligible_lambda"}


def load_auxiliary_corn_finalize_spec(path: str | Path) -> dict[str, Any]:
    resolved = _repo_path(path)
    document = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    required = {"experiment", "source", "fallback", "audit", "protocol"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Fallback finalization spec is missing sections: {missing}")
    if document["experiment"].get("type") != "auxiliary_corn_nested_lambda_finalize":
        raise ValueError(
            "experiment.type must be 'auxiliary_corn_nested_lambda_finalize'"
        )
    if document["fallback"].get("strategy") != "paired_categorical_baseline":
        raise ValueError("fallback.strategy must be paired_categorical_baseline")
    if not bool(document["fallback"].get("post_execution_protocol_amendment")):
        raise ValueError("Fallback must be marked as a post-execution amendment")
    if document["protocol"].get("selection_information") != "inner_validation_only":
        raise ValueError("selection_information must remain inner_validation_only")
    if not bool(document["protocol"].get("reuse_existing_outer_predictions")):
        raise ValueError("Finalization must reuse existing outer predictions")
    if bool(document["protocol"].get("model_training_performed", True)):
        raise ValueError("Finalization must not train models")
    if int(document["audit"].get("expected_selection_units", -1)) != 30:
        raise ValueError("audit.expected_selection_units must be 30")
    if int(document["audit"].get("expected_candidate_fold_fits", -1)) != 90:
        raise ValueError("audit.expected_candidate_fold_fits must be 90")
    if int(document["audit"].get("expected_sequences_per_policy_model", -1)) <= 0:
        raise ValueError("audit.expected_sequences_per_policy_model must be positive")
    if int(document["audit"].get("expected_subjects_per_policy_model", -1)) <= 0:
        raise ValueError("audit.expected_subjects_per_policy_model must be positive")
    return document


def _parse_selection_id(selection_id: str) -> tuple[str, int, int]:
    match = _SELECTION_RE.fullmatch(str(selection_id))
    if match is None:
        raise ValueError(f"Invalid selection_id: {selection_id!r}")
    return (
        match.group("feature_group"),
        int(match.group("seed")),
        int(match.group("fold")),
    )


def _single_baseline_prediction_file(run_directory: Path) -> Path:
    candidates = list(run_directory.glob("**/group_kfold_subject/predictions.parquet"))
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one categorical baseline prediction file under "
            f"{run_directory}, found {len(candidates)}"
        )
    return candidates[0]


@dataclass(frozen=True)
class FinalizeUnitPlan:
    selection_id: str
    feature_group: str
    seed: int
    outer_fold: int
    source_status: str
    source_prediction: Path
    source_run_directory: Path
    source_selected_checkpoint: Path | None
    target_root: Path
    source_outcome: Mapping[str, Any]

    @property
    def policy_branch(self) -> str:
        return (
            "joint_selected"
            if self.source_status == "completed"
            else "categorical_fallback"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "feature_group": self.feature_group,
            "seed": self.seed,
            "outer_fold": self.outer_fold,
            "source_status": self.source_status,
            "policy_branch": self.policy_branch,
            "source_prediction": _relative_path(self.source_prediction),
            "source_run_directory": _relative_path(self.source_run_directory),
            "source_selected_checkpoint": (
                _relative_path(self.source_selected_checkpoint)
                if self.source_selected_checkpoint is not None
                else None
            ),
            "target_root": _relative_path(self.target_root),
        }


@dataclass(frozen=True)
class FinalizePlan:
    units: tuple[FinalizeUnitPlan, ...]
    source_summary: Path
    output_root: Path
    expected_candidate_fold_fits: int

    @property
    def joint_units(self) -> int:
        return sum(item.policy_branch == "joint_selected" for item in self.units)

    @property
    def fallback_units(self) -> int:
        return sum(item.policy_branch == "categorical_fallback" for item in self.units)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_summary": _relative_path(self.source_summary),
            "output_root": _relative_path(self.output_root),
            "selection_units": len(self.units),
            "joint_units": self.joint_units,
            "fallback_units": self.fallback_units,
            "expected_candidate_fold_fits": self.expected_candidate_fold_fits,
            "units": [item.to_dict() for item in self.units],
        }




def _canonical_target_sample_id(value: Any) -> str:
    """Normalize legacy numeric and string sample identifiers equivalently."""
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return str(numeric)
        rounded = round(numeric)
        if np.isclose(numeric, rounded, rtol=0.0, atol=1e-9):
            return str(int(rounded))
        return format(numeric, ".12g")
    text = str(value)
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if np.isfinite(numeric):
        rounded = round(numeric)
        if np.isclose(numeric, rounded, rtol=0.0, atol=1e-9):
            return str(int(rounded))
        return format(numeric, ".12g")
    return text


def _semantic_outer_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    """Compare outer identities while tolerating storage-only dtype differences."""
    missing_reference = sorted(set(OUTER_ALIGNMENT_COLUMNS) - set(reference.columns))
    missing_candidate = sorted(set(OUTER_ALIGNMENT_COLUMNS) - set(candidate.columns))
    if missing_reference or missing_candidate:
        return {
            "exact": False,
            "missing_reference": missing_reference,
            "missing_candidate": missing_candidate,
        }
    left = reference.loc[:, list(OUTER_ALIGNMENT_COLUMNS)].sort_values(
        "sequence_id", kind="mergesort"
    ).reset_index(drop=True)
    right = candidate.loc[:, list(OUTER_ALIGNMENT_COLUMNS)].sort_values(
        "sequence_id", kind="mergesort"
    ).reset_index(drop=True)
    duplicate_sequence_ids = {
        "reference": int(left["sequence_id"].duplicated().sum()),
        "candidate": int(right["sequence_id"].duplicated().sum()),
    }
    mismatches: dict[str, int] = {"row_count": int(abs(len(left) - len(right)))}
    examples: dict[str, list[dict[str, Any]]] = {}
    if len(left) == len(right):
        for column in OUTER_ALIGNMENT_COLUMNS:
            if column == "target_time":
                left_values = left[column].to_numpy(dtype=float)
                right_values = right[column].to_numpy(dtype=float)
                mask = ~np.isclose(
                    left_values, right_values, rtol=1e-9, atol=1e-8, equal_nan=True
                )
            elif column in {"fold", "y_true"}:
                left_values = pd.to_numeric(left[column], errors="coerce").to_numpy()
                right_values = pd.to_numeric(right[column], errors="coerce").to_numpy()
                mask = left_values != right_values
            elif column == "target_sample_id":
                left_values = left[column].map(_canonical_target_sample_id).to_numpy()
                right_values = right[column].map(_canonical_target_sample_id).to_numpy()
                mask = left_values != right_values
            else:
                left_values = left[column].astype(str).to_numpy()
                right_values = right[column].astype(str).to_numpy()
                mask = left_values != right_values
            mismatches[column] = int(np.count_nonzero(mask))
            if mismatches[column]:
                indices = np.flatnonzero(mask)[:5]
                examples[column] = [
                    {
                        "sequence_id": str(left.iloc[index]["sequence_id"]),
                        "reference": str(left_values[index]),
                        "candidate": str(right_values[index]),
                    }
                    for index in indices
                ]
    else:
        for column in OUTER_ALIGNMENT_COLUMNS:
            mismatches[column] = max(len(left), len(right))
    exact = bool(
        not any(mismatches.values())
        and not any(duplicate_sequence_ids.values())
    )
    return {
        "exact": exact,
        "reference_rows": int(len(left)),
        "candidate_rows": int(len(right)),
        "duplicate_sequence_ids": duplicate_sequence_ids,
        "mismatches": mismatches,
        "examples": examples,
    }


def _canonical_outer_identity_sha256(frame: pd.DataFrame) -> str:
    """Hash semantic identity after canonicalizing legacy parquet dtypes."""
    missing = sorted(set(OUTER_ALIGNMENT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Outer prediction frame is missing identity columns: {missing}")
    selected = frame.loc[:, list(OUTER_ALIGNMENT_COLUMNS)].sort_values(
        "sequence_id", kind="mergesort"
    ).reset_index(drop=True)
    digest = hashlib.sha256()
    for row in selected.itertuples(index=False, name=None):
        values = dict(zip(OUTER_ALIGNMENT_COLUMNS, row))
        normalized = [
            str(values["sequence_id"]),
            int(values["fold"]),
            str(values["subject_id"]),
            str(values["record_id"]),
            str(values["source"]),
            _canonical_target_sample_id(values["target_sample_id"]),
            None if pd.isna(values["target_time"]) else round(float(values["target_time"]), 8),
            int(values["y_true"]),
            str(values["split"]),
        ]
        digest.update(json.dumps(
            normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _primary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    probabilities = frame[[f"proba_{index}" for index in range(5)]].to_numpy(
        dtype=np.float64
    )
    return MetricsCalculator.calculate_all_metrics(
        frame["y_true"].to_numpy(dtype=np.int64),
        frame["y_pred"].to_numpy(dtype=np.int64),
        probabilities,
        expected_rank=frame["categorical_expected_rank"].to_numpy(
            dtype=np.float64
        ),
    )


def _joint_subset_auxiliary_metrics(frame: pd.DataFrame) -> dict[str, Any] | None:
    if "aux_available" not in frame.columns:
        return None
    subset = frame.loc[frame["aux_available"].astype(bool)].copy()
    if subset.empty:
        return None
    probabilities = subset[
        [f"aux_class_probability_{index}" for index in range(5)]
    ].to_numpy(dtype=np.float64)
    metrics = MetricsCalculator.calculate_all_metrics(
        subset["y_true"].to_numpy(dtype=np.int64),
        subset["aux_ordinal_prediction"].to_numpy(dtype=np.int64),
        probabilities,
        expected_rank=subset["aux_expected_rank"].to_numpy(dtype=np.float64),
    )
    metrics["coverage_rows"] = int(len(subset))
    metrics["coverage_fraction"] = float(len(subset) / len(frame))
    return metrics


def _ensure_identity_columns(frame: pd.DataFrame) -> None:
    missing = sorted(set(OUTER_ALIGNMENT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Outer prediction frame is missing identity columns: {missing}")
    if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
        raise ValueError("Outer prediction sequence_id values must be complete and unique")


def _normalize_joint_frame(
    frame: pd.DataFrame,
    plan: FinalizeUnitPlan,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["fold"] = int(plan.outer_fold)
    normalized["outer_fold"] = int(plan.outer_fold)
    normalized["split"] = "outer_test"
    normalized["feature_group"] = plan.feature_group
    normalized["seed"] = int(plan.seed)
    normalized["selection_id"] = plan.selection_id
    normalized["policy_branch"] = "joint_selected"
    normalized["fallback_applied"] = False
    normalized["aux_available"] = True
    normalized["selected_model_type"] = "categorical_corn"
    normalized["source_prediction"] = _relative_path(plan.source_prediction)
    selected = plan.source_outcome["selection"]["selected"]
    selected_weight = float(selected["auxiliary_weight"])
    normalized["selected_auxiliary_weight"] = selected_weight
    normalized["auxiliary_weight"] = selected_weight
    for index in range(5):
        proba = f"proba_{index}"
        cls = f"class_probability_{index}"
        if proba not in normalized.columns and cls in normalized.columns:
            normalized[proba] = normalized[cls]
        if cls not in normalized.columns and proba in normalized.columns:
            normalized[cls] = normalized[proba]
    if "categorical_expected_rank" not in normalized.columns:
        probabilities = normalized[
            [f"proba_{index}" for index in range(5)]
        ].to_numpy(dtype=np.float64)
        normalized["categorical_expected_rank"] = probabilities @ np.arange(5)
    required_aux = {
        "aux_expected_rank",
        "aux_ordinal_prediction",
        "aux_ordinal_argmax",
        *{f"aux_class_probability_{index}" for index in range(5)},
        *{f"aux_threshold_probability_{index}" for index in range(4)},
    }
    missing_aux = sorted(required_aux - set(normalized.columns))
    if missing_aux:
        raise ValueError(f"Selected joint predictions are missing: {missing_aux}")
    _validate_probabilities(normalized, require_auxiliary=True)
    _ensure_identity_columns(normalized)
    return normalized.sort_values("sequence_id", kind="mergesort").reset_index(drop=True)


def _normalize_categorical_fallback_frame(
    frame: pd.DataFrame,
    plan: FinalizeUnitPlan,
) -> pd.DataFrame:
    if "fold" not in frame.columns:
        raise ValueError("Categorical baseline predictions do not contain fold")
    normalized = frame.loc[
        frame["fold"].astype(int) == int(plan.outer_fold)
    ].copy()
    if normalized.empty:
        raise ValueError(
            f"Categorical baseline has no rows for {plan.selection_id}"
        )
    normalized["fold"] = int(plan.outer_fold)
    normalized["outer_fold"] = int(plan.outer_fold)
    normalized["split"] = "outer_test"
    normalized["feature_group"] = plan.feature_group
    normalized["seed"] = int(plan.seed)
    normalized["selection_id"] = plan.selection_id
    normalized["head_type"] = "categorical"
    normalized["policy_branch"] = "categorical_fallback"
    normalized["fallback_applied"] = True
    normalized["aux_available"] = False
    normalized["selected_model_type"] = "categorical"
    normalized["selected_auxiliary_weight"] = np.nan
    normalized["auxiliary_weight"] = np.nan
    normalized["source_prediction"] = _relative_path(plan.source_prediction)
    for index in range(5):
        proba = f"proba_{index}"
        cls = f"class_probability_{index}"
        if proba not in normalized.columns and cls in normalized.columns:
            normalized[proba] = normalized[cls]
        if proba not in normalized.columns:
            raise ValueError(f"Categorical fallback is missing {proba}")
        normalized[cls] = normalized[proba]
    probabilities = normalized[
        [f"proba_{index}" for index in range(5)]
    ].to_numpy(dtype=np.float64)
    normalized["categorical_expected_rank"] = probabilities @ np.arange(5)
    normalized["aux_expected_rank"] = np.nan
    normalized["aux_ordinal_prediction"] = np.nan
    normalized["aux_ordinal_argmax"] = np.nan
    for index in range(5):
        normalized[f"aux_class_probability_{index}"] = np.nan
    for index in range(4):
        normalized[f"aux_threshold_probability_{index}"] = np.nan
        normalized[f"aux_threshold_logit_{index}"] = np.nan
    _validate_probabilities(normalized, require_auxiliary=False)
    _ensure_identity_columns(normalized)
    return normalized.sort_values("sequence_id", kind="mergesort").reset_index(drop=True)


def _validate_probabilities(
    frame: pd.DataFrame,
    *,
    require_auxiliary: bool,
    tolerance: float = 1e-6,
) -> None:
    probabilities = frame[[f"proba_{index}" for index in range(5)]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(probabilities).all():
        raise ValueError("Primary class probabilities are not finite")
    if float(np.min(probabilities)) < -tolerance:
        raise ValueError("Primary class probabilities contain negative values")
    if float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) > tolerance:
        raise ValueError("Primary class probabilities do not sum to one")
    recomputed = probabilities.argmax(axis=1)
    if np.count_nonzero(recomputed != frame["y_pred"].to_numpy(dtype=np.int64)):
        raise ValueError("Primary predictions differ from probability argmax")
    if not require_auxiliary:
        return
    auxiliary = frame[
        [f"aux_class_probability_{index}" for index in range(5)]
    ].to_numpy(dtype=np.float64)
    thresholds = frame[
        [f"aux_threshold_probability_{index}" for index in range(4)]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(auxiliary).all() or not np.isfinite(thresholds).all():
        raise ValueError("Joint auxiliary probabilities are not finite")
    if float(np.max(np.abs(auxiliary.sum(axis=1) - 1.0))) > tolerance:
        raise ValueError("Joint auxiliary class probabilities do not sum to one")
    if float(np.max(thresholds[:, 1:] - thresholds[:, :-1])) > tolerance:
        raise ValueError("Joint auxiliary threshold probabilities are not monotone")


class AuxiliaryCornNestedLambdaFinalizeExperiment:
    """Materialize a 30-unit policy with categorical fallback where required."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.document = load_auxiliary_corn_finalize_spec(self.spec_path)
        self.source_summary_path = _repo_path(self.document["source"]["summary_path"])
        self.output_root = _repo_path(
            output_dir or self.document["experiment"]["output_dir"]
        )

    def _load_source_summary(self) -> dict[str, Any]:
        if not self.source_summary_path.is_file():
            raise FileNotFoundError(
                f"Nested source summary not found: {self.source_summary_path}"
            )
        summary = json.loads(self.source_summary_path.read_text(encoding="utf-8"))
        if summary.get("experiment") != self.document["source"]["experiment_name"]:
            raise ValueError("Unexpected source experiment name")
        if not bool(summary.get("outer_test_selected_only", False)):
            raise ValueError("Source experiment did not preserve selected-only outer use")
        return summary

    def plan(self) -> FinalizePlan:
        summary = self._load_source_summary()
        source_plans = {
            str(item["selection_id"]): item for item in summary["plan"]["folds"]
        }
        outcomes = {str(item["selection_id"]): item for item in summary["outcomes"]}
        expected = int(self.document["audit"]["expected_selection_units"])
        if len(source_plans) != expected or len(outcomes) != expected:
            raise ValueError(
                f"Expected {expected} source units, got plans={len(source_plans)} "
                f"outcomes={len(outcomes)}"
            )
        rows: list[FinalizeUnitPlan] = []
        for selection_id in sorted(source_plans):
            source_plan = source_plans[selection_id]
            outcome = outcomes[selection_id]
            status = str(outcome.get("status"))
            if status not in _REQUIRED_SOURCE_STATUSES:
                raise ValueError(
                    f"Unsupported source status for {selection_id}: {status!r}"
                )
            group, seed, fold = _parse_selection_id(selection_id)
            if (
                str(source_plan["feature_group"]) != group
                or int(source_plan["seed"]) != seed
                or int(source_plan["outer_fold"]) != fold
            ):
                raise ValueError(f"Source plan identity mismatch for {selection_id}")
            run_directory = _repo_path(source_plan["baseline_run_directory"])
            selected_checkpoint: Path | None = None
            if status == "completed":
                prediction = _repo_path(
                    outcome["selected_outer"]["artifacts"]["predictions"]
                )
                selected_checkpoint = _repo_path(
                    outcome["selected_outer"]["selected_checkpoint"]
                )
            else:
                prediction = _single_baseline_prediction_file(run_directory)
            rows.append(FinalizeUnitPlan(
                selection_id=selection_id,
                feature_group=group,
                seed=seed,
                outer_fold=fold,
                source_status=status,
                source_prediction=prediction,
                source_run_directory=run_directory,
                source_selected_checkpoint=selected_checkpoint,
                target_root=(
                    self.output_root
                    / "selected"
                    / group
                    / f"seed_{seed}"
                    / f"fold_{fold:02d}"
                ),
                source_outcome=outcome,
            ))
        plan = FinalizePlan(
            units=tuple(rows),
            source_summary=self.source_summary_path,
            output_root=self.output_root,
            expected_candidate_fold_fits=int(
                self.document["audit"]["expected_candidate_fold_fits"]
            ),
        )
        expected_joint = int(self.document["audit"].get("expected_joint_units", plan.joint_units))
        expected_fallback = int(
            self.document["audit"].get("expected_fallback_units", plan.fallback_units)
        )
        if plan.joint_units != expected_joint or plan.fallback_units != expected_fallback:
            raise ValueError(
                "Unexpected source policy counts: "
                f"joint={plan.joint_units}/{expected_joint}, "
                f"fallback={plan.fallback_units}/{expected_fallback}"
            )
        return plan

    @staticmethod
    def render_plan(plan: FinalizePlan) -> str:
        fallback_ids = [
            item.selection_id
            for item in plan.units
            if item.policy_branch == "categorical_fallback"
        ]
        lines = [
            "# Auxiliary-CORN nested policy finalization plan",
            "",
            "No models are trained. Existing outer-test artifacts are reused only after the original inner-validation decisions.",
            "",
            f"Selection units: {len(plan.units)}.",
            f"Joint selections: {plan.joint_units}.",
            f"Categorical fallbacks: {plan.fallback_units}.",
            f"Candidate artifacts to audit: {plan.expected_candidate_fold_fits}.",
            "Fallback units:",
        ]
        lines.extend(f"- `{selection_id}`" for selection_id in fallback_ids)
        return "\n".join(lines)

    def _audit_candidate_artifacts(
        self,
        source_summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        audited: list[dict[str, Any]] = []
        for source_plan in source_summary["plan"]["folds"]:
            selection_id = str(source_plan["selection_id"])
            candidate_root = _repo_path(source_plan["candidate_root"])
            for weight in AUXILIARY_WEIGHTS:
                root = candidate_root / f"lambda_{_lambda_token(weight)}"
                manifest_path = root / "candidate_manifest.json"
                required = {
                    "model": root / "model.pt",
                    "validation_predictions": root / "validation_predictions.parquet",
                    "validation_metrics": root / "validation_metrics.json",
                }
                missing = [name for name, path in required.items() if not path.is_file()]
                if not manifest_path.is_file() or missing:
                    raise FileNotFoundError(
                        f"Incomplete candidate artifact {selection_id} lambda={weight}: "
                        f"manifest={manifest_path.is_file()} missing={missing}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "completed" or manifest.get(
                    "outer_test_used", True
                ):
                    raise ValueError(
                        f"Invalid candidate manifest {selection_id} lambda={weight}"
                    )
                if (root / "outer_test_predictions.parquet").exists():
                    raise ValueError(
                        f"Candidate unexpectedly contains outer predictions: {root}"
                    )
                audited.append({
                    "selection_id": selection_id,
                    "auxiliary_weight": float(weight),
                    "candidate_manifest": _relative_path(manifest_path),
                    "validation_rows": int(manifest["validation_rows"]),
                })
        expected = int(self.document["audit"]["expected_candidate_fold_fits"])
        if len(audited) != expected:
            raise ValueError(f"Expected {expected} candidate artifacts, got {len(audited)}")
        return {
            "status": "completed",
            "candidate_fold_fits_completed": len(audited),
            "all_outer_test_used_false": True,
            "candidates": audited,
        }

    def execute(
        self,
        plan: FinalizePlan | None = None,
        *,
        resume: bool = True,
    ) -> dict[str, Any]:
        del resume  # deterministic materialization; writes are idempotent
        resolved = self.plan() if plan is None else plan
        source_summary = self._load_source_summary()
        candidate_audit = self._audit_candidate_artifacts(source_summary)
        outcomes: list[dict[str, Any]] = []
        frames_by_fold: dict[int, list[pd.DataFrame]] = {}
        for unit in resolved.units:
            if not unit.source_prediction.is_file():
                raise FileNotFoundError(
                    f"Source outer predictions not found: {unit.source_prediction}"
                )
            source_frame = pd.read_parquet(unit.source_prediction)
            if unit.policy_branch == "joint_selected":
                frame = _normalize_joint_frame(source_frame, unit)
                selected_weight: float | None = float(
                    unit.source_outcome["selection"]["selected"]["auxiliary_weight"]
                )
                fallback_reason = None
            else:
                frame = _normalize_categorical_fallback_frame(source_frame, unit)
                selected_weight = None
                fallback_reason = str(unit.source_outcome.get("reason", ""))
            metrics = _primary_metrics(frame)
            auxiliary_metrics = _joint_subset_auxiliary_metrics(frame)
            unit.target_root.mkdir(parents=True, exist_ok=True)
            predictions_path = unit.target_root / "outer_test_predictions.parquet"
            metrics_path = unit.target_root / "outer_test_metrics.json"
            decision_path = unit.target_root / "policy_decision.json"
            manifest_path = unit.target_root / "finalized_unit_manifest.json"
            frame.to_parquet(predictions_path, index=False)
            metric_payload = {
                "primary": metrics,
                "joint_subset_auxiliary": auxiliary_metrics,
            }
            _write_json(metrics_path, metric_payload)
            decision = {
                "selection_id": unit.selection_id,
                "policy_branch": unit.policy_branch,
                "selected_model_type": (
                    "categorical_corn"
                    if unit.policy_branch == "joint_selected"
                    else "categorical"
                ),
                "selected_auxiliary_weight": selected_weight,
                "fallback_reason": fallback_reason,
                "selection_information": "inner_validation_only",
                "source_status_preserved": unit.source_status,
                "post_execution_protocol_amendment": True,
                "outer_test_used_for_decision": False,
                "existing_outer_test_artifact_reused": True,
                "source_prediction": _relative_path(unit.source_prediction),
            }
            _write_json(decision_path, decision)
            manifest = {
                "status": "completed",
                **decision,
                "feature_group": unit.feature_group,
                "seed": unit.seed,
                "outer_fold": unit.outer_fold,
                "outer_test_rows": int(len(frame)),
                "outer_test_identity_sha256": _canonical_outer_identity_sha256(frame),
                "source_outer_test_identity_sha256_raw": stable_frame_sha256(
                    frame, OUTER_ALIGNMENT_COLUMNS
                ),
                "source_selected_checkpoint": (
                    _relative_path(unit.source_selected_checkpoint)
                    if unit.source_selected_checkpoint is not None
                    else None
                ),
                "metrics": metric_payload,
                "artifacts": {
                    "predictions": _relative_path(predictions_path),
                    "metrics": _relative_path(metrics_path),
                    "policy_decision": _relative_path(decision_path),
                },
            }
            _write_json(manifest_path, manifest)
            outcomes.append(manifest)
            frames_by_fold.setdefault(unit.outer_fold, []).append(frame)

        alignment: dict[str, Any] = {}
        for fold, frames in sorted(frames_by_fold.items()):
            if len(frames) != 6:
                raise ValueError(f"Outer fold {fold} must have six policy frames")
            ordered = sorted(
                frames, key=lambda frame: str(frame["selection_id"].iloc[0])
            )
            reference = ordered[0]
            reference_id = str(reference["selection_id"].iloc[0])
            comparisons = {
                str(frame["selection_id"].iloc[0]): _semantic_outer_alignment(
                    reference, frame
                )
                for frame in ordered
            }
            canonical_hashes = {
                str(frame["selection_id"].iloc[0]): _canonical_outer_identity_sha256(
                    frame
                )
                for frame in ordered
            }
            exact = all(item["exact"] for item in comparisons.values())
            alignment_entry = {
                "exact": exact,
                "reference_selection_id": reference_id,
                "canonical_hashes": canonical_hashes,
                "comparisons": comparisons,
            }
            alignment[f"fold_{fold:02d}"] = alignment_entry
            if not exact:
                diagnostics_path = (
                    self.output_root / "cross_policy_outer_alignment_failure.json"
                )
                _write_json(diagnostics_path, {
                    "status": "failed",
                    "failed_fold": int(fold),
                    "alignment": alignment,
                })
                raise ValueError(
                    "Cross-policy outer identity mismatch in fold "
                    f"{fold}; diagnostics: {_relative_path(diagnostics_path)}"
                )

        aggregated: dict[str, Any] = {}
        all_policy_frames: list[pd.DataFrame] = []
        for group in ("eeg_pow", "eeg_only"):
            for seed in (7, 42, 123):
                rows = [
                    item
                    for item in outcomes
                    if item["feature_group"] == group and int(item["seed"]) == seed
                ]
                if len(rows) != 5:
                    raise ValueError(f"Expected five finalized folds for {group} seed={seed}")
                frames = [pd.read_parquet(item["artifacts"]["predictions"]) for item in rows]
                combined = pd.concat(frames, ignore_index=True)
                if combined["sequence_id"].duplicated().any():
                    raise ValueError(f"Duplicate outer sequences for {group} seed={seed}")
                expected_sequences = int(
                    self.document["audit"]["expected_sequences_per_policy_model"]
                )
                expected_subjects = int(
                    self.document["audit"]["expected_subjects_per_policy_model"]
                )
                if len(combined) != expected_sequences:
                    raise ValueError(
                        f"Unexpected sequence count for {group} seed={seed}: "
                        f"{len(combined)} != {expected_sequences}"
                    )
                if int(combined["subject_id"].nunique()) != expected_subjects:
                    raise ValueError(
                        f"Unexpected subject count for {group} seed={seed}: "
                        f"{combined['subject_id'].nunique()} != {expected_subjects}"
                    )
                combined = combined.sort_values(
                    ["fold", "sequence_id"], kind="mergesort"
                ).reset_index(drop=True)
                output = self.output_root / "selected" / group / f"seed_{seed}"
                output.mkdir(parents=True, exist_ok=True)
                predictions_path = output / "outer_test_predictions.parquet"
                metrics_path = output / "outer_test_metrics.json"
                combined.to_parquet(predictions_path, index=False)
                metric_payload = {
                    "primary": _primary_metrics(combined),
                    "joint_subset_auxiliary": _joint_subset_auxiliary_metrics(combined),
                }
                _write_json(metrics_path, metric_payload)
                branches = combined[["outer_fold", "policy_branch", "selected_auxiliary_weight"]].drop_duplicates()
                aggregated[f"{group}_seed_{seed}"] = {
                    "folds_completed": 5,
                    "joint_folds": int((branches["policy_branch"] == "joint_selected").sum()),
                    "fallback_folds": int((branches["policy_branch"] == "categorical_fallback").sum()),
                    "metrics": metric_payload,
                    "predictions": _relative_path(predictions_path),
                    "selection_information": "inner_validation_only",
                }
                all_policy_frames.append(combined)

        subject_input = pd.concat(all_policy_frames, ignore_index=True)
        subject_input_path = self.output_root / "subject_level_analysis_input.parquet"
        subject_input.to_parquet(subject_input_path, index=False)
        selection_rows = [{
            "selection_id": item["selection_id"],
            "feature_group": item["feature_group"],
            "seed": item["seed"],
            "outer_fold": item["outer_fold"],
            "policy_branch": item["policy_branch"],
            "selected_model_type": item["selected_model_type"],
            "selected_auxiliary_weight": item["selected_auxiliary_weight"],
            "fallback_reason": item["fallback_reason"],
        } for item in outcomes]
        selection_table = pd.DataFrame(selection_rows).sort_values(
            ["feature_group", "seed", "outer_fold"], kind="mergesort"
        )
        selection_table_path = self.output_root / "selection_policy.csv"
        selection_table.to_csv(selection_table_path, index=False)

        joint = [item for item in outcomes if item["policy_branch"] == "joint_selected"]
        fallback = [
            item for item in outcomes if item["policy_branch"] == "categorical_fallback"
        ]
        lambda_counts: dict[str, int] = {}
        for item in joint:
            key = str(item["selected_auxiliary_weight"])
            lambda_counts[key] = lambda_counts.get(key, 0) + 1
        source_reported = int(
            source_summary.get("candidate_fold_fits_trained_this_run", 0)
        ) + int(source_summary.get("candidate_fold_fits_resumed", 0))
        summary = {
            "schema_version": 1,
            "status": "completed",
            "experiment": self.document["experiment"]["name"],
            "protocol_amendment": {
                "type": "post_execution_safe_categorical_fallback",
                "reason": (
                    "The original protective protocol aborted units with no eligible "
                    "joint lambda. Finalization adds the paired categorical baseline "
                    "as a safe policy branch without changing any validation decision."
                ),
                "source_status_preserved": source_summary["status"],
                "outer_test_used_for_selection": False,
                "model_training_performed": False,
            },
            "plan": resolved.to_dict(),
            "outcomes": outcomes,
            "aggregated": aggregated,
            "candidate_audit": candidate_audit,
            "source_candidate_fold_fits_reported": source_reported,
            "candidate_fold_fits_completed": candidate_audit[
                "candidate_fold_fits_completed"
            ],
            "candidate_counter_correction": int(
                candidate_audit["candidate_fold_fits_completed"] - source_reported
            ),
            "selection_units_completed": len(outcomes),
            "selection_units_joint": len(joint),
            "selection_units_fallback": len(fallback),
            "selection_units_aborted": 0,
            "selected_lambda_counts": lambda_counts,
            "fallback_selection_ids": [item["selection_id"] for item in fallback],
            "outer_test_selected_only": True,
            "existing_outer_test_artifacts_reused": True,
            "model_training_performed": False,
            "cross_policy_outer_alignment": alignment,
            "canonical_output_audit": {
                "policy_models": 6,
                "sequences_per_policy_model": int(
                    self.document["audit"]["expected_sequences_per_policy_model"]
                ),
                "subjects_per_policy_model": int(
                    self.document["audit"]["expected_subjects_per_policy_model"]
                ),
                "subject_level_input_rows": int(len(subject_input)),
            },
            "artifacts": {
                "subject_level_analysis_input": _relative_path(subject_input_path),
                "selection_policy": _relative_path(selection_table_path),
            },
            "ready_for_subject_level_analysis": bool(
                len(outcomes) == 30
                and len(joint) + len(fallback) == 30
                and candidate_audit["candidate_fold_fits_completed"] == 90
            ),
        }
        summary_path = _repo_path(self.document["experiment"]["summary_path"])
        report_path = _repo_path(self.document["experiment"]["report_path"])
        _write_json(summary_path, summary)
        fallback_lines = [
            f"| `{item['selection_id']}` | {item['fallback_reason']} |"
            for item in fallback
        ]
        report_lines = [
            "# Finalized auxiliary-CORN selection policy",
            "",
            "> Protocol amendment: the original aborted units are preserved in the source report. This finalization adds the already trained paired categorical baseline as a safe fallback. No outer-test result was used to make a selection decision.",
            "",
            f"- Selection units completed: {len(outcomes)}/30.",
            f"- Joint auxiliary-CORN selections: {len(joint)}.",
            f"- Categorical fallbacks: {len(fallback)}.",
            f"- Candidate fold fits audited: {candidate_audit['candidate_fold_fits_completed']}/90.",
            f"- Corrected source candidate counter by: {summary['candidate_counter_correction']}.",
            f"- Selected joint lambda counts: {lambda_counts}.",
            "- Model training performed during finalization: false.",
            "- Ready for subject-level analysis: true.",
            "",
            "## Fallback units",
            "",
            "| Selection unit | Original guard outcome |",
            "| --- | --- |",
            *fallback_lines,
        ]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "completed",
            "config": _relative_path(self.spec_path),
            "summary": _relative_path(summary_path),
            "report": _relative_path(report_path),
            "output_directory": _relative_path(self.output_root),
            "selection_units_completed": len(outcomes),
            "selection_units_joint": len(joint),
            "selection_units_fallback": len(fallback),
            "candidate_fold_fits_completed": candidate_audit[
                "candidate_fold_fits_completed"
            ],
            "outer_test_selected_only": True,
            "model_training_performed": False,
            "ready_for_subject_level_analysis": summary[
                "ready_for_subject_level_analysis"
            ],
        }
        _write_json(
            self.output_root / "auxiliary_corn_nested_lambda_finalize_manifest.json",
            manifest,
        )
        return manifest


__all__ = [
    "AuxiliaryCornNestedLambdaFinalizeExperiment",
    "FinalizePlan",
    "FinalizeUnitPlan",
    "load_auxiliary_corn_finalize_spec",
]
