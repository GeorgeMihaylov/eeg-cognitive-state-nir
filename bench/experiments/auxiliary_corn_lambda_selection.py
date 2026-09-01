"""Leakage-safe setup and deterministic selection for auxiliary-CORN lambda."""

from __future__ import annotations

import inspect
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from bench.core.abstract_task import TaskSplit
from bench.experiments.ordinal_transformer import (
    _relative_path,
    _repo_path,
    _write_json,
    stable_frame_sha256,
)
from bench.experiments.ordinal_transformer_full import (
    FULL_FOLDS,
    OrdinalTransformerFullExperiment,
)
from bench.validation.metrics import MetricsCalculator
from cogstate.model_zoo import build_model
from cogstate.model_zoo.base import BaseModelAdapter


AUXILIARY_WEIGHTS = (0.25, 0.5, 1.0)
FEATURE_GROUPS = ("eeg_pow", "eeg_only")
SEEDS = (7, 42, 123)
INNER_ALIGNMENT_COLUMNS = (
    "sequence_id",
    "subject_id",
    "record_id",
    "record_group_id",
    "source",
    "target_sample_id",
    "target_time",
    "y_true",
    "outer_fold",
    "split",
)


class NoEligibleAuxiliaryWeightError(RuntimeError):
    """Raised when every joint candidate violates the categorical BA guard."""


@dataclass(frozen=True)
class LambdaValidationResult:
    auxiliary_weight: float
    balanced_accuracy: float
    severe_error_rate: float
    ordinal_mae: float
    macro_f1: float | None = None
    artifact: str | None = None

    def __post_init__(self) -> None:
        values = {
            "auxiliary_weight": self.auxiliary_weight,
            "balanced_accuracy": self.balanced_accuracy,
            "severe_error_rate": self.severe_error_rate,
            "ordinal_mae": self.ordinal_mae,
        }
        invalid = [name for name, value in values.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"Lambda validation result has non-finite fields: {invalid}")
        if self.auxiliary_weight not in AUXILIARY_WEIGHTS:
            raise ValueError(
                "auxiliary_weight must be one of "
                f"{list(AUXILIARY_WEIGHTS)}, got {self.auxiliary_weight}"
            )
        if not 0.0 <= self.balanced_accuracy <= 1.0:
            raise ValueError("balanced_accuracy must be in [0,1]")
        if not 0.0 <= self.severe_error_rate <= 1.0:
            raise ValueError("severe_error_rate must be in [0,1]")
        if self.ordinal_mae < 0.0:
            raise ValueError("ordinal_mae must be non-negative")
        if self.macro_f1 is not None and not math.isfinite(self.macro_f1):
            raise ValueError("macro_f1 must be finite when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "auxiliary_weight": float(self.auxiliary_weight),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_f1": None if self.macro_f1 is None else float(self.macro_f1),
            "severe_error_rate": float(self.severe_error_rate),
            "ordinal_mae": float(self.ordinal_mae),
            "artifact": self.artifact,
        }


@dataclass(frozen=True)
class LambdaSelectionDecision:
    baseline_balanced_accuracy: float
    ba_tolerance: float
    minimum_allowed_balanced_accuracy: float
    selected: LambdaValidationResult
    eligible: tuple[LambdaValidationResult, ...]
    rejected: tuple[LambdaValidationResult, ...]
    selection_order: tuple[str, ...] = (
        "balanced_accuracy_guard",
        "minimum_severe_error_rate",
        "minimum_ordinal_mae",
        "minimum_auxiliary_weight",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_balanced_accuracy": self.baseline_balanced_accuracy,
            "ba_tolerance": self.ba_tolerance,
            "minimum_allowed_balanced_accuracy": (
                self.minimum_allowed_balanced_accuracy
            ),
            "selected": self.selected.to_dict(),
            "eligible": [item.to_dict() for item in self.eligible],
            "rejected": [item.to_dict() for item in self.rejected],
            "selection_order": list(self.selection_order),
            "outer_test_used": False,
        }


def select_auxiliary_weight(
    baseline_metrics: Mapping[str, Any],
    candidates: Sequence[LambdaValidationResult | Mapping[str, Any]],
    *,
    ba_tolerance: float = 0.0100,
) -> LambdaSelectionDecision:
    """Select lambda using inner-validation metrics only.

    A candidate first has to remain within the absolute balanced-accuracy
    tolerance relative to the paired categorical baseline. Eligible candidates
    are ordered by severe-error rate, ordinal MAE, and finally lower lambda.
    """
    baseline_ba = float(baseline_metrics["balanced_accuracy"])
    tolerance = float(ba_tolerance)
    if not math.isfinite(baseline_ba) or not 0.0 <= baseline_ba <= 1.0:
        raise ValueError("Baseline balanced_accuracy must be finite and in [0,1]")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("ba_tolerance must be finite and non-negative")
    resolved = tuple(
        item
        if isinstance(item, LambdaValidationResult)
        else LambdaValidationResult(**dict(item))
        for item in candidates
    )
    if tuple(sorted(item.auxiliary_weight for item in resolved)) != AUXILIARY_WEIGHTS:
        raise ValueError(
            "Candidates must contain exactly one result for each lambda in "
            f"{list(AUXILIARY_WEIGHTS)}"
        )
    minimum_ba = baseline_ba - tolerance
    eligible = tuple(
        item for item in resolved if item.balanced_accuracy >= minimum_ba
    )
    rejected = tuple(
        item for item in resolved if item.balanced_accuracy < minimum_ba
    )
    if not eligible:
        raise NoEligibleAuxiliaryWeightError(
            "No auxiliary-CORN lambda satisfies the inner-validation BA guard: "
            f"baseline={baseline_ba:.6f}, tolerance={tolerance:.6f}, "
            f"minimum={minimum_ba:.6f}"
        )
    selected = min(
        eligible,
        key=lambda item: (
            item.severe_error_rate,
            item.ordinal_mae,
            item.auxiliary_weight,
        ),
    )
    return LambdaSelectionDecision(
        baseline_balanced_accuracy=baseline_ba,
        ba_tolerance=tolerance,
        minimum_allowed_balanced_accuracy=minimum_ba,
        selected=selected,
        eligible=eligible,
        rejected=rejected,
    )


def load_auxiliary_corn_lambda_setup_spec(path: str | Path) -> dict[str, Any]:
    resolved = _repo_path(path)
    document = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    required = {
        "experiment",
        "dataset",
        "task",
        "feature_groups",
        "feature_definitions",
        "auxiliary_weights",
        "seeds",
        "categorical_baseline_index",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "selection",
        "protocol",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Auxiliary-CORN lambda setup is missing sections: {missing}")
    if document["experiment"].get("type") != "auxiliary_corn_lambda_selection_setup":
        raise ValueError(
            "experiment.type must be 'auxiliary_corn_lambda_selection_setup'"
        )
    if tuple(document["feature_groups"]) != FEATURE_GROUPS:
        raise ValueError("feature_groups must be eeg_pow, eeg_only")
    if tuple(float(value) for value in document["auxiliary_weights"]) != AUXILIARY_WEIGHTS:
        raise ValueError("auxiliary_weights must be [0.25, 0.5, 1.0]")
    if tuple(int(value) for value in document["seeds"]) != SEEDS:
        raise ValueError("seeds must be [7, 42, 123]")
    if tuple(int(value) for value in document["evaluation"]["folds"]) != FULL_FOLDS:
        raise ValueError("evaluation.folds must be [1,2,3,4,5]")
    if int(document["validation"]["random_state"]) != 42:
        raise ValueError("validation.random_state must remain 42")
    if int(document["evaluation"]["random_state"]) != 42:
        raise ValueError("evaluation.random_state must remain 42")
    tolerance = float(document["selection"]["balanced_accuracy_tolerance"])
    if tolerance != 0.0100:
        raise ValueError("balanced_accuracy_tolerance must be exactly 0.0100")
    order = tuple(document["selection"]["order"])
    expected_order = (
        "balanced_accuracy_guard",
        "severe_error_rate",
        "ordinal_mae",
        "lower_auxiliary_weight",
    )
    if order != expected_order:
        raise ValueError(f"selection.order must be {list(expected_order)}")
    if document["selection"].get("no_eligible_action") != "abort_fold":
        raise ValueError("selection.no_eligible_action must be 'abort_fold'")
    if bool(document["protocol"].get("outer_test_during_setup", True)):
        raise ValueError("Setup must not evaluate outer-test predictions")
    return document


@dataclass(frozen=True)
class CategoricalBaselineReference:
    feature_group: str
    seed: int
    run_directory: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_group": self.feature_group,
            "seed": self.seed,
            "run_directory": _relative_path(self.run_directory),
        }


def load_categorical_baseline_references(
    summary_path: str | Path,
) -> tuple[CategoricalBaselineReference, ...]:
    resolved = _repo_path(summary_path)
    document = json.loads(resolved.read_text(encoding="utf-8"))
    references: dict[tuple[str, int], CategoricalBaselineReference] = {}
    for row in document.get("run_index", []):
        if str(row.get("method")) != "categorical":
            continue
        group = str(row.get("feature_group"))
        seed = int(row.get("seed"))
        key = (group, seed)
        if key in references:
            raise ValueError(f"Duplicate categorical baseline reference: {key}")
        references[key] = CategoricalBaselineReference(
            feature_group=group,
            seed=seed,
            run_directory=_repo_path(row["run_directory"]),
        )
    expected = {(group, seed) for group in FEATURE_GROUPS for seed in SEEDS}
    missing = sorted(expected - set(references))
    extra = sorted(set(references) - expected)
    if missing or extra:
        raise ValueError(
            f"Categorical baseline index mismatch: missing={missing}, extra={extra}"
        )
    return tuple(references[key] for key in sorted(expected))


@dataclass(frozen=True)
class AuxiliaryCornLambdaSetupPlan:
    baselines: tuple[CategoricalBaselineReference, ...]
    baseline_fold_materializations: int
    future_candidate_fold_fits: int
    feature_groups: tuple[str, ...]
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    auxiliary_weights: tuple[float, ...]
    ba_tolerance: float
    output_directory: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "baselines": [item.to_dict() for item in self.baselines],
            "baseline_fold_materializations": self.baseline_fold_materializations,
            "future_candidate_fold_fits": self.future_candidate_fold_fits,
            "feature_groups": list(self.feature_groups),
            "seeds": list(self.seeds),
            "folds": list(self.folds),
            "auxiliary_weights": list(self.auxiliary_weights),
            "balanced_accuracy_tolerance": self.ba_tolerance,
            "output_directory": _relative_path(self.output_directory),
            "outer_test_used": False,
        }


def _find_fold_directory(run_directory: Path, fold: int) -> Path:
    matches = list(
        run_directory.glob(
            f"**/group_kfold_subject/fold_{int(fold):02d}"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one fold_{fold:02d} directory in {run_directory}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _configure_group_validation(
    model: Any,
    split: TaskSplit,
    validation_config: Mapping[str, Any],
) -> None:
    strategy = str(validation_config.get("strategy", "group_record"))
    if strategy != "group_record":
        raise ValueError("Nested lambda setup requires group_record validation")
    group_column = str(validation_config.get("group_column", "record_group_id"))
    if group_column == "record_id":
        train_groups = np.asarray(split.record_id_train).astype(str)
    elif group_column == "subject_id":
        train_groups = np.asarray(split.subject_train).astype(str)
    elif group_column in split.row_metadata_train:
        train_groups = np.asarray(split.row_metadata_train[group_column]).astype(str)
    else:
        raise ValueError(
            f"Validation group column {group_column!r} is unavailable"
        )
    model.set_validation_groups(
        train_groups,
        subject_ids=np.asarray(split.subject_train).astype(str),
        record_ids=np.asarray(split.record_id_train).astype(str),
        outer_test_record_ids=np.asarray(split.record_id_test).astype(str),
        strategy=strategy,
        group_column=group_column,
        validation_size=float(validation_config.get("validation_size", 0.15)),
        random_state=int(validation_config.get("random_state", 42)),
    )


def build_inner_validation_prediction_frame(
    split: TaskSplit,
    detailed: Mapping[str, Any],
    *,
    feature_group: str,
    seed: int,
    outer_fold: int,
) -> pd.DataFrame:
    indices = np.asarray(detailed["indices"], dtype=np.int64)
    y_true = np.asarray(detailed["y_true"], dtype=np.int64)
    y_pred = np.asarray(detailed["y_pred"], dtype=np.int64)
    probabilities = np.asarray(detailed["class_probabilities"], dtype=np.float64)
    if probabilities.shape != (len(indices), 5):
        raise ValueError(
            "Inner-validation class probabilities must have shape "
            f"{(len(indices), 5)}, got {probabilities.shape}"
        )
    if y_true.shape != (len(indices),) or y_pred.shape != (len(indices),):
        raise ValueError("Inner-validation labels and predictions have invalid shape")
    data: dict[str, Any] = {
        "outer_train_index": indices,
        "outer_fold": int(outer_fold),
        "fold": int(outer_fold),
        "split": "inner_validation",
        "feature_group": feature_group,
        "seed": int(seed),
        "head_type": str(detailed.get("head_type", "categorical")),
        "sample_id": np.asarray(split.sample_id_train)[indices],
        "subject_id": np.asarray(split.subject_train).astype(str)[indices],
        "record_id": np.asarray(split.record_id_train).astype(str)[indices],
        "y_true": y_true,
        "y_pred": y_pred,
    }
    for name, values in split.row_metadata_train.items():
        selected = np.asarray(values)[indices]
        if name not in data:
            data[name] = selected
    frame = pd.DataFrame(data)
    for class_index in range(5):
        frame[f"proba_{class_index}"] = probabilities[:, class_index]
        frame[f"class_probability_{class_index}"] = probabilities[:, class_index]
    expected = detailed.get("categorical_expected_rank")
    if expected is None:
        expected = (probabilities * np.arange(5, dtype=np.float64)).sum(axis=1)
    frame["categorical_expected_rank"] = np.asarray(expected, dtype=np.float64)
    identity = "sequence_id" if "sequence_id" in frame.columns else "sample_id"
    if frame[identity].duplicated().any():
        raise ValueError("Inner-validation prediction identities are duplicated")
    return frame.sort_values(identity, kind="mergesort").reset_index(drop=True)


def validation_metrics_from_frame(frame: pd.DataFrame) -> dict[str, Any]:
    probabilities = frame[[f"proba_{index}" for index in range(5)]].to_numpy(
        dtype=np.float64
    )
    return MetricsCalculator.calculate_all_metrics(
        frame["y_true"].to_numpy(dtype=np.int64),
        frame["y_pred"].to_numpy(dtype=np.int64),
        probabilities,
        expected_rank=frame["categorical_expected_rank"].to_numpy(dtype=np.float64),
    )


def materialize_categorical_baseline_validation(
    reference: CategoricalBaselineReference,
    *,
    output_root: Path,
    split_builder: Callable[[Mapping[str, Any]], Mapping[str, TaskSplit]] = (
        OrdinalTransformerFullExperiment._rebuild_splits
    ),
) -> dict[str, Any]:
    """Reconstruct validation predictions from a completed categorical run.

    The existing checkpoints are loaded strictly. No optimizer, fitting call, or
    outer-test prediction is executed.
    """
    run_directory = reference.run_directory
    config_path = run_directory / "config.yaml"
    manifest_path = run_directory / "run_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Incomplete categorical baseline: {run_directory}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = config["models"]["torch_transformer"]["params"]
    dataset_config = next(iter(config.get("datasets", {}).values()), {})
    observed_group = str(dataset_config.get("feature_group", ""))
    if not observed_group:
        observed_group = {
            "eeg_only": "eeg_only",
            "eeg_pow": "eeg_pow",
            "pow_plus_eeg": "eeg_pow",
        }.get(str(dataset_config.get("feature_set", "")), "")
    if observed_group != reference.feature_group:
        raise ValueError(
            "Baseline feature group does not match the reference index: "
            f"{observed_group!r} != {reference.feature_group!r}"
        )
    if manifest.get("status") != "completed":
        raise ValueError(f"Categorical baseline is not completed: {run_directory}")
    if str(params.get("head_type", "categorical")) != "categorical":
        raise ValueError("Baseline checkpoint must use head_type=categorical")
    if int(params.get("random_state", -1)) != reference.seed:
        raise ValueError("Baseline model seed does not match the reference index")
    if int(config["validation"].get("random_state", -1)) != 42:
        raise ValueError("Baseline inner-validation split seed is not canonical 42")
    if int(config["evaluation"].get("random_state", -1)) != 42:
        raise ValueError("Baseline outer split seed is not canonical 42")
    if tuple(int(value) for value in config["evaluation"].get("folds", FULL_FOLDS)) != FULL_FOLDS:
        raise ValueError("Baseline does not contain the five canonical folds")
    if int(config.get("task_config", {}).get("random_state", -1)) != 42:
        raise ValueError("Baseline task split seed is not canonical 42")

    splits = dict(split_builder(config))
    fold_rows: list[dict[str, Any]] = []
    for fold in FULL_FOLDS:
        fold_name = f"fold_{fold:02d}"
        split = splits[fold_name]
        fold_directory = _find_fold_directory(run_directory, fold)
        checkpoint = fold_directory / "model.pt"
        validation_split_path = fold_directory / "validation_split.json"
        normalization_path = fold_directory / "normalization_stats.json"
        if not checkpoint.is_file() or not validation_split_path.is_file():
            raise ValueError(f"Baseline fold is incomplete: {fold_directory}")
        model = build_model(
            model_name="torch_transformer",
            task_type="classification",
            input_shape=tuple(split.X_train.shape[1:]),
            num_outputs=5,
            params=deepcopy(params),
        )
        if not isinstance(model, BaseModelAdapter):
            raise TypeError("Categorical Transformer did not build a model adapter")
        _configure_group_validation(model, split, config["validation"])
        model.load(checkpoint)
        _, validation_indices = model.resolve_validation_indices(split.y_train)
        saved_validation = json.loads(
            validation_split_path.read_text(encoding="utf-8")
        )
        group_column = str(config["validation"]["group_column"])
        observed_groups = sorted(
            np.unique(
                np.asarray(split.row_metadata_train[group_column]).astype(str)[
                    validation_indices
                ]
            ).tolist()
        )
        expected_groups = sorted(
            str(value)
            for value in saved_validation.get("inner_validation_group_ids", [])
        )
        if observed_groups != expected_groups:
            raise ValueError(
                f"Rebuilt validation groups differ for {reference.feature_group} "
                f"seed {reference.seed} fold {fold}"
            )
        detailed = model.validation_partition_detailed(
            split.X_train,
            split.y_train,
            validation_indices=validation_indices,
        )
        frame = build_inner_validation_prediction_frame(
            split,
            detailed,
            feature_group=reference.feature_group,
            seed=reference.seed,
            outer_fold=fold,
        )
        metrics = validation_metrics_from_frame(frame)
        target = (
            output_root
            / "baselines"
            / reference.feature_group
            / f"seed_{reference.seed}"
            / fold_name
        )
        target.mkdir(parents=True, exist_ok=True)
        predictions_path = target / "validation_predictions.parquet"
        metrics_path = target / "validation_metrics.json"
        frame.to_parquet(predictions_path, index=False)
        _write_json(metrics_path, metrics)
        normalization_audit: dict[str, Any] = {
            "artifact_present": normalization_path.is_file(),
            "feature_order_equal": None,
            "mean_max_abs_delta": None,
            "scale_max_abs_delta": None,
        }
        if normalization_path.is_file():
            saved_normalization = json.loads(
                normalization_path.read_text(encoding="utf-8")
            )
            saved_mean = np.asarray(saved_normalization["mean"], dtype=np.float64)
            saved_scale = np.asarray(saved_normalization["scale"], dtype=np.float64)
            current_mean = np.asarray(model.feature_mean_, dtype=np.float64)
            current_scale = np.asarray(model.feature_scale_, dtype=np.float64)
            normalization_audit.update({
                "feature_order_equal": list(saved_normalization["feature_names"])
                == list(split.feature_names or []),
                "mean_max_abs_delta": float(
                    np.max(np.abs(saved_mean - current_mean), initial=0.0)
                ),
                "scale_max_abs_delta": float(
                    np.max(np.abs(saved_scale - current_scale), initial=0.0)
                ),
            })
            if not normalization_audit["feature_order_equal"]:
                raise ValueError("Baseline feature order changed during reconstruction")
            if normalization_audit["mean_max_abs_delta"] != 0.0:
                raise ValueError("Baseline normalization mean changed")
            if normalization_audit["scale_max_abs_delta"] != 0.0:
                raise ValueError("Baseline normalization scale changed")
        identity_hash = stable_frame_sha256(frame, INNER_ALIGNMENT_COLUMNS)
        fold_manifest = {
            "feature_group": reference.feature_group,
            "seed": reference.seed,
            "outer_fold": fold,
            "run_directory": _relative_path(run_directory),
            "checkpoint": _relative_path(checkpoint),
            "validation_predictions": _relative_path(predictions_path),
            "validation_metrics": _relative_path(metrics_path),
            "validation_rows": int(len(frame)),
            "validation_identity_sha256": identity_hash,
            "metrics": metrics,
            "normalization_audit": normalization_audit,
            "outer_test_used": False,
            "strict_checkpoint_load": True,
        }
        _write_json(target / "validation_manifest.json", fold_manifest)
        fold_rows.append(fold_manifest)
    return {
        "feature_group": reference.feature_group,
        "seed": reference.seed,
        "run_directory": _relative_path(run_directory),
        "folds": fold_rows,
        "outer_test_used": False,
    }


class AuxiliaryCornLambdaSelectionSetupExperiment:
    """Materialize paired categorical validation references without training."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        baseline_materializer: Callable[..., Mapping[str, Any]] = (
            materialize_categorical_baseline_validation
        ),
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.document = load_auxiliary_corn_lambda_setup_spec(self.spec_path)
        self.output_root = _repo_path(
            output_dir or self.document["experiment"]["output_dir"]
        )
        self.baseline_materializer = baseline_materializer

    def plan(self) -> AuxiliaryCornLambdaSetupPlan:
        references = load_categorical_baseline_references(
            self.document["categorical_baseline_index"]
        )
        return AuxiliaryCornLambdaSetupPlan(
            baselines=references,
            baseline_fold_materializations=(
                len(FEATURE_GROUPS) * len(SEEDS) * len(FULL_FOLDS)
            ),
            future_candidate_fold_fits=(
                len(FEATURE_GROUPS)
                * len(SEEDS)
                * len(FULL_FOLDS)
                * len(AUXILIARY_WEIGHTS)
            ),
            feature_groups=FEATURE_GROUPS,
            seeds=SEEDS,
            folds=FULL_FOLDS,
            auxiliary_weights=AUXILIARY_WEIGHTS,
            ba_tolerance=float(
                self.document["selection"]["balanced_accuracy_tolerance"]
            ),
            output_directory=self.output_root,
        )

    @staticmethod
    def render_plan(plan: AuxiliaryCornLambdaSetupPlan) -> str:
        lines = [
            "# Auxiliary-CORN nested lambda setup plan",
            "",
            "This setup reconstructs inner-validation predictions from completed "
            "categorical checkpoints. It performs no model fitting and never reads "
            "outer-test predictions for lambda selection.",
            "",
            "| Group | Seed | Baseline run | Folds |",
            "| --- | ---: | --- | ---: |",
        ]
        for reference in plan.baselines:
            lines.append(
                f"| {reference.feature_group} | {reference.seed} | "
                f"`{_relative_path(reference.run_directory)}` | {len(plan.folds)} |"
            )
        lines.extend([
            "",
            f"Baseline fold materializations: {plan.baseline_fold_materializations}.",
            f"Future candidate fold fits: {plan.future_candidate_fold_fits}.",
            f"Lambda grid: {list(plan.auxiliary_weights)}.",
            f"BA tolerance: {plan.ba_tolerance:.4f} absolute.",
            "No eligible candidate causes the fold to abort before outer-test use.",
        ])
        return "\n".join(lines)

    def execute(
        self,
        plan: AuxiliaryCornLambdaSetupPlan | None = None,
        *,
        resume: bool = True,
    ) -> dict[str, Any]:
        resolved_plan = self.plan() if plan is None else plan
        outcomes: list[dict[str, Any]] = []
        for reference in resolved_plan.baselines:
            summary_path = (
                self.output_root
                / "baselines"
                / reference.feature_group
                / f"seed_{reference.seed}"
                / "baseline_validation_summary.json"
            )
            if resume and summary_path.is_file():
                result = json.loads(summary_path.read_text(encoding="utf-8"))
                outcome = "resumed"
            else:
                result = dict(
                    self.baseline_materializer(
                        reference,
                        output_root=self.output_root,
                    )
                )
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(summary_path, result)
                outcome = "materialized"
            outcomes.append({
                "feature_group": reference.feature_group,
                "seed": reference.seed,
                "outcome": outcome,
                "summary": _relative_path(summary_path),
                "result": result,
            })

        alignment: dict[str, Any] = {}
        for group in FEATURE_GROUPS:
            for fold in FULL_FOLDS:
                hashes: dict[str, str] = {}
                for seed in SEEDS:
                    row = next(
                        item
                        for item in outcomes
                        if item["feature_group"] == group and item["seed"] == seed
                    )
                    fold_row = next(
                        item
                        for item in row["result"]["folds"]
                        if int(item["outer_fold"]) == fold
                    )
                    hashes[str(seed)] = str(
                        fold_row["validation_identity_sha256"]
                    )
                exact = len(set(hashes.values())) == 1
                alignment[f"{group}_fold_{fold:02d}"] = {
                    "exact": exact,
                    "hashes": hashes,
                }
                if not exact:
                    raise ValueError(
                        f"Inner-validation identities differ across seeds: "
                        f"{group} fold {fold}"
                    )

        summary = {
            "schema_version": 1,
            "status": "completed",
            "experiment": self.document["experiment"]["name"],
            "plan": resolved_plan.to_dict(),
            "outcomes": outcomes,
            "cross_seed_validation_alignment": alignment,
            "selection_rule": {
                "balanced_accuracy_tolerance": resolved_plan.ba_tolerance,
                "order": list(self.document["selection"]["order"]),
                "no_eligible_action": "abort_fold",
            },
            "outer_test_used": False,
            "model_training_performed": False,
            "ready_for_nested_candidate_training": True,
        }
        summary_path = _repo_path(self.document["experiment"]["summary_path"])
        report_path = _repo_path(self.document["experiment"]["report_path"])
        _write_json(summary_path, summary)
        report_lines = [
            "# Auxiliary-CORN nested lambda selection setup",
            "",
            "The six categorical baselines were loaded from completed checkpoints. "
            "Their deterministic inner-validation partitions were reconstructed "
            "without fitting and without using outer-test predictions.",
            "",
            f"- Baseline folds materialized: {resolved_plan.baseline_fold_materializations}.",
            f"- Future candidate fold fits: {resolved_plan.future_candidate_fold_fits}.",
            f"- Lambda grid: {list(resolved_plan.auxiliary_weights)}.",
            f"- BA tolerance: {resolved_plan.ba_tolerance:.4f} absolute.",
            "- No-eligible action: abort the fold before outer-test evaluation.",
            "- Cross-seed inner-validation identity alignment: exact.",
            "",
            "## Selection rule",
            "",
            "1. Reject candidates below categorical validation BA minus 0.0100.",
            "2. Minimize validation severe-error rate.",
            "3. Minimize validation ordinal MAE.",
            "4. Break exact ties with the lower lambda.",
            "",
            "No lambda has been selected and no joint model has been trained in this setup task.",
        ]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        manifest = {
            "status": "completed",
            "config": _relative_path(self.spec_path),
            "summary": _relative_path(summary_path),
            "report": _relative_path(report_path),
            "output_directory": _relative_path(self.output_root),
            "outer_test_used": False,
            "model_training_performed": False,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.output_root / "auxiliary_corn_lambda_selection_setup_manifest.json",
            manifest,
        )
        return manifest


def experiment_contains_training_loop() -> bool:
    """Return whether the setup layer accidentally contains optimization code."""
    source = inspect.getsource(AuxiliaryCornLambdaSelectionSetupExperiment)
    forbidden = ("torch.optim", "loss.backward", ".fit(", "runner.run(")
    return any(token in source for token in forbidden)


__all__ = [
    "AUXILIARY_WEIGHTS",
    "FEATURE_GROUPS",
    "SEEDS",
    "AuxiliaryCornLambdaSelectionSetupExperiment",
    "AuxiliaryCornLambdaSetupPlan",
    "CategoricalBaselineReference",
    "LambdaSelectionDecision",
    "LambdaValidationResult",
    "NoEligibleAuxiliaryWeightError",
    "build_inner_validation_prediction_frame",
    "experiment_contains_training_loop",
    "load_auxiliary_corn_lambda_setup_spec",
    "load_categorical_baseline_references",
    "materialize_categorical_baseline_validation",
    "select_auxiliary_weight",
    "validation_metrics_from_frame",
]
