"""Technical one-fold smoke experiment for the auxiliary-CORN Transformer."""

from __future__ import annotations

import gc
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from bench.bench_runner import (
    BenchmarkRunner,
    CompletedBenchmarkRun,
    benchmark_config_hash,
)
from cogstate.model_zoo import build_model

from .ordinal_transformer import (
    SMOKE_ALIGNMENT_COLUMNS,
    OrdinalTransformerSmokeExperiment,
    _jsonable,
    _relative_path,
    _repo_path,
    _write_json,
    prediction_alignment,
)


AUXILIARY_WEIGHTS = (0.25, 0.5, 1.0)
JOINT_HEAD_TYPE = "categorical_corn"


def load_auxiliary_corn_smoke_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate the fixed task-7V technical smoke specification."""
    resolved = _repo_path(path)
    document = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    required = {
        "experiment",
        "dataset",
        "task",
        "feature_group",
        "auxiliary_weights",
        "seeds",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "protocol",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(
            f"Auxiliary-CORN smoke experiment is missing sections: {missing}"
        )
    experiment_type = str(document["experiment"].get("type", "")).strip()
    if experiment_type != "auxiliary_corn_transformer_smoke":
        raise ValueError(
            "experiment.type must be 'auxiliary_corn_transformer_smoke'"
        )
    weights = tuple(float(value) for value in document["auxiliary_weights"])
    if weights != AUXILIARY_WEIGHTS:
        raise ValueError(
            "Technical smoke auxiliary_weights must be exactly "
            "[0.25, 0.5, 1.0] in ascending order"
        )
    if not all(math.isfinite(value) and value >= 0 for value in weights):
        raise ValueError("auxiliary_weights must be finite and non-negative")
    seeds = [int(value) for value in document["seeds"]]
    if seeds != [42]:
        raise ValueError("Technical auxiliary-CORN smoke supports only seed 42")
    folds = [int(value) for value in document["evaluation"].get("folds", [])]
    if folds != [1]:
        raise ValueError("Technical auxiliary-CORN smoke supports only outer fold 1")
    max_epochs = int(document["protocol"].get("max_epochs", 3))
    if max_epochs < 1 or max_epochs > 3:
        raise ValueError("Technical auxiliary-CORN smoke max_epochs must be in [1,3]")
    feature = document["feature_group"]
    if str(feature.get("name", "")).strip().lower() != "eeg_pow":
        raise ValueError("Technical auxiliary-CORN smoke requires feature group eeg_pow")
    if int(feature.get("feature_count", -1)) != 448:
        raise ValueError("Technical auxiliary-CORN smoke requires 448 EEG+POW features")
    model = document["model"]
    if str(model.get("type", "")) != "torch_transformer":
        raise ValueError("Technical auxiliary-CORN smoke requires torch_transformer")
    return document


def _weight_token(weight: float) -> str:
    return f"{float(weight):g}".replace("-", "m").replace(".", "p")


def audit_auxiliary_corn_probabilities(
    predictions: pd.DataFrame,
    *,
    expected_weight: float,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate primary categorical and auxiliary CORN prediction artifacts."""
    primary_columns = [f"class_probability_{index}" for index in range(5)]
    legacy_columns = [f"proba_{index}" for index in range(5)]
    aux_class_columns = [f"aux_class_probability_{index}" for index in range(5)]
    aux_threshold_columns = [
        f"aux_threshold_probability_{index}" for index in range(4)
    ]
    required = {
        *primary_columns,
        *legacy_columns,
        *aux_class_columns,
        *aux_threshold_columns,
        "categorical_expected_rank",
        "aux_expected_rank",
        "aux_ordinal_prediction",
        "aux_ordinal_argmax",
        "auxiliary_weight",
        "head_type",
        "y_pred",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Joint predictions are missing columns: {missing}")
    if len(predictions) == 0:
        raise ValueError("Joint predictions cannot be empty")
    head_values = set(predictions["head_type"].astype(str).str.lower())
    if head_values != {JOINT_HEAD_TYPE}:
        raise ValueError(f"Unexpected joint prediction head_type values: {head_values}")
    weights = predictions["auxiliary_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or not np.allclose(
        weights, float(expected_weight), atol=0.0, rtol=0.0
    ):
        raise ValueError("Saved auxiliary_weight does not match the resolved trial")

    primary = predictions[primary_columns].to_numpy(dtype=np.float64)
    legacy = predictions[legacy_columns].to_numpy(dtype=np.float64)
    aux_class = predictions[aux_class_columns].to_numpy(dtype=np.float64)
    cumulative = predictions[aux_threshold_columns].to_numpy(dtype=np.float64)
    for name, values, width in (
        ("primary class probabilities", primary, 5),
        ("legacy primary probabilities", legacy, 5),
        ("auxiliary class probabilities", aux_class, 5),
        ("auxiliary cumulative probabilities", cumulative, 4),
    ):
        if values.shape != (len(predictions), width):
            raise ValueError(f"{name} has invalid shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contain NaN or infinite values")
    minimum_probability = float(min(primary.min(), aux_class.min()))
    if minimum_probability < -tolerance:
        raise ValueError(
            f"Class probability is negative beyond tolerance: {minimum_probability}"
        )
    primary_sum_error = float(
        np.max(np.abs(primary.sum(axis=1) - 1.0), initial=0.0)
    )
    aux_sum_error = float(
        np.max(np.abs(aux_class.sum(axis=1) - 1.0), initial=0.0)
    )
    if primary_sum_error > tolerance or aux_sum_error > tolerance:
        raise ValueError(
            "Primary or auxiliary class probabilities do not sum to one within "
            f"tolerance: primary={primary_sum_error}, auxiliary={aux_sum_error}"
        )
    legacy_delta = float(np.max(np.abs(primary - legacy), initial=0.0))
    if legacy_delta > tolerance:
        raise ValueError(
            "Legacy proba_* columns do not match primary class probabilities: "
            f"maximum delta={legacy_delta}"
        )
    monotonicity = cumulative[:, 1:] - cumulative[:, :-1]
    maximum_violation = float(max(0.0, monotonicity.max(initial=0.0)))
    if maximum_violation > tolerance:
        raise ValueError(
            "Auxiliary cumulative probabilities violate monotonicity: "
            f"maximum violation={maximum_violation}"
        )

    primary_prediction = predictions["y_pred"].to_numpy(dtype=np.int64)
    aux_prediction = predictions["aux_ordinal_prediction"].to_numpy(dtype=np.int64)
    aux_argmax = predictions["aux_ordinal_argmax"].to_numpy(dtype=np.int64)
    categorical_expected = predictions["categorical_expected_rank"].to_numpy(
        dtype=np.float64
    )
    aux_expected = predictions["aux_expected_rank"].to_numpy(dtype=np.float64)
    if not np.isfinite(categorical_expected).all() or not np.isfinite(aux_expected).all():
        raise ValueError("Expected-rank columns contain non-finite values")
    if np.any((primary_prediction < 0) | (primary_prediction > 4)):
        raise ValueError("Primary predictions fall outside [0,4]")
    if np.any((aux_prediction < 0) | (aux_prediction > 4)):
        raise ValueError("Auxiliary predictions fall outside [0,4]")
    if np.any((aux_argmax < 0) | (aux_argmax > 4)):
        raise ValueError("Auxiliary argmax predictions fall outside [0,4]")

    class_ids = np.arange(5, dtype=np.float64)
    recomputed_primary = primary.argmax(axis=1)
    recomputed_categorical_expected = (primary * class_ids).sum(axis=1)
    recomputed_aux_prediction = (cumulative >= 0.5).sum(axis=1)
    recomputed_aux_expected = cumulative.sum(axis=1)
    recomputed_aux_argmax = aux_class.argmax(axis=1)
    primary_mismatches = int(np.count_nonzero(recomputed_primary != primary_prediction))
    aux_mismatches = int(np.count_nonzero(recomputed_aux_prediction != aux_prediction))
    aux_argmax_mismatches = int(np.count_nonzero(recomputed_aux_argmax != aux_argmax))
    categorical_expected_delta = float(
        np.max(
            np.abs(recomputed_categorical_expected - categorical_expected),
            initial=0.0,
        )
    )
    aux_expected_delta = float(
        np.max(np.abs(recomputed_aux_expected - aux_expected), initial=0.0)
    )
    if primary_mismatches or aux_mismatches or aux_argmax_mismatches:
        raise ValueError(
            "Saved joint predictions do not match their registered decoding rules: "
            f"primary={primary_mismatches}, auxiliary={aux_mismatches}, "
            f"aux_argmax={aux_argmax_mismatches}"
        )
    if categorical_expected_delta > tolerance or aux_expected_delta > tolerance:
        raise ValueError(
            "Saved expected ranks do not match probability-derived values: "
            f"categorical={categorical_expected_delta}, auxiliary={aux_expected_delta}"
        )

    raw_aux_class = np.concatenate(
        [
            1.0 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ],
        axis=1,
    )
    return {
        "rows": int(len(predictions)),
        "head_type": JOINT_HEAD_TYPE,
        "auxiliary_weight": float(expected_weight),
        "primary_probability_shape": list(primary.shape),
        "auxiliary_probability_shape": list(aux_class.shape),
        "auxiliary_threshold_shape": list(cumulative.shape),
        "minimum_class_probability": minimum_probability,
        "maximum_primary_probability_sum_error": primary_sum_error,
        "maximum_auxiliary_probability_sum_error": aux_sum_error,
        "maximum_legacy_probability_delta": legacy_delta,
        "maximum_auxiliary_monotonicity_violation": maximum_violation,
        "round_off_correction_count": int(np.count_nonzero(raw_aux_class < 0.0)),
        "primary_prediction_recomputation_mismatches": primary_mismatches,
        "auxiliary_prediction_recomputation_mismatches": aux_mismatches,
        "auxiliary_argmax_recomputation_mismatches": aux_argmax_mismatches,
        "maximum_categorical_expected_rank_delta": categorical_expected_delta,
        "maximum_auxiliary_expected_rank_delta": aux_expected_delta,
        "categorical_aux_prediction_agreement": float(
            np.mean(primary_prediction == aux_prediction)
        ),
    }


@dataclass(frozen=True)
class AuxiliaryCornTrialPlan:
    trial_id: str
    auxiliary_weight: float
    feature_group: str
    feature_count: int
    feature_list_sha256: str
    input_shape: tuple[int, int]
    sequence_length: int
    full_sequence_count: int
    full_sequence_index_sha256: str
    smoke_sequence_subset_sha256: str
    outer_fold: int
    outer_train_sequences: int
    train_sequences: int
    validation_sequences: int
    test_sequences: int
    train_subjects: int
    validation_groups: int
    test_subjects: int
    class_counts: Mapping[str, Mapping[str, int]]
    model_parameter_count: int
    maximum_epochs: int
    output_dir: Path
    status: str
    invalid_reasons: tuple[str, ...]
    action: str
    resolved_config: Mapping[str, Any]
    config_hash: str
    completed_run: CompletedBenchmarkRun | None = None

    def to_dict(self, *, include_config: bool = False) -> dict[str, Any]:
        payload = {
            "trial_id": self.trial_id,
            "head_type": JOINT_HEAD_TYPE,
            "auxiliary_weight": self.auxiliary_weight,
            "feature_group": self.feature_group,
            "feature_count": self.feature_count,
            "feature_list_sha256": self.feature_list_sha256,
            "input_shape": list(self.input_shape),
            "sequence_length": self.sequence_length,
            "full_sequence_count": self.full_sequence_count,
            "full_sequence_index_sha256": self.full_sequence_index_sha256,
            "smoke_sequence_subset_sha256": self.smoke_sequence_subset_sha256,
            "outer_fold": self.outer_fold,
            "outer_train_sequences": self.outer_train_sequences,
            "train_sequences": self.train_sequences,
            "validation_sequences": self.validation_sequences,
            "test_sequences": self.test_sequences,
            "train_subjects": self.train_subjects,
            "validation_groups": self.validation_groups,
            "test_subjects": self.test_subjects,
            "class_counts": _jsonable(self.class_counts),
            "model_parameter_count": self.model_parameter_count,
            "maximum_epochs": self.maximum_epochs,
            "output_directory": _relative_path(self.output_dir),
            "validity_status": self.status,
            "invalid_reasons": list(self.invalid_reasons),
            "action": self.action,
            "reusable_completed_run": (
                None
                if self.completed_run is None
                else _relative_path(self.completed_run.run_directory)
            ),
            "config_hash": self.config_hash,
        }
        if include_config:
            payload["resolved_config"] = _jsonable(self.resolved_config)
        return payload


class AuxiliaryCornTransformerSmokeExperiment(OrdinalTransformerSmokeExperiment):
    """Resolve three auxiliary weights and delegate training to BenchmarkRunner."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        runner_factory: Callable[[dict[str, Any]], Any] = BenchmarkRunner,
        completed_run_finder: Callable[..., CompletedBenchmarkRun | None] = (
            BenchmarkRunner.find_completed_run
        ),
        context_builder: Callable[[], Mapping[str, Any]] | None = None,
        trial_auditor: Callable[
            [AuxiliaryCornTrialPlan, CompletedBenchmarkRun, Any],
            Mapping[str, Any],
        ]
        | None = None,
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.document = load_auxiliary_corn_smoke_spec(self.spec_path)
        self.output_root = _repo_path(
            output_dir or self.document["experiment"]["output_dir"]
        )
        self.runner_factory = runner_factory
        self.completed_run_finder = completed_run_finder
        self.context_builder = context_builder
        self.trial_auditor = trial_auditor
        self._context: dict[str, Any] | None = None

    def _resolved_config(
        self,
        auxiliary_weight: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        dataset = self.document["dataset"]
        feature = self.document["feature_group"]
        model = self.document["model"]
        trial_id = f"categorical_corn_eeg_pow_lambda_{_weight_token(auxiliary_weight)}"
        params = deepcopy(model["params"])
        params.update(
            {
                "head_type": JOINT_HEAD_TYPE,
                "auxiliary_weight": float(auxiliary_weight),
                "num_classes": 5,
                "max_epochs": int(self.document["protocol"]["max_epochs"]),
                "random_state": 42,
            }
        )
        return {
            "output_dir": str(self.output_root / "runs" / trial_id),
            "datasets": {
                str(dataset["name"]): {
                    "data_path": str(self.data_path),
                    "feature_set": str(feature["feature_set"]),
                    "feature_group": str(feature["name"]),
                    "target_col": str(dataset["target"]),
                    "subject_col": str(dataset.get("subject_col", "subject_id")),
                    "n_classes": 5,
                    "discretize": False,
                    "max_features": int(feature["feature_count"]),
                    "expected_feature_count": int(feature["feature_count"]),
                    "feature_list_sha256": str(feature["feature_list_sha256"]),
                }
            },
            "tasks": [str(self.document["task"]["benchmark_task"])],
            "models": {
                str(model["name"]): {
                    "type": str(model["type"]),
                    "task_type": "classification",
                    "params": params,
                }
            },
            "sequence": deepcopy(self.document["sequence"]),
            "validation": deepcopy(self.document["validation"]),
            "evaluation": deepcopy(self.document["evaluation"]),
            "task_config": {"random_state": 42},
            "run_within_subject": False,
            "run_loso": False,
            "experiment": {
                "name": str(self.document["experiment"]["name"]),
                "type": "auxiliary_corn_transformer_smoke",
                "trial_id": trial_id,
                "head_type": JOINT_HEAD_TYPE,
                "auxiliary_weight": float(auxiliary_weight),
                "feature_group": str(feature["name"]),
                "seed": 42,
                "outer_fold": int(context["outer_fold"]),
                "full_sequence_index_sha256": str(
                    context["full_sequence_index_sha256"]
                ),
                "smoke_sequence_subset_sha256": str(
                    context["smoke_sequence_subset_sha256"]
                ),
                "technical_only": True,
                "lambda_selection_performed": False,
            },
        }

    def plan(self) -> list[AuxiliaryCornTrialPlan]:
        context = self._build_context()
        invalid_common: list[str] = []
        expected = self.document["dataset"]
        feature = self.document["feature_group"]
        if context["supervised_rows"] != int(expected["expected_supervised_rows"]):
            invalid_common.append("supervised row count mismatch")
        if len(context["canonical"]) != int(expected["expected_sequences"]):
            invalid_common.append("canonical sequence count mismatch")
        if context["full_sequence_index_sha256"] != str(
            expected["sequence_index_sha256"]
        ):
            invalid_common.append("canonical sequence-index hash mismatch")
        if context["source_parquet_sha256"] != str(expected["parquet_sha256"]):
            invalid_common.append("source Parquet hash mismatch")
        if len(context["feature_names"]) != int(feature["feature_count"]):
            invalid_common.append("feature count mismatch")
        if context["feature_list_sha256"] != str(feature["feature_list_sha256"]):
            invalid_common.append("feature-list hash mismatch")
        if context["outer_subject_overlap"]:
            invalid_common.append("outer train/test subject overlap")
        if context["validation_summary"]["group_overlap"]:
            invalid_common.append("inner train/validation group overlap")
        if any(len(counts) != 5 for counts in context["class_counts"].values()):
            invalid_common.append("one or more splits do not contain all five classes")

        plans: list[AuxiliaryCornTrialPlan] = []
        for auxiliary_weight in AUXILIARY_WEIGHTS:
            config = self._resolved_config(auxiliary_weight, context)
            output = Path(config["output_dir"])
            completed = self.completed_run_finder(
                config, search_directories=[output]
            )
            adapter = build_model(
                "torch_transformer",
                "classification",
                input_shape=(
                    int(self.document["sequence"]["length"]),
                    int(feature["feature_count"]),
                ),
                num_outputs=5,
                params=config["models"]["torch_transformer"]["params"],
            )
            reasons = list(invalid_common)
            plans.append(
                AuxiliaryCornTrialPlan(
                    trial_id=str(config["experiment"]["trial_id"]),
                    auxiliary_weight=float(auxiliary_weight),
                    feature_group=str(feature["name"]),
                    feature_count=int(feature["feature_count"]),
                    feature_list_sha256=str(context["feature_list_sha256"]),
                    input_shape=(
                        int(self.document["sequence"]["length"]),
                        int(feature["feature_count"]),
                    ),
                    sequence_length=int(self.document["sequence"]["length"]),
                    full_sequence_count=int(len(context["canonical"])),
                    full_sequence_index_sha256=str(
                        context["full_sequence_index_sha256"]
                    ),
                    smoke_sequence_subset_sha256=str(
                        context["smoke_sequence_subset_sha256"]
                    ),
                    outer_fold=int(context["outer_fold"]),
                    outer_train_sequences=int(context["outer_train_sequences"]),
                    train_sequences=int(context["train_sequences"]),
                    validation_sequences=int(context["validation_sequences"]),
                    test_sequences=int(context["test_sequences"]),
                    train_subjects=int(context["train_subjects"]),
                    validation_groups=int(context["validation_groups"]),
                    test_subjects=int(context["test_subjects"]),
                    class_counts=deepcopy(context["class_counts"]),
                    model_parameter_count=int(
                        adapter.model_metadata["parameter_count"]
                    ),
                    maximum_epochs=int(self.document["protocol"]["max_epochs"]),
                    output_dir=output,
                    status="valid" if not reasons else "invalid",
                    invalid_reasons=tuple(reasons),
                    action="reuse" if completed is not None else "run",
                    resolved_config=config,
                    config_hash=benchmark_config_hash(config),
                    completed_run=completed,
                )
            )
            del adapter
        return plans

    @staticmethod
    def render_plan(plans: Sequence[AuxiliaryCornTrialPlan]) -> str:
        lines = [
            "# Auxiliary-CORN Transformer technical smoke plan",
            "",
            "| Trial | Lambda | Features/hash | Input | Full sequences/hash | "
            "Smoke hash | Fold | Train/val/test | Subjects/groups | Params | "
            "Epochs | Output | Reusable | Status |",
            "| --- | ---: | --- | --- | --- | --- | ---: | --- | --- | ---: | "
            "---: | --- | --- | --- |",
        ]
        for plan in plans:
            lines.append(
                f"| `{plan.trial_id}` | {plan.auxiliary_weight:g} | "
                f"{plan.feature_count} / `{plan.feature_list_sha256[:12]}` | "
                f"`{list(plan.input_shape)}` | {plan.full_sequence_count} / "
                f"`{plan.full_sequence_index_sha256[:12]}` | "
                f"`{plan.smoke_sequence_subset_sha256[:12]}` | "
                f"{plan.outer_fold} | {plan.train_sequences}/"
                f"{plan.validation_sequences}/{plan.test_sequences} | "
                f"{plan.train_subjects}/{plan.validation_groups}/"
                f"{plan.test_subjects} | {plan.model_parameter_count} | "
                f"{plan.maximum_epochs} | `{_relative_path(plan.output_dir)}` | "
                f"{'yes' if plan.completed_run else 'no'} | {plan.status} |"
            )
        lines.extend(
            [
                "",
                "All three trials use one canonical outer fold, one inner split, "
                "and one normalization scope.",
                "This technical smoke does not select a lambda and does not support "
                "a scientific quality claim.",
                "Plan-only performs no training and writes no benchmark artifacts.",
            ]
        )
        return "\n".join(lines)

    def _audit_trial(
        self,
        plan: AuxiliaryCornTrialPlan,
        completed: CompletedBenchmarkRun,
        split: Any,
    ) -> dict[str, Any]:
        _, fold = self._fold_result(completed)
        artifacts = {key: Path(value) for key, value in fold["artifacts"].items()}
        required_artifacts = {
            "predictions",
            "model",
            "training_log",
            "normalization_stats",
            "validation_split",
            "auxiliary_corn_metadata",
        }
        missing_artifacts = sorted(required_artifacts - set(artifacts))
        if missing_artifacts:
            raise ValueError(
                f"Auxiliary-CORN smoke artifacts are missing: {missing_artifacts}"
            )
        predictions = pd.read_parquet(artifacts["predictions"])
        probability = audit_auxiliary_corn_probabilities(
            predictions, expected_weight=plan.auxiliary_weight
        )
        training_log = pd.read_csv(artifacts["training_log"])
        numeric_columns = (
            "train_loss",
            "validation_loss",
            "train_total_loss",
            "train_categorical_loss",
            "train_ordinal_loss",
            "validation_total_loss",
            "validation_categorical_loss",
            "validation_ordinal_loss",
            "learning_rate",
        )
        for column in numeric_columns:
            if column not in training_log or not np.isfinite(
                training_log[column].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"Training log has a missing or non-finite {column!r} column"
                )
        if "early_stopping_metric" not in training_log:
            raise ValueError("Training log is missing early_stopping_metric")
        if set(training_log["early_stopping_metric"].astype(str)) != {
            "validation_categorical_loss"
        }:
            raise ValueError(
                "Joint smoke must early-stop on validation_categorical_loss"
            )
        if len(training_log) > plan.maximum_epochs:
            raise ValueError("Training exceeded the smoke maximum epoch count")
        if len(training_log) > 1 and float(
            np.ptp(training_log["train_total_loss"].to_numpy(dtype=float))
        ) == 0.0:
            raise ValueError("Joint training total loss is exactly constant")
        train_equation_error = float(
            np.max(
                np.abs(
                    training_log["train_total_loss"].to_numpy(dtype=float)
                    - training_log["train_categorical_loss"].to_numpy(dtype=float)
                    - plan.auxiliary_weight
                    * training_log["train_ordinal_loss"].to_numpy(dtype=float)
                ),
                initial=0.0,
            )
        )
        validation_equation_error = float(
            np.max(
                np.abs(
                    training_log["validation_total_loss"].to_numpy(dtype=float)
                    - training_log["validation_categorical_loss"].to_numpy(
                        dtype=float
                    )
                    - plan.auxiliary_weight
                    * training_log["validation_ordinal_loss"].to_numpy(dtype=float)
                ),
                initial=0.0,
            )
        )
        alias_error = float(
            max(
                np.max(
                    np.abs(
                        training_log["train_loss"].to_numpy(dtype=float)
                        - training_log["train_total_loss"].to_numpy(dtype=float)
                    ),
                    initial=0.0,
                ),
                np.max(
                    np.abs(
                        training_log["validation_loss"].to_numpy(dtype=float)
                        - training_log["validation_total_loss"].to_numpy(dtype=float)
                    ),
                    initial=0.0,
                ),
            )
        )
        if max(train_equation_error, validation_equation_error, alias_error) > 1e-6:
            raise ValueError(
                "Joint training loss columns violate the registered loss equation"
            )

        config = yaml.safe_load(
            (completed.run_directory / "config.yaml").read_text(encoding="utf-8")
        )
        params = config["models"]["torch_transformer"]["params"]
        if str(params.get("head_type")) != JOINT_HEAD_TYPE:
            raise ValueError("Resolved config does not preserve categorical_corn")
        if float(params.get("auxiliary_weight", float("nan"))) != plan.auxiliary_weight:
            raise ValueError("Resolved config does not preserve auxiliary_weight")

        model = build_model(
            "torch_transformer",
            "classification",
            input_shape=tuple(split.X_test.shape[1:]),
            num_outputs=5,
            params=params,
        )
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.model.state_dict().items()
        }
        try:
            checkpoint = torch.load(
                artifacts["model"], map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(artifacts["model"], map_location="cpu")
        if checkpoint.get("head_type") != JOINT_HEAD_TYPE:
            raise ValueError("Checkpoint head_type does not match categorical_corn")
        checkpoint_weight = float(
            checkpoint.get("training_config", {}).get(
                "auxiliary_weight", float("nan")
            )
        )
        if checkpoint_weight != plan.auxiliary_weight:
            raise ValueError("Checkpoint auxiliary_weight does not match the trial")
        if not self._finite_checkpoint_state(checkpoint):
            raise ValueError("Checkpoint contains non-finite model parameters")
        model.load(artifacts["model"])
        detailed = model.predict_detailed(split.X_test)
        reloaded = pd.DataFrame(
            {
                "sequence_id": np.asarray(
                    split.row_metadata_test["sequence_id"]
                ).astype(str),
                "y_pred_reloaded": detailed["y_pred"],
                "categorical_expected_rank_reloaded": detailed[
                    "categorical_expected_rank"
                ],
                "aux_expected_rank_reloaded": detailed["aux_expected_rank"],
                "aux_ordinal_prediction_reloaded": detailed[
                    "aux_ordinal_prediction"
                ],
                "aux_ordinal_argmax_reloaded": detailed[
                    "aux_ordinal_argmax"
                ],
            }
        )
        for index in range(5):
            reloaded[f"class_probability_{index}_reloaded"] = detailed[
                "class_probabilities"
            ][:, index]
            reloaded[f"aux_class_probability_{index}_reloaded"] = detailed[
                "aux_class_probabilities"
            ][:, index]
        for index in range(4):
            reloaded[f"aux_threshold_probability_{index}_reloaded"] = detailed[
                "aux_threshold_probabilities"
            ][:, index]
        merged = predictions.merge(reloaded, on="sequence_id", how="inner")
        if len(merged) != len(predictions):
            raise ValueError("Checkpoint reload predictions are not exactly aligned")
        y_mismatches = int(
            np.count_nonzero(
                merged["y_pred"].to_numpy()
                != merged["y_pred_reloaded"].to_numpy()
            )
        )
        aux_prediction_mismatches = int(
            np.count_nonzero(
                merged["aux_ordinal_prediction"].to_numpy()
                != merged["aux_ordinal_prediction_reloaded"].to_numpy()
            )
        )
        aux_argmax_mismatches = int(
            np.count_nonzero(
                merged["aux_ordinal_argmax"].to_numpy()
                != merged["aux_ordinal_argmax_reloaded"].to_numpy()
            )
        )
        primary_probability_delta = float(
            max(
                np.max(
                    np.abs(
                        merged[f"class_probability_{index}"].to_numpy(dtype=float)
                        - merged[
                            f"class_probability_{index}_reloaded"
                        ].to_numpy(dtype=float)
                    ),
                    initial=0.0,
                )
                for index in range(5)
            )
        )
        auxiliary_probability_delta = float(
            max(
                np.max(
                    np.abs(
                        merged[f"aux_class_probability_{index}"].to_numpy(
                            dtype=float
                        )
                        - merged[
                            f"aux_class_probability_{index}_reloaded"
                        ].to_numpy(dtype=float)
                    ),
                    initial=0.0,
                )
                for index in range(5)
            )
        )
        threshold_probability_delta = float(
            max(
                np.max(
                    np.abs(
                        merged[f"aux_threshold_probability_{index}"].to_numpy(
                            dtype=float
                        )
                        - merged[
                            f"aux_threshold_probability_{index}_reloaded"
                        ].to_numpy(dtype=float)
                    ),
                    initial=0.0,
                )
                for index in range(4)
            )
        )
        categorical_expected_delta = float(
            np.max(
                np.abs(
                    merged["categorical_expected_rank"].to_numpy(dtype=float)
                    - merged["categorical_expected_rank_reloaded"].to_numpy(
                        dtype=float
                    )
                ),
                initial=0.0,
            )
        )
        aux_expected_delta = float(
            np.max(
                np.abs(
                    merged["aux_expected_rank"].to_numpy(dtype=float)
                    - merged["aux_expected_rank_reloaded"].to_numpy(dtype=float)
                ),
                initial=0.0,
            )
        )
        if any(
            value != 0
            for value in (y_mismatches, aux_prediction_mismatches, aux_argmax_mismatches)
        ) or max(
            primary_probability_delta,
            auxiliary_probability_delta,
            threshold_probability_delta,
            categorical_expected_delta,
            aux_expected_delta,
        ) > 1e-7:
            raise ValueError("Checkpoint reload does not reproduce joint predictions")

        changed: dict[str, dict[str, Any]] = {}
        for prefix in ("classifier.", "auxiliary_ordinal_head."):
            keys = [key for key in checkpoint["model_state_dict"] if key.startswith(prefix)]
            deltas = [
                float(
                    torch.max(
                        torch.abs(
                            checkpoint["model_state_dict"][key].cpu()
                            - initial_state[key]
                        )
                    )
                )
                for key in keys
                if torch.is_floating_point(checkpoint["model_state_dict"][key])
            ]
            changed[prefix] = {
                "keys": keys,
                "changed_parameter_count": int(sum(delta > 0 for delta in deltas)),
                "maximum_parameter_delta": float(max(deltas, default=0.0)),
            }
            if not keys or changed[prefix]["changed_parameter_count"] == 0:
                raise ValueError(f"No trained parameter changed under prefix {prefix!r}")

        training_summary = fold["training"]
        if training_summary.get("early_stopping_monitor") != (
            "validation_categorical_loss"
        ):
            raise ValueError("Training summary has an invalid early-stopping monitor")
        objective_diagnostics = dict(
            training_summary.get("objective_training_diagnostics", {})
        )
        risk_counts = [
            int(objective_diagnostics[f"aux_risk_count_{index}"])
            for index in range(4)
        ]
        if risk_counts[0] <= 0 or not np.all(np.diff(risk_counts) <= 0):
            raise ValueError("Auxiliary CORN risk counts are invalid")

        required_primary = {
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "auc",
            "kappa",
            "quadratic_weighted_kappa",
            "ordinal_mae",
            "adjacent_accuracy",
            "severe_error_rate",
            "expected_rank_mae",
            "expected_rank_spearman",
        }
        required_auxiliary = {
            f"aux_{name}"
            for name in required_primary
        }
        required_metrics = {
            *required_primary,
            *required_auxiliary,
            "categorical_aux_prediction_agreement",
        }
        missing_metrics = sorted(required_metrics - set(fold["metrics"]))
        if missing_metrics:
            raise ValueError(f"Joint smoke metrics are missing: {missing_metrics}")
        nonfinite_metrics = sorted(
            name
            for name in required_metrics
            if not np.isfinite(float(fold["metrics"][name]))
        )
        if nonfinite_metrics:
            raise ValueError(f"Joint smoke metrics are non-finite: {nonfinite_metrics}")

        validation_split = json.loads(
            artifacts["validation_split"].read_text(encoding="utf-8")
        )
        if validation_split["group_overlap"]:
            raise ValueError("Inner train/validation groups overlap")
        if validation_split["outer_test_record_overlap"]:
            raise ValueError("Inner data overlap outer-test records")
        if fold["split_metadata"].get("subject_overlap"):
            raise ValueError("Outer train/test subjects overlap")
        auxiliary_metadata = json.loads(
            artifacts["auxiliary_corn_metadata"].read_text(encoding="utf-8")
        )
        if auxiliary_metadata.get("head_type") != JOINT_HEAD_TYPE:
            raise ValueError("Auxiliary metadata has the wrong head type")
        if float(auxiliary_metadata.get("auxiliary_weight")) != plan.auxiliary_weight:
            raise ValueError("Auxiliary metadata has the wrong auxiliary weight")

        diagnostics = {
            "trial_id": plan.trial_id,
            "head_type": JOINT_HEAD_TYPE,
            "auxiliary_weight": plan.auxiliary_weight,
            "run_directory": str(completed.run_directory),
            "artifacts": {key: str(value) for key, value in artifacts.items()},
            "epochs_trained": int(len(training_log)),
            "training_log": training_log.to_dict(orient="records"),
            "best_epoch": int(training_summary["best_epoch"]),
            "best_validation_loss": float(
                training_summary["best_validation_loss"]
            ),
            "best_validation_components": dict(
                training_summary.get("best_validation_components", {})
            ),
            "early_stopping_monitor": str(
                training_summary["early_stopping_monitor"]
            ),
            "stopping_reason": str(training_summary["stopping_reason"]),
            "training_time_seconds": float(fold["training_time"]),
            "device": str(training_summary["device"]),
            "device_name": str(training_summary["device_name"]),
            "parameter_count": int(training_summary["trainable_parameter_count"]),
            "objective_training_diagnostics": objective_diagnostics,
            "risk_counts": risk_counts,
            "probability_validation": probability,
            "loss_equation_audit": {
                "maximum_train_equation_error": train_equation_error,
                "maximum_validation_equation_error": validation_equation_error,
                "maximum_legacy_alias_error": alias_error,
            },
            "metrics": fold["metrics"],
            "checkpoint_reload": {
                "strict_load": True,
                "primary_prediction_mismatches": y_mismatches,
                "auxiliary_prediction_mismatches": aux_prediction_mismatches,
                "auxiliary_argmax_mismatches": aux_argmax_mismatches,
                "maximum_primary_class_probability_delta": (
                    primary_probability_delta
                ),
                "maximum_auxiliary_class_probability_delta": (
                    auxiliary_probability_delta
                ),
                "maximum_auxiliary_threshold_probability_delta": (
                    threshold_probability_delta
                ),
                "maximum_categorical_expected_rank_delta": (
                    categorical_expected_delta
                ),
                "maximum_auxiliary_expected_rank_delta": aux_expected_delta,
            },
            "checkpoint": {
                "head_type": checkpoint.get("head_type"),
                "auxiliary_weight": checkpoint_weight,
                "all_parameters_finite": True,
                "output_head_parameter_changes": changed,
                "categorical_classifier_keys_preserved": sorted(
                    key
                    for key in checkpoint["model_state_dict"]
                    if key.startswith("classifier.")
                ),
                "auxiliary_head_keys": sorted(
                    key
                    for key in checkpoint["model_state_dict"]
                    if key.startswith("auxiliary_ordinal_head.")
                ),
            },
            "validation_split": validation_split,
            "auxiliary_metadata": auxiliary_metadata,
        }
        artifact_dir = artifacts["predictions"].parent
        probability_path = artifact_dir / "joint_probability_validation_summary.json"
        checkpoint_path = artifact_dir / "joint_checkpoint_reload_audit.json"
        objective_path = artifact_dir / "joint_objective_audit.json"
        fold_manifest_path = artifact_dir / "auxiliary_corn_fold_manifest.json"
        _write_json(probability_path, probability)
        _write_json(checkpoint_path, diagnostics["checkpoint_reload"])
        _write_json(
            objective_path,
            {
                "auxiliary_weight": plan.auxiliary_weight,
                "early_stopping_monitor": diagnostics["early_stopping_monitor"],
                "risk_counts": risk_counts,
                "loss_equation_audit": diagnostics["loss_equation_audit"],
                "best_validation_components": diagnostics[
                    "best_validation_components"
                ],
            },
        )
        diagnostics["technical_artifacts"] = {
            "probability_validation": str(probability_path),
            "checkpoint_reload_audit": str(checkpoint_path),
            "objective_audit": str(objective_path),
            "fold_manifest": str(fold_manifest_path),
        }
        _write_json(
            fold_manifest_path,
            {
                "schema_version": 1,
                "status": "completed",
                "trial_id": plan.trial_id,
                "head_type": JOINT_HEAD_TYPE,
                "auxiliary_weight": plan.auxiliary_weight,
                "outer_fold": plan.outer_fold,
                "smoke_sequence_subset_sha256": (
                    plan.smoke_sequence_subset_sha256
                ),
                "split_metadata": fold["split_metadata"],
                "training": training_summary,
                "metrics": fold["metrics"],
                "standard_artifacts": diagnostics["artifacts"],
                "technical_artifacts": diagnostics["technical_artifacts"],
                "probability_validation": probability,
                "checkpoint_reload": diagnostics["checkpoint_reload"],
            },
        )
        with (completed.run_directory / "resolved_config.yaml").open(
            "w", encoding="utf-8"
        ) as output:
            yaml.safe_dump(_jsonable(config), output, sort_keys=False)
        _write_json(
            completed.run_directory / "auxiliary_corn_smoke_trial_manifest.json",
            diagnostics,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return diagnostics

    def _combined_audit(
        self,
        plans: Sequence[AuxiliaryCornTrialPlan],
        completed: Mapping[str, CompletedBenchmarkRun],
        audits: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        predictions: dict[float, pd.DataFrame] = {}
        normalizations: dict[float, dict[str, Any]] = {}
        validations: dict[float, dict[str, Any]] = {}
        for plan in plans:
            _, fold = self._fold_result(completed[plan.trial_id])
            artifacts = {key: Path(value) for key, value in fold["artifacts"].items()}
            predictions[plan.auxiliary_weight] = pd.read_parquet(
                artifacts["predictions"]
            )
            normalizations[plan.auxiliary_weight] = json.loads(
                artifacts["normalization_stats"].read_text(encoding="utf-8")
            )
            validations[plan.auxiliary_weight] = json.loads(
                artifacts["validation_split"].read_text(encoding="utf-8")
            )
        reference_weight = AUXILIARY_WEIGHTS[0]
        reference_predictions = predictions[reference_weight]
        reference_normalization = normalizations[reference_weight]
        reference_validation = validations[reference_weight]
        alignments = {
            f"lambda_{_weight_token(weight)}": prediction_alignment(
                reference_predictions, frame
            )
            for weight, frame in predictions.items()
        }
        normalization_deltas: dict[str, Any] = {}
        validation_equal: dict[str, bool] = {}
        for weight in AUXILIARY_WEIGHTS:
            normalization = normalizations[weight]
            key = f"lambda_{_weight_token(weight)}"
            normalization_deltas[key] = {
                "mean_max_abs_delta": float(
                    np.max(
                        np.abs(
                            np.asarray(normalization["mean"], dtype=float)
                            - np.asarray(reference_normalization["mean"], dtype=float)
                        ),
                        initial=0.0,
                    )
                ),
                "scale_max_abs_delta": float(
                    np.max(
                        np.abs(
                            np.asarray(normalization["scale"], dtype=float)
                            - np.asarray(reference_normalization["scale"], dtype=float)
                        ),
                        initial=0.0,
                    )
                ),
                "feature_order_equal": (
                    normalization["feature_names"]
                    == reference_normalization["feature_names"]
                ),
            }
            validation_equal[key] = validations[weight] == reference_validation
        subset_hashes = {
            f"lambda_{_weight_token(plan.auxiliary_weight)}": (
                plan.smoke_sequence_subset_sha256
            )
            for plan in plans
        }
        config_hashes = {plan.config_hash for plan in plans}
        ready = bool(
            all(value["exact_match"] for value in alignments.values())
            and all(
                value["mean_max_abs_delta"] == 0.0
                and value["scale_max_abs_delta"] == 0.0
                and value["feature_order_equal"]
                for value in normalization_deltas.values()
            )
            and all(validation_equal.values())
            and len(set(subset_hashes.values())) == 1
            and len(config_hashes) == len(plans)
            and all(
                audit["checkpoint_reload"]["primary_prediction_mismatches"] == 0
                and audit["checkpoint_reload"][
                    "auxiliary_prediction_mismatches"
                ]
                == 0
                for audit in audits.values()
            )
        )
        return {
            "status": "completed" if ready else "invalid",
            "technical_only": True,
            "scientific_quality_claim": False,
            "lambda_selection_performed": False,
            "ready_for_nested_lambda_experiment": ready,
            "sequence_alignment": alignments,
            "normalization_deltas": normalization_deltas,
            "validation_splits_equal": validation_equal,
            "smoke_subset_hashes": subset_hashes,
            "all_smoke_subset_hashes_equal": len(set(subset_hashes.values())) == 1,
            "distinct_config_hashes": len(config_hashes) == len(plans),
            "trials": _jsonable(audits),
        }

    def _write_reports(
        self,
        plans: Sequence[AuxiliaryCornTrialPlan],
        manifest: Mapping[str, Any],
    ) -> None:
        report_path = _repo_path(
            self.document["experiment"].get(
                "report_path",
                "reports/auxiliary_corn_transformer_smoke_results.md",
            )
        )
        summary_path = _repo_path(
            self.document["experiment"].get(
                "summary_path",
                "reports/auxiliary_corn_transformer_smoke_results.json",
            )
        )
        audit = dict(manifest.get("audit", {}))
        trials = dict(audit.get("trials", {}))
        rows: list[dict[str, Any]] = []
        for plan in plans:
            trial = dict(trials.get(plan.trial_id, {}))
            metrics = dict(trial.get("metrics", {}))
            rows.append(
                {
                    "trial_id": plan.trial_id,
                    "auxiliary_weight": plan.auxiliary_weight,
                    "outcome": next(
                        (
                            item.get("outcome")
                            for item in manifest.get("outcomes", [])
                            if item.get("trial_id") == plan.trial_id
                        ),
                        None,
                    ),
                    "epochs_trained": trial.get("epochs_trained"),
                    "best_epoch": trial.get("best_epoch"),
                    "best_validation_loss": trial.get("best_validation_loss"),
                    "training_time_seconds": trial.get("training_time_seconds"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "ordinal_mae": metrics.get("ordinal_mae"),
                    "severe_error_rate": metrics.get("severe_error_rate"),
                    "aux_ordinal_mae": metrics.get("aux_ordinal_mae"),
                    "aux_severe_error_rate": metrics.get(
                        "aux_severe_error_rate"
                    ),
                    "categorical_aux_prediction_agreement": metrics.get(
                        "categorical_aux_prediction_agreement"
                    ),
                    "probability_validation": trial.get(
                        "probability_validation"
                    ),
                    "checkpoint_reload": trial.get("checkpoint_reload"),
                    "risk_counts": trial.get("risk_counts"),
                }
            )
        summary = {
            "schema_version": 1,
            "experiment": self.document["experiment"]["name"],
            "technical_only": True,
            "lambda_selection_performed": False,
            "source_parquet_sha256": manifest.get("source_parquet_sha256"),
            "full_sequence_index_sha256": manifest.get(
                "full_sequence_index_sha256"
            ),
            "smoke_sequence_subset_sha256": manifest.get(
                "smoke_sequence_subset_sha256"
            ),
            "audit_status": audit.get("status"),
            "ready_for_nested_lambda_experiment": audit.get(
                "ready_for_nested_lambda_experiment"
            ),
            "trials": rows,
        }
        _write_json(summary_path, summary)

        lines = [
            "# Auxiliary-CORN Transformer technical smoke",
            "",
            "> These results use one outer fold and at most three epochs. They are "
            "technical pipeline checks, not a lambda selection or a scientific "
            "comparison.",
            "",
            "## Protocol",
            "",
            f"- Feature group: `{self.document['feature_group']['name']}`.",
            f"- Input: `[8, {self.document['feature_group']['feature_count']}]`.",
            "- Seed: `42`.",
            "- Outer fold: `1`.",
            f"- Auxiliary weights: `{list(AUXILIARY_WEIGHTS)}`.",
            f"- Maximum epochs: `{self.document['protocol']['max_epochs']}`.",
            "- Early stopping monitor: `validation_categorical_loss`.",
            "- Lambda selection performed: `false`.",
            "",
            "## Technical outcomes",
            "",
            "| Lambda | Epochs/best | Validation categorical loss | BA | Macro F1 | "
            "Ordinal MAE | Severe error | Aux ordinal MAE | Aux severe error | "
            "Head agreement |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            def fmt(value: Any) -> str:
                return "—" if value is None else f"{float(value):.6f}"

            epoch_text = (
                "—"
                if row["epochs_trained"] is None
                else f"{row['epochs_trained']}/{row['best_epoch']}"
            )
            lines.append(
                f"| {row['auxiliary_weight']:.2f} | {epoch_text} | "
                f"{fmt(row['best_validation_loss'])} | "
                f"{fmt(row['balanced_accuracy'])} | {fmt(row['macro_f1'])} | "
                f"{fmt(row['ordinal_mae'])} | {fmt(row['severe_error_rate'])} | "
                f"{fmt(row['aux_ordinal_mae'])} | "
                f"{fmt(row['aux_severe_error_rate'])} | "
                f"{fmt(row['categorical_aux_prediction_agreement'])} |"
            )
        lines.extend(
            [
                "",
                "## Audit conclusion",
                "",
                f"- Status: `{audit.get('status', 'unknown')}`.",
                "- Exact sequence alignment is required across all three weights.",
                "- Inner splits and normalization statistics are required to be "
                "identical across weights.",
                "- Primary and auxiliary probabilities, loss decomposition, and "
                "strict checkpoint reload are audited per trial.",
                f"- Ready for the nested lambda experiment: "
                f"`{audit.get('ready_for_nested_lambda_experiment')}`.",
                "",
                "No preferred lambda is selected from this smoke run.",
            ]
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def execute(
        self,
        plans: Sequence[AuxiliaryCornTrialPlan],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        invalid = [plan for plan in plans if plan.status != "valid"]
        if invalid:
            raise ValueError(
                "Invalid auxiliary-CORN smoke trials: "
                + "; ".join(
                    f"{plan.trial_id}: {', '.join(plan.invalid_reasons)}"
                    for plan in invalid
                )
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        context = self._build_context()
        split_manifest_path = (
            self.output_root / "auxiliary_corn_smoke_sequence_split.parquet"
        )
        context["split_manifest"].to_parquet(split_manifest_path, index=False)
        completed: dict[str, CompletedBenchmarkRun] = {}
        outcomes: list[dict[str, Any]] = []
        for plan in plans:
            existing = self.completed_run_finder(
                plan.resolved_config, search_directories=[plan.output_dir]
            )
            if resume and existing is not None:
                completed[plan.trial_id] = existing
                outcomes.append({**plan.to_dict(), "outcome": "resumed"})
                continue
            runner = self.runner_factory(deepcopy(dict(plan.resolved_config)))
            runner.run()
            completed_run = runner.completed_run()
            completed[plan.trial_id] = completed_run
            outcomes.append({**plan.to_dict(), "outcome": "completed"})
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        audits: dict[str, Mapping[str, Any]] = {}
        split = None
        if self.trial_auditor is None:
            split = self._rebuild_test_split(plans[0].resolved_config)
        for plan in plans:
            auditor = self.trial_auditor or self._audit_trial
            audits[plan.trial_id] = dict(
                auditor(plan, completed[plan.trial_id], split)
            )
        if self.trial_auditor is None:
            combined = self._combined_audit(plans, completed, audits)
        else:
            combined = {
                "status": "completed",
                "technical_only": True,
                "scientific_quality_claim": False,
                "lambda_selection_performed": False,
                "ready_for_nested_lambda_experiment": True,
                "trials": _jsonable(audits),
            }
        manifest = {
            "experiment": str(self.document["experiment"]["name"]),
            "source_parquet_sha256": str(context["source_parquet_sha256"]),
            "full_sequence_index_sha256": str(
                context["full_sequence_index_sha256"]
            ),
            "smoke_sequence_subset_sha256": str(
                context["smoke_sequence_subset_sha256"]
            ),
            "sequence_split_manifest": str(split_manifest_path),
            "lambda_selection_performed": False,
            "outcomes": outcomes,
            "audit": combined,
        }
        manifest_path = (
            self.output_root / "auxiliary_corn_transformer_smoke_manifest.json"
        )
        _write_json(manifest_path, manifest)
        self._write_reports(plans, manifest)
        return manifest


__all__ = [
    "AUXILIARY_WEIGHTS",
    "AuxiliaryCornTransformerSmokeExperiment",
    "AuxiliaryCornTrialPlan",
    "JOINT_HEAD_TYPE",
    "audit_auxiliary_corn_probabilities",
    "load_auxiliary_corn_smoke_spec",
]
