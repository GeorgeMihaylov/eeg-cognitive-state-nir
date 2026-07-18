"""Five-fold seed-42 study for ordinal Transformer classification heads."""

from __future__ import annotations

import gc
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold

from bench.bench_runner import (
    BenchmarkRunner,
    CompletedBenchmarkRun,
    benchmark_config_hash,
)
from bench.datasets.base_eeg_data_loader import (
    feature_list_sha256,
    resolve_feature_columns,
)
from bench.experiments.ordinal_transformer import (
    _file_sha256,
    _jsonable,
    _relative_path,
    _repo_path,
    _write_json,
    audit_prediction_probabilities,
)
from bench.tasks.tasks_registry import get_task
from bench.validation.cross_val import CrossValidator
from bench.validation.metrics import MetricsCalculator
from model_zoo import build_model
from model_zoo.DL.sequence_utils import build_sequences, sequence_index_sha256


FULL_HEAD_TYPES = ("coral", "corn")
FULL_FEATURE_GROUPS = ("eeg_only", "eeg_pow")
FULL_FOLDS = (1, 2, 3, 4, 5)
ALIGNMENT_COLUMNS = (
    "sequence_id",
    "fold",
    "subject_id",
    "record_id",
    "source",
    "target_sample_id",
    "target_time",
    "y_true",
)
METRIC_NAMES = (
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
)


def load_ordinal_transformer_full_spec(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the fixed seed-42 full-study matrix."""
    spec_path = _repo_path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Ordinal full experiment not found: {spec_path}")
    document = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    required = {
        "experiment", "dataset", "task", "feature_groups",
        "feature_definitions", "head_types", "seeds",
        "categorical_references", "model", "sequence", "validation",
        "evaluation", "protocol",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Ordinal full experiment is missing sections: {missing}")
    if str(document["experiment"].get("type")) != "ordinal_transformer_full":
        raise ValueError("Full study requires experiment.type=ordinal_transformer_full")
    if tuple(document["feature_groups"]) != FULL_FEATURE_GROUPS:
        raise ValueError("Full study feature_groups must be eeg_only, eeg_pow")
    if tuple(document["head_types"]) != FULL_HEAD_TYPES:
        raise ValueError("Full study head_types must be coral, corn")
    if [int(value) for value in document["seeds"]] != [42]:
        raise ValueError("Full ordinal study supports only seed 42")
    folds = tuple(int(value) for value in document["evaluation"].get("folds", []))
    if folds != FULL_FOLDS:
        raise ValueError("Full ordinal study requires all five canonical folds")
    if tuple(int(value) for value in document["protocol"]["outer_folds"]) != FULL_FOLDS:
        raise ValueError("protocol.outer_folds must contain folds 1..5")
    params = document["model"]["params"]
    if int(params["max_epochs"]) != int(document["protocol"]["max_epochs"]):
        raise ValueError("Model and protocol max_epochs disagree")
    if int(params["max_epochs"]) != 15:
        raise ValueError("Full study must use the canonical 15-epoch limit")
    return document


def full_prediction_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    """Compare the canonical sequence identities used by full studies."""
    missing_left = sorted(set(ALIGNMENT_COLUMNS) - set(reference.columns))
    missing_right = sorted(set(ALIGNMENT_COLUMNS) - set(candidate.columns))
    if missing_left or missing_right:
        return {
            "exact_match": False,
            "missing_reference": missing_left,
            "missing_candidate": missing_right,
        }
    left = reference.loc[:, list(ALIGNMENT_COLUMNS)].sort_values(
        "sequence_id", kind="mergesort"
    ).reset_index(drop=True)
    right = candidate.loc[:, list(ALIGNMENT_COLUMNS)].sort_values(
        "sequence_id", kind="mergesort"
    ).reset_index(drop=True)
    mismatches: dict[str, int] = {
        "row_count": int(abs(len(left) - len(right)))
    }
    for column in ALIGNMENT_COLUMNS:
        if len(left) != len(right):
            mismatches[column] = max(len(left), len(right))
        elif column in {"target_time", "y_true"}:
            mismatches[column] = int(np.count_nonzero(~np.isclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                equal_nan=True,
            )))
        else:
            mismatches[column] = int(np.count_nonzero(
                left[column].astype(str).to_numpy()
                != right[column].astype(str).to_numpy()
            ))
    duplicates = {
        "reference": int(left["sequence_id"].duplicated().sum()),
        "candidate": int(right["sequence_id"].duplicated().sum()),
    }
    return {
        "exact_match": bool(not any(mismatches.values()) and not any(duplicates.values())),
        "reference_rows": int(len(left)),
        "candidate_rows": int(len(right)),
        "duplicate_sequence_ids": duplicates,
        "mismatches": mismatches,
    }


@dataclass(frozen=True)
class OrdinalTransformerFullTrialPlan:
    trial_id: str
    head_type: str
    feature_group: str
    feature_count: int
    feature_list_sha256: str
    input_shape: tuple[int, int]
    sequence_count: int
    subject_count: int
    sequence_index_sha256: str
    folds: tuple[int, ...]
    fold_summaries: Mapping[str, Mapping[str, Any]]
    model_parameter_count: int
    maximum_epochs: int
    patience: int
    seed: int
    output_dir: Path
    config_hash: str
    status: str
    invalid_reasons: tuple[str, ...]
    action: str
    resolved_config: Mapping[str, Any]
    completed_run: CompletedBenchmarkRun | None = None

    def to_dict(self, *, include_config: bool = False) -> dict[str, Any]:
        payload = {
            "trial_id": self.trial_id,
            "head_type": self.head_type,
            "feature_group": self.feature_group,
            "feature_count": self.feature_count,
            "feature_list_sha256": self.feature_list_sha256,
            "input_shape": list(self.input_shape),
            "sequence_count": self.sequence_count,
            "subject_count": self.subject_count,
            "sequence_index_sha256": self.sequence_index_sha256,
            "folds": list(self.folds),
            "fold_runs": len(self.folds),
            "fold_summaries": _jsonable(self.fold_summaries),
            "model_parameter_count": self.model_parameter_count,
            "maximum_epochs": self.maximum_epochs,
            "patience": self.patience,
            "seed": self.seed,
            "output_directory": _relative_path(self.output_dir),
            "config_hash": self.config_hash,
            "validity_status": self.status,
            "invalid_reasons": list(self.invalid_reasons),
            "action": self.action,
            "reusable_completed_run": (
                None if self.completed_run is None
                else _relative_path(self.completed_run.run_directory)
            ),
        }
        if include_config:
            payload["resolved_config"] = _jsonable(self.resolved_config)
        return payload


class OrdinalTransformerFullExperiment:
    """Resolve four full trials and delegate all fitting to BenchmarkRunner."""

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
        trial_auditor: Callable[..., Mapping[str, Any]] | None = None,
        reference_auditor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.document = load_ordinal_transformer_full_spec(self.spec_path)
        self.output_root = _repo_path(
            output_dir or self.document["experiment"]["output_dir"]
        )
        self.runner_factory = runner_factory
        self.completed_run_finder = completed_run_finder
        self.context_builder = context_builder
        self.trial_auditor = trial_auditor
        self.reference_auditor = reference_auditor
        self._context: dict[str, Any] | None = None

    @property
    def data_path(self) -> Path:
        return _repo_path(self.document["dataset"]["data_path"])

    def _build_context(self) -> dict[str, Any]:
        if self._context is not None:
            return self._context
        if self.context_builder is not None:
            self._context = dict(self.context_builder())
            return self._context

        schema = list(pq.ParquetFile(self.data_path).schema.names)
        frame = pd.read_parquet(
            self.data_path,
            columns=["subject_id", "record_id", "source", "t_start", "label_q5"],
        )
        frame.insert(0, "sample_id", np.arange(len(frame), dtype=np.int64))
        supervised = frame.loc[frame["label_q5"].notna()].copy()
        supervised["label_q5"] = supervised["label_q5"].astype(np.int64)
        supervised["fold"] = 0
        splitter = GroupKFold(n_splits=5)
        for fold, (_, test_index) in enumerate(splitter.split(
            supervised, supervised["label_q5"], supervised["subject_id"]
        ), start=1):
            supervised.iloc[test_index, supervised.columns.get_loc("fold")] = fold

        sequence = self.document["sequence"]
        built = build_sequences(
            X=np.zeros((len(supervised), 1), dtype=np.float32),
            y=supervised["label_q5"].to_numpy(dtype=np.int64),
            metadata=supervised[[
                "source", "subject_id", "record_id", "sample_id", "t_start"
            ]],
            sequence_length=int(sequence["length"]),
            stride=int(sequence.get("stride", 1)),
            target_position=str(sequence.get("target_position", "last")),
            expected_step_seconds=sequence.get("expected_step_seconds"),
            max_gap_seconds=sequence.get("max_gap_seconds"),
        )
        canonical = built.metadata.copy()
        subject_folds = supervised.drop_duplicates("subject_id").set_index(
            "subject_id"
        )["fold"]
        canonical["fold"] = canonical["subject_id"].map(subject_folds).astype(int)
        canonical["y_true"] = built.y.astype(np.int64)
        canonical = canonical.sort_values(
            ["fold", "sequence_id"], kind="mergesort"
        ).reset_index(drop=True)

        feature_context: dict[str, Any] = {}
        for name in FULL_FEATURE_GROUPS:
            definition = self.document["feature_definitions"][name]
            names = resolve_feature_columns(schema, str(definition["feature_set"]))
            feature_context[name] = {
                "names": names,
                "count": len(names),
                "sha256": feature_list_sha256(names),
            }

        fold_summaries: dict[str, Any] = {}
        preview_params = deepcopy(self.document["model"]["params"])
        preview_params.update({"head_type": "coral", "device": "cpu"})
        preview = build_model(
            "torch_transformer", "classification",
            input_shape=(int(sequence["length"]), feature_context["eeg_only"]["count"]),
            num_outputs=5,
            params=preview_params,
        )
        for fold in FULL_FOLDS:
            outer_train = canonical.loc[canonical["fold"] != fold].copy()
            outer_test = canonical.loc[canonical["fold"] == fold].copy()
            preview.set_validation_groups(
                outer_train["record_group_id"].astype(str).to_numpy(),
                subject_ids=outer_train["subject_id"].astype(str).to_numpy(),
                record_ids=outer_train["record_id"].astype(str).to_numpy(),
                outer_test_record_ids=outer_test["record_id"].astype(str).to_numpy(),
                strategy=str(self.document["validation"]["strategy"]),
                group_column=str(self.document["validation"]["group_column"]),
                validation_size=float(self.document["validation"]["validation_size"]),
                random_state=42,
            )
            labels = outer_train["y_true"].to_numpy(dtype=np.int64)
            train_index, validation_index = preview._group_validation_indices(labels)
            validation = preview._validation_summary(
                labels, train_index, validation_index
            )
            fold_summaries[f"fold_{fold:02d}"] = {
                "outer_train_sequences": int(len(outer_train)),
                "inner_train_sequences": int(len(train_index)),
                "validation_sequences": int(len(validation_index)),
                "test_sequences": int(len(outer_test)),
                "outer_train_subjects": int(outer_train["subject_id"].nunique()),
                "test_subjects": int(outer_test["subject_id"].nunique()),
                "inner_validation_groups": int(
                    len(validation["inner_validation_group_ids"])
                ),
                "outer_subject_overlap": sorted(
                    set(outer_train["subject_id"].astype(str))
                    & set(outer_test["subject_id"].astype(str))
                ),
                "inner_group_overlap": list(validation["group_overlap"]),
                "inner_validation_group_ids": list(
                    validation["inner_validation_group_ids"]
                ),
            }
        del preview

        self._context = {
            "supervised_rows": int(len(supervised)),
            "canonical": canonical,
            "sequence_count": int(len(canonical)),
            "subject_count": int(canonical["subject_id"].nunique()),
            "sequence_index_sha256": sequence_index_sha256(canonical),
            "source_parquet_sha256": _file_sha256(self.data_path),
            "sequence_build_stats": built.stats,
            "features": feature_context,
            "fold_summaries": fold_summaries,
        }
        return self._context

    def _categorical_reference_audit(
        self, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        audits: dict[str, Any] = {}
        canonical = context["canonical"]
        for group in FULL_FEATURE_GROUPS:
            run_dir = _repo_path(
                self.document["categorical_references"][group]["run_directory"]
            )
            config_path = run_dir / "config.yaml"
            metrics_path = run_dir / "metrics.json"
            candidates = list(run_dir.glob("**/group_kfold_subject/predictions.parquet"))
            if not config_path.is_file() or not metrics_path.is_file() or len(candidates) != 1:
                raise ValueError(f"Incomplete categorical reference for {group}: {run_dir}")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            params = config["models"]["torch_transformer"]["params"]
            predictions = pd.read_parquet(candidates[0])
            alignment = full_prediction_alignment(canonical, predictions)
            checks = {
                "seed": int(params.get("random_state", -1)) == 42,
                "head_type": str(params.get("head_type", "categorical")) == "categorical",
                "sequence_length": int(params.get("sequence_length", -1)) == 8,
                "folds": sorted(predictions["fold"].astype(int).unique()) == list(FULL_FOLDS),
                "subjects": int(predictions["subject_id"].nunique()) == 53,
                "sequences": int(len(predictions)) == 44142,
                "alignment": bool(alignment["exact_match"]),
            }
            if not all(checks.values()):
                raise ValueError(
                    f"Categorical reference {group} is not canonical: {checks}"
                )
            model = json.loads(metrics_path.read_text(encoding="utf-8"))[
                "emotiv_cognitive"
            ]["models"]["cognitive_load_5class"]["torch_transformer"][
                "group_kfold_subject"
            ]
            audits[group] = {
                "run_directory": _relative_path(run_dir),
                "prediction_file": _relative_path(candidates[0]),
                "checks": checks,
                "alignment": alignment,
                "config_hash": json.loads(
                    (run_dir / "run_manifest.json").read_text(encoding="utf-8")
                )["config_hash"],
                "aggregated_metrics": model["aggregated"],
            }
        return audits

    def _resolved_config(
        self,
        head_type: str,
        feature_group: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        dataset = self.document["dataset"]
        feature = self.document["feature_definitions"][feature_group]
        model = self.document["model"]
        trial_id = f"{head_type}_{feature_group}"
        params = deepcopy(model["params"])
        params.update({"head_type": head_type, "random_state": 42})
        return {
            "output_dir": str(self.output_root / "runs" / trial_id),
            "datasets": {
                str(dataset["name"]): {
                    "data_path": str(self.data_path),
                    "feature_set": str(feature["feature_set"]),
                    "feature_group": feature_group,
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
                "type": "ordinal_transformer_full",
                "trial_id": trial_id,
                "head_type": head_type,
                "feature_group": feature_group,
                "seed": 42,
                "required_folds": list(FULL_FOLDS),
                "full_sequence_index_sha256": str(
                    context["sequence_index_sha256"]
                ),
                "categorical_reference": str(
                    self.document["categorical_references"][feature_group][
                        "run_directory"
                    ]
                ),
            },
        }

    @staticmethod
    def _result_model(completed: CompletedBenchmarkRun) -> dict[str, Any]:
        results = json.loads(completed.result_file.read_text(encoding="utf-8"))
        return results["emotiv_cognitive"]["models"]["cognitive_load_5class"][
            "torch_transformer"
        ]["group_kfold_subject"]

    def _completed_is_reusable(
        self,
        completed: CompletedBenchmarkRun | None,
        config: Mapping[str, Any],
    ) -> bool:
        if completed is None or completed.config_hash != benchmark_config_hash(config):
            return False
        try:
            manifest = json.loads(
                (completed.run_directory / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            saved = yaml.safe_load(
                (completed.run_directory / "config.yaml").read_text(encoding="utf-8")
            )
            model = self._result_model(completed)
            fold_names = sorted(model["folds"])
            return bool(
                manifest.get("status") == "completed"
                and saved["experiment"]["type"] == "ordinal_transformer_full"
                and saved["experiment"]["head_type"] == config["experiment"]["head_type"]
                and saved["experiment"]["feature_group"] == config["experiment"]["feature_group"]
                and saved["experiment"]["full_sequence_index_sha256"]
                == config["experiment"]["full_sequence_index_sha256"]
                and fold_names == [f"fold_{fold:02d}" for fold in FULL_FOLDS]
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def plan(self) -> list[OrdinalTransformerFullTrialPlan]:
        context = self._build_context()
        expected = self.document["dataset"]
        reasons: list[str] = []
        if context["supervised_rows"] != int(expected["expected_supervised_rows"]):
            reasons.append("supervised row count mismatch")
        if context["sequence_count"] != int(expected["expected_sequences"]):
            reasons.append("canonical sequence count mismatch")
        if context["subject_count"] != int(expected["expected_subjects"]):
            reasons.append("canonical subject count mismatch")
        if context["sequence_index_sha256"] != str(expected["sequence_index_sha256"]):
            reasons.append("canonical sequence-index hash mismatch")
        if context["source_parquet_sha256"] != str(expected["parquet_sha256"]):
            reasons.append("source Parquet hash mismatch")
        if any(
            summary["outer_subject_overlap"] or summary["inner_group_overlap"]
            for summary in context["fold_summaries"].values()
        ):
            reasons.append("split leakage detected")
        references = dict(
            self.reference_auditor(context)
            if self.reference_auditor is not None
            else self._categorical_reference_audit(context)
        )
        plans: list[OrdinalTransformerFullTrialPlan] = []
        for head_type in FULL_HEAD_TYPES:
            for feature_group in FULL_FEATURE_GROUPS:
                feature = context["features"][feature_group]
                definition = self.document["feature_definitions"][feature_group]
                trial_reasons = list(reasons)
                if feature["count"] != int(definition["feature_count"]):
                    trial_reasons.append("feature count mismatch")
                if feature["sha256"] != str(definition["feature_list_sha256"]):
                    trial_reasons.append("feature-list hash mismatch")
                config = self._resolved_config(head_type, feature_group, context)
                output = Path(config["output_dir"])
                found = self.completed_run_finder(
                    config, search_directories=[output]
                )
                completed = found if self._completed_is_reusable(found, config) else None
                adapter = build_model(
                    "torch_transformer", "classification",
                    input_shape=(8, int(feature["count"])),
                    num_outputs=5,
                    params=config["models"]["torch_transformer"]["params"],
                )
                plans.append(OrdinalTransformerFullTrialPlan(
                    trial_id=str(config["experiment"]["trial_id"]),
                    head_type=head_type,
                    feature_group=feature_group,
                    feature_count=int(feature["count"]),
                    feature_list_sha256=str(feature["sha256"]),
                    input_shape=(8, int(feature["count"])),
                    sequence_count=int(context["sequence_count"]),
                    subject_count=int(context["subject_count"]),
                    sequence_index_sha256=str(context["sequence_index_sha256"]),
                    folds=FULL_FOLDS,
                    fold_summaries=deepcopy(context["fold_summaries"]),
                    model_parameter_count=int(
                        adapter.model_metadata["parameter_count"]
                    ),
                    maximum_epochs=int(self.document["protocol"]["max_epochs"]),
                    patience=int(
                        self.document["model"]["params"]["early_stopping_patience"]
                    ),
                    seed=42,
                    output_dir=output,
                    config_hash=benchmark_config_hash(config),
                    status="valid" if not trial_reasons else "invalid",
                    invalid_reasons=tuple(trial_reasons),
                    action="reuse" if completed else "run",
                    resolved_config=config,
                    completed_run=completed,
                ))
                del adapter
        context["categorical_references"] = references
        return plans

    @staticmethod
    def render_plan(plans: Sequence[OrdinalTransformerFullTrialPlan]) -> str:
        lines = [
            "# Ordinal Transformer full seed-42 plan",
            "",
            "| Trial | Head | Group | Features/hash | Input | Sequences/subjects | Folds | Params | Epochs/patience | Output | Reusable | Status |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
        for plan in plans:
            lines.append(
                f"| `{plan.trial_id}` | {plan.head_type} | {plan.feature_group} | "
                f"{plan.feature_count} / `{plan.feature_list_sha256[:12]}` | "
                f"`{list(plan.input_shape)}` | {plan.sequence_count}/{plan.subject_count} | "
                f"{list(plan.folds)} | {plan.model_parameter_count} | "
                f"{plan.maximum_epochs}/{plan.patience} | "
                f"`{_relative_path(plan.output_dir)}` | "
                f"{'yes' if plan.completed_run else 'no'} | {plan.status} |"
            )
        lines.extend([
            "",
            f"Trials: {len(plans)}; fold-runs: {sum(len(plan.folds) for plan in plans)}.",
            "No sequence limit is applied. Plan-only does not write checkpoints or predictions.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _rebuild_splits(config: Mapping[str, Any]) -> dict[str, Any]:
        runner = BenchmarkRunner(deepcopy(dict(config)))
        dataset_name = next(iter(config["datasets"]))
        data = runner.load_dataset(dataset_name)
        task_name = str(config["tasks"][0])
        task = get_task(task_name, data, dict(config.get("task_config", {})))
        splits = CrossValidator(task).run_group_kfold(
            group_column=str(config["evaluation"]["group_column"]),
            n_splits=int(config["evaluation"]["n_splits"]),
            random_state=int(config["evaluation"].get("random_state", 42)),
        )
        return {
            fold_name: runner._build_sequence_split(splits[fold_name])
            for fold_name in [f"fold_{fold:02d}" for fold in FULL_FOLDS]
        }

    @staticmethod
    def _finite_checkpoint_state(payload: Mapping[str, Any]) -> bool:
        return all(
            not torch.is_floating_point(value) or bool(torch.isfinite(value).all())
            for value in payload["model_state_dict"].values()
        )

    @staticmethod
    def _metric_payload(frame: pd.DataFrame) -> dict[str, Any]:
        probability_columns = [f"proba_{index}" for index in range(5)]
        return MetricsCalculator.calculate_all_metrics(
            frame["y_true"].to_numpy(dtype=int),
            frame["y_pred"].to_numpy(dtype=int),
            frame[probability_columns].to_numpy(dtype=float),
            expected_rank=frame["expected_rank"].to_numpy(dtype=float),
        )

    @staticmethod
    def _subject_metrics(
        predictions: pd.DataFrame,
        plan: OrdinalTransformerFullTrialPlan,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for subject_id, group in predictions.groupby("subject_id", sort=True):
            y_true = group["y_true"].to_numpy(dtype=int)
            y_pred = group["y_pred"].to_numpy(dtype=int)
            expected = group["expected_rank"].to_numpy(dtype=float)
            reasons: list[str] = []
            qwk = np.nan
            if len(np.unique(y_true)) < 2:
                reasons.append("qwk_single_true_class")
            else:
                qwk = float(cohen_kappa_score(
                    y_true, y_pred, labels=list(range(5)), weights="quadratic"
                ))
                if not np.isfinite(qwk):
                    reasons.append("qwk_undefined")
            rank_spearman = np.nan
            if len(y_true) < 2 or np.ptp(y_true) == 0:
                reasons.append("expected_rank_spearman_no_target_variation")
            elif np.ptp(expected) == 0:
                reasons.append("expected_rank_spearman_no_prediction_variation")
            else:
                rank_spearman = float(spearmanr(y_true, expected).statistic)
            distance = np.abs(y_pred - y_true)
            source_membership = "+".join(
                sorted(group["source"].dropna().astype(str).unique())
            )
            fold_values = group["fold"].astype(int).unique()
            if len(fold_values) != 1:
                raise ValueError("A subject appears in multiple outer folds")
            present_classes = np.unique(y_true)
            balanced = float(np.mean([
                np.mean(y_pred[y_true == class_id] == class_id)
                for class_id in present_classes
            ]))
            rows.append({
                "trial_id": plan.trial_id,
                "head_type": plan.head_type,
                "feature_group": plan.feature_group,
                "seed": plan.seed,
                "subject_id": str(subject_id),
                "fold": int(fold_values[0]),
                "source_membership": source_membership,
                "n_sequences": int(len(group)),
                "balanced_accuracy": balanced,
                "macro_f1": float(f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                )),
                "quadratic_weighted_kappa": qwk,
                "ordinal_mae": float(distance.mean()),
                "adjacent_accuracy": float(np.mean(distance <= 1)),
                "severe_error_rate": float(np.mean(distance >= 2)),
                "expected_rank_mae": float(np.mean(np.abs(expected - y_true))),
                "expected_rank_spearman": rank_spearman,
                "undefined_metric_reason": ";".join(reasons) if reasons else None,
            })
        return pd.DataFrame(rows)

    @classmethod
    def _source_metrics(cls, predictions: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source, group in predictions.groupby("source", sort=True):
            metrics = cls._metric_payload(group)
            rows.append({
                "source": str(source),
                "sequences": int(len(group)),
                "subjects": int(group["subject_id"].nunique()),
                **{name: metrics.get(name) for name in METRIC_NAMES},
            })
        return rows

    @staticmethod
    def _class_metrics(predictions: pd.DataFrame) -> list[dict[str, Any]]:
        return MetricsCalculator.calculate_class_metrics(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["y_pred"].to_numpy(dtype=int),
            labels=np.arange(5),
        )

    @staticmethod
    def _reference_fold_directory(run_dir: Path, fold_name: str) -> Path:
        matches = list(run_dir.glob(f"**/group_kfold_subject/{fold_name}"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one categorical {fold_name} directory in {run_dir}"
            )
        return matches[0]

    def _audit_fold(
        self,
        plan: OrdinalTransformerFullTrialPlan,
        fold_name: str,
        fold: Mapping[str, Any],
        split: Any,
        categorical_reference_run: Path | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        artifacts = {key: Path(value) for key, value in fold["artifacts"].items()}
        required_artifacts = {
            "predictions", "metrics", "class_metrics", "feature_manifest",
            "sequence_stats", "sequence_index_manifest", "validation_split",
            "model", "training_log", "normalization_stats", "ordinal_metadata",
        }
        missing_artifacts = sorted(required_artifacts - set(artifacts))
        missing_files = sorted(
            key for key, value in artifacts.items()
            if key in required_artifacts and not value.is_file()
        )
        if missing_artifacts or missing_files:
            raise ValueError(
                f"{plan.trial_id}/{fold_name} missing artifacts: "
                f"keys={missing_artifacts}, files={missing_files}"
            )
        predictions = pd.read_parquet(artifacts["predictions"])
        probability = audit_prediction_probabilities(predictions, plan.head_type)
        training_log = pd.read_csv(artifacts["training_log"])
        for column in ("train_loss", "validation_loss", "learning_rate"):
            if column not in training_log or not np.isfinite(
                training_log[column].to_numpy(dtype=float)
            ).all():
                raise ValueError(f"Non-finite or missing {column} in {fold_name}")
        if not 1 <= len(training_log) <= plan.maximum_epochs:
            raise ValueError(f"Invalid epoch count in {fold_name}: {len(training_log)}")
        best_epoch = int(fold["training"]["best_epoch"])
        best_validation_loss = float(fold["training"]["best_validation_loss"])
        observed_best = int(training_log.loc[
            training_log["validation_loss"].idxmin(), "epoch"
        ])
        if best_epoch != observed_best:
            raise ValueError(f"Checkpoint best epoch mismatch in {fold_name}")
        if not np.isclose(
            best_validation_loss,
            float(training_log["validation_loss"].min()),
            atol=1e-12,
        ):
            raise ValueError(f"Checkpoint best loss mismatch in {fold_name}")

        params = plan.resolved_config["models"]["torch_transformer"]["params"]
        model = build_model(
            "torch_transformer", "classification",
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
        if checkpoint.get("head_type") != plan.head_type:
            raise ValueError(f"Checkpoint head mismatch in {fold_name}")
        if not self._finite_checkpoint_state(checkpoint):
            raise ValueError(f"Non-finite checkpoint parameter in {fold_name}")
        summary = checkpoint.get("training_summary", {})
        if int(summary.get("best_epoch", -1)) != best_epoch:
            raise ValueError(f"Saved checkpoint is not tagged with best epoch in {fold_name}")
        model.load(artifacts["model"])
        detailed = model.predict_detailed(split.X_test)
        reloaded = pd.DataFrame({
            "sequence_id": np.asarray(split.row_metadata_test["sequence_id"]).astype(str),
            "y_pred_reloaded": detailed["y_pred"],
            "expected_rank_reloaded": detailed["expected_rank"],
        })
        for index in range(5):
            reloaded[f"class_probability_{index}_reloaded"] = detailed[
                "class_probabilities"
            ][:, index]
        for index in range(4):
            reloaded[f"threshold_probability_{index}_reloaded"] = detailed[
                "threshold_probabilities"
            ][:, index]
        compared = predictions.merge(
            reloaded, on="sequence_id", how="outer", validate="one_to_one",
            indicator=True,
        )
        if not compared["_merge"].eq("both").all():
            raise ValueError(f"Reloaded membership mismatch in {fold_name}")
        y_mismatches = int(np.count_nonzero(
            compared["y_pred"].to_numpy(dtype=int)
            != compared["y_pred_reloaded"].to_numpy(dtype=int)
        ))
        class_delta = float(max(
            np.max(np.abs(
                compared[f"class_probability_{index}"].to_numpy(dtype=float)
                - compared[f"class_probability_{index}_reloaded"].to_numpy(dtype=float)
            ), initial=0.0)
            for index in range(5)
        ))
        threshold_delta = float(max(
            np.max(np.abs(
                compared[f"threshold_probability_{index}"].to_numpy(dtype=float)
                - compared[f"threshold_probability_{index}_reloaded"].to_numpy(dtype=float)
            ), initial=0.0)
            for index in range(4)
        ))
        expected_delta = float(np.max(np.abs(
            compared["expected_rank"].to_numpy(dtype=float)
            - compared["expected_rank_reloaded"].to_numpy(dtype=float)
        ), initial=0.0))
        if y_mismatches or max(class_delta, threshold_delta, expected_delta) > 1e-7:
            raise ValueError(f"Strict checkpoint reload mismatch in {fold_name}")

        wrong_head = "corn" if plan.head_type == "coral" else "coral"
        wrong_params = deepcopy(dict(params))
        wrong_params["head_type"] = wrong_head
        wrong_model = build_model(
            "torch_transformer", "classification",
            input_shape=tuple(split.X_test.shape[1:]),
            num_outputs=5,
            params=wrong_params,
        )
        incompatible_rejected = False
        try:
            wrong_model.load(artifacts["model"])
        except ValueError:
            incompatible_rejected = True
        if not incompatible_rejected:
            raise ValueError("Incompatible ordinal head checkpoint loaded silently")

        changed = 0
        maximum_parameter_delta = 0.0
        for key, trained in checkpoint["model_state_dict"].items():
            if not key.startswith("ordinal_head.") or not torch.is_floating_point(trained):
                continue
            delta = float(torch.max(torch.abs(
                trained.detach().cpu() - initial_state[key]
            )).item())
            maximum_parameter_delta = max(maximum_parameter_delta, delta)
            changed += int(delta > 0)
        if not changed:
            raise ValueError(f"No output parameter changed in {fold_name}")

        validation = json.loads(
            artifacts["validation_split"].read_text(encoding="utf-8")
        )
        if validation["group_overlap"] or validation["outer_test_record_overlap"]:
            raise ValueError(f"Inner leakage detected in {fold_name}")
        if fold["split_metadata"].get("subject_overlap"):
            raise ValueError(f"Outer subject leakage detected in {fold_name}")
        reference_run = (
            categorical_reference_run
            if categorical_reference_run is not None
            else _repo_path(
                self.document["categorical_references"][plan.feature_group][
                    "run_directory"
                ]
            )
        )
        reference_fold = self._reference_fold_directory(reference_run, fold_name)
        reference_validation = json.loads(
            (reference_fold / "validation_split.json").read_text(encoding="utf-8")
        )
        normalization = json.loads(
            artifacts["normalization_stats"].read_text(encoding="utf-8")
        )
        reference_normalization = json.loads(
            (reference_fold / "normalization_stats.json").read_text(encoding="utf-8")
        )
        normalization_comparison = {
            "feature_order_equal": normalization["feature_names"]
            == reference_normalization["feature_names"],
            "mean_max_abs_delta": float(np.max(np.abs(
                np.asarray(normalization["mean"], dtype=float)
                - np.asarray(reference_normalization["mean"], dtype=float)
            ), initial=0.0)),
            "scale_max_abs_delta": float(np.max(np.abs(
                np.asarray(normalization["scale"], dtype=float)
                - np.asarray(reference_normalization["scale"], dtype=float)
            ), initial=0.0)),
            "validation_groups_equal": validation["inner_validation_group_ids"]
            == reference_validation["inner_validation_group_ids"],
        }
        if (
            not normalization_comparison["feature_order_equal"]
            or not normalization_comparison["validation_groups_equal"]
            or normalization_comparison["mean_max_abs_delta"] > 1e-12
            or normalization_comparison["scale_max_abs_delta"] > 1e-12
        ):
            raise ValueError(f"Categorical normalization mismatch in {fold_name}")

        head_diagnostics = dict(fold["training"].get("head_diagnostics", {}))
        objective_diagnostics = dict(
            fold["training"].get("objective_training_diagnostics", {})
        )
        if plan.head_type == "coral":
            cutpoints = [head_diagnostics[f"cutpoint_{index}"] for index in range(4)]
            if not np.all(np.diff(cutpoints) > 0):
                raise ValueError(f"Unordered CORAL cutpoints in {fold_name}")
        else:
            risk_counts = [
                objective_diagnostics[f"risk_count_{index}"] for index in range(4)
            ]
            if risk_counts[0] <= 0 or not np.all(np.diff(risk_counts) <= 0):
                raise ValueError(f"Invalid CORN risk counts in {fold_name}")

        fold_audit = {
            "fold": fold_name,
            "epochs_trained": int(len(training_log)),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "stopping_reason": str(
                fold["training"].get("stopping_reason", summary.get("stopping_reason"))
            ),
            "training_duration_seconds": float(fold["training_time"]),
            "device": fold["training"]["device"],
            "device_name": fold["training"]["device_name"],
            "parameter_count": int(fold["training"]["trainable_parameter_count"]),
            "training_log": training_log.to_dict(orient="records"),
            "metrics": {name: fold["metrics"].get(name) for name in METRIC_NAMES},
            "head_diagnostics": head_diagnostics,
            "objective_training_diagnostics": objective_diagnostics,
            "probability_audit": probability,
            "checkpoint_reload": {
                "strict_load": True,
                "y_pred_mismatches": y_mismatches,
                "maximum_class_probability_delta": class_delta,
                "maximum_threshold_probability_delta": threshold_delta,
                "maximum_expected_rank_delta": expected_delta,
                "incompatible_head_rejected": incompatible_rejected,
            },
            "checkpoint": {
                "all_parameters_finite": True,
                "best_epoch_matches": True,
                "changed_output_parameters": changed,
                "maximum_output_parameter_delta": maximum_parameter_delta,
            },
            "leakage": {
                "outer_subject_overlap": [],
                "inner_group_overlap": list(validation["group_overlap"]),
                "outer_test_record_overlap": list(
                    validation["outer_test_record_overlap"]
                ),
            },
            "categorical_normalization_comparison": normalization_comparison,
            "standard_artifacts": {
                key: _relative_path(value) for key, value in artifacts.items()
            },
        }
        artifact_dir = artifacts["predictions"].parent
        technical_paths = {
            "probability_audit": artifact_dir / "probability_validation_summary.json",
            "checkpoint_reload_audit": artifact_dir / "checkpoint_reload_audit.json",
            "ordinal_diagnostics": artifact_dir / "ordinal_diagnostics.json",
            "fold_manifest": artifact_dir / "fold_manifest.json",
        }
        _write_json(technical_paths["probability_audit"], probability)
        _write_json(technical_paths["checkpoint_reload_audit"], fold_audit["checkpoint_reload"])
        _write_json(technical_paths["ordinal_diagnostics"], {
            "head_type": plan.head_type,
            "head_diagnostics": head_diagnostics,
            "objective_training_diagnostics": objective_diagnostics,
        })
        fold_audit["technical_artifacts"] = {
            key: _relative_path(value) for key, value in technical_paths.items()
        }
        _write_json(technical_paths["fold_manifest"], {
            "schema_version": 1,
            "status": "completed",
            "trial_id": plan.trial_id,
            "head_type": plan.head_type,
            "feature_group": plan.feature_group,
            "sequence_index_sha256": plan.sequence_index_sha256,
            **fold_audit,
        })
        del model, wrong_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return fold_audit, predictions

    def _audit_trial(
        self,
        plan: OrdinalTransformerFullTrialPlan,
        completed: CompletedBenchmarkRun,
        splits: Mapping[str, Any],
        categorical_reference_run: Path | None = None,
    ) -> dict[str, Any]:
        model_result = self._result_model(completed)
        expected_folds = [f"fold_{fold:02d}" for fold in FULL_FOLDS]
        if sorted(model_result["folds"]) != expected_folds:
            raise ValueError(f"{plan.trial_id} does not contain all five folds")
        fold_audits: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        for fold_name in expected_folds:
            audit, predictions = self._audit_fold(
                plan,
                fold_name,
                model_result["folds"][fold_name],
                splits[fold_name],
                categorical_reference_run,
            )
            fold_audits.append(audit)
            prediction_frames.append(predictions)
        combined = pd.concat(prediction_frames, ignore_index=True)
        if len(combined) != plan.sequence_count:
            raise ValueError(f"{plan.trial_id} prediction row count mismatch")
        if combined["sequence_id"].duplicated().any():
            raise ValueError(f"{plan.trial_id} has duplicate sequence_id values")
        combined = combined.sort_values(
            ["fold", "sequence_id"], kind="mergesort"
        ).reset_index(drop=True)
        aggregate_metrics = self._metric_payload(combined)
        fold_metrics: list[dict[str, Any]] = []
        for fold_name, group in combined.groupby("fold", sort=True):
            metrics = self._metric_payload(group)
            fold_metrics.append({
                "fold": int(fold_name),
                "sequences": int(len(group)),
                **{name: metrics.get(name) for name in METRIC_NAMES},
            })
        fold_frame = pd.DataFrame(fold_metrics)
        aggregate_by_fold: dict[str, Any] = {}
        for name in METRIC_NAMES:
            values = fold_frame[name].to_numpy(dtype=float)
            aggregate_by_fold[f"{name}_mean"] = float(np.nanmean(values))
            aggregate_by_fold[f"{name}_std"] = float(np.nanstd(values, ddof=0))
        subject_metrics = self._subject_metrics(combined, plan)
        source_metrics = self._source_metrics(combined)
        class_metrics = self._class_metrics(combined)

        reference_run = (
            categorical_reference_run
            if categorical_reference_run is not None
            else _repo_path(
                self.document["categorical_references"][plan.feature_group][
                    "run_directory"
                ]
            )
        )
        reference_prediction_path = next(
            reference_run.glob("**/group_kfold_subject/predictions.parquet")
        )
        reference_predictions = pd.read_parquet(reference_prediction_path)
        reference_metrics = MetricsCalculator.calculate_all_metrics(
            reference_predictions["y_true"].to_numpy(dtype=int),
            reference_predictions["y_pred"].to_numpy(dtype=int),
            reference_predictions[
                [f"proba_{index}" for index in range(5)]
            ].to_numpy(dtype=float),
        )
        comparison_metric_names = (
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            "auc", "kappa", "quadratic_weighted_kappa", "ordinal_mae",
            "adjacent_accuracy", "severe_error_rate",
        )
        categorical_deltas = {
            name: float(aggregate_metrics[name] - reference_metrics[name])
            for name in comparison_metric_names
        }
        alignment = full_prediction_alignment(reference_predictions, combined)
        if not alignment["exact_match"]:
            raise ValueError(f"Categorical alignment failed for {plan.trial_id}")

        run_dir = completed.run_directory
        subject_path = run_dir / "subject_metrics.parquet"
        class_path = run_dir / "class_metrics_full.parquet"
        subject_metrics.to_parquet(subject_path, index=False)
        pd.DataFrame(class_metrics).to_parquet(class_path, index=False)
        aggregate_path = run_dir / "aggregate_metrics.json"
        source_path = run_dir / "source_metrics.json"
        probability_path = run_dir / "probability_audit.json"
        checkpoint_path = run_dir / "checkpoint_reload_audit.json"
        ordinal_path = run_dir / "ordinal_diagnostics.json"
        _write_json(aggregate_path, {
            "window_sequence_aggregate": aggregate_metrics,
            "fold_aggregate": aggregate_by_fold,
            "fold_metrics": fold_metrics,
        })
        _write_json(source_path, {"rows": source_metrics})
        _write_json(probability_path, {
            row["fold"]: row["probability_audit"] for row in fold_audits
        })
        _write_json(checkpoint_path, {
            row["fold"]: row["checkpoint_reload"] for row in fold_audits
        })
        _write_json(ordinal_path, {
            row["fold"]: {
                "head_diagnostics": row["head_diagnostics"],
                "objective_training_diagnostics": row[
                    "objective_training_diagnostics"
                ],
            }
            for row in fold_audits
        })

        audit = {
            "trial_id": plan.trial_id,
            "head_type": plan.head_type,
            "feature_group": plan.feature_group,
            "run_directory": _relative_path(run_dir),
            "config_hash": plan.config_hash,
            "sequence_count": int(len(combined)),
            "subject_count": int(combined["subject_id"].nunique()),
            "parameter_count": plan.model_parameter_count,
            "folds": fold_audits,
            "fold_metrics": fold_metrics,
            "fold_aggregate": aggregate_by_fold,
            "window_sequence_aggregate": {
                name: aggregate_metrics.get(name) for name in METRIC_NAMES
            },
            "subject_level_summary": {
                "rows": int(len(subject_metrics)),
                "undefined_qwk": int(
                    subject_metrics["quadratic_weighted_kappa"].isna().sum()
                ),
                "undefined_expected_rank_spearman": int(
                    subject_metrics["expected_rank_spearman"].isna().sum()
                ),
                "metric_means": {
                    name: float(subject_metrics[name].mean(skipna=True))
                    for name in (
                        "balanced_accuracy", "macro_f1",
                        "quadratic_weighted_kappa", "ordinal_mae",
                        "adjacent_accuracy", "severe_error_rate",
                        "expected_rank_mae", "expected_rank_spearman",
                    )
                },
            },
            "source_level_results": source_metrics,
            "class_level_results": class_metrics,
            "categorical_reference": {
                "run_directory": _relative_path(reference_run),
                "alignment": alignment,
                "metrics": {
                    name: reference_metrics.get(name)
                    for name in comparison_metric_names
                },
                "descriptive_deltas_ordinal_minus_categorical": categorical_deltas,
            },
            "generated_artifacts": {
                "subject_metrics": _relative_path(subject_path),
                "class_metrics": _relative_path(class_path),
                "aggregate_metrics": _relative_path(aggregate_path),
                "source_metrics": _relative_path(source_path),
                "probability_audit": _relative_path(probability_path),
                "checkpoint_reload_audit": _relative_path(checkpoint_path),
                "ordinal_diagnostics": _relative_path(ordinal_path),
            },
        }
        _write_json(run_dir / "ordinal_full_trial_manifest.json", audit)
        return audit

    def _combined_summary(
        self,
        plans: Sequence[OrdinalTransformerFullTrialPlan],
        completed: Mapping[str, CompletedBenchmarkRun],
        audits: Mapping[str, Mapping[str, Any]],
        outcomes: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        context = self._build_context()
        frames: dict[str, pd.DataFrame] = {}
        for plan in plans:
            model = self._result_model(completed[plan.trial_id])
            path = Path(model["artifacts"]["predictions"])
            frames[plan.trial_id] = pd.read_parquet(path)
        for group in FULL_FEATURE_GROUPS:
            run_dir = _repo_path(
                self.document["categorical_references"][group]["run_directory"]
            )
            path = next(run_dir.glob("**/group_kfold_subject/predictions.parquet"))
            frames[f"categorical_{group}"] = pd.read_parquet(path)
        reference = frames["categorical_eeg_only"]
        alignments = {
            name: full_prediction_alignment(reference, frame)
            for name, frame in frames.items()
        }
        exact = all(value["exact_match"] for value in alignments.values())
        if not exact:
            raise ValueError("Six-method exact alignment audit failed")

        cutpoint_rows: list[dict[str, Any]] = []
        risk_rows: list[dict[str, Any]] = []
        for plan in plans:
            for fold in audits[plan.trial_id]["folds"]:
                if plan.head_type == "coral":
                    diagnostics = fold["head_diagnostics"]
                    cutpoint_rows.append({
                        "feature_group": plan.feature_group,
                        "fold": fold["fold"],
                        "best_epoch": fold["best_epoch"],
                        **{
                            f"cutpoint_{index}": diagnostics[f"cutpoint_{index}"]
                            for index in range(4)
                        },
                        "minimum_gap": diagnostics["cutpoint_min_gap"],
                    })
                else:
                    diagnostics = fold["objective_training_diagnostics"]
                    counts = [
                        int(diagnostics[f"risk_count_{index}"])
                        for index in range(4)
                    ]
                    risk_rows.append({
                        "feature_group": plan.feature_group,
                        "fold": fold["fold"],
                        **{
                            f"risk_count_{index}": counts[index]
                            for index in range(4)
                        },
                        **{
                            f"risk_fraction_{index}": counts[index] / counts[0]
                            for index in range(4)
                        },
                    })
        cutpoint_summary: dict[str, Any] = {}
        cutpoint_frame = pd.DataFrame(cutpoint_rows)
        for group, rows in cutpoint_frame.groupby("feature_group", sort=True):
            cutpoint_summary[str(group)] = {
                f"cutpoint_{index}": {
                    "mean": float(rows[f"cutpoint_{index}"].mean()),
                    "std": float(rows[f"cutpoint_{index}"].std(ddof=0)),
                }
                for index in range(4)
            }

        summary = {
            "schema_version": 1,
            "status": "completed",
            "experiment": str(self.document["experiment"]["name"]),
            "seed": 42,
            "canonical_data": {
                "supervised_rows": context["supervised_rows"],
                "sequences": context["sequence_count"],
                "subjects": context["subject_count"],
                "sequence_length": 8,
                "sequence_index_sha256": context["sequence_index_sha256"],
                "source_parquet_sha256": context["source_parquet_sha256"],
            },
            "matrix": {
                "trials": 4,
                "folds_per_trial": 5,
                "fold_runs": 20,
                "head_types": list(FULL_HEAD_TYPES),
                "feature_groups": list(FULL_FEATURE_GROUPS),
            },
            "categorical_reference_audit": context["categorical_references"],
            "outcomes": _jsonable(outcomes),
            "exact_alignment": {
                "all_exact": exact,
                "methods": alignments,
            },
            "cutpoints_by_fold": cutpoint_rows,
            "cutpoint_summary": cutpoint_summary,
            "risk_counts_by_fold": risk_rows,
            "trials": _jsonable(audits),
            "statistical_inference_performed": False,
            "winner_selected": False,
            "ready_for_task_6d": True,
        }
        return summary

    @staticmethod
    def _render_report(summary: Mapping[str, Any]) -> str:
        metric_lines = []
        fold_metric_lines = []
        for trial_id, trial in summary["trials"].items():
            values = trial["fold_aggregate"]
            metric_lines.append(
                f"| {trial_id} | {values['balanced_accuracy_mean']:.4f} ± "
                f"{values['balanced_accuracy_std']:.4f} | "
                f"{values['macro_f1_mean']:.4f} ± {values['macro_f1_std']:.4f} | "
                f"{values['quadratic_weighted_kappa_mean']:.4f} ± "
                f"{values['quadratic_weighted_kappa_std']:.4f} | "
                f"{values['ordinal_mae_mean']:.4f} ± {values['ordinal_mae_std']:.4f} |"
            )
            for row in trial["fold_metrics"]:
                fold_metric_lines.append(
                    f"| {trial_id} | {row['fold']} | {row['balanced_accuracy']:.4f} | "
                    f"{row['macro_f1']:.4f} | {row['quadratic_weighted_kappa']:.4f} | "
                    f"{row['ordinal_mae']:.4f} | {row['expected_rank_mae']:.4f} | "
                    f"{row['expected_rank_spearman']:.4f} |"
                )
        training_lines = []
        for trial_id, trial in summary["trials"].items():
            epochs = [row["epochs_trained"] for row in trial["folds"]]
            best = [row["best_epoch"] for row in trial["folds"]]
            losses = [round(row["best_validation_loss"], 6) for row in trial["folds"]]
            duration = sum(row["training_duration_seconds"] for row in trial["folds"])
            training_lines.append(
                f"| {trial_id} | {trial['parameter_count']} | {epochs} | {best} | "
                f"{losses} | {duration:.1f} |"
            )
        cutpoint_lines = [
            "| {feature_group} | {fold} | {best_epoch} | {cutpoint_0:.6f} | "
            "{cutpoint_1:.6f} | {cutpoint_2:.6f} | {cutpoint_3:.6f} | "
            "{minimum_gap:.6f} |".format(**row)
            for row in summary["cutpoints_by_fold"]
        ]
        risk_lines = [
            f"| {row['feature_group']} | {row['fold']} | "
            f"{row['risk_count_0']} | {row['risk_count_1']} | "
            f"{row['risk_count_2']} | {row['risk_count_3']} | "
            f"{row['risk_fraction_1']:.4f} | {row['risk_fraction_2']:.4f} | "
            f"{row['risk_fraction_3']:.4f} |"
            for row in summary["risk_counts_by_fold"]
        ]
        cutpoint_summary_lines = []
        for group, values in summary["cutpoint_summary"].items():
            cutpoint_summary_lines.append(
                f"| {group} | "
                + " | ".join(
                    f"{values[f'cutpoint_{index}']['mean']:.6f} ± "
                    f"{values[f'cutpoint_{index}']['std']:.6f}"
                    for index in range(4)
                )
                + " |"
            )
        probability_lines = []
        checkpoint_lines = []
        for trial_id, trial in summary["trials"].items():
            probability_rows = [row["probability_audit"] for row in trial["folds"]]
            checkpoint_rows = [row["checkpoint_reload"] for row in trial["folds"]]
            probability_lines.append(
                f"| {trial_id} | "
                f"{min(row['minimum_class_probability'] for row in probability_rows):.3g} | "
                f"{max(row['maximum_class_probability_sum_error'] for row in probability_rows):.3g} | "
                f"{max(row['maximum_monotonicity_violation'] for row in probability_rows):.3g} | "
                f"{sum(row['round_off_correction_count'] for row in probability_rows)} | "
                f"{np.mean([row['ordinal_argmax_disagreement_fraction'] for row in probability_rows]):.4f} |"
            )
            checkpoint_lines.append(
                f"| {trial_id} | "
                f"{sum(row['y_pred_mismatches'] for row in checkpoint_rows)} | "
                f"{max(row['maximum_class_probability_delta'] for row in checkpoint_rows):.3g} | "
                f"{max(row['maximum_threshold_probability_delta'] for row in checkpoint_rows):.3g} | "
                f"{max(row['maximum_expected_rank_delta'] for row in checkpoint_rows):.3g} | "
                f"{all(row['incompatible_head_rejected'] for row in checkpoint_rows)} |"
            )
        subject_lines = []
        source_lines = []
        class_lines = []
        delta_lines = []
        for trial_id, trial in summary["trials"].items():
            subject = trial["subject_level_summary"]
            subject_lines.append(
                f"| {trial_id} | {subject['rows']} | "
                f"{subject['metric_means']['balanced_accuracy']:.4f} | "
                f"{subject['metric_means']['ordinal_mae']:.4f} | "
                f"{subject['undefined_qwk']} | "
                f"{subject['undefined_expected_rank_spearman']} |"
            )
            for row in trial["source_level_results"]:
                source_lines.append(
                    f"| {trial_id} | {row['source']} | {row['sequences']} | "
                    f"{row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | "
                    f"{row['ordinal_mae']:.4f} |"
                )
            for row in trial["class_level_results"]:
                class_lines.append(
                    f"| {trial_id} | {row['class_id']} | {row['support']} | "
                    f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
                )
            delta = trial["categorical_reference"][
                "descriptive_deltas_ordinal_minus_categorical"
            ]
            delta_lines.append(
                f"| {trial_id} | {delta['balanced_accuracy']:+.4f} | "
                f"{delta['macro_f1']:+.4f} | {delta['quadratic_weighted_kappa']:+.4f} | "
                f"{delta['ordinal_mae']:+.4f} | {delta['severe_error_rate']:+.4f} |"
            )
        return "\n".join([
            "# Ordinal Transformer full seed-42 study",
            "",
            "## 1. Цель эксперимента",
            "",
            "Выполнено первичное описательное сравнение CORAL и CORN на EEG-only и EEG+POW.",
            "",
            "> Выбор лучшего метода и статистические выводы выполняются отдельно в задаче 6Д. Средние показатели этого отчёта являются предварительными описательными результатами.",
            "",
            "## 2. Канонические данные и последовательности",
            "",
            f"Использованы {summary['canonical_data']['sequences']} последовательности длины 8, "
            f"{summary['canonical_data']['subjects']} субъекта и hash "
            f"`{summary['canonical_data']['sequence_index_sha256']}`.",
            "",
            "## 3. Группы признаков",
            "",
            "EEG-only содержит 168 признаков на окно; EEG+POW — 448.",
            "",
            "## 4. Архитектуры выходных частей",
            "",
            "Общий Transformer encoder неизменён; различаются только CORAL и CORN heads и размер входа.",
            "",
            "## 5. Протокол разбиения",
            "",
            "Seed 42, пять outer GroupKFold по subject_id; inner validation — record_group_id.",
            "",
            "## 6. Проверка утечек",
            "",
            "Во всех 20 fold-runs outer subject overlap и inner record-group overlap равны нулю.",
            "",
            "## 7. Exact alignment",
            "",
            f"Шесть методов: exact={summary['exact_alignment']['all_exact']}; 44 142 общих sequence IDs, расхождений нет.",
            "",
            "## 8. Процесс обучения",
            "",
            "| Trial | Parameters | Epochs by fold | Best epochs | Best validation losses | Training seconds |",
            "| --- | ---: | --- | --- | --- | ---: |",
            *training_lines,
            "",
            "## 9. CORAL cutpoints",
            "",
            "Cutpoints находятся в пространстве скрытой оценки модели и не являются границами target_focus.",
            "",
            "| Group | Fold | Best epoch | c0 | c1 | c2 | c3 | Minimum gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *cutpoint_lines,
            "",
            "Среднее ± стандартное отклонение между folds:",
            "",
            "| Group | c0 | c1 | c2 | c3 |",
            "| --- | ---: | ---: | ---: | ---: |",
            *cutpoint_summary_lines,
            "",
            "## 10. CORN risk sets",
            "",
            "| Group | Fold | risk0 | risk1 | risk2 | risk3 | fraction1 | fraction2 | fraction3 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *risk_lines,
            "",
            "## 11. Проверка вероятностей",
            "",
            "Все probabilities конечны, неотрицательны, нормированы; cumulative probabilities монотонны.",
            "",
            "| Trial | Minimum class p | Maximum sum error | Maximum monotonicity violation | Corrections | Mean argmax disagreement |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *probability_lines,
            "",
            "## 12. Проверка checkpoints",
            "",
            "Все 20 checkpoint загружены strict=True; predictions совпали, несовместимые heads отклонены.",
            "",
            "| Trial | y_pred mismatches | Maximum class p delta | Maximum threshold p delta | Maximum rank delta | Wrong head rejected |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            *checkpoint_lines,
            "",
            "## 13. Fold-level метрики",
            "",
            "| Trial | Fold | Balanced accuracy | Macro F1 | QWK | Ordinal MAE | Rank MAE | Rank Spearman |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *fold_metric_lines,
            "",
            "## 14. Агрегированные метрики",
            "",
            "| Trial | Balanced accuracy | Macro F1 | QWK | Ordinal MAE |",
            "| --- | ---: | ---: | ---: | ---: |",
            *metric_lines,
            "",
            "## 15. Subject-level descriptive summary",
            "",
            "| Trial | Subjects | Mean balanced accuracy | Mean ordinal MAE | Undefined QWK | Undefined rank Spearman |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *subject_lines,
            "",
            "## 16. Source-level descriptive summary",
            "",
            "| Trial | Source | Sequences | Balanced accuracy | Macro F1 | Ordinal MAE |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *source_lines,
            "",
            "## 17. Class-level результаты",
            "",
            "| Trial | Class | Support | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *class_lines,
            "",
            "## 18. Описательные дельты к categorical",
            "",
            "| Trial | Δ balanced accuracy | Δ macro F1 | Δ QWK | Δ ordinal MAE | Δ severe error |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *delta_lines,
            "",
            "## 19. Ограничения",
            "",
            "Использован один seed; статистические тесты, дополнительные seeds и выбор победителя не выполнялись.",
            "",
            "## 20. Подготовленность к статистическому анализу",
            "",
            "Сохранены aligned predictions и subject/source/class-level таблицы. Материалы готовы к отдельной задаче 6Д.",
            "",
        ])

    def execute(
        self,
        plans: Sequence[OrdinalTransformerFullTrialPlan],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        invalid = [plan for plan in plans if plan.status != "valid"]
        if invalid:
            raise ValueError(
                "Invalid ordinal full trials: "
                + "; ".join(
                    f"{plan.trial_id}: {', '.join(plan.invalid_reasons)}"
                    for plan in invalid
                )
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        context = self._build_context()
        canonical_path = self.output_root / "canonical_sequence_index.parquet"
        context["canonical"].to_parquet(canonical_path, index=False)
        completed: dict[str, CompletedBenchmarkRun] = {}
        outcomes: list[dict[str, Any]] = []
        for plan in plans:
            found = self.completed_run_finder(
                plan.resolved_config, search_directories=[plan.output_dir]
            )
            reusable = found if self._completed_is_reusable(
                found, plan.resolved_config
            ) else None
            if resume and reusable is not None:
                completed[plan.trial_id] = reusable
                outcomes.append({**plan.to_dict(), "outcome": "resumed"})
                continue
            runner = self.runner_factory(deepcopy(dict(plan.resolved_config)))
            runner.run()
            run = runner.completed_run()
            if not self._completed_is_reusable(run, plan.resolved_config):
                raise ValueError(f"New run is incomplete: {plan.trial_id}")
            completed[plan.trial_id] = run
            outcomes.append({**plan.to_dict(), "outcome": "completed"})
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        audits: dict[str, Mapping[str, Any]] = {}
        split_cache: dict[str, Mapping[str, Any]] = {}
        for plan in plans:
            if self.trial_auditor is not None:
                audits[plan.trial_id] = dict(
                    self.trial_auditor(plan, completed[plan.trial_id], None)
                )
                continue
            if plan.feature_group not in split_cache:
                split_cache[plan.feature_group] = self._rebuild_splits(
                    plan.resolved_config
                )
            audits[plan.trial_id] = self._audit_trial(
                plan, completed[plan.trial_id], split_cache[plan.feature_group]
            )
        if self.trial_auditor is None:
            summary = self._combined_summary(plans, completed, audits, outcomes)
            report_path = _repo_path("reports/ordinal_transformer_full_seed42.md")
            summary_path = _repo_path(
                "reports/ordinal_transformer_full_seed42_summary.json"
            )
            report_path.write_text(self._render_report(summary), encoding="utf-8")
            _write_json(summary_path, summary)
        else:
            summary = {
                "status": "completed",
                "matrix": {"trials": 4, "folds_per_trial": 5, "fold_runs": 20},
                "outcomes": outcomes,
                "trials": _jsonable(audits),
            }
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "experiment": str(self.document["experiment"]["name"]),
            "config_file": _relative_path(self.spec_path),
            "canonical_sequence_index": _relative_path(canonical_path),
            "sequence_index_sha256": context["sequence_index_sha256"],
            "source_parquet_sha256": context["source_parquet_sha256"],
            "outcomes": outcomes,
            "summary": summary,
        }
        _write_json(
            self.output_root / "ordinal_transformer_full_seed42_manifest.json",
            manifest,
        )
        return manifest


__all__ = [
    "ALIGNMENT_COLUMNS",
    "FULL_FEATURE_GROUPS",
    "FULL_FOLDS",
    "FULL_HEAD_TYPES",
    "OrdinalTransformerFullExperiment",
    "OrdinalTransformerFullTrialPlan",
    "full_prediction_alignment",
    "load_ordinal_transformer_full_spec",
]
