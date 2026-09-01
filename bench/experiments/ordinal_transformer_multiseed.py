"""Comparable three-seed categorical/CORAL/CORN Transformer experiment."""

from __future__ import annotations

import gc
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from bench.bench_runner import BenchmarkRunner, CompletedBenchmarkRun, benchmark_config_hash
from bench.experiments.ordinal_transformer import (
    _jsonable,
    _relative_path,
    _repo_path,
    _write_json,
)
from bench.experiments.ordinal_transformer_full import (
    FULL_FEATURE_GROUPS,
    FULL_FOLDS,
    OrdinalTransformerFullExperiment,
    OrdinalTransformerFullTrialPlan,
    full_prediction_alignment,
)
from bench.validation.metrics import MetricsCalculator
from cogstate.model_zoo import build_model


HEAD_TYPES = ("categorical", "coral", "corn")
NEW_SEEDS = (7, 123)
ALL_SEEDS = (7, 42, 123)
ARCHITECTURE_KEYS = (
    "sequence_length", "d_model", "nhead", "num_layers", "dim_feedforward",
    "dropout", "activation", "pooling", "positional_encoding",
    "batch_size", "max_epochs", "learning_rate", "weight_decay",
    "validation_size", "early_stopping_patience", "standardize", "num_workers",
)


def load_ordinal_transformer_multiseed_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _repo_path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Ordinal multiseed experiment not found: {spec_path}")
    document = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    required = {
        "experiment", "dataset", "task", "feature_groups", "feature_definitions",
        "head_types", "seeds", "categorical_references", "ordinal_seed42_references",
        "categorical_search_roots", "model", "sequence", "validation",
        "evaluation", "protocol",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Ordinal multiseed experiment is missing sections: {missing}")
    if document["experiment"].get("type") != "ordinal_transformer_multiseed":
        raise ValueError("Expected experiment.type=ordinal_transformer_multiseed")
    if tuple(document["feature_groups"]) != FULL_FEATURE_GROUPS:
        raise ValueError("feature_groups must be eeg_only, eeg_pow")
    if tuple(document["head_types"]) != ("coral", "corn"):
        raise ValueError("head_types must be coral, corn")
    if tuple(int(value) for value in document["seeds"]) != NEW_SEEDS:
        raise ValueError("New ordinal seeds must be exactly 7 and 123")
    if tuple(int(value) for value in document["evaluation"]["folds"]) != FULL_FOLDS:
        raise ValueError("All five canonical folds are required")
    if int(document["validation"]["random_state"]) != 42:
        raise ValueError("Inner validation split seed must remain fixed at 42")
    if int(document["evaluation"]["random_state"]) != 42:
        raise ValueError("Outer split seed must remain fixed at 42")
    if int(document["model"]["params"]["max_epochs"]) != 15:
        raise ValueError("Comparable full runs require max_epochs=15")
    return document


@dataclass(frozen=True)
class CategoricalCandidateAudit:
    feature_group: str
    seed: int
    run_directory: Path
    eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_group": self.feature_group,
            "seed": self.seed,
            "run_directory": _relative_path(self.run_directory),
            "eligible": self.eligible,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class OrdinalTransformerMultiseedPlan:
    trials: tuple[OrdinalTransformerFullTrialPlan, ...]
    categorical_candidates: tuple[CategoricalCandidateAudit, ...]
    selected_categorical_references: Mapping[str, str]

    @property
    def fold_runs(self) -> int:
        return sum(len(trial.folds) for trial in self.trials)


class OrdinalTransformerMultiseedExperiment(OrdinalTransformerFullExperiment):
    """Run only missing comparable seeds while preserving canonical splits."""

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
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.document = load_ordinal_transformer_multiseed_spec(self.spec_path)
        self.output_root = _repo_path(
            output_dir or self.document["experiment"]["output_dir"]
        )
        self.runner_factory = runner_factory
        self.completed_run_finder = completed_run_finder
        self.context_builder = context_builder
        self.trial_auditor = None
        self.reference_auditor = None
        self._context: dict[str, Any] | None = None

    @staticmethod
    def _prediction_file(run_directory: Path) -> Path | None:
        matches = list(run_directory.glob("**/group_kfold_subject/predictions.parquet"))
        return matches[0] if len(matches) == 1 else None

    def _candidate_group(self, config: Mapping[str, Any]) -> str | None:
        experiment = config.get("experiment", {})
        group = str(experiment.get("feature_group", ""))
        if group in FULL_FEATURE_GROUPS:
            return group
        dataset = next(iter(config.get("datasets", {}).values()), {})
        feature_set = str(dataset.get("feature_set", ""))
        return {
            "eeg_only": "eeg_only",
            "eeg_pow": "eeg_pow",
            "pow_plus_eeg": "eeg_pow",
        }.get(feature_set)

    def _audit_categorical_candidate(
        self, run_directory: Path, group: str, seed: int
    ) -> CategoricalCandidateAudit:
        reasons: list[str] = []
        try:
            manifest = json.loads(
                (run_directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            config = yaml.safe_load(
                (run_directory / "config.yaml").read_text(encoding="utf-8")
            )
            params = config["models"]["torch_transformer"]["params"]
            dataset = next(iter(config["datasets"].values()))
            prediction_path = self._prediction_file(run_directory)
            if manifest.get("status") != "completed":
                reasons.append("run manifest is not completed")
            if self._candidate_group(config) != group:
                reasons.append("feature group mismatch")
            if str(params.get("head_type", "categorical")) != "categorical":
                reasons.append("not a categorical head")
            if int(params.get("random_state", -1)) != seed:
                reasons.append("model seed mismatch")
            for key in ARCHITECTURE_KEYS:
                expected = self.document["model"]["params"][key]
                observed = params.get(key)
                if observed != expected:
                    reasons.append(f"model parameter {key} mismatch")
            if int(config.get("validation", {}).get("random_state", -1)) != 42:
                reasons.append("inner validation split seed is not 42")
            if int(config.get("evaluation", {}).get("random_state", -1)) != 42:
                reasons.append("outer split seed is not 42")
            if int(config.get("task_config", {}).get("random_state", -1)) != 42:
                reasons.append("task split seed is not 42")
            definition = self.document["feature_definitions"][group]
            if int(dataset.get("expected_feature_count", dataset.get("max_features", -1))) != int(definition["feature_count"]):
                reasons.append("feature count mismatch")
            if prediction_path is None:
                reasons.append("missing unique unified predictions")
            else:
                predictions = pd.read_parquet(prediction_path)
                if len(predictions) != int(self.document["dataset"]["expected_sequences"]):
                    reasons.append("sequence count mismatch")
                if sorted(predictions["fold"].astype(int).unique()) != list(FULL_FOLDS):
                    reasons.append("not all five folds")
                if predictions["subject_id"].nunique() != int(self.document["dataset"]["expected_subjects"]):
                    reasons.append("subject count mismatch")
                if self._context is not None and not full_prediction_alignment(
                    self._context["canonical"], predictions
                )["exact_match"]:
                    reasons.append("canonical sequence-index mismatch")
            feature_manifests = list(
                run_directory.glob("**/group_kfold_subject/fold_*/feature_manifest.json")
            )
            if len(feature_manifests) != len(FULL_FOLDS):
                reasons.append("missing fold feature manifests")
            else:
                expected_feature_hash = str(definition["feature_list_sha256"])
                for feature_manifest in feature_manifests:
                    value = json.loads(feature_manifest.read_text(encoding="utf-8"))
                    if value.get("feature_list_sha256") != expected_feature_hash:
                        reasons.append("feature-list hash mismatch")
                        break
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            reasons.append(f"unreadable candidate: {type(error).__name__}: {error}")
        return CategoricalCandidateAudit(
            feature_group=group,
            seed=seed,
            run_directory=run_directory,
            eligible=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def _discover_categorical_candidates(
        self,
    ) -> tuple[list[CategoricalCandidateAudit], dict[tuple[str, int], Path]]:
        audits: list[CategoricalCandidateAudit] = []
        selected: dict[tuple[str, int], Path] = {}
        seen: set[Path] = set()
        for group in FULL_FEATURE_GROUPS:
            canonical = _repo_path(
                self.document["categorical_references"][group]["run_directory"]
            )
            audit = self._audit_categorical_candidate(canonical, group, 42)
            audits.append(audit)
            seen.add(canonical.resolve())
            if audit.eligible:
                selected[(group, 42)] = canonical
        for root_value in self.document["categorical_search_roots"]:
            root = _repo_path(root_value)
            if not root.is_dir():
                continue
            for manifest_path in root.rglob("run_manifest.json"):
                run_directory = manifest_path.parent
                if run_directory.resolve() in seen:
                    continue
                config_path = run_directory / "config.yaml"
                if not config_path.is_file():
                    continue
                try:
                    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                    group = self._candidate_group(config)
                    params = config["models"]["torch_transformer"]["params"]
                    seed = int(params.get("random_state", -1))
                    head = str(params.get("head_type", "categorical"))
                except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
                    continue
                if group not in FULL_FEATURE_GROUPS or seed not in ALL_SEEDS or head != "categorical":
                    continue
                audit = self._audit_categorical_candidate(run_directory, group, seed)
                audits.append(audit)
                seen.add(run_directory.resolve())
                if audit.eligible and (group, seed) not in selected:
                    selected[(group, seed)] = run_directory
        return audits, selected

    def _resolved_config(
        self,
        head_type: str,
        feature_group: str,
        seed: int,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        dataset = self.document["dataset"]
        feature = self.document["feature_definitions"][feature_group]
        model = self.document["model"]
        trial_id = f"{head_type}_{feature_group}_seed{seed}"
        params = deepcopy(model["params"])
        params.update({"head_type": head_type, "random_state": seed})
        validation = deepcopy(self.document["validation"])
        evaluation = deepcopy(self.document["evaluation"])
        validation["random_state"] = 42
        evaluation["random_state"] = 42
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
            "validation": validation,
            "evaluation": evaluation,
            "task_config": {"random_state": 42},
            "run_within_subject": False,
            "run_loso": False,
            "experiment": {
                "name": str(self.document["experiment"]["name"]),
                "type": "ordinal_transformer_multiseed",
                "trial_id": trial_id,
                "head_type": head_type,
                "feature_group": feature_group,
                "seed": seed,
                "required_folds": list(FULL_FOLDS),
                "full_sequence_index_sha256": str(context["sequence_index_sha256"]),
                "split_seed": 42,
            },
        }

    def _completed_is_reusable(
        self, completed: CompletedBenchmarkRun | None, config: Mapping[str, Any]
    ) -> bool:
        if completed is None or completed.config_hash != benchmark_config_hash(config):
            return False
        try:
            manifest = json.loads(
                (completed.run_directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            saved = yaml.safe_load(
                (completed.run_directory / "config.yaml").read_text(encoding="utf-8")
            )
            folds = sorted(self._result_model(completed)["folds"])
            return bool(
                manifest.get("status") == "completed"
                and saved["experiment"]["type"] == "ordinal_transformer_multiseed"
                and saved["experiment"]["trial_id"] == config["experiment"]["trial_id"]
                and saved["experiment"]["split_seed"] == 42
                and folds == [f"fold_{fold:02d}" for fold in FULL_FOLDS]
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def plan(self) -> OrdinalTransformerMultiseedPlan:
        context = self._build_context()
        expected = self.document["dataset"]
        common_reasons: list[str] = []
        for key, observed_key in (
            ("expected_supervised_rows", "supervised_rows"),
            ("expected_sequences", "sequence_count"),
            ("expected_subjects", "subject_count"),
        ):
            if int(expected[key]) != int(context[observed_key]):
                common_reasons.append(f"{observed_key} mismatch")
        if expected["sequence_index_sha256"] != context["sequence_index_sha256"]:
            common_reasons.append("sequence-index hash mismatch")
        if expected["parquet_sha256"] != context["source_parquet_sha256"]:
            common_reasons.append("source Parquet hash mismatch")
        candidate_audits, selected = self._discover_categorical_candidates()
        trials: list[OrdinalTransformerFullTrialPlan] = []
        combinations = [
            ("categorical", group, seed)
            for seed in NEW_SEEDS for group in FULL_FEATURE_GROUPS
        ] + [
            (head, group, seed)
            for seed in NEW_SEEDS
            for head in ("coral", "corn")
            for group in FULL_FEATURE_GROUPS
        ]
        for head, group, seed in combinations:
            feature = context["features"][group]
            definition = self.document["feature_definitions"][group]
            reasons = list(common_reasons)
            if feature["count"] != int(definition["feature_count"]):
                reasons.append("feature count mismatch")
            if feature["sha256"] != str(definition["feature_list_sha256"]):
                reasons.append("feature-list hash mismatch")
            config = self._resolved_config(head, group, seed, context)
            output = Path(config["output_dir"])
            found = self.completed_run_finder(config, search_directories=[output])
            completed = found if self._completed_is_reusable(found, config) else None
            adapter = build_model(
                "torch_transformer", "classification",
                input_shape=(8, int(feature["count"])), num_outputs=5,
                params=config["models"]["torch_transformer"]["params"],
            )
            trials.append(OrdinalTransformerFullTrialPlan(
                trial_id=config["experiment"]["trial_id"],
                head_type=head,
                feature_group=group,
                feature_count=int(feature["count"]),
                feature_list_sha256=str(feature["sha256"]),
                input_shape=(8, int(feature["count"])),
                sequence_count=int(context["sequence_count"]),
                subject_count=int(context["subject_count"]),
                sequence_index_sha256=str(context["sequence_index_sha256"]),
                folds=FULL_FOLDS,
                fold_summaries=deepcopy(context["fold_summaries"]),
                model_parameter_count=int(adapter.model_metadata["parameter_count"]),
                maximum_epochs=15,
                patience=int(self.document["model"]["params"]["early_stopping_patience"]),
                seed=seed,
                output_dir=output,
                config_hash=benchmark_config_hash(config),
                status="valid" if not reasons else "invalid",
                invalid_reasons=tuple(reasons),
                action="reuse" if completed else "run",
                resolved_config=config,
                completed_run=completed,
            ))
            del adapter
        selected_text = {
            f"{group}_seed{seed}": _relative_path(path)
            for (group, seed), path in sorted(selected.items())
        }
        return OrdinalTransformerMultiseedPlan(
            trials=tuple(trials),
            categorical_candidates=tuple(candidate_audits),
            selected_categorical_references=selected_text,
        )

    @staticmethod
    def render_plan(plan: OrdinalTransformerMultiseedPlan) -> str:
        rows = [
            "# Ordinal Transformer multiseed plan", "",
            "| Trial | Head/group/seed | Features/hash | Input | Sequences/subjects | Sequence hash | Folds | Parameters | Epochs/patience | Output | Config hash | Action/status |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
        for trial in plan.trials:
            rows.append(
                f"| `{trial.trial_id}` | {trial.head_type}/{trial.feature_group}/{trial.seed} | "
                f"{trial.feature_count}/`{trial.feature_list_sha256[:12]}` | "
                f"`{list(trial.input_shape)}` | {trial.sequence_count}/{trial.subject_count} | "
                f"`{trial.sequence_index_sha256[:12]}` | {list(trial.folds)} | "
                f"{trial.model_parameter_count} | {trial.maximum_epochs}/{trial.patience} | "
                f"`{_relative_path(trial.output_dir)}` | `{trial.config_hash[:12]}` | "
                f"{trial.action}/{trial.status} |"
            )
        ordinal_count = sum(trial.head_type != "categorical" for trial in plan.trials)
        categorical_count = sum(trial.head_type == "categorical" for trial in plan.trials)
        rows.extend([
            "", f"New ordinal trials: {ordinal_count}; missing categorical trials: {categorical_count}.",
            f"Total new/reusable trials: {len(plan.trials)}; fold-runs: {plan.fold_runs}.",
            "Model seeds vary; outer, task, and inner-validation split seeds stay fixed at 42.",
            "", "## Categorical baseline slots", "",
            "| Group | Seed | Validity | Reused or missing | Run | Reason |",
            "| --- | ---: | --- | --- | --- | --- |",
        ])
        eligible = {
            (audit.feature_group, audit.seed): audit
            for audit in plan.categorical_candidates if audit.eligible
        }
        rejected = {
            (audit.feature_group, audit.seed): audit
            for audit in plan.categorical_candidates if not audit.eligible
        }
        for group in FULL_FEATURE_GROUPS:
            for seed in ALL_SEEDS:
                audit = eligible.get((group, seed)) or rejected.get((group, seed))
                if (group, seed) in eligible:
                    validity, action, reason = "valid", "reused", "comparable"
                else:
                    validity, action = "missing", "scheduled"
                    reason = "; ".join(audit.reasons) if audit else "no completed candidate found"
                path = "-" if audit is None else f"`{_relative_path(audit.run_directory)}`"
                rows.append(
                    f"| {group} | {seed} | {validity} | {action} | {path} | {reason} |"
                )
        rows.extend(["", "## All categorical candidates inspected", "",
            "| Group | Seed | Eligible | Run | Reason |",
            "| --- | ---: | --- | --- | --- |",
        ])
        for audit in plan.categorical_candidates:
            rows.append(
                f"| {audit.feature_group} | {audit.seed} | {audit.eligible} | "
                f"`{_relative_path(audit.run_directory)}` | "
                f"{'; '.join(audit.reasons) or 'comparable'} |"
            )
        rows.append("\nPlan-only does not train models or write experiment artifacts.")
        return "\n".join(rows)

    def _audit_categorical_trial(
        self,
        plan: OrdinalTransformerFullTrialPlan,
        completed: CompletedBenchmarkRun,
        reference_run: Path,
    ) -> dict[str, Any]:
        result = self._result_model(completed)
        expected_folds = [f"fold_{fold:02d}" for fold in FULL_FOLDS]
        if sorted(result["folds"]) != expected_folds:
            raise ValueError(f"Incomplete categorical trial: {plan.trial_id}")
        frames: list[pd.DataFrame] = []
        fold_rows: list[dict[str, Any]] = []
        for fold_name in expected_folds:
            fold = result["folds"][fold_name]
            artifacts = {name: Path(path) for name, path in fold["artifacts"].items()}
            required = {
                "predictions", "metrics", "class_metrics", "feature_manifest",
                "sequence_index_manifest", "validation_split", "model",
                "training_log", "normalization_stats",
            }
            if sorted(required - set(artifacts)) or any(
                not artifacts[name].is_file() for name in required
            ):
                raise ValueError(f"Missing standard categorical artifacts in {fold_name}")
            frame = pd.read_parquet(artifacts["predictions"])
            probabilities = frame[[f"proba_{index}" for index in range(5)]].to_numpy(float)
            if not np.isfinite(probabilities).all() or np.min(probabilities) < -1e-8:
                raise ValueError(f"Invalid categorical probabilities in {fold_name}")
            sum_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
            if sum_error > 1e-6:
                raise ValueError(f"Categorical probabilities do not sum to one in {fold_name}")
            validation = json.loads(artifacts["validation_split"].read_text(encoding="utf-8"))
            if validation["group_overlap"] or validation["outer_test_record_overlap"]:
                raise ValueError(f"Inner leakage in {fold_name}")
            reference_fold = self._reference_fold_directory(reference_run, fold_name)
            ref_validation = json.loads(
                (reference_fold / "validation_split.json").read_text(encoding="utf-8")
            )
            normalization = json.loads(
                artifacts["normalization_stats"].read_text(encoding="utf-8")
            )
            ref_normalization = json.loads(
                (reference_fold / "normalization_stats.json").read_text(encoding="utf-8")
            )
            mean_delta = float(np.max(np.abs(
                np.asarray(normalization["mean"]) - np.asarray(ref_normalization["mean"])
            ), initial=0.0))
            scale_delta = float(np.max(np.abs(
                np.asarray(normalization["scale"]) - np.asarray(ref_normalization["scale"])
            ), initial=0.0))
            if (
                validation["inner_validation_group_ids"]
                != ref_validation["inner_validation_group_ids"]
                or normalization["feature_names"] != ref_normalization["feature_names"]
                or max(mean_delta, scale_delta) > 1e-12
            ):
                raise ValueError(f"Categorical split/normalization mismatch in {fold_name}")
            log = pd.read_csv(artifacts["training_log"])
            if not np.isfinite(log[["train_loss", "validation_loss"]].to_numpy()).all():
                raise ValueError(f"Non-finite categorical loss in {fold_name}")
            fold_rows.append({
                "fold": fold_name,
                "epochs_trained": int(len(log)),
                "best_epoch": int(fold["training"]["best_epoch"]),
                "best_validation_loss": float(fold["training"]["best_validation_loss"]),
                "training_duration_seconds": float(fold["training_time"]),
                "maximum_probability_sum_error": sum_error,
                "inner_group_overlap": [],
                "outer_subject_overlap": list(fold["split_metadata"].get("subject_overlap", [])),
                "normalization_mean_max_abs_delta": mean_delta,
                "normalization_scale_max_abs_delta": scale_delta,
            })
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        reference_predictions = pd.read_parquet(self._prediction_file(reference_run))
        alignment = full_prediction_alignment(reference_predictions, combined)
        if not alignment["exact_match"]:
            raise ValueError(f"Categorical seed alignment failed: {plan.trial_id}")
        probability = combined[[f"proba_{index}" for index in range(5)]].to_numpy(float)
        metrics = MetricsCalculator.calculate_all_metrics(
            combined["y_true"].to_numpy(int), combined["y_pred"].to_numpy(int), probability
        )
        audit = {
            "trial_id": plan.trial_id,
            "head_type": "categorical",
            "feature_group": plan.feature_group,
            "seed": plan.seed,
            "run_directory": _relative_path(completed.run_directory),
            "sequence_count": int(len(combined)),
            "subject_count": int(combined["subject_id"].nunique()),
            "parameter_count": plan.model_parameter_count,
            "folds": fold_rows,
            "alignment_to_seed42": alignment,
            "window_sequence_aggregate": metrics,
        }
        _write_json(completed.run_directory / "categorical_multiseed_trial_manifest.json", audit)
        return audit

    def execute(
        self, plan: OrdinalTransformerMultiseedPlan, *, resume: bool
    ) -> dict[str, Any]:
        invalid = [trial for trial in plan.trials if trial.status != "valid"]
        if invalid:
            raise ValueError("Invalid trials: " + "; ".join(
                f"{trial.trial_id}: {', '.join(trial.invalid_reasons)}" for trial in invalid
            ))
        self.output_root.mkdir(parents=True, exist_ok=True)
        context = self._build_context()
        canonical_path = self.output_root / "canonical_sequence_index.parquet"
        context["canonical"].to_parquet(canonical_path, index=False)
        completed: dict[str, CompletedBenchmarkRun] = {}
        outcomes: list[dict[str, Any]] = []
        ordered = sorted(plan.trials, key=lambda item: item.head_type != "categorical")
        for trial in ordered:
            found = self.completed_run_finder(
                trial.resolved_config, search_directories=[trial.output_dir]
            )
            reusable = found if self._completed_is_reusable(found, trial.resolved_config) else None
            if resume and reusable is not None:
                run = reusable
                outcome = "resumed"
            else:
                runner = self.runner_factory(deepcopy(dict(trial.resolved_config)))
                runner.run()
                run = runner.completed_run()
                if not self._completed_is_reusable(run, trial.resolved_config):
                    raise ValueError(f"New run is incomplete: {trial.trial_id}")
                del runner
                outcome = "completed"
            completed[trial.trial_id] = run
            outcomes.append({**trial.to_dict(), "outcome": outcome, "run_directory": _relative_path(run.run_directory)})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        baseline_runs: dict[tuple[str, int], Path] = {
            (group, 42): _repo_path(
                self.document["categorical_references"][group]["run_directory"]
            ) for group in FULL_FEATURE_GROUPS
        }
        for key, value in plan.selected_categorical_references.items():
            group, seed_text = key.rsplit("_seed", 1)
            baseline_runs[(group, int(seed_text))] = _repo_path(value)
        for trial in plan.trials:
            if trial.head_type == "categorical":
                baseline_runs[(trial.feature_group, trial.seed)] = completed[trial.trial_id].run_directory

        audits: dict[str, Any] = {}
        split_cache: dict[str, Mapping[str, Any]] = {}
        for trial in plan.trials:
            reference = baseline_runs[(trial.feature_group, trial.seed)]
            if trial.head_type == "categorical":
                seed42 = baseline_runs[(trial.feature_group, 42)]
                audits[trial.trial_id] = self._audit_categorical_trial(
                    trial, completed[trial.trial_id], seed42
                )
            else:
                if trial.feature_group not in split_cache:
                    split_cache[trial.feature_group] = self._rebuild_splits(
                        trial.resolved_config
                    )
                audits[trial.trial_id] = self._audit_trial(
                    trial, completed[trial.trial_id], split_cache[trial.feature_group], reference
                )

        run_index: list[dict[str, Any]] = []
        frames: dict[str, pd.DataFrame] = {}
        for group in FULL_FEATURE_GROUPS:
            for seed in ALL_SEEDS:
                categorical_path = baseline_runs[(group, seed)]
                key = f"categorical_{group}_seed{seed}"
                frames[key] = pd.read_parquet(self._prediction_file(categorical_path))
                run_index.append({"method": "categorical", "feature_group": group, "seed": seed, "run_directory": _relative_path(categorical_path)})
                for head in ("coral", "corn"):
                    if seed == 42:
                        path = _repo_path(
                            self.document["ordinal_seed42_references"][head][group]
                        )
                    else:
                        path = completed[f"{head}_{group}_seed{seed}"].run_directory
                    method_key = f"{head}_{group}_seed{seed}"
                    frames[method_key] = pd.read_parquet(self._prediction_file(path))
                    run_index.append({"method": head, "feature_group": group, "seed": seed, "run_directory": _relative_path(path)})
        reference = frames["categorical_eeg_only_seed42"]
        alignments = {
            key: full_prediction_alignment(reference, frame) for key, frame in frames.items()
        }
        if not all(value["exact_match"] for value in alignments.values()):
            raise ValueError("Three-seed exact alignment audit failed")
        summary = {
            "schema_version": 1,
            "status": "completed",
            "experiment": self.document["experiment"]["name"],
            "new_trials": len(plan.trials),
            "new_fold_runs": plan.fold_runs,
            "seeds": list(ALL_SEEDS),
            "split_seed": 42,
            "canonical_data": {
                "sequences": context["sequence_count"],
                "subjects": context["subject_count"],
                "sequence_index_sha256": context["sequence_index_sha256"],
                "source_parquet_sha256": context["source_parquet_sha256"],
            },
            "categorical_candidate_audit": [value.to_dict() for value in plan.categorical_candidates],
            "run_index": run_index,
            "alignment": {"all_exact": True, "runs": alignments},
            "outcomes": outcomes,
            "new_trial_audits": audits,
        }
        summary_path = _repo_path(self.document["experiment"]["summary_path"])
        report_path = _repo_path(self.document["experiment"]["report_path"])
        _write_json(summary_path, summary)
        report_lines = [
            "# Ordinal Transformer multiseed runs", "",
            "Seeds 7 and 123 were trained with the canonical seed-42 outer and inner splits. Seed 42 was reused.", "",
            f"New trials: {len(plan.trials)} ({plan.fold_runs} fold-runs). All 18 method/group/seed prediction artifacts align exactly.", "",
            "## Categorical baseline audit", "",
            "Existing seed-7/123 EEG+POW runs were excluded when their validation/task split seed differed from 42; comparable baselines were rerun.", "",
            "| Group | Seed | Eligible | Candidate | Reason |",
            "| --- | ---: | --- | --- | --- |",
        ]
        report_lines.extend(
            f"| {row.feature_group} | {row.seed} | {row.eligible} | `{_relative_path(row.run_directory)}` | "
            f"{'; '.join(row.reasons) or 'comparable'} |"
            for row in plan.categorical_candidates
        )
        report_lines.extend([
            "", "## New and reused runs", "",
            "## Runs", "",
            "| Method | Feature group | Seed | Run directory |", "| --- | --- | ---: | --- |",
        ])
        report_lines.extend(
            f"| {row['method']} | {row['feature_group']} | {row['seed']} | `{row['run_directory']}` |"
            for row in run_index
        )
        report_lines.extend([
            "", "## Split, normalization, probability, and checkpoint audits", "",
            "All outer subject overlaps and inner record-group overlaps are zero. For every matching feature group/fold, inner validation groups, feature order, normalization mean, and normalization scale match the canonical seed-42 baseline exactly.", "",
            "All new ordinal checkpoints loaded strictly into a fresh factory model; recomputed predictions match saved predictions within 1e-7. Class probabilities are finite, non-negative, normalized; cumulative probabilities are monotone.", "",
            "## Aggregate metrics for new trials", "",
            "| Trial | Balanced accuracy | Macro F1 | QWK | Ordinal MAE | Severe error |", "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for trial_id, audit in sorted(audits.items()):
            metrics = audit["window_sequence_aggregate"]
            report_lines.append(
                f"| {trial_id} | {metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{metrics['quadratic_weighted_kappa']:.4f} | {metrics['ordinal_mae']:.4f} | "
                f"{metrics['severe_error_rate']:.4f} |"
            )
        report_lines.extend([
            "", "## Training by fold", "",
            "| Trial | Epochs | Best epochs | Best validation loss | Training seconds |", "| --- | --- | --- | --- | ---: |",
        ])
        for trial_id, audit in sorted(audits.items()):
            folds = audit["folds"]
            report_lines.append(
                f"| {trial_id} | {'/'.join(str(row['epochs_trained']) for row in folds)} | "
                f"{'/'.join(str(row['best_epoch']) for row in folds)} | "
                f"{'/'.join(format(row['best_validation_loss'], '.4f') for row in folds)} | "
                f"{sum(row['training_duration_seconds'] for row in folds):.1f} |"
            )
        report_lines.extend([
            "", "## CORAL cutpoints and CORN risk sets", "",
            "CORAL cutpoints remain strictly ordered in every audited fold. CORN risk counts remain positive and non-increasing across thresholds. Absolute CORAL and CORN loss values are not compared.", "",
            "## Source- and class-level artifacts", "",
            "Each ordinal trial manifest stores source-level metrics and per-class precision/recall/F1. Categorical and ordinal unified predictions remain available for the common downstream multiseed analysis.",
        ])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "config_file": _relative_path(self.spec_path),
            "canonical_sequence_index": _relative_path(canonical_path),
            "summary": _relative_path(summary_path),
            "report": _relative_path(report_path),
            "run_index": run_index,
            "outcomes": outcomes,
        }
        _write_json(self.output_root / "ordinal_transformer_multiseed_manifest.json", manifest)
        return manifest


__all__ = [
    "ALL_SEEDS",
    "CategoricalCandidateAudit",
    "OrdinalTransformerMultiseedExperiment",
    "OrdinalTransformerMultiseedPlan",
    "load_ordinal_transformer_multiseed_spec",
]
