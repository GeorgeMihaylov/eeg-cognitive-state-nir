"""Optuna orchestration for nested trials executed by ``BenchmarkRunner``."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import optuna
import yaml
from optuna.trial import TrialState

from bench.bench_runner import BenchmarkRunner
from bench.datasets.datasets_registry import get_dataset
from bench.tasks.tasks_registry import get_task
from bench.validation.cross_val import CrossValidator

from .artifacts import initialize_study_artifacts, update_study_artifacts
from .objective import (
    AutoMLTrialResult,
    NestedBenchmarkObjective,
    extract_group_result,
)
from .search_space import AutoMLStudySpec, SearchParameterSpec, SearchSpaceSpec
from .trial_resolver import resolve_outer_evaluation_config


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMER_SEARCH_PATHS = frozenset({
    "model.params.d_model",
    "model.params.nhead",
    "model.params.num_layers",
    "model.params.dim_feedforward",
    "model.params.dropout",
    "training.learning_rate",
    "training.weight_decay",
    "training.batch_size",
})


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_automl_study_spec(path: str | Path) -> AutoMLStudySpec:
    spec_path = _repo_path(path)
    with open(spec_path, encoding="utf-8") as input_file:
        document = yaml.safe_load(input_file) or {}
    required = {"study", "base_config", "evaluation", "search", "search_space"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"AutoML study spec is missing keys: {missing}")
    study = document["study"]
    evaluation = document["evaluation"]
    search = document["search"]
    artifacts = document.get("artifacts", {})
    if evaluation.get("nested") is not True:
        raise ValueError("evaluation.nested must be true")
    constraints = tuple(
        document.get("constraints", ["d_model_divisible_by_nhead"])
    )
    return AutoMLStudySpec(
        name=str(study["name"]),
        objective_metric=str(study.get("objective_metric", "")),
        direction=str(study.get("direction", "")),
        sampler_seed=int(study.get("sampler_seed", 42)),
        base_config_path=_repo_path(document["base_config"]["path"]),
        output_root=_repo_path(
            artifacts.get(
                "output_root",
                "benchmark_results/automl/transformer_label_q5",
            )
        ),
        storage_path=_repo_path(
            artifacts.get(
                "storage",
                "benchmark_results/automl/transformer_label_q5/study.db",
            )
        ),
        inner_protocol=str(evaluation.get("inner_protocol", "")),
        inner_splits=int(evaluation.get("inner_splits", 3)),
        n_trials=int(search.get("n_trials", 10)),
        timeout_seconds=(
            None
            if search.get("timeout_seconds") is None
            else float(search["timeout_seconds"])
        ),
        max_epochs=(
            None if search.get("max_epochs") is None else int(search["max_epochs"])
        ),
        max_windows=(
            None if search.get("max_windows") is None else int(search["max_windows"])
        ),
        evaluate_best=bool(evaluation.get("evaluate_best", True)),
        search_space=SearchSpaceSpec.from_dict(
            document["search_space"], constraints=constraints
        ),
    )


class AutoMLStudyRunner:
    """Persistent study controller; all model work is delegated to the benchmark."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        outer_fold: int = 1,
        study_name: str | None = None,
        storage: str | Path | None = None,
        n_trials: int | None = None,
        timeout_seconds: float | None = None,
        seed: int | None = None,
        inner_splits: int | None = None,
        max_epochs: int | None = None,
        max_windows: int | None = None,
        evaluate_best: bool | None = None,
        outer_folds_provider: Callable[[], Mapping[str, Any]] | None = None,
        objective_factory: Callable[..., Any] | None = None,
    ) -> None:
        spec = load_automl_study_spec(spec_path)
        overrides: dict[str, Any] = {}
        if study_name is not None:
            overrides["name"] = study_name
        if storage is not None:
            overrides["storage_path"] = _repo_path(storage)
        if n_trials is not None:
            overrides["n_trials"] = int(n_trials)
        if timeout_seconds is not None:
            overrides["timeout_seconds"] = float(timeout_seconds)
        if seed is not None:
            overrides["sampler_seed"] = int(seed)
        if inner_splits is not None:
            overrides["inner_splits"] = int(inner_splits)
        if max_epochs is not None:
            overrides["max_epochs"] = int(max_epochs)
        if max_windows is not None:
            overrides["max_windows"] = int(max_windows)
        if evaluate_best is not None:
            overrides["evaluate_best"] = bool(evaluate_best)
        self.spec = replace(spec, **overrides)
        self.outer_fold = int(outer_fold)
        if self.outer_fold < 1:
            raise ValueError("outer_fold must be positive")
        self.base_config = self._load_base_config()
        self._validate_initial_track()
        self.study_dir = self.spec.output_root / self.spec.name
        self.outer_folds_provider = outer_folds_provider
        self.objective_factory = objective_factory
        self._outer_folds_cache: Mapping[str, Any] | None = None

    def _load_base_config(self) -> dict[str, Any]:
        with open(self.spec.base_config_path, encoding="utf-8") as input_file:
            config = yaml.safe_load(input_file) or {}
        if not isinstance(config, dict):
            raise ValueError("base_config must contain a mapping")
        return config

    def _validate_initial_track(self) -> None:
        if set(self.spec.search_space.paths) != TRANSFORMER_SEARCH_PATHS:
            raise ValueError(
                "Initial Transformer search space must contain exactly: "
                f"{sorted(TRANSFORMER_SEARCH_PATHS)}"
            )
        if len(self.base_config.get("models", {})) != 1:
            raise ValueError("Initial AutoML track requires exactly one model")
        model = next(iter(self.base_config["models"].values()))
        params = model.get("params", {})
        if model.get("type") != "torch_transformer":
            raise ValueError("Initial AutoML track requires torch_transformer")
        fixed = {
            "sequence_length": 8,
            "pooling": "last",
            "positional_encoding": "learned",
        }
        mismatched = {
            key: params.get(key)
            for key, expected in fixed.items()
            if params.get(key) != expected
        }
        if mismatched:
            raise ValueError(f"Fixed Transformer choices changed: {mismatched}")
        dataset = next(iter(self.base_config.get("datasets", {}).values()))
        if dataset.get("feature_set") != "pow_plus_eeg":
            raise ValueError("Initial AutoML dataset must use EEG + POW features")
        if dataset.get("target_col") != "label_q5":
            raise ValueError("Initial AutoML target must be label_q5")

    def _build_outer_folds(self) -> Mapping[str, Any]:
        if self.outer_folds_provider is not None:
            return self.outer_folds_provider()
        dataset_name = next(iter(self.base_config["datasets"]))
        dataset_config = deepcopy(self.base_config["datasets"][dataset_name])
        dataset_config.pop("max_windows", None)
        dataset_config.pop("include_subject_ids", None)
        dataset_config["data_path"] = _repo_path(dataset_config["data_path"])
        dataset = get_dataset(dataset_name, dataset_config)
        data = dataset.load()
        task_name = str(self.base_config["tasks"][0])
        task = get_task(task_name, data, self.base_config.get("task_config", {}))
        evaluation = self.base_config["evaluation"]
        splits = CrossValidator(task).run_group_kfold(
            group_column=str(evaluation["group_column"]),
            n_splits=int(evaluation.get("n_splits", 5)),
            random_state=int(evaluation.get("random_state", 42)),
            precomputed_fold_column=evaluation.get("precomputed_fold_column"),
        )
        return {
            "protocol": "group_kfold_subject",
            "group_column": str(evaluation["group_column"]),
            "n_splits": len(splits),
            "dataset": dataset_name,
            "n_samples": int(data.n_samples),
            "n_subjects": int(data.n_subjects),
            "folds": {
                fold_name: {
                    "fold": int(split.metadata["fold"]),
                    "train_subject_ids": split.metadata["train_subject_ids"],
                    "test_subject_ids": split.metadata["test_subject_ids"],
                    "n_train_rows": int(len(split.y_train)),
                    "n_test_rows": int(len(split.y_test)),
                    "subject_overlap": split.metadata["subject_overlap"],
                }
                for fold_name, split in splits.items()
            },
        }

    @property
    def outer_folds(self) -> Mapping[str, Any]:
        if self._outer_folds_cache is None:
            self._outer_folds_cache = self._build_outer_folds()
        return self._outer_folds_cache

    @property
    def selected_fold(self) -> Mapping[str, Any]:
        name = f"fold_{self.outer_fold:02d}"
        try:
            return self.outer_folds["folds"][name]
        except KeyError as exc:
            raise ValueError(
                f"Outer fold {self.outer_fold} is unavailable; available="
                f"{sorted(self.outer_folds['folds'])}"
            ) from exc

    def _readonly_trial_counts(self) -> dict[str, int]:
        if not self.spec.storage_path.is_file():
            return {}
        try:
            with sqlite3.connect(
                f"file:{self.spec.storage_path.as_posix()}?mode=ro", uri=True
            ) as connection:
                rows = connection.execute(
                    """
                    SELECT trials.state, COUNT(*)
                    FROM trials
                    JOIN studies ON studies.study_id = trials.study_id
                    WHERE studies.study_name = ?
                    GROUP BY trials.state
                    """,
                    (self.spec.name,),
                ).fetchall()
            return {str(state): int(count) for state, count in rows}
        except sqlite3.Error:
            return {"unreadable": -1}

    def _estimated_runtime_seconds(self) -> float | None:
        completed = BenchmarkRunner.find_completed_run(
            self.base_config,
            search_directories=[self.base_config.get("output_dir", ".")],
        )
        if completed is None:
            return None
        metrics = json.loads(
            (completed.run_directory / "metrics.json").read_text(encoding="utf-8")
        )
        group = extract_group_result(metrics, self.base_config)
        total = float(group.get("aggregated", {}).get("training_time_total", 0.0))
        n_folds = max(1, int(group.get("n_folds", 1)))
        base_epochs = int(
            next(iter(self.base_config["models"].values()))
            .get("params", {})
            .get("max_epochs", 1)
        )
        epoch_scale = (
            1.0
            if self.spec.max_epochs is None
            else self.spec.max_epochs / max(1, base_epochs)
        )
        window_scale = 1.0
        if self.spec.max_windows is not None:
            window_scale = min(
                1.0,
                self.spec.max_windows / max(1, int(self.outer_folds["n_samples"])),
            )
        return (
            total
            / n_folds
            * self.spec.inner_splits
            * self.spec.n_trials
            * epoch_scale
            * window_scale
        )

    def plan(self) -> dict[str, Any]:
        selected = self.selected_fold
        existing = self._readonly_trial_counts()
        reusable = 0
        runs_root = self.study_dir / "benchmark_runs"
        if runs_root.exists():
            reusable = sum(1 for _ in runs_root.glob("**/run_manifest.json"))
        return {
            "study_name": self.spec.name,
            "base_config": str(self.spec.base_config_path),
            "base_config_hash": BenchmarkRunner.config_hash_for(self.base_config),
            "outer_fold": self.outer_fold,
            "outer_train_subjects": selected["train_subject_ids"],
            "outer_test_subjects": selected["test_subject_ids"],
            "outer_subject_overlap": selected.get("subject_overlap", []),
            "inner_grouping": "subject_id",
            "inner_splits": self.spec.inner_splits,
            "search_parameters": list(self.spec.search_space.paths),
            "constraints": list(self.spec.search_space.constraints),
            "estimated_trials": self.spec.n_trials,
            "estimated_runtime_seconds": self._estimated_runtime_seconds(),
            "storage_path": str(self.spec.storage_path),
            "existing_trials": existing,
            "reusable_configs": reusable,
            "max_epochs": self.spec.max_epochs,
            "max_windows": self.spec.max_windows,
            "evaluate_best": self.spec.evaluate_best,
        }

    @staticmethod
    def render_plan(plan: Mapping[str, Any]) -> str:
        runtime = plan.get("estimated_runtime_seconds")
        runtime_text = "unknown" if runtime is None else f"{runtime:.1f} s"
        return "\n".join([
            "# AutoML plan",
            "",
            f"- Study: `{plan['study_name']}`",
            f"- Base config: `{plan['base_config']}`",
            f"- Base config hash: `{plan['base_config_hash']}`",
            f"- Outer fold: {plan['outer_fold']}",
            "- Outer train subjects: " + ", ".join(plan["outer_train_subjects"]),
            "- Outer test subjects: " + ", ".join(plan["outer_test_subjects"]),
            f"- Inner grouping: {plan['inner_grouping']} ({plan['inner_splits']} folds)",
            "- Search parameters: " + ", ".join(plan["search_parameters"]),
            "- Constraints: " + ", ".join(plan["constraints"]),
            f"- Estimated trials: {plan['estimated_trials']}",
            f"- Estimated runtime: {runtime_text}",
            f"- Storage: `{plan['storage_path']}`",
            f"- Existing trials: {json.dumps(plan['existing_trials'], sort_keys=True)}",
            f"- Reusable configs: {plan['reusable_configs']}",
            f"- Max epochs: {plan['max_epochs']}",
            f"- Max windows: {plan['max_windows']}",
            f"- Evaluate selected trial on outer test: {plan['evaluate_best']}",
        ])

    @staticmethod
    def _suggest_parameter(
        trial: optuna.Trial,
        parameter: SearchParameterSpec,
    ) -> Any:
        if parameter.type == "categorical":
            return trial.suggest_categorical(parameter.path, list(parameter.choices))
        if parameter.type == "integer":
            return trial.suggest_int(
                parameter.path, int(parameter.low), int(parameter.high)
            )
        return trial.suggest_float(
            parameter.path,
            float(parameter.low),
            float(parameter.high),
            log=parameter.type == "log_float",
        )

    def _make_objective(self) -> Any:
        selected = self.selected_fold
        kwargs = {
            "study_name": self.spec.name,
            "base_config": self.base_config,
            "search_space": self.spec.search_space,
            "outer_fold": self.outer_fold,
            "outer_train_subjects": selected["train_subject_ids"],
            "outer_test_subjects": selected["test_subject_ids"],
            "inner_splits": self.spec.inner_splits,
            "random_state": self.spec.sampler_seed,
            "benchmark_runs_root": self.study_dir / "benchmark_runs",
            "max_epochs": self.spec.max_epochs,
            "max_windows": self.spec.max_windows,
        }
        if self.objective_factory is not None:
            return self.objective_factory(**kwargs)
        return NestedBenchmarkObjective(**kwargs)

    @staticmethod
    def _set_result_attributes(
        trial: optuna.Trial,
        result: AutoMLTrialResult,
    ) -> None:
        trial.set_user_attr("resolved_config_hash", result.resolved_config_hash)
        trial.set_user_attr("failure_reason", result.failure_reason)
        trial.set_user_attr("automl_result", result.to_dict())

    @staticmethod
    def _study_results(study: optuna.Study) -> list[AutoMLTrialResult]:
        results = []
        for trial in study.get_trials(deepcopy=False):
            value = trial.user_attrs.get("automl_result")
            if isinstance(value, Mapping):
                results.append(AutoMLTrialResult.from_dict(value))
        return sorted(results, key=lambda result: result.trial_number)

    def _storage_url(self) -> str:
        return "sqlite:///" + self.spec.storage_path.resolve().as_posix()

    def _recover_interrupted(self, study: optuna.Study) -> int:
        interrupted = study.get_trials(
            deepcopy=False, states=(TrialState.RUNNING,)
        )
        for frozen in interrupted:
            if frozen.params:
                study.enqueue_trial(
                    frozen.params,
                    user_attrs={"retry_of_interrupted_trial": frozen.number},
                )
            study.tell(frozen.number, state=TrialState.FAIL)
        return len(interrupted)

    def _run_outer_evaluation(
        self,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        resolved, config_hash = resolve_outer_evaluation_config(
            base_config=self.base_config,
            trial_parameters=parameters,
            search_space=self.spec.search_space,
            outer_fold=self.outer_fold,
            random_state=self.spec.sampler_seed,
            benchmark_runs_root=self.study_dir / "outer_evaluation",
            max_epochs=self.spec.max_epochs,
            max_windows=None,
        )
        completed = BenchmarkRunner.find_completed_run(
            resolved, search_directories=[resolved["output_dir"]]
        )
        reused = completed is not None
        if completed is None:
            runner = BenchmarkRunner(resolved)
            runner.run()
            completed = runner.completed_run()
            results = runner.results
        else:
            results = json.loads(
                (completed.run_directory / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
        group = extract_group_result(results, resolved)
        fold_name = f"fold_{self.outer_fold:02d}"
        fold = group["folds"][fold_name]
        return {
            "outer_fold": self.outer_fold,
            "resolved_config_hash": config_hash,
            "metrics": fold["metrics"],
            "training_time": fold["training_time"],
            "training": fold.get("training", {}),
            "benchmark_run_reference": completed.to_dict(),
            "reused": reused,
        }

    def execute(self, *, resume: bool = False) -> dict[str, Any]:
        self.spec.storage_path.parent.mkdir(parents=True, exist_ok=True)
        initialize_study_artifacts(self.study_dir, self.spec, self.outer_folds)
        sampler = optuna.samplers.TPESampler(seed=self.spec.sampler_seed)
        study = optuna.create_study(
            study_name=self.spec.name,
            storage=self._storage_url(),
            sampler=sampler,
            direction=self.spec.direction,
            load_if_exists=resume,
        )
        recovered = self._recover_interrupted(study) if resume else 0
        objective = self._make_objective()
        previous_results = self._study_results(study)
        completed_by_hash = {
            result.resolved_config_hash: result
            for result in previous_results
            if result.state == "COMPLETE" and result.resolved_config_hash
        }
        terminal_before = len(study.get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED),
        )) - recovered
        remaining = max(0, self.spec.n_trials - terminal_before) + recovered
        deadline = (
            None
            if self.spec.timeout_seconds is None
            else time.monotonic() + self.spec.timeout_seconds
        )
        for _ in range(remaining):
            if deadline is not None and time.monotonic() >= deadline:
                break
            trial = study.ask()
            parameters = {
                parameter.path: self._suggest_parameter(trial, parameter)
                for parameter in self.spec.search_space.parameters
            }
            try:
                self.spec.search_space.validate_parameters(parameters)
                _, config_hash = objective.resolve(parameters)
            except Exception as exc:
                rejected = AutoMLTrialResult(
                    study_name=self.spec.name,
                    trial_number=trial.number,
                    trial_parameters=parameters,
                    resolved_config={},
                    resolved_config_hash="",
                    outer_fold=self.outer_fold,
                    inner_split=f"aggregate_{self.spec.inner_splits}fold",
                    objective_value=None,
                    state="REJECTED",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
                self._set_result_attributes(trial, rejected)
                study.tell(trial, state=TrialState.PRUNED)
                continue
            previous = completed_by_hash.get(config_hash)
            if previous is not None:
                result = replace(
                    previous,
                    trial_number=trial.number,
                    trial_parameters=parameters,
                    runtime_seconds=0.0,
                    reused=True,
                )
                self._set_result_attributes(trial, result)
                trial.set_user_attr("reused_trial_number", previous.trial_number)
                study.tell(trial, float(result.objective_value))
                continue
            result = objective.execute(trial.number, parameters)
            self._set_result_attributes(trial, result)
            if result.state == "COMPLETE":
                study.tell(trial, float(result.objective_value))
                completed_by_hash[result.resolved_config_hash] = result
            else:
                study.tell(trial, state=TrialState.FAIL)

            current_results = self._study_results(study)
            update_study_artifacts(
                self.study_dir,
                results=current_results,
                summary={
                    "study_name": self.spec.name,
                    "status": "running",
                    "completed_trials": sum(
                        result.state == "COMPLETE" for result in current_results
                    ),
                    "failed_trials": sum(
                        result.state == "FAIL" for result in current_results
                    ),
                },
                best_trials={},
            )

        results = self._study_results(study)
        complete_trials = study.get_trials(
            deepcopy=False, states=(TrialState.COMPLETE,)
        )
        best_payload: dict[str, Any] = {}
        outer_evaluation = None
        parameter_importance: dict[str, float] = {}
        if complete_trials:
            best = study.best_trial
            best_result = next(
                result for result in results if result.trial_number == best.number
            )
            if len(complete_trials) >= 2:
                try:
                    parameter_importance = {
                        key: float(value)
                        for key, value in optuna.importance.get_param_importances(
                            study,
                            evaluator=optuna.importance.FanovaImportanceEvaluator(
                                seed=self.spec.sampler_seed
                            ),
                        ).items()
                    }
                except Exception as exc:
                    logger.warning("Could not calculate parameter importance: %s", exc)
            if self.spec.evaluate_best:
                outer_evaluation = self._run_outer_evaluation(best.params)
            best_payload = {
                f"outer_fold_{self.outer_fold:02d}": {
                    "trial_number": best.number,
                    "parameters": dict(best.params),
                    "inner_objective": float(best.value),
                    "inner_secondary_metrics": dict(
                        best_result.secondary_metrics
                    ),
                    "resolved_config_hash": best_result.resolved_config_hash,
                    "benchmark_run_reference": (
                        best_result.benchmark_run_reference
                    ),
                    "outer_evaluation": outer_evaluation,
                }
            }
        summary = {
            "study_name": self.spec.name,
            "study_spec_hash": self.spec.config_hash(),
            "storage": str(self.spec.storage_path),
            "outer_fold": self.outer_fold,
            "inner_splits": self.spec.inner_splits,
            "objective_metric": self.spec.objective_metric,
            "sampler_seed": self.spec.sampler_seed,
            "requested_trials": self.spec.n_trials,
            "recorded_trials": len(study.trials),
            "completed_trials": len(complete_trials),
            "failed_trials": sum(
                trial.state == TrialState.FAIL for trial in study.trials
            ),
            "pruned_or_rejected_trials": sum(
                trial.state == TrialState.PRUNED for trial in study.trials
            ),
            "recovered_interrupted_trials": recovered,
            "best_trial_number": (
                None if not complete_trials else study.best_trial.number
            ),
            "best_value": None if not complete_trials else float(study.best_value),
            "best_parameters": (
                {} if not complete_trials else dict(study.best_params)
            ),
            "parameter_importance": parameter_importance,
            "outer_evaluation": outer_evaluation,
        }
        update_study_artifacts(
            self.study_dir,
            results=results,
            summary=summary,
            best_trials=best_payload,
        )
        return summary
