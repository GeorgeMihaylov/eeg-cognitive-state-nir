from __future__ import annotations

from pathlib import Path

import yaml

import cli
from bench.automl.objective import AutoMLTrialResult
from bench.automl.search_space import stable_hash
from bench.automl.study_runner import AutoMLStudyRunner


def _write_specs(tmp_path: Path, *, seed: int = 42) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base_path = tmp_path / "base.yaml"
    base = {
        "output_dir": str(tmp_path / "base-results"),
        "datasets": {
            "emotiv_cognitive": {
                "data_path": str(tmp_path / "unused.parquet"),
                "feature_set": "pow_plus_eeg",
                "target_col": "label_q5",
            }
        },
        "tasks": ["cognitive_load_5class"],
        "sequence": {"length": 8},
        "models": {
            "torch_transformer": {
                "type": "torch_transformer",
                "task_type": "classification",
                "params": {
                    "sequence_length": 8,
                    "pooling": "last",
                    "positional_encoding": "learned",
                },
            }
        },
        "evaluation": {
            "protocol": "group_kfold_subject",
            "group_column": "subject_id",
            "n_splits": 5,
        },
    }
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    one = lambda value: {"type": "categorical", "choices": [value]}
    spec = {
        "study": {
            "name": "unit-study",
            "objective_metric": "balanced_accuracy",
            "direction": "maximize",
            "sampler_seed": seed,
        },
        "base_config": {"path": str(base_path)},
        "artifacts": {
            "output_root": str(tmp_path / "automl"),
            "storage": str(tmp_path / "study.db"),
        },
        "evaluation": {
            "nested": True,
            "inner_protocol": "group_kfold_subject",
            "inner_splits": 2,
            "evaluate_best": False,
        },
        "search": {"n_trials": 2, "timeout_seconds": None},
        "search_space": {
            "model.params.d_model": one(16),
            "model.params.nhead": one(4),
            "model.params.num_layers": one(1),
            "model.params.dim_feedforward": one(32),
            "model.params.dropout": one(0.1),
            "training.learning_rate": one(0.001),
            "training.weight_decay": one(0.0001),
            "training.batch_size": one(16),
        },
        "constraints": ["d_model_divisible_by_nhead"],
    }
    spec_path = tmp_path / "study.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return spec_path


def _folds() -> dict:
    return {
        "protocol": "group_kfold_subject",
        "group_column": "subject_id",
        "n_splits": 1,
        "dataset": "emotiv_cognitive",
        "n_samples": 30,
        "n_subjects": 3,
        "folds": {
            "fold_01": {
                "fold": 1,
                "train_subject_ids": ["S1", "S2"],
                "test_subject_ids": ["S3"],
                "n_train_rows": 20,
                "n_test_rows": 10,
                "subject_overlap": [],
            }
        },
    }


class _FakeObjective:
    calls = 0

    def __init__(self, **kwargs) -> None:
        self.study_name = kwargs["study_name"]
        self.outer_fold = kwargs["outer_fold"]

    def resolve(self, parameters: dict) -> tuple[dict, str]:
        return {"parameters": parameters}, stable_hash(parameters)

    def execute(self, trial_number: int, parameters: dict) -> AutoMLTrialResult:
        type(self).calls += 1
        config, config_hash = self.resolve(parameters)
        return AutoMLTrialResult(
            study_name=self.study_name,
            trial_number=trial_number,
            trial_parameters=parameters,
            resolved_config=config,
            resolved_config_hash=config_hash,
            outer_fold=self.outer_fold,
            inner_split="fold_01,fold_02",
            objective_value=0.4,
            secondary_metrics={"macro_f1": 0.38},
            benchmark_run_reference={"status": "completed", "path": "run"},
        )


def test_plan_only_has_no_filesystem_side_effects(tmp_path) -> None:
    spec_path = _write_specs(tmp_path)
    runner = AutoMLStudyRunner(
        spec_path,
        outer_folds_provider=_folds,
        objective_factory=_FakeObjective,
    )
    plan = runner.plan()
    assert plan["outer_train_subjects"] == ["S1", "S2"]
    assert plan["outer_test_subjects"] == ["S3"]
    assert not runner.spec.storage_path.exists()
    assert not runner.study_dir.exists()


def test_sqlite_resume_and_duplicate_hash_reuse(tmp_path) -> None:
    _FakeObjective.calls = 0
    spec_path = _write_specs(tmp_path)
    runner = AutoMLStudyRunner(
        spec_path,
        outer_folds_provider=_folds,
        objective_factory=_FakeObjective,
    )
    first = runner.execute(resume=False)
    assert first["completed_trials"] == 2
    assert _FakeObjective.calls == 1
    assert runner.spec.storage_path.is_file()
    assert (runner.study_dir / "trials.parquet").is_file()
    assert (runner.study_dir / "study_summary.json").is_file()

    resumed = AutoMLStudyRunner(
        spec_path,
        outer_folds_provider=_folds,
        objective_factory=_FakeObjective,
    ).execute(resume=True)
    assert resumed["recorded_trials"] == 2
    assert _FakeObjective.calls == 1


def test_sampler_seed_is_reproducible(tmp_path) -> None:
    first_path = _write_specs(tmp_path / "first", seed=7)
    second_path = _write_specs(tmp_path / "second", seed=7)
    first = AutoMLStudyRunner(
        first_path,
        n_trials=1,
        outer_folds_provider=_folds,
        objective_factory=_FakeObjective,
    ).execute()
    second = AutoMLStudyRunner(
        second_path,
        n_trials=1,
        outer_folds_provider=_folds,
        objective_factory=_FakeObjective,
    ).execute()
    assert first["best_parameters"] == second["best_parameters"]


def test_cli_plan_only_uses_main_study_runner(monkeypatch, capsys) -> None:
    calls = {}

    class _CLIStudy:
        def __init__(self, path, **kwargs) -> None:
            calls["path"] = path
            calls["kwargs"] = kwargs

        def plan(self) -> dict:
            return {"ok": True}

        @staticmethod
        def render_plan(plan: dict) -> str:
            return f"plan={plan['ok']}"

    monkeypatch.setattr(
        "bench.automl.study_runner.AutoMLStudyRunner", _CLIStudy
    )
    cli.main([
        "--automl-study",
        "study.yaml",
        "--outer-fold",
        "1",
        "--plan-only",
    ])
    assert calls["path"] == "study.yaml"
    assert calls["kwargs"]["outer_fold"] == 1
    assert "plan=True" in capsys.readouterr().out
