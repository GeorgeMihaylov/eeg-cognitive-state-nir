"""Synthetic tests for the personalization execution bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import cli
from bench.core.artifact_paths import PORTABLE_PATH_LIMIT, absolute_path_length

from bench.experiments.personalization_calibration import (
    PersonalizationCalibrationPlanner,
    PlanFilters,
)
from bench.experiments.personalization_calibration_execution import (
    BaseRunHandle,
    BenchmarkPersonalizationBackend,
    PersonalizationCalibrationExecutor,
    _adapter_state_hash,
    adapter_normalization_hash,
    aggregate_execution_results,
    base_run_directory,
    base_unit_id,
    base_unit_key,
    build_eligibility_table,
    execution_scope_directory,
    participant_execution_identity,
    participant_run_directory,
    temporal_adaptation_split,
    validate_base_checkpoint_manifest,
    validate_participant_resume_result,
)
from bench.experiments.personalization_calibration import stable_hash
from model_zoo.factory import build_model


CONFIG = Path("experiments/calibration/personalization_calibration_v1.json")


def test_personalization_direct_artifact_paths_are_portable(
    tmp_path: Path,
) -> None:
    missing = 145 - absolute_path_length(tmp_path) - 1
    output = tmp_path if missing <= 0 else tmp_path / ("p" * missing)
    base = base_run_directory(output, "b" * 20)
    participant = participant_run_directory(output, "p" * 24)

    assert base.parent.name == "_b"
    assert participant.parent.name == "_p"
    assert absolute_path_length(
        base / "base_checkpoint_manifest.json"
    ) <= PORTABLE_PATH_LIMIT
    assert absolute_path_length(
        participant / "predictions.parquet"
    ) <= PORTABLE_PATH_LIMIT
    assert base == base_run_directory(output, "b" * 20)
    assert participant == participant_run_directory(output, "p" * 24)
    assert base_run_directory(Path("o"), "unit") == Path("o/base_runs/unit")
    assert participant_run_directory(Path("o"), "run") == Path(
        "o/participant_runs/run"
    )


def _metadata(n: int, *, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{index}" for index in range(n)],
        "absolute_t_start": offset + np.arange(n, dtype=float) * 10.0,
        "source": "synthetic",
        "record_id": "record",
        "t_start": np.arange(n, dtype=float) * 10.0,
    })


def test_temporal_validation_is_calibration_only_and_ordered() -> None:
    calibration = _metadata(10)
    evaluation = _metadata(5, offset=200.0).assign(
        sample_id=[f"e{index}" for index in range(5)]
    )
    train, validation, audit = temporal_adaptation_split(
        calibration,
        validation_fraction=0.2,
        minimum_train_windows=4,
        minimum_validation_windows=1,
        evaluation_metadata=evaluation,
    )
    assert len(train) == 8 and len(validation) == 2
    assert set(train).isdisjoint(validation)
    assert audit["train_max_absolute_t_start"] < audit[
        "validation_min_absolute_t_start"
    ] < audit["evaluation_min_absolute_t_start"]
    assert not audit["random_split"]


def test_temporal_validation_marks_tiny_prefix_insufficient() -> None:
    with pytest.raises(ValueError, match="insufficient_data"):
        temporal_adaptation_split(
            _metadata(4), validation_fraction=0.2,
            minimum_train_windows=4, minimum_validation_windows=1,
        )


def test_torch_clone_adaptations_are_independent_and_keep_base_unchanged() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 6)).astype(np.float32)
    y = np.asarray([0, 1, 2] * 10, dtype=np.int64)
    base = build_model(
        model_name="torch_mlp", task_type="classification",
        input_shape=(6,), num_outputs=3,
        params={
            "hidden_dims": [8], "dropout": 0.0, "batch_size": 8,
            "max_epochs": 1, "early_stopping_patience": 1,
            "validation_size": 0.2, "device": "cpu", "random_state": 42,
        },
    )
    base.fit(X, y)
    base_hash = _adapter_state_hash(base)
    normalization_hash = adapter_normalization_hash(base)
    participant_a = base.clone()
    participant_b = base.clone()
    participant_a.fine_tune(
        X[:12], y[:12], mode="head_only",
        X_validation=X[12:18], y_validation=y[12:18], max_epochs=1,
        early_stopping_patience=1, random_state=7,
    )
    assert _adapter_state_hash(base) == base_hash
    assert _adapter_state_hash(participant_b) == base_hash
    assert _adapter_state_hash(participant_a) != base_hash
    assert adapter_normalization_hash(base) == normalization_hash
    assert adapter_normalization_hash(participant_a) == normalization_hash


@pytest.mark.parametrize("mode", ["head_only", "full_model"])
def test_shallowconvnet_classification_adaptation_paths(mode: str) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 1, 4, 128)).astype(np.float32)
    y = np.tile(np.arange(3), 10).astype(np.int64)
    base = build_model(
        model_name="torch_shallow_convnet", task_type="classification",
        input_shape=(1, 4, 128), num_outputs=3,
        params={
            "n_filters": 4, "temporal_kernel_samples": 9,
            "pool_size": 16, "pool_stride": 4, "dropout": 0.0,
            "batch_size": 8, "max_epochs": 1,
            "early_stopping_patience": 1, "validation_size": 0.2,
            "device": "cpu", "random_state": 42,
        },
    )
    base.fit(X, y)
    base_hash = _adapter_state_hash(base)
    adapted = base.clone()
    adapted.fine_tune(
        X[:18], y[:18], mode=mode,
        X_validation=X[18:24], y_validation=y[18:24],
        max_epochs=1, early_stopping_patience=1, random_state=7,
    )

    assert _adapter_state_hash(base) == base_hash
    assert _adapter_state_hash(adapted) != base_hash
    assert adapted.predict(X[24:]).shape == (6,)
    probabilities = adapted.predict_proba(X[24:])
    assert probabilities.shape == (6, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def _matrix_and_participants() -> tuple[pd.DataFrame, pd.DataFrame]:
    common = {
        "pm": "focus", "task_type": "classification",
        "target_id": "pm_focus_q3_fold_local",
        "model": "torch_mlp", "input_family": "features",
        "outer_fold": 1, "seed": 42, "q3_transform_hash": "q3",
        "participants_total": 2, "participants_sufficient": 2,
        "participants_insufficient": 0, "participant_execution_count": 2,
        "status": "planned", "reason": "",
    }
    matrix = pd.DataFrame([
        {**common, "mode": "zero_shot", "budget_fraction": 0.0,
         "condition_id": "zero"},
        {**common, "mode": "head_only", "budget_fraction": 0.01,
         "condition_id": "head1"},
        {**common, "mode": "head_only", "budget_fraction": 0.05,
         "condition_id": "head5"},
    ])
    participants = []
    for subject in ("A", "B"):
        for budget in (0.0, 0.01, 0.05):
            participants.append({
                "pm": "focus", "outer_fold": 1, "subject_id": subject,
                "budget_fraction": budget, "status": "planned",
                "calibration_sample_hash": f"cal-{subject}-{budget}",
                "evaluation_sample_hash": f"eval-{subject}",
                "q3_transform_hash": "q3",
            })
    return matrix, pd.DataFrame(participants)


class _FakeBackend:
    def __init__(self) -> None:
        self.base_calls = 0
        self.base_models: list[str] = []
        self.participant_models: list[str] = []
        self.condition_ids: list[str] = []
        self.execution_calls: list[tuple[str, str, float, bool]] = []

    def ensure_base(self, base, *, resume):
        self.base_calls += 1
        self.base_models.append(str(base["model"]))
        return {"base": base_unit_id(base), "state": 0}

    def execute_participant(self, handle, condition, participant, *, resume):
        self.participant_models.append(str(condition["model"]))
        self.condition_ids.append(str(condition["condition_id"]))
        key = (
            str(participant["subject_id"]), str(condition["mode"]),
            float(condition["budget_fraction"]), bool(resume),
        )
        self.execution_calls.append(key)
        gain = 0.0 if condition["mode"] == "zero_shot" else float(
            condition["budget_fraction"]
        )
        return {
            "execution_id": "-".join(map(str, key[:3])),
            "status": "completed", "reason": "",
            "evaluation_sample_hash": participant["evaluation_sample_hash"],
            "base_checkpoint_sha256": "base", "adapted_checkpoint_sha256": "adapted",
            **{
                f"zero_shot_{metric}": 0.4
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
            },
            **{
                f"adapted_{metric}": 0.4 + gain
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
            },
            **{
                f"delta_{metric}": gain
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
            },
        }


def _multi_model_matrix_and_participants() -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, participants = _matrix_and_participants()
    rows = []
    for model, family in (
        ("torch_shallow_convnet", "raw"),
        ("torch_eegnet", "raw"),
        ("torch_mlp", "features"),
    ):
        for row in matrix.to_dict("records"):
            rows.append({
                **row,
                "model": model,
                "input_family": family,
                "condition_id": f"{model}-{row['condition_id']}",
            })
    return pd.DataFrame(rows), participants


def test_executor_reuses_one_base_for_participants_and_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = PersonalizationCalibrationPlanner(
        CONFIG, data_root=tmp_path, output_dir=tmp_path / "out"
    )
    matrix, participants = _matrix_and_participants()
    monkeypatch.setattr(planner, "materialize_tables", lambda **_: {
        "run_matrix": matrix, "participants": participants,
        "compatibility": pd.DataFrame(), "transforms": {}, "cohorts": {},
        "filters": PlanFilters(),
    })
    backend = _FakeBackend()
    manifest = PersonalizationCalibrationExecutor(planner).run(
        backend=backend, resume=True
    )
    assert backend.base_calls == 1
    assert len(backend.execution_calls) == 6
    assert manifest["completed_participant_executions"] == 6
    assert manifest["formal_criteria"]["classification_accuracy_threshold"] == 0.75
    assert manifest["formal_criteria"]["threshold_role"] == (
        "report_only_not_for_selection"
    )
    results = pd.read_csv(tmp_path / "out" / "participant_results.csv")
    assert results.groupby(["outer_fold", "subject_id"])[
        "evaluation_sample_hash"
    ].nunique().max() == 1
    assert (tmp_path / "out" / "aggregate_results.csv").is_file()
    assert (tmp_path / "out" / "budget_curve.csv").is_file()


def test_execution_model_filters_after_full_plan_without_changing_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = PersonalizationCalibrationPlanner(
        CONFIG, data_root=tmp_path, output_dir=tmp_path / "out"
    )
    matrix, participants = _multi_model_matrix_and_participants()
    monkeypatch.setattr(planner, "materialize_tables", lambda **_: {
        "run_matrix": matrix, "participants": participants,
        "compatibility": pd.DataFrame(), "transforms": {}, "cohorts": {},
        "filters": PlanFilters(),
    })
    protocol_hash = planner.protocol_hash
    full_condition_ids = set(
        matrix.loc[
            matrix["model"].eq("torch_shallow_convnet"), "condition_id"
        ].astype(str)
    )
    backend = _FakeBackend()

    manifest = PersonalizationCalibrationExecutor(planner).run(
        backend=backend,
        resume=True,
        execution_model="torch_shallow_convnet",
    )

    assert planner.protocol_hash == protocol_hash == (
        "a3723e8f77ec1a9eeef21a2b5a88660d9cd42a717084e6e1aadb12429085d0d4"
    )
    assert manifest["plan_hash"] == (
        "d8c7430e75e692fcf8cf53b7052d48faa2c8f392bb2cd7a049657204d3396412"
    )
    assert set(backend.condition_ids) == full_condition_ids
    assert backend.base_models == ["torch_shallow_convnet"]
    assert set(backend.participant_models) == {"torch_shallow_convnet"}
    assert manifest["full_plan_conditions"] == len(matrix)
    assert manifest["selected_execution_conditions"] == len(full_condition_ids)
    scope = execution_scope_directory(
        planner.output_dir, "torch_shallow_convnet"
    )
    assert (scope / "participant_results.csv").is_file()
    assert not (planner.output_dir / "participant_results.csv").exists()


def test_unfiltered_execution_keeps_previous_all_model_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = PersonalizationCalibrationPlanner(
        CONFIG, data_root=tmp_path, output_dir=tmp_path / "out"
    )
    matrix, participants = _multi_model_matrix_and_participants()
    monkeypatch.setattr(planner, "materialize_tables", lambda **_: {
        "run_matrix": matrix, "participants": participants,
        "compatibility": pd.DataFrame(), "transforms": {}, "cohorts": {},
        "filters": PlanFilters(),
    })
    backend = _FakeBackend()

    manifest = PersonalizationCalibrationExecutor(planner).run(
        backend=backend, resume=True
    )

    assert set(backend.base_models) == {
        "torch_shallow_convnet", "torch_eegnet", "torch_mlp"
    }
    assert set(backend.participant_models) == set(backend.base_models)
    assert manifest["plan_hash"] == (
        "d8c7430e75e692fcf8cf53b7052d48faa2c8f392bb2cd7a049657204d3396412"
    )
    assert "execution_filter" not in manifest
    assert (planner.output_dir / "participant_results.csv").is_file()


def test_participant_macro_and_paired_common_cohort() -> None:
    matrix, participant_plan = _matrix_and_participants()
    eligibility = build_eligibility_table(matrix, participant_plan)
    rows = []
    for subject in ("A", "B"):
        for budget in (0.0, 0.01, 0.05):
            mode = "zero_shot" if budget == 0 else "head_only"
            gain = budget + (0.01 if subject == "B" else 0.0)
            row = {
                "pm": "focus", "task_type": "classification",
                "model": "torch_mlp", "mode": mode,
                "budget_fraction": budget, "outer_fold": 1,
                "subject_id": subject, "status": "completed",
            }
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
                row[f"zero_shot_{metric}"] = 0.4
                row[f"adapted_{metric}"] = 0.4 + gain
                row[f"delta_{metric}"] = gain
            rows.append(row)
    aggregate, curve = aggregate_execution_results(pd.DataFrame(rows), eligibility)
    selected = aggregate.loc[
        aggregate["mode"].eq("head_only")
        & aggregate["budget_fraction"].eq(0.05)
        & aggregate["metric"].eq("accuracy")
        & aggregate["value_kind"].eq("delta")
    ].iloc[0]
    assert selected["mean"] == pytest.approx(0.055)
    paired = curve.loc[
        curve["budget_fraction"].eq(0.05)
        & curve["metric"].eq("accuracy")
    ].iloc[0]
    assert paired["n_paired_common_cohort"] == 2
    assert paired["mean_delta_paired_common_cohort"] == pytest.approx(0.055)


def test_incompatible_base_checkpoint_manifest_is_rejected() -> None:
    matrix, _ = _matrix_and_participants()
    unit = matrix.iloc[0].to_dict()
    complete = {
        "protocol_hash": "expected", "base_unit": base_unit_key(unit),
        "plan_hash": "plan", "preprocessing_hashes": [],
        "benchmark_config_hash": "a", "sample_universe_hash": "b",
        "input_shape": [448], "normalization_hash": "c",
        "task_type": "classification", "num_outputs": 3, "seed": 42,
        "model_config_hash": "d", "target_transform_hash": "q3",
        "checkpoint_sha256": "e",
    }
    complete["checkpoint_identity_hash"] = stable_hash(complete)
    validate_base_checkpoint_manifest(
        complete, protocol_hash="expected", unit=unit
    )
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        validate_base_checkpoint_manifest(
            complete, protocol_hash="different", unit=unit
        )


def test_participant_resume_requires_exact_identity_and_checkpoint_hash(
    tmp_path: Path,
) -> None:
    identity = {"condition_id": "c", "subject_id": "s"}
    result = {
        "execution_identity": identity,
        "status": "completed",
        "adapted_checkpoint": None,
    }
    validate_participant_resume_result(result, identity)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_participant_resume_result(result, {**identity, "subject_id": "x"})


def test_execution_scope_reuses_existing_participant_identity_without_rewrite(
    tmp_path: Path,
) -> None:
    planner = PersonalizationCalibrationPlanner(
        CONFIG, data_root=tmp_path, output_dir=tmp_path / "out"
    )
    plan_hash = (
        "d8c7430e75e692fcf8cf53b7052d48faa2c8f392bb2cd7a049657204d3396412"
    )
    condition = _multi_model_matrix_and_participants()[0].loc[
        lambda frame: frame["model"].eq("torch_shallow_convnet")
    ].iloc[0].to_dict()
    participant = _matrix_and_participants()[1].iloc[0].to_dict()
    handle = BaseRunHandle(
        unit={}, adapter=None, split=None,
        checkpoint_path=tmp_path / "model.pt", checkpoint_sha256="checkpoint",
        checkpoint_identity_hash="base-identity", normalization_hash="normalization",
        target_transform_hash="q3", resumed=True,
    )
    identity = participant_execution_identity(
        protocol_hash=planner.protocol_hash,
        plan_hash=plan_hash,
        base_checkpoint_identity_hash=handle.checkpoint_identity_hash,
        condition=condition,
        participant=participant,
    )
    execution_id = stable_hash(identity)[:24]
    result_path = (
        participant_run_directory(planner.output_dir, execution_id) / "result.json"
    )
    result_path.parent.mkdir(parents=True)
    payload = {
        **identity,
        "execution_identity": identity,
        "execution_id": execution_id,
        "status": "completed",
        "adapted_checkpoint": None,
    }
    original = json.dumps(payload, indent=2) + "\n"
    result_path.write_text(original, encoding="utf-8")
    backend = BenchmarkPersonalizationBackend(planner, plan_hash=plan_hash)

    resumed = backend.execute_participant(
        handle, condition, participant, resume=True
    )

    assert resumed["participant_resumed"] is True
    assert resumed["execution_identity"] == identity
    assert result_path.read_text(encoding="utf-8") == original


def test_interrupted_participant_directory_without_result_is_not_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = PersonalizationCalibrationPlanner(
        CONFIG, data_root=tmp_path, output_dir=tmp_path / "out"
    )
    plan_hash = (
        "d8c7430e75e692fcf8cf53b7052d48faa2c8f392bb2cd7a049657204d3396412"
    )
    condition = _multi_model_matrix_and_participants()[0].loc[
        lambda frame: frame["model"].eq("torch_shallow_convnet")
    ].iloc[0].to_dict()
    participant = _matrix_and_participants()[1].iloc[0].to_dict()
    handle = BaseRunHandle(
        unit={}, adapter=None, split=None,
        checkpoint_path=tmp_path / "model.pt", checkpoint_sha256="checkpoint",
        checkpoint_identity_hash="base-identity", normalization_hash="normalization",
        target_transform_hash="q3", resumed=True,
    )
    identity = participant_execution_identity(
        protocol_hash=planner.protocol_hash,
        plan_hash=plan_hash,
        base_checkpoint_identity_hash=handle.checkpoint_identity_hash,
        condition=condition,
        participant=participant,
    )
    run_dir = participant_run_directory(
        planner.output_dir, stable_hash(identity)[:24]
    )
    run_dir.mkdir(parents=True)
    marker = run_dir / "interrupted.marker"
    marker.write_text("partial", encoding="utf-8")
    monkeypatch.setattr(
        planner,
        "_load_target_frame",
        lambda pm: pd.DataFrame({
            "outer_fold": [1], "subject_id": [participant["subject_id"]],
        }),
    )
    monkeypatch.setattr(
        "bench.experiments.personalization_calibration_execution._participant_partition",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("continued interrupted execution")
        ),
    )
    backend = BenchmarkPersonalizationBackend(planner, plan_hash=plan_hash)

    with pytest.raises(RuntimeError, match="continued interrupted execution"):
        backend.execute_participant(handle, condition, participant, resume=True)

    assert marker.read_text(encoding="utf-8") == "partial"
    assert not (run_dir / "result.json").exists()


def test_dry_execution_does_not_train(tmp_path: Path, monkeypatch) -> None:
    planner = PersonalizationCalibrationPlanner(
        CONFIG, data_root=tmp_path, output_dir=tmp_path / "dry"
    )
    matrix, participants = _matrix_and_participants()
    monkeypatch.setattr(planner, "materialize_tables", lambda **_: {
        "run_matrix": matrix, "participants": participants,
        "compatibility": pd.DataFrame(), "transforms": {}, "cohorts": {},
        "filters": PlanFilters(),
    })
    report = PersonalizationCalibrationExecutor(planner).dry_execution()
    assert not report["training_executed"]
    assert report["base_training_units"] == 1
    assert report["zero_shot_shared_eval_inferences"] == 2
    assert report["head_only_adaptation_trainings"] == 4
    assert report["formal_criteria"] == {
        "classification_accuracy_threshold": 0.75,
        "aggregation": "participant_macro",
        "threshold_role": "report_only_not_for_selection",
    }


def test_dry_execution_model_uses_full_plan_hash_and_scoped_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = PersonalizationCalibrationPlanner(
        CONFIG, data_root=tmp_path, output_dir=tmp_path / "dry"
    )
    matrix, participants = _multi_model_matrix_and_participants()
    monkeypatch.setattr(planner, "materialize_tables", lambda **_: {
        "run_matrix": matrix, "participants": participants,
        "compatibility": pd.DataFrame(), "transforms": {}, "cohorts": {},
        "filters": PlanFilters(),
    })

    report = PersonalizationCalibrationExecutor(planner).dry_execution(
        execution_model="torch_shallow_convnet"
    )

    assert report["protocol_hash"] == planner.protocol_hash
    assert report["plan_hash"] == (
        "d8c7430e75e692fcf8cf53b7052d48faa2c8f392bb2cd7a049657204d3396412"
    )
    assert report["full_plan_conditions"] == len(matrix)
    assert report["selected_execution_conditions"] == len(matrix) // 3
    assert report["base_training_units"] == 1
    scope = execution_scope_directory(
        planner.output_dir, "torch_shallow_convnet"
    )
    assert (scope / "dry_execution.json").is_file()
    assert not (planner.output_dir / "dry_execution.json").exists()


def test_cli_routes_dry_and_run_modes_without_implicit_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        PersonalizationCalibrationExecutor,
        "dry_execution",
        lambda self, **kwargs: calls.append(("dry", kwargs)) or {
            "training_executed": False
        },
    )
    monkeypatch.setattr(
        PersonalizationCalibrationExecutor,
        "run",
        lambda self, **kwargs: calls.append(("run", kwargs)) or {
            "training_executed": True
        },
    )
    cli.main([
        "--personalization-calibration", str(CONFIG),
        "--dry-execution", "--output-dir", str(tmp_path / "dry"),
    ])
    cli.main([
        "--personalization-calibration", str(CONFIG), "--run",
        "--outer-fold", "1", "--pm", "focus",
        "--task-type", "classification", "--models", "torch_mlp",
        "--calibration-mode", "head_only",
        "--calibration-budget-fraction", "0.05",
        "--subject-limit", "1", "--max-calibration-epochs", "1",
        "--device", "cpu", "--output-dir", str(tmp_path / "run"),
    ])
    cli.main([
        "--personalization-calibration", str(CONFIG), "--run", "--resume",
        "--execution-model", "torch_shallow_convnet",
        "--output-dir", str(tmp_path / "scoped"),
    ])
    assert calls[0][0] == "dry"
    assert calls[1][0] == "run"
    assert calls[1][1]["filters"].model == "torch_mlp"
    assert calls[1][1]["filters"].budget_fraction == 0.05
    assert calls[1][1]["device"] == "cpu"
    assert calls[2][0] == "run"
    assert calls[2][1]["filters"].model is None
    assert calls[2][1]["execution_model"] == "torch_shallow_convnet"
    assert '"training_executed": false' in capsys.readouterr().out.lower()


def test_cli_rejects_execution_model_during_plan_only() -> None:
    with pytest.raises(SystemExit):
        cli.main([
            "--personalization-calibration", str(CONFIG), "--plan-only",
            "--execution-model", "torch_shallow_convnet",
        ])
