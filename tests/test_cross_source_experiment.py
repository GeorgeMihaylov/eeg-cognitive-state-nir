from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from bench.experiments.cross_source_generalization import CrossSourceExperiment
from bench.datasets.emotiv_loader import EmotivDataset


def write_experiment(tmp_path: Path) -> Path:
    rows = []
    logical_rows = []

    def add_record(source, subject, logical_id, duplicate=False):
        record_id = f"{source}__{logical_id}"
        start = len(rows)
        for offset in range(25):
            rows.append({
                "record_id": record_id,
                "source": source,
                "subject_id": subject,
                "t_start": float(offset * 10),
                "t_center": float(offset * 10 + 5),
                "t_end": float(offset * 10 + 10),
                "label_q5": offset % 5,
                "EEG.F0__mean": float(start + offset),
                "EEG.F1__mean": float(offset % 5),
            })
        return record_id

    for index in range(5):
        logical = f"g_{index}"
        record = add_record("gpn_data", f"G{index}", logical)
        logical_rows.append((logical, [record], False))
    for index in range(3):
        logical = f"o_{index}"
        record = add_record("Old_EEG", f"O{index}", logical)
        logical_rows.append((logical, [record], False))
    for index in range(3):
        logical = f"duplicate_{index}"
        records = [
            add_record("gpn_data", f"X{index}", logical, True),
            add_record("Old_EEG", f"X{index}", logical, True),
        ]
        logical_rows.append((logical, records, True))
        extra = f"g_extra_{index}"
        record = add_record("gpn_data", f"X{index}", extra)
        logical_rows.append((extra, [record], False))
    record = add_record("Old_EEG", "X0", "old_extra_0")
    logical_rows.append(("old_extra_0", [record], False))

    data_path = tmp_path / "features.parquet"
    map_path = tmp_path / "logical.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)
    pd.DataFrame([
        {
            "record_group_id": logical,
            "source_record_ids": records,
            "present_in_both_sources": duplicate,
        }
        for logical, records, duplicate in logical_rows
    ]).to_parquet(map_path, index=False)
    spec = {
        "experiment": {
            "output_dir": str(tmp_path / "outputs"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "dataset": {
            "name": "emotiv_cognitive",
            "data_path": str(data_path),
            "logical_recording_map_path": str(map_path),
            "feature_set": "pow_plus_eeg",
            "target": "label_q5",
            "subject_col": "subject_id",
            "task": "cognitive_load_5class",
            "n_classes": 5,
            "max_features": 2,
        },
        "sequence": {
            "length": 2,
            "stride": 1,
            "target_position": "last",
            "expected_step_seconds": 10.0,
            "max_gap_seconds": 10.5,
        },
        "validation_by_subject_mode": {
            "source_exclusive": {
                "strategy": "group_record",
                "group_column": "subject_id",
                "validation_size": 0.2,
            },
            "shared_subject": {
                "strategy": "group_record",
                "group_column": "record_group_id",
                "validation_size": 0.2,
            },
        },
        "models": {
            "random_forest": {
                "type": "random_forest",
                "task_type": "classification",
                "params": {"n_estimators": 2, "random_state": 42},
            },
            "torch_transformer": {
                "type": "torch_transformer",
                "task_type": "classification",
                "params": {
                    "sequence_length": 2,
                    "d_model": 8,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 16,
                    "max_epochs": 2,
                    "random_state": 42,
                },
            },
        },
        "evaluation": {
            "protocol": "cross_source_holdout",
            "remove_logical_duplicates": True,
            "thresholds": {
                "minimum_train_subjects": 3,
                "minimum_test_subjects": 3,
                "minimum_train_classes": 5,
                "minimum_test_classes": 2,
                "minimum_predictions_per_test_subject": 20,
            },
        },
        "matrix": {
            "directions": ["gpn_data->Old_EEG", "Old_EEG->gpn_data"],
            "subject_modes": ["source_exclusive", "shared_subject"],
            "models": ["random_forest", "torch_transformer"],
        },
        "runtime_estimates_seconds": {
            "random_forest": 1,
            "torch_transformer": 2,
        },
    }
    spec_path = tmp_path / "experiment.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return spec_path


def no_completed(*args, **kwargs):
    return None


def test_plan_expands_bounded_matrix_and_marks_invalid_trials(tmp_path):
    experiment = CrossSourceExperiment(
        write_experiment(tmp_path), completed_run_finder=no_completed
    )
    plans = experiment.plan()

    assert len(plans) == 8
    assert sum(plan.status == "valid" for plan in plans) == 4
    assert sum(plan.status == "invalid" for plan in plans) == 4
    assert all(
        plan.action == "skip_invalid"
        for plan in plans if plan.status == "invalid"
    )
    for plan in plans:
        counts = plan.counts
        assert sum(counts["train_class_distribution"].values()) == counts[
            "train_predictions"
        ]
        assert sum(counts["test_class_distribution"].values()) == counts[
            "test_predictions"
        ]
        assert counts["train_logical_recordings"] >= 1
        assert counts["test_logical_recordings"] >= 1
        assert len(counts["removed_duplicate_logical_recordings"]) == 3
        assert len(counts["shared_subject_ids"]) == 3
        if plan.subject_mode == "shared_subject":
            assert counts["eligible_shared_subject_ids"] == ["X0"]


def test_plan_only_writes_reports_without_constructing_runner(tmp_path):
    def forbidden_runner(config):
        raise AssertionError("plan-only constructed BenchmarkRunner")

    experiment = CrossSourceExperiment(
        write_experiment(tmp_path),
        runner_factory=forbidden_runner,
        completed_run_finder=no_completed,
    )
    gitignore = Path(".gitignore").read_bytes()
    plans = experiment.plan(models=["random_forest"])
    reports = experiment.write_plan_reports(plans)

    assert Path(reports["markdown"]).is_file()
    assert Path(reports["json"]).is_file()
    assert Path(".gitignore").read_bytes() == gitignore


def test_direction_and_smoke_limits_change_config_hash(tmp_path):
    experiment = CrossSourceExperiment(
        write_experiment(tmp_path), completed_run_finder=no_completed
    )
    full = experiment.plan(
        subject_modes=["source_exclusive"], models=["random_forest"]
    )
    smoke = experiment.plan(
        directions=["gpn_data->Old_EEG"],
        subject_modes=["source_exclusive"],
        models=["random_forest"],
        max_train_windows=100,
        max_test_windows=60,
    )

    assert full[0].config_hash != full[1].config_hash
    assert smoke[0].config_hash != full[0].config_hash


def test_execute_invokes_runner_only_for_valid_trials(tmp_path, monkeypatch):
    calls = []
    completed = SimpleNamespace(to_dict=lambda: {"status": "completed"})

    class FakeRunner:
        def __init__(self, config):
            calls.append(config)

        def run(self):
            return None

        def completed_run(self):
            return completed

    experiment = CrossSourceExperiment(
        write_experiment(tmp_path),
        runner_factory=FakeRunner,
        completed_run_finder=no_completed,
    )
    monkeypatch.setattr(
        experiment,
        "write_result_reports",
        lambda plans, completed_by_trial: {},
    )
    plans = experiment.plan(
        directions=["gpn_data->Old_EEG"], models=["random_forest"]
    )
    result = experiment.execute(plans)

    assert len(calls) == 1
    assert len(result["trials"]) == 2
    assert {row["status"] for row in result["trials"]} == {
        "completed", "invalid"
    }


def test_dataset_source_filter_is_canonical_and_preserves_logical_ids(tmp_path):
    spec = yaml.safe_load(write_experiment(tmp_path).read_text(encoding="utf-8"))
    dataset = spec["dataset"]
    data = EmotivDataset({
        "data_path": dataset["data_path"],
        "logical_recording_map_path": dataset["logical_recording_map_path"],
        "include_sources": ["Old_EEG"],
        "feature_set": "pow_plus_eeg",
        "target_col": "label_q5",
        "discretize": False,
        "max_features": 2,
    }).load()

    assert set(data.row_metadata["source"]) == {"Old_EEG"}
    assert data.metadata["include_sources"] == ["Old_EEG"]
    assert not pd.isna(data.row_metadata["record_group_id"]).any()
