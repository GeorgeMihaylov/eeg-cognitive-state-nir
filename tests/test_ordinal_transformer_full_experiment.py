from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from bench.bench_runner import CompletedBenchmarkRun, benchmark_config_hash
from bench.experiments.ordinal_transformer import build_ordinal_transformer_experiment
from bench.experiments.ordinal_transformer_full import (
    OrdinalTransformerFullExperiment,
)


FULL_SPEC = Path("experiments/ordinal_transformer_full_seed42.yaml")
SMOKE_SPEC = Path("experiments/ordinal_transformer_smoke.yaml")


def _context() -> dict:
    document = yaml.safe_load(FULL_SPEC.read_text(encoding="utf-8"))
    canonical = pd.DataFrame({
        "sequence_id": [f"s-{index}" for index in range(10)],
        "fold": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "subject_id": [f"u-{index}" for index in range(10)],
        "record_id": [f"r-{index}" for index in range(10)],
        "record_group_id": [f"g-{index}" for index in range(10)],
        "source": ["synthetic"] * 10,
        "target_sample_id": list(range(10)),
        "target_time": [float(index * 10) for index in range(10)],
        "y_true": [index % 5 for index in range(10)],
    })
    fold_summaries = {
        f"fold_{fold:02d}": {
            "outer_train_sequences": 8,
            "inner_train_sequences": 6,
            "validation_sequences": 2,
            "test_sequences": 2,
            "outer_train_subjects": 8,
            "test_subjects": 2,
            "inner_validation_groups": 2,
            "outer_subject_overlap": [],
            "inner_group_overlap": [],
            "inner_validation_group_ids": [f"g-{fold}"],
        }
        for fold in range(1, 6)
    }
    return {
        "supervised_rows": document["dataset"]["expected_supervised_rows"],
        "canonical": canonical,
        "sequence_count": document["dataset"]["expected_sequences"],
        "subject_count": document["dataset"]["expected_subjects"],
        "sequence_index_sha256": document["dataset"]["sequence_index_sha256"],
        "source_parquet_sha256": document["dataset"]["parquet_sha256"],
        "sequence_build_stats": {},
        "features": {
            name: {
                "names": [f"{name}-{index}" for index in range(definition["feature_count"])],
                "count": definition["feature_count"],
                "sha256": definition["feature_list_sha256"],
            }
            for name, definition in document["feature_definitions"].items()
        },
        "fold_summaries": fold_summaries,
    }


def _spec(tmp_path: Path) -> Path:
    document = yaml.safe_load(FULL_SPEC.read_text(encoding="utf-8"))
    document["experiment"]["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "full.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _reference_audit(context: dict) -> dict:
    return {
        group: {"checks": {"canonical": True}, "alignment": {"exact_match": True}}
        for group in ("eeg_only", "eeg_pow")
    }


def test_full_matrix_has_four_trials_and_twenty_fold_runs(tmp_path: Path) -> None:
    output = tmp_path / "planned-output"
    experiment = OrdinalTransformerFullExperiment(
        _spec(tmp_path),
        output_dir=output,
        context_builder=_context,
        reference_auditor=_reference_audit,
        completed_run_finder=lambda *args, **kwargs: None,
    )
    plans = experiment.plan()
    assert [(plan.head_type, plan.feature_group) for plan in plans] == [
        ("coral", "eeg_only"),
        ("coral", "eeg_pow"),
        ("corn", "eeg_only"),
        ("corn", "eeg_pow"),
    ]
    assert len(plans) == 4
    assert sum(len(plan.folds) for plan in plans) == 20
    assert all(plan.folds == (1, 2, 3, 4, 5) for plan in plans)
    assert all(plan.seed == 42 and plan.maximum_epochs == 15 for plan in plans)
    assert all(plan.status == "valid" for plan in plans)
    assert all(
        plan.resolved_config["models"]["torch_transformer"]["params"][
            "head_type"
        ] == plan.head_type
        for plan in plans
    )
    assert len({plan.config_hash for plan in plans}) == 4
    smoke_like = deepcopy(dict(plans[0].resolved_config))
    smoke_like["experiment"]["type"] = "ordinal_transformer_smoke"
    smoke_like["evaluation"]["folds"] = [1]
    smoke_like["models"]["torch_transformer"]["params"]["max_epochs"] = 3
    assert benchmark_config_hash(smoke_like) != plans[0].config_hash
    assert not output.exists()


def test_builder_distinguishes_full_and_smoke_configs() -> None:
    full = build_ordinal_transformer_experiment(FULL_SPEC)
    smoke = build_ordinal_transformer_experiment(SMOKE_SPEC)
    assert isinstance(full, OrdinalTransformerFullExperiment)
    assert type(full) is not type(smoke)


def test_resume_requires_completed_five_fold_result(tmp_path: Path) -> None:
    experiment = OrdinalTransformerFullExperiment(
        _spec(tmp_path),
        context_builder=_context,
        reference_auditor=_reference_audit,
        completed_run_finder=lambda *args, **kwargs: None,
    )
    plan = experiment.plan()[0]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dict(plan.resolved_config), sort_keys=False), encoding="utf-8"
    )
    result_file = tmp_path / "result.json"

    def write_result(folds: list[int]) -> None:
        result_file.write_text(json.dumps({
            "emotiv_cognitive": {"models": {"cognitive_load_5class": {
                "torch_transformer": {"group_kfold_subject": {
                    "folds": {f"fold_{fold:02d}": {} for fold in folds}
                }}
            }}}
        }), encoding="utf-8")

    completed = CompletedBenchmarkRun(
        config_hash=plan.config_hash,
        run_directory=run_dir,
        result_file=result_file,
        summary_file=None,
        manifest_file=run_dir / "run_manifest.json",
    )
    write_result([1, 2, 3, 4])
    assert experiment._completed_is_reusable(completed, plan.resolved_config) is False
    write_result([1, 2, 3, 4, 5])
    assert experiment._completed_is_reusable(completed, plan.resolved_config) is True


def test_full_layer_delegates_training_and_requires_standard_artifacts() -> None:
    source = inspect.getsource(OrdinalTransformerFullExperiment)
    assert "torch.optim" not in source
    assert "loss.backward" not in source
    assert "runner.run()" in source
    for artifact in (
        "predictions", "metrics", "class_metrics", "feature_manifest",
        "sequence_index_manifest", "validation_split", "model",
        "training_log", "normalization_stats", "ordinal_metadata",
    ):
        assert f'"{artifact}"' in source
    assert "_audit_fold" in source
    assert "_audit_trial" in source
    assert "subject_metrics.parquet" in source
    assert "audit_prediction_probabilities" in source
    assert "checkpoint_reload_audit" in source


def test_report_contains_required_inference_disclaimer() -> None:
    source = inspect.getsource(OrdinalTransformerFullExperiment._render_report)
    assert "Выбор лучшего метода и статистические выводы" in source
    assert "Subject-level" in source
    assert "Source-level" in source
    assert "Class-level" in source
