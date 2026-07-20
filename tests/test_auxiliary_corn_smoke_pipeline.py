from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.bench_runner import CompletedBenchmarkRun
from bench.experiments.auxiliary_corn_transformer import (
    AUXILIARY_WEIGHTS,
    AuxiliaryCornTransformerSmokeExperiment,
    audit_auxiliary_corn_probabilities,
    load_auxiliary_corn_smoke_spec,
)
from bench.experiments.ordinal_transformer import (
    SMOKE_ALIGNMENT_COLUMNS,
    build_ordinal_transformer_experiment,
    stable_frame_sha256,
)


SPEC = Path("experiments/auxiliary_corn_transformer_smoke.yaml")


def _feature_hash(names: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{name}\n" for name in names).encode("utf-8")
    ).hexdigest()


def _small_context() -> dict:
    rows = []
    for index in range(15):
        rows.append(
            {
                "sequence_id": f"sequence-{index}",
                "fold": 1 if index >= 10 else 2,
                "source": "synthetic",
                "subject_id": f"subject-{index // 3}",
                "record_id": f"record-{index // 3}",
                "record_group_id": f"group-{index // 3}",
                "target_sample_id": index,
                "target_time": float(index * 10),
                "y_true": index % 5,
                "outer_fold": 1,
                "split": (
                    "test"
                    if index >= 10
                    else ("validation" if index in {5, 6, 7, 8, 9} else "train")
                ),
            }
        )
    split = pd.DataFrame(rows)
    feature_names = [f"feature-{index}" for index in range(448)]
    return {
        "supervised_rows": 15,
        "canonical": split.copy(),
        "split_manifest": split,
        "full_sequence_index_sha256": "a" * 64,
        "smoke_sequence_subset_sha256": stable_frame_sha256(
            split, SMOKE_ALIGNMENT_COLUMNS
        ),
        "feature_names": feature_names,
        "feature_list_sha256": _feature_hash(feature_names),
        "outer_fold": 1,
        "outer_train_sequences": 10,
        "train_sequences": 5,
        "validation_sequences": 5,
        "test_sequences": 5,
        "train_subjects": 2,
        "validation_groups": 2,
        "test_subjects": 2,
        "test_subject_ids": ["subject-3", "subject-4"],
        "class_counts": {
            "train": {str(index): 1 for index in range(5)},
            "validation": {str(index): 1 for index in range(5)},
            "test": {str(index): 1 for index in range(5)},
        },
        "validation_summary": {"group_overlap": []},
        "outer_subject_overlap": [],
        "source_parquet_sha256": "b" * 64,
        "sequence_build_stats": {"sequences_created": 15},
    }


def _small_spec(tmp_path: Path, context: dict) -> Path:
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    document["experiment"]["output_dir"] = str(tmp_path / "outputs")
    document["experiment"]["report_path"] = str(tmp_path / "report.md")
    document["experiment"]["summary_path"] = str(tmp_path / "summary.json")
    document["dataset"].update(
        {
            "expected_supervised_rows": context["supervised_rows"],
            "expected_sequences": len(context["canonical"]),
            "sequence_index_sha256": context["full_sequence_index_sha256"],
            "parquet_sha256": context["source_parquet_sha256"],
        }
    )
    document["feature_group"]["feature_list_sha256"] = context[
        "feature_list_sha256"
    ]
    path = tmp_path / "auxiliary-smoke.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _joint_predictions(weight: float = 0.5) -> pd.DataFrame:
    primary = np.asarray(
        [
            [0.05, 0.15, 0.55, 0.20, 0.05],
            [0.10, 0.15, 0.20, 0.25, 0.30],
        ],
        dtype=float,
    )
    cumulative = np.asarray(
        [
            [0.90, 0.70, 0.30, 0.10],
            [0.80, 0.60, 0.40, 0.20],
        ],
        dtype=float,
    )
    auxiliary = np.column_stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2] - cumulative[:, 3],
            cumulative[:, 3],
        ]
    )
    frame = pd.DataFrame(
        {
            "sequence_id": ["s0", "s1"],
            "head_type": ["categorical_corn", "categorical_corn"],
            "auxiliary_weight": [weight, weight],
            "y_pred": primary.argmax(axis=1),
            "categorical_expected_rank": (
                primary * np.arange(5, dtype=float)
            ).sum(axis=1),
            "aux_expected_rank": cumulative.sum(axis=1),
            "aux_ordinal_prediction": (cumulative >= 0.5).sum(axis=1),
            "aux_ordinal_argmax": auxiliary.argmax(axis=1),
        }
    )
    for index in range(5):
        frame[f"proba_{index}"] = primary[:, index]
        frame[f"class_probability_{index}"] = primary[:, index]
        frame[f"aux_class_probability_{index}"] = auxiliary[:, index]
    for index in range(4):
        frame[f"aux_threshold_probability_{index}"] = cumulative[:, index]
    return frame


def test_spec_and_builder_resolve_auxiliary_smoke() -> None:
    document = load_auxiliary_corn_smoke_spec(SPEC)
    assert tuple(document["auxiliary_weights"]) == AUXILIARY_WEIGHTS
    experiment = build_ordinal_transformer_experiment(SPEC)
    assert isinstance(experiment, AuxiliaryCornTransformerSmokeExperiment)


def test_plan_builds_three_weights_with_one_split_and_distinct_hashes(
    tmp_path: Path,
) -> None:
    context = _small_context()
    spec = _small_spec(tmp_path, context)
    output = tmp_path / "plan-only"
    experiment = AuxiliaryCornTransformerSmokeExperiment(
        spec,
        output_dir=output,
        context_builder=lambda: context,
        completed_run_finder=lambda *args, **kwargs: None,
    )
    plans = experiment.plan()
    assert [plan.auxiliary_weight for plan in plans] == list(AUXILIARY_WEIGHTS)
    assert len({plan.smoke_sequence_subset_sha256 for plan in plans}) == 1
    assert len({plan.config_hash for plan in plans}) == 3
    assert all(plan.status == "valid" for plan in plans)
    assert all(plan.outer_fold == 1 and plan.maximum_epochs == 3 for plan in plans)
    assert all(plan.feature_group == "eeg_pow" for plan in plans)
    assert all(plan.model_parameter_count == plans[0].model_parameter_count for plan in plans)
    for plan in plans:
        params = plan.resolved_config["models"]["torch_transformer"]["params"]
        assert params["head_type"] == "categorical_corn"
        assert params["auxiliary_weight"] == plan.auxiliary_weight
        assert plan.resolved_config["experiment"]["lambda_selection_performed"] is False
    assert not output.exists()


def test_execute_uses_runner_and_resume_skips_completed(tmp_path: Path) -> None:
    context = _small_context()
    spec = _small_spec(tmp_path, context)
    created: list[dict] = []
    completed = CompletedBenchmarkRun(
        config_hash="synthetic",
        run_directory=tmp_path / "completed-run",
        result_file=tmp_path / "results.json",
        summary_file=None,
        manifest_file=None,
    )

    class FakeRunner:
        def __init__(self, config):
            created.append(config)

        def run(self):
            return pd.DataFrame()

        def completed_run(self):
            return completed

    experiment = AuxiliaryCornTransformerSmokeExperiment(
        spec,
        output_dir=tmp_path / "execute",
        context_builder=lambda: context,
        runner_factory=FakeRunner,
        completed_run_finder=lambda *args, **kwargs: None,
        trial_auditor=lambda plan, run, split: {
            "trial_id": plan.trial_id,
            "auxiliary_weight": plan.auxiliary_weight,
            "audited": True,
        },
    )
    result = experiment.execute(experiment.plan(), resume=False)
    assert len(created) == 3
    assert result["lambda_selection_performed"] is False
    assert all(item["outcome"] == "completed" for item in result["outcomes"])
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "summary.json").is_file()

    resumed = AuxiliaryCornTransformerSmokeExperiment(
        spec,
        output_dir=tmp_path / "resume",
        context_builder=lambda: context,
        runner_factory=lambda config: (_ for _ in ()).throw(
            AssertionError("resume retrained a completed trial")
        ),
        completed_run_finder=lambda *args, **kwargs: completed,
        trial_auditor=lambda plan, run, split: {"resumed": True},
    )
    resumed_result = resumed.execute(resumed.plan(), resume=True)
    assert all(item["outcome"] == "resumed" for item in resumed_result["outcomes"])


def test_probability_audit_accepts_joint_predictions() -> None:
    result = audit_auxiliary_corn_probabilities(
        _joint_predictions(), expected_weight=0.5
    )
    assert result["primary_probability_shape"] == [2, 5]
    assert result["auxiliary_threshold_shape"] == [2, 4]
    assert result["primary_prediction_recomputation_mismatches"] == 0
    assert result["auxiliary_prediction_recomputation_mismatches"] == 0
    assert result["maximum_auxiliary_monotonicity_violation"] == 0.0


def test_probability_audit_detects_primary_and_auxiliary_corruption() -> None:
    primary = _joint_predictions()
    primary.loc[0, "y_pred"] = 4
    with pytest.raises(ValueError, match="decoding rules"):
        audit_auxiliary_corn_probabilities(primary, expected_weight=0.5)

    ordinal = _joint_predictions()
    ordinal.loc[0, "aux_threshold_probability_2"] = 0.8
    with pytest.raises(ValueError, match="monotonicity"):
        audit_auxiliary_corn_probabilities(ordinal, expected_weight=0.5)


def test_experiment_layer_contains_no_training_loop() -> None:
    source = inspect.getsource(AuxiliaryCornTransformerSmokeExperiment)
    assert "torch.optim" not in source
    assert "loss.backward" not in source
    assert "runner.run()" in source
