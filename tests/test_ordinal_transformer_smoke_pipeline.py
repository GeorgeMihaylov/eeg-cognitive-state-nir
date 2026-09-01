from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bench.bench_runner import CompletedBenchmarkRun
from bench.experiments.ordinal_transformer import (
    OrdinalTransformerSmokeExperiment,
    SMOKE_ALIGNMENT_COLUMNS,
    stable_frame_sha256,
)


SPEC = Path("experiments/ordinal_transformer_smoke.yaml")


def _small_context() -> dict:
    rows = []
    for index in range(10):
        rows.append({
            "sequence_id": f"sequence-{index}",
            "fold": 1 if index >= 8 else 2,
            "source": "synthetic",
            "subject_id": f"subject-{index // 2}",
            "record_id": f"record-{index // 2}",
            "record_group_id": f"group-{index // 2}",
            "target_sample_id": index,
            "target_time": float(index * 10),
            "y_true": index % 5,
            "outer_fold": 1,
            "split": "test" if index >= 8 else (
                "validation" if index in {6, 7} else "train"
            ),
        })
    split = pd.DataFrame(rows)
    feature_names = [f"EEG.synthetic_{index}" for index in range(168)]
    feature_hash = hashlib.sha256(
        "".join(f"{name}\n" for name in feature_names).encode("utf-8")
    ).hexdigest()
    full_hash = "a" * 64
    return {
        "supervised_rows": 10,
        "canonical": split.copy(),
        "split_manifest": split,
        "full_sequence_index_sha256": full_hash,
        "smoke_sequence_subset_sha256": stable_frame_sha256(
            split, SMOKE_ALIGNMENT_COLUMNS
        ),
        "feature_names": feature_names,
        "feature_list_sha256": feature_hash,
        "outer_fold": 1,
        "outer_train_sequences": 8,
        "train_sequences": 6,
        "validation_sequences": 2,
        "test_sequences": 2,
        "train_subjects": 3,
        "validation_groups": 1,
        "test_subjects": 1,
        "test_subject_ids": ["subject-4"],
        "class_counts": {
            "train": {str(index): 1 for index in range(5)},
            "validation": {str(index): 1 for index in range(5)},
            "test": {str(index): 1 for index in range(5)},
        },
        "validation_summary": {"group_overlap": []},
        "outer_subject_overlap": [],
        "source_parquet_sha256": "b" * 64,
        "sequence_build_stats": {"sequences_created": 10},
    }


def _small_spec(tmp_path: Path, context: dict) -> Path:
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    document["experiment"]["output_dir"] = str(tmp_path / "outputs")
    document["dataset"].update({
        "expected_supervised_rows": context["supervised_rows"],
        "expected_sequences": len(context["canonical"]),
        "sequence_index_sha256": context["full_sequence_index_sha256"],
        "parquet_sha256": context["source_parquet_sha256"],
    })
    document["feature_group"]["feature_list_sha256"] = context[
        "feature_list_sha256"
    ]
    path = tmp_path / "ordinal-smoke.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_plan_builds_three_heads_with_one_subset_and_writes_nothing(
    tmp_path: Path,
) -> None:
    context = _small_context()
    spec = _small_spec(tmp_path, context)
    output = tmp_path / "planned-output"
    experiment = OrdinalTransformerSmokeExperiment(
        spec,
        output_dir=output,
        context_builder=lambda: context,
        completed_run_finder=lambda *args, **kwargs: None,
    )
    plans = experiment.plan()
    assert [plan.head_type for plan in plans] == ["categorical", "coral", "corn"]
    assert len({plan.smoke_sequence_subset_sha256 for plan in plans}) == 1
    assert all(plan.status == "valid" for plan in plans)
    assert all(plan.outer_fold == 1 and plan.maximum_epochs == 3 for plan in plans)
    assert all(
        plan.resolved_config["models"]["torch_transformer"]["params"][
            "head_type"
        ] == plan.head_type
        for plan in plans
    )
    assert not output.exists()


def test_execute_calls_benchmark_runner_and_resume_skips_completed(
    tmp_path: Path,
) -> None:
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

    experiment = OrdinalTransformerSmokeExperiment(
        spec,
        output_dir=tmp_path / "execute",
        context_builder=lambda: context,
        runner_factory=FakeRunner,
        completed_run_finder=lambda *args, **kwargs: None,
        trial_auditor=lambda plan, run, split: {
            "trial_id": plan.trial_id,
            "audited": True,
        },
    )
    plans = experiment.plan()
    result = experiment.execute(plans, resume=False)
    assert len(created) == 3
    assert all(item["outcome"] == "completed" for item in result["outcomes"])

    resumed = OrdinalTransformerSmokeExperiment(
        spec,
        output_dir=tmp_path / "resume",
        context_builder=lambda: context,
        runner_factory=lambda config: (_ for _ in ()).throw(
            AssertionError("resume retrained a completed trial")
        ),
        completed_run_finder=lambda *args, **kwargs: completed,
        trial_auditor=lambda plan, run, split: {"resumed": True},
    )
    result = resumed.execute(resumed.plan(), resume=True)
    assert all(item["outcome"] == "resumed" for item in result["outcomes"])


def test_experiment_layer_contains_no_training_loop() -> None:
    source = inspect.getsource(OrdinalTransformerSmokeExperiment)
    assert "torch.optim" not in source
    assert "loss.backward" not in source
    assert "runner.run()" in source
