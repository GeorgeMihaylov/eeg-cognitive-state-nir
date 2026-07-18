from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bench.bench_runner import CompletedBenchmarkRun
from bench.datasets.base_eeg_data_loader import (
    feature_list_sha256,
    resolve_feature_columns,
)
from bench.experiments.feature_group_ablation import (
    FeatureGroupTransformerExperiment,
    build_feature_group_experiment,
    load_feature_group_spec,
    resolve_trial_config,
    sequence_prediction_alignment,
)
from model_zoo import build_model
from model_zoo.DL.adapter import TorchClassificationAdapter
from model_zoo.DL.sequence_utils import build_sequences, sequence_index_sha256


SPEC = Path("experiments/feature_group_transformer_ablation.yaml")
RF_SPEC = Path("experiments/feature_group_rf_ablation.yaml")
DATA = Path("data/processed/windowed_eeg_pm_dataset_w10.parquet")


def _sequence_metadata() -> pd.DataFrame:
    rows = []
    for source, subject, record in (
        ("gpn_data", "S1", "R1"),
        ("Old_EEG", "S2", "R2"),
    ):
        for index in range(10):
            rows.append({
                "source": source,
                "subject_id": subject,
                "record_id": record,
                "sample_id": f"{record}-{index}",
                "t_start": float(index * 10),
            })
    return pd.DataFrame(rows)


def test_transformer_groups_reuse_rf_counts_hashes_and_resolver() -> None:
    transformer = load_feature_group_spec(SPEC)
    rf = load_feature_group_spec(RF_SPEC)
    columns = list(pq.ParquetFile(DATA).schema.names)
    for group, count in (("eeg_only", 168), ("pow_only", 280), ("eeg_pow", 448)):
        transformer_group = transformer["feature_groups"][group]
        rf_group = rf["feature_groups"][group]
        names = resolve_feature_columns(columns, transformer_group["feature_set"])
        assert len(names) == count
        assert feature_list_sha256(names) == transformer_group["feature_list_sha256"]
        assert transformer_group == rf_group


def test_resolved_transformer_trials_have_fixed_architecture_and_shapes(tmp_path) -> None:
    document = load_feature_group_spec(SPEC)
    for group, width in (("eeg_only", 168), ("pow_only", 280), ("eeg_pow", 448)):
        config = resolve_trial_config(
            document,
            task_name="classification",
            feature_group=group,
            seed=42,
            output_root=tmp_path,
        )
        assert config["sequence"]["length"] == 8
        assert config["validation"]["strategy"] == "group_record"
        assert config["models"]["torch_transformer"]["params"]["d_model"] == 128
        adapter = build_model(
            "torch_transformer",
            "classification",
            input_shape=(8, width),
            num_outputs=5,
            params={
                **config["models"]["torch_transformer"]["params"],
                "device": "cpu",
                "max_epochs": 1,
            },
        )
        assert isinstance(adapter, TorchClassificationAdapter)
        assert adapter.input_shape == (8, width)


def test_common_sequence_index_is_feature_independent_and_respects_boundaries() -> None:
    metadata = _sequence_metadata()
    labels = np.arange(len(metadata), dtype=np.int64) % 5
    first = build_sequences(
        np.zeros((len(metadata), 1), dtype=np.float32),
        labels,
        metadata,
        sequence_length=8,
        expected_step_seconds=10.0,
        max_gap_seconds=10.5,
    )
    second = build_sequences(
        np.arange(len(metadata) * 7, dtype=np.float32).reshape(len(metadata), 7),
        labels,
        metadata,
        sequence_length=8,
        expected_step_seconds=10.0,
        max_gap_seconds=10.5,
    )
    for result in (first, second):
        result.metadata["fold"] = result.metadata["subject_id"].map({"S1": 1, "S2": 2})
        result.metadata["y_true"] = result.y
    pd.testing.assert_frame_equal(first.metadata, second.metadata)
    assert sequence_index_sha256(first.metadata) == sequence_index_sha256(second.metadata)
    assert len(first.metadata) == 6
    assert first.metadata.groupby("sequence_id")["source"].nunique().eq(1).all()
    assert first.metadata.groupby("sequence_id")["subject_id"].nunique().eq(1).all()
    assert first.metadata.groupby("sequence_id")["record_id"].nunique().eq(1).all()
    assert first.metadata["max_internal_gap"].le(10.5).all()


def test_sequence_hash_and_alignment_detect_semantic_changes() -> None:
    metadata = _sequence_metadata()
    built = build_sequences(
        np.ones((len(metadata), 1), dtype=np.float32),
        np.arange(len(metadata), dtype=np.int64) % 5,
        metadata,
        sequence_length=8,
        expected_step_seconds=10.0,
        max_gap_seconds=10.5,
    )
    frame = built.metadata.copy()
    frame["fold"] = frame["subject_id"].map({"S1": 1, "S2": 2})
    frame["y_true"] = built.y
    shuffled = frame.sample(frac=1.0, random_state=42)
    assert sequence_index_sha256(frame) == sequence_index_sha256(shuffled)
    assert sequence_prediction_alignment(frame, shuffled)["exact_match"]
    changed = frame.copy()
    changed.loc[0, "y_true"] = (int(changed.loc[0, "y_true"]) + 1) % 5
    assert sequence_index_sha256(frame) != sequence_index_sha256(changed)
    assert not sequence_prediction_alignment(frame, changed)["exact_match"]


def test_plan_matches_canonical_baseline_and_writes_nothing(tmp_path) -> None:
    output = tmp_path / "transformer-feature-groups"
    gitignore_before = Path(".gitignore").read_bytes()
    data_stat = DATA.stat()
    experiment = build_feature_group_experiment(SPEC, output_dir=output)
    assert isinstance(experiment, FeatureGroupTransformerExperiment)
    plans = experiment.plan(seed=42)
    assert len(plans) == 3
    assert all(plan.status == "valid" for plan in plans)
    assert [plan.input_shape for plan in plans] == [(8, 168), (8, 280), (8, 448)]
    assert all(plan.sequence_count == 44_142 for plan in plans)
    assert all(plan.subjects == 53 for plan in plans)
    assert len({plan.sequence_index_sha256 for plan in plans}) == 1
    assert experiment._canonical_alignment["exact_match"] is True
    assert experiment._canonical_alignment["subjects_without_sequences"] == [
        "9192c107"
    ]
    assert all(
        not overlap
        for overlap in experiment._canonical_alignment[
            "train_test_subject_overlap"
        ].values()
    )
    assert not output.exists()
    assert Path(".gitignore").read_bytes() == gitignore_before
    assert DATA.stat().st_size == data_stat.st_size
    assert DATA.stat().st_mtime_ns == data_stat.st_mtime_ns


def test_transformer_matrix_uses_runner_and_resume(tmp_path) -> None:
    created: list[dict] = []
    dummy = CompletedBenchmarkRun(
        config_hash="dummy",
        run_directory=tmp_path / "run",
        result_file=tmp_path / "result.json",
        summary_file=None,
        manifest_file=tmp_path / "manifest.json",
    )

    class FakeRunner:
        def __init__(self, config):
            created.append(config)

        def run(self):
            return pd.DataFrame()

        def completed_run(self):
            return dummy

    experiment = FeatureGroupTransformerExperiment(
        SPEC,
        output_dir=tmp_path / "output",
        runner_factory=FakeRunner,
        completed_run_finder=lambda *args, **kwargs: None,
    )
    plan = experiment.plan(feature_groups=["eeg_only"], seed=42)
    result = experiment.execute(plan, resume=False)
    assert len(created) == 1
    assert created[0]["models"]["torch_transformer"]["type"] == "torch_transformer"
    assert result["analysis"]["status"] == "partial_matrix"

    resumed = FeatureGroupTransformerExperiment(
        SPEC,
        output_dir=tmp_path / "resume",
        runner_factory=lambda config: (_ for _ in ()).throw(AssertionError("trained")),
        completed_run_finder=lambda *args, **kwargs: dummy,
    )
    resumed_plan = resumed.plan(feature_groups=["eeg_only"], seed=42)
    resumed_result = resumed.execute(resumed_plan, resume=True)
    assert resumed_result["trials"][0]["outcome"] == "resumed"
