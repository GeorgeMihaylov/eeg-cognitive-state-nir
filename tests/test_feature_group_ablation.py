from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from bench.bench_runner import CompletedBenchmarkRun
from bench.datasets.base_eeg_data_loader import (
    feature_list_sha256,
    resolve_feature_columns,
)
from bench.experiments.feature_group_ablation import (
    FeatureGroupRFExperiment,
    load_feature_group_spec,
    prediction_alignment,
    resolve_trial_config,
)


SPEC = Path("experiments/feature_group_rf_ablation.yaml")
DATA = Path("data/processed/windowed_eeg_pm_dataset_w10.parquet")


def test_canonical_feature_groups_have_exact_counts_hashes_and_no_leakage() -> None:
    document = load_feature_group_spec(SPEC)
    columns = list(pq.ParquetFile(DATA).schema.names)
    forbidden = {
        "target_focus", "label_q5", "subject_id", "record_id", "source",
        "sample_id", "t_start", "t_end", "t_center",
    }
    for group, expected_count in (("eeg_only", 168), ("pow_only", 280), ("eeg_pow", 448)):
        spec = document["feature_groups"][group]
        first = resolve_feature_columns(columns, spec["feature_set"])
        second = resolve_feature_columns(columns, spec["feature_set"])
        assert first == second
        assert len(first) == expected_count
        assert feature_list_sha256(first) == spec["feature_list_sha256"]
        assert not (set(first) & forbidden)
        assert not any(name.startswith(("target_", "PM.")) for name in first)
    assert all(name.startswith("EEG.") for name in resolve_feature_columns(columns, "eeg_only"))
    assert all(name.startswith("POW.") for name in resolve_feature_columns(columns, "pow_only"))


def test_resolve_trial_config_uses_canonical_rf_and_regression_factory_path(tmp_path) -> None:
    document = load_feature_group_spec(SPEC)
    classification = resolve_trial_config(
        document,
        task_name="classification",
        feature_group="eeg_pow",
        seed=42,
        output_root=tmp_path,
    )
    regression = resolve_trial_config(
        document,
        task_name="regression",
        feature_group="pow_only",
        seed=42,
        output_root=tmp_path,
    )
    assert classification["tasks"] == ["cognitive_load_5class"]
    assert classification["models"]["random_forest_classifier"]["params"] == {
        "n_estimators": 50,
        "max_depth": 12,
        "random_state": 42,
        "n_jobs": -1,
    }
    assert regression["tasks"] == ["focus_regression"]
    assert regression["models"]["random_forest_regressor"]["task_type"] == "regression"
    assert regression["datasets"]["emotiv_cognitive"]["target_col"] == "target_focus"


def test_plan_is_canonical_valid_and_writes_nothing(tmp_path) -> None:
    output = tmp_path / "feature-output"
    gitignore_before = Path(".gitignore").read_bytes()
    data_stat = DATA.stat()
    experiment = FeatureGroupRFExperiment(SPEC, output_dir=output)
    plans = experiment.plan(seed=42)
    assert len(plans) == 6
    assert all(plan.status == "valid" for plan in plans)
    assert all(plan.fold_count == 5 for plan in plans)
    assert all(plan.rows == 45_384 and plan.subjects == 54 for plan in plans)
    assert experiment._canonical_alignment["exact_match"] is True
    assert all(
        not overlap
        for overlap in experiment._canonical_alignment["train_test_subject_overlap"].values()
    )
    assert not output.exists()
    assert Path(".gitignore").read_bytes() == gitignore_before
    assert DATA.stat().st_size == data_stat.st_size
    assert DATA.stat().st_mtime_ns == data_stat.st_mtime_ns


def test_prediction_alignment_checks_metadata_and_targets() -> None:
    left = pd.DataFrame({
        "sample_id": [1, 2],
        "fold": [1, 2],
        "subject_id": ["s1", "s2"],
        "record_id": ["r1", "r2"],
        "source": ["a", "b"],
        "y_true": [0, 1],
    })
    assert prediction_alignment(left, left.copy(), compare_target=True)["exact_match"]
    changed = left.copy()
    changed.loc[1, "record_id"] = "different"
    result = prediction_alignment(left, changed, compare_target=True)
    assert result["exact_match"] is False
    assert result["mismatches"]["record_id"] == 1


def test_execute_uses_runner_and_resume_skips_completed(tmp_path) -> None:
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

    experiment = FeatureGroupRFExperiment(
        SPEC,
        output_dir=tmp_path / "outputs",
        runner_factory=FakeRunner,
        completed_run_finder=lambda *args, **kwargs: None,
    )
    plan = experiment.plan(
        feature_groups=["eeg_only"], tasks=["classification"], seed=42
    )
    result = experiment.execute(plan, resume=False)
    assert len(created) == 1
    assert result["analysis"]["status"] == "partial_matrix"

    resumed = FeatureGroupRFExperiment(
        SPEC,
        output_dir=tmp_path / "resume",
        runner_factory=lambda config: (_ for _ in ()).throw(AssertionError("trained")),
        completed_run_finder=lambda *args, **kwargs: dummy,
    )
    resume_plan = resumed.plan(
        feature_groups=["eeg_only"], tasks=["classification"], seed=42
    )
    result = resumed.execute(resume_plan, resume=True)
    assert result["trials"][0]["outcome"] == "resumed"
