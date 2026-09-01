from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bench.experiments.artifact_removal_ablation import (
    build_run_matrix as build_v1_matrix,
    load_config as load_v1_config,
)
from bench.experiments.artifact_removal_ablation_v2 import (
    ARTIFACT_VARIANTS_V2,
    ArtifactRunSpecV2,
    _paired_raw_metrics,
    _task_arrays,
    _thresholds_for_metric_fold,
    build_run_matrix,
    calculate_coverage,
    load_config,
    load_signal_universe,
    protocol_plan,
    run_specification_hash,
    smoke_sample_ids,
)
from bench.preprocessing import artifact_removal_cache_v2 as cache_module
from bench.preprocessing.artifact_removal_cache_v2 import (
    CachedArtifactWindowView,
    _process_record,
    build_variant_cache,
    preprocessing_cache_hash,
    variant_config_hash,
)
from bench.tasks.target_registry import PM_METRICS
from cogstate.model_zoo import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preprocessing" / "artifact_removal_ablation_v2.yaml"
V1_CONFIG = ROOT / "experiments" / "preprocessing" / "artifact_removal_ablation_v1.yaml"


def _require_real_data(config: dict) -> None:
    path = ROOT / config["dataset"]["raw_window_index"]
    if not path.is_file():
        pytest.skip(f"canonical raw EEG index is not available in this worktree: {path}")


@pytest.fixture(scope="module")
def real_config() -> dict:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def real_universe(real_config: dict):
    _require_real_data(real_config)
    return load_signal_universe(real_config)


class FakeRawView:
    def __init__(self, values: np.ndarray, manifest: pd.DataFrame) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.manifest = manifest.reset_index(drop=True)
        self.shape = self.values.shape

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        if np.isscalar(index):
            return self.values[int(index)]
        positions = np.arange(len(self))[index]
        return FakeRawView(
            self.values[np.asarray(positions, dtype=np.int64)],
            self.manifest.iloc[np.asarray(positions, dtype=np.int64)],
        )


def _synthetic_manifest(tmp_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(6, 1, 14, 2560)).astype(np.float32)
    source_shard = tmp_path / "source.npy"
    np.save(source_shard, values[:, 0], allow_pickle=False)
    frame = pd.DataFrame(
        {
            "sample_id": np.arange(100, 106),
            "subject_id": ["s1"] * 3 + ["s2"] * 3,
            "record_id": ["r1"] * 3 + ["r2"] * 3,
            "record_group_id": ["g1"] * 3 + ["g2"] * 3,
            "source": ["a"] * 3 + ["b"] * 3,
            "outer_fold": [1] * 3 + [2] * 3,
            "t_start": [0.0, 10.0, 20.0] * 2,
            "t_end": [10.0, 20.0, 30.0] * 2,
            "cache_file": [str(source_shard)] * 6,
            "cache_offset": np.arange(6),
            "preprocessing_hash": ["raw-contract"] * 6,
        }
    )
    return values, frame


def test_run_matrix_is_280_with_all_pm_variants_and_tasks(real_config: dict) -> None:
    specs = build_run_matrix(real_config)
    assert len(specs) == 7 * 4 * 5 * 2 == 280
    assert tuple(dict.fromkeys(spec.metric for spec in specs)) == PM_METRICS
    assert set(spec.variant for spec in specs) == set(ARTIFACT_VARIANTS_V2)
    assert set(spec.task_type for spec in specs) == {"classification", "regression"}


def test_smoke_matrix_logically_contains_all_56_combinations(real_config: dict) -> None:
    specs = build_run_matrix(real_config)
    smoke = [spec for spec in specs if spec.fold == 1]
    assert len(smoke) == 7 * 4 * 2 == 56
    assert len({(spec.metric, spec.variant, spec.task_type) for spec in smoke}) == 56


def test_real_plan_uses_target_independent_union_and_expected_counts(real_config: dict) -> None:
    _require_real_data(real_config)
    plan = protocol_plan(CONFIG)
    assert plan["signal_universe"]["windows"] == 30_958
    assert plan["signal_universe"]["subjects"] == 54
    assert plan["signal_universe"]["record_group_ids"] == 86
    assert plan["signal_universe"]["input_shape"] == [1, 14, 2560]
    assert plan["target_cohorts"]["attention"]["sample_count"] == 29_569
    assert all(
        plan["target_cohorts"][metric]["sample_count"] == 30_958
        for metric in PM_METRICS[1:]
    )
    assert plan["all_target_cohorts_subset_of_signal_universe"] is True
    assert plan["target_mask_application_order"] == "after_preprocessing_cache"


def test_real_outer_folds_are_participant_disjoint_and_match_reference(real_config: dict) -> None:
    _require_real_data(real_config)
    plan = protocol_plan(CONFIG)
    audit = plan["fixed_outer_folds"]
    assert audit["subject_fold_assignments_match"] is True
    assert all(not fold["participant_overlap"] for fold in audit["folds"].values())


def test_smoke_selection_uses_all_pm_without_leaving_signal_universe(
    real_config: dict, real_universe
) -> None:
    selected = set(smoke_sample_ids(real_universe, real_config))
    signal_ids = set(real_universe.manifest["sample_id"].astype(str))
    assert selected
    assert selected.issubset(signal_ids)
    for metric in PM_METRICS:
        assert selected & real_universe.target_sample_ids[metric]


def test_q3_thresholds_are_fitted_before_rejection_and_shared_by_variants(
    real_universe,
) -> None:
    expected = _thresholds_for_metric_fold(real_universe, "attention", 1)
    retained = real_universe.manifest.iloc[::2].copy()
    values = []
    for variant in ARTIFACT_VARIANTS_V2:
        variant_manifest = retained.copy()
        variant_manifest["variant"] = variant
        _, _, _, thresholds = _task_arrays(
            real_universe,
            variant_manifest,
            ArtifactRunSpecV2("attention", variant, 1, "classification"),
        )
        values.append(thresholds)
    assert values == [expected] * 4


def test_record_cache_never_mixes_record_group_ids(
    tmp_path: Path, real_config: dict
) -> None:
    values, manifest = _synthetic_manifest(tmp_path)
    result = build_variant_cache(
        FakeRawView(values, manifest),
        variant="raw",
        variant_root=tmp_path / "raw",
        config=real_config["preprocessing"],
        source_hash="source-hash",
        resume=False,
    )
    assert result["record_group_count"] == 2
    reports = pd.read_parquet(tmp_path / "raw" / "record_diagnostics.parquet")
    assert set(reports["record_group_id"]) == {"g1", "g2"}
    assert set(reports["input_window_count"]) == {3}


def test_full_faster_retained_indices_map_to_original_sample_ids(
    monkeypatch: pytest.MonkeyPatch, real_config: dict
) -> None:
    tensor = np.zeros((4, 1, 14, 2560), dtype=np.float32)
    report = SimpleNamespace(
        kept_epoch_indices=[0, 2],
        bad_channels=[],
        bad_epochs=[1, 3],
        bad_components=[],
        bad_channel_epoch_pairs_original=[],
        channel_bads_by_metric={},
        epoch_bads_by_metric={"variance": [1, 3]},
        component_bads_by_metric={},
        ica_fitted=True,
        ica_converged=True,
        interpolation_method="mean",
    )

    def fake_run_faster(epochs, config, *, sample_rate):
        return epochs[[0, 2]], report

    monkeypatch.setattr(cache_module, "run_faster", fake_run_faster)
    cleaned, retained, diagnostics = _process_record(
        tensor, "faster_full_record_local", real_config["preprocessing"]
    )
    assert retained.tolist() == [0, 2]
    assert cleaned.shape[0] == 2
    assert diagnostics["bad_epochs"] == [1, 3]


def test_rejected_epochs_are_absent_from_cached_dataset_view(tmp_path: Path) -> None:
    values = np.zeros((2, 1, 14, 2560), dtype=np.float32)
    np.save(tmp_path / "record.npy", values, allow_pickle=False)
    manifest = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "subject_id": ["s"] * 3,
            "record_id": ["r"] * 3,
            "record_group_id": ["g"] * 3,
            "source": ["x"] * 3,
            "outer_fold": [1] * 3,
            "variant": ["faster_full_record_local"] * 3,
            "retained": [True, False, True],
            "shard": ["record.npy"] * 3,
            "offset": [0, None, 1],
            "storage_root": ["variant"] * 3,
            "preprocessing_hash": ["h"] * 3,
            "t_start": [0.0, 10.0, 20.0],
            "t_end": [10.0, 20.0, 30.0],
        }
    )
    view = CachedArtifactWindowView(
        manifest, repo_root=tmp_path, variant_root=tmp_path
    )
    assert len(view) == 2
    assert view.manifest["sample_id"].tolist() == [1, 3]


def test_cache_resume_does_not_reprocess_completed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_config: dict
) -> None:
    values, manifest = _synthetic_manifest(tmp_path)
    calls = []

    def fake_process(tensor, variant, config):
        calls.append(len(tensor))
        return tensor.copy(), np.arange(len(tensor)), {
            "input_window_count": len(tensor),
            "retained_window_count": len(tensor),
            "rejected_window_count": 0,
            "coverage": 1.0,
            "elapsed_seconds": 0.0,
            "epoch_zscore_structurally_limited": True,
            "average_reference": False,
        }

    monkeypatch.setattr(cache_module, "_process_record", fake_process)
    kwargs = dict(
        base_view=FakeRawView(values, manifest),
        variant="faster_online",
        variant_root=tmp_path / "cache",
        config=real_config["preprocessing"],
        source_hash="source-hash",
    )
    build_variant_cache(**kwargs, resume=False)
    assert calls == [3, 3]
    calls.clear()
    result = build_variant_cache(**kwargs, resume=True)
    assert calls == []
    assert result["resumed_record_count"] == 2


def test_preprocessing_hash_changes_with_scientific_config(real_config: dict) -> None:
    base = real_config["preprocessing"]
    changed = json.loads(json.dumps(base))
    changed["ica"]["max_iter"] = 999
    assert preprocessing_cache_hash(base, "source") != preprocessing_cache_hash(
        changed, "source"
    )
    assert variant_config_hash("raw", base, "source") != variant_config_hash(
        "raw", changed, "source"
    )


def test_raw_coverage_is_exactly_one() -> None:
    coverage, rejected = calculate_coverage(["1", "2"], ["1", "2"], variant="raw")
    assert coverage == 1.0
    assert rejected == []
    with pytest.raises(RuntimeError, match="Raw coverage"):
        calculate_coverage(["1"], ["1", "2"], variant="raw")


def test_specification_hash_covers_protocol_cache_and_run_spec() -> None:
    spec = ArtifactRunSpecV2("focus", "raw", 1, "classification")
    base = run_specification_hash(spec, protocol_hash="p", cache_hash="c")
    assert base == run_specification_hash(spec, protocol_hash="p", cache_hash="c")
    assert base != run_specification_hash(spec, protocol_hash="p2", cache_hash="c")
    assert base != run_specification_hash(spec, protocol_hash="p", cache_hash="c2")
    changed = ArtifactRunSpecV2("focus", "raw", 1, "regression")
    assert base != run_specification_hash(changed, protocol_hash="p", cache_hash="c")


def test_paired_raw_uses_exact_same_sample_ids(tmp_path: Path) -> None:
    config = {"output_dir": str(tmp_path)}
    raw_spec = ArtifactRunSpecV2("focus", "raw", 1, "classification")
    raw_dir = tmp_path / "smoke" / raw_spec.run_id
    raw_dir.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "subject_id": ["a", "a", "b"],
            "y_true": [0, 1, 2],
            "y_pred": [0, 0, 2],
        }
    )
    raw.to_parquet(raw_dir / "predictions.parquet", index=False)
    current = pd.DataFrame(
        {
            "sample_id": [3, 1],
            "subject_id": ["b", "a"],
            "y_true": [2, 0],
            "y_pred": [1, 0],
        }
    )
    spec = ArtifactRunSpecV2(
        "focus", "faster_full_record_local", 1, "classification"
    )
    window, macro, participant = _paired_raw_metrics(
        config, "smoke", spec, current
    )
    assert window["accuracy"] == 1.0
    assert macro["accuracy"] == 1.0
    assert set(participant["subject_id"]) == {"a", "b"}


def test_shared_torch_adapter_keeps_inner_record_groups_disjoint() -> None:
    rng = np.random.default_rng(8)
    groups = np.repeat([f"g{index}" for index in range(6)], 3)
    subjects = np.repeat([f"s{index}" for index in range(6)], 3)
    records = np.repeat([f"r{index}" for index in range(6)], 3)
    labels = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 6)
    inputs = rng.normal(size=(18, 1, 14, 2560)).astype(np.float32)
    model = build_model(
        "torch_shallow_convnet",
        "classification",
        (1, 14, 2560),
        3,
        {
            "n_filters": 2,
            "temporal_kernel_samples": 5,
            "pool_size": 25,
            "pool_stride": 10,
            "dropout": 0.0,
            "batch_size": 6,
            "max_epochs": 1,
            "validation_size": 0.34,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
            "standardize": True,
            "num_workers": 0,
        },
    )
    model.set_validation_groups(
        groups,
        subject_ids=subjects,
        record_ids=records,
        outer_test_record_ids=np.asarray(["outer-r"]),
        outer_test_group_ids=np.asarray(["outer-g"]),
        strategy="group_record",
        group_column="record_group_id",
        validation_size=0.34,
        random_state=42,
    )
    model.fit(inputs, labels)
    assert model.validation_split_["inner_group_overlap"] == 0
    assert model.validation_split_["outer_test_group_overlap"] == 0


def test_v1_matrix_contract_remains_unchanged() -> None:
    assert len(build_v1_matrix(load_v1_config(V1_CONFIG))) == 140

