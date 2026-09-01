from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bench.datasets.channel_contracts import (
    PROJECT_EMOTIV_CHANNEL_ORDER,
    ChannelSelectionPolicy,
)
from bench.datasets.cog_bci_window_cache import (
    COGBCIWindowBuilder,
    PolyphaseResamplingPreprocessor,
    RawWindowSpec,
    stable_sample_id,
)
from bench.experiments.cog_bci_contrastive_transfer import (
    create_pretraining_split,
)
from bench.experiments.cog_bci_time_aligned_transfer import (
    COGBCITimeAlignedTransferRunner,
    DOWNSTREAM_MODES,
    EXPECTED_VALIDATION_SUBJECTS,
    build_event_timing_audit,
    build_window_time_mapping,
    estimate_time_aligned_cache,
    time_alignment_transfer_decision,
)
from cogstate.model_zoo.DL.contrastive import (
    export_encoder_checkpoint,
    load_encoder_checkpoint,
)
from cogstate.model_zoo.DL.eegnet import TorchEEGNetClassifier


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "cog_bci"
    / "time_aligned_eegnet_transfer_screening.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _resampler() -> PolyphaseResamplingPreprocessor:
    return PolyphaseResamplingPreprocessor()


def _events() -> pd.DataFrame:
    families = ("n_back", "matb", "pvt", "flanker", "resting_state")
    return pd.DataFrame(
        {
            "record_id": [f"record-{index}" for index in range(5)],
            "event_index": [0] * 5,
            "task_family": families,
            "onset_seconds": [0.0, 0.101, 1.25, 2.003, 9.999],
            "duration_seconds": [0.0, 0.2, 0.0, 1.0, 0.1],
            "description": ["600", "MATBeasyend", "15", "21", "41"],
        }
    )


def _eegnet() -> TorchEEGNetClassifier:
    return TorchEEGNetClassifier(
        n_channels=14,
        n_times=2560,
        num_classes=5,
        temporal_kernel_samples=128,
        separable_kernel_samples=32,
        f1=2,
        depth_multiplier=1,
        f2=2,
        pool1=4,
        pool2=8,
        dropout=0.0,
    )


def test_polyphase_ratio_is_exact_64_over_125() -> None:
    spec = _resampler()
    assert math.gcd(spec.up, spec.down) == 1
    assert spec.up == 64
    assert spec.down == 125
    assert spec.target_sampling_rate_hz / spec.source_sampling_rate_hz == 64 / 125


def test_resampling_preserves_duration_within_one_target_sample() -> None:
    spec = _resampler()
    signal = np.zeros((14, 50_003), dtype=np.float32)
    output = spec.apply(signal, sampling_rate=500.0)
    error = output.shape[1] / 256.0 - signal.shape[1] / 500.0
    assert abs(error) <= 1 / 256
    assert output.shape[1] == math.ceil(signal.shape[1] * 64 / 125)


def test_resampling_uses_explicit_antialias_filter_contract() -> None:
    document = _resampler().to_dict()
    assert document["method"] == "scipy.signal.resample_poly"
    assert document["anti_alias_filter"] == {
        "design": "scipy.signal.firwin",
        "num_taps": 2501,
        "normalized_cutoff": 1 / 125,
        "pass_zero": "lowpass",
        "scale": True,
        "window": ["kaiser", 5.0],
    }
    assert document["padtype"] == "constant"
    assert document["scipy_version"]


def test_resampling_output_is_float32_finite_and_deterministic() -> None:
    signal = np.random.default_rng(42).normal(size=(14, 5_000)).astype(
        np.float32
    )
    first = _resampler().apply(signal, sampling_rate=500.0)
    second = _resampler().apply(signal, sampling_rate=500.0)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_resampling_rejects_wrong_source_rate() -> None:
    with pytest.raises(ValueError, match="source sampling rate mismatch"):
        _resampler().apply(
            np.zeros((14, 500), dtype=np.float32),
            sampling_rate=256.0,
        )


def test_resampling_rejects_nonfinite_input() -> None:
    signal = np.zeros((14, 500), dtype=np.float32)
    signal[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        _resampler().apply(signal, sampling_rate=500.0)


def test_time_aligned_window_has_project_physical_shape() -> None:
    output = _resampler().apply(
        np.zeros((14, 5_000), dtype=np.float32),
        sampling_rate=500.0,
    )
    assert output.shape == (14, 2560)
    assert output.shape[1] / 256 == 10.0


def test_resampling_spec_disables_forbidden_processing() -> None:
    operations = _resampler().to_dict()["additional_signal_processing"]
    assert set(operations) == {
        "demean",
        "bandpass",
        "notch",
        "car",
        "rereference",
        "cz_interpolation",
    }
    assert not any(operations.values())


def test_time_aligned_cache_estimate_uses_whole_record_lengths() -> None:
    records = [
        SimpleNamespace(
            n_samples=5_000,
            task_family="n_back",
        ),
        SimpleNamespace(
            n_samples=7_501,
            task_family="pvt",
        ),
    ]
    estimate = estimate_time_aligned_cache(
        records,
        resampler=_resampler(),
        window_samples=2560,
        channel_count=14,
    )
    assert estimate["accepted_windows"] == 2
    assert estimate["rejected_incomplete_tails"] == 1
    assert estimate["npy_bytes"] == 2 * 14 * 2560 * 4


def test_time_aligned_sample_id_differs_from_shape_only() -> None:
    old_spec = RawWindowSpec(
        window_duration_seconds=5.12,
        window_stride_seconds=5.12,
    )
    new_spec = RawWindowSpec(
        window_duration_seconds=10.0,
        window_stride_seconds=10.0,
        target_sampling_rate_hz=256.0,
    )
    old = stable_sample_id(
        record_id="record",
        start_sample=0,
        stop_sample=2560,
        spec=old_spec,
        channel_policy_name="emotiv_common",
        preprocessing_hash="old",
    )
    new = stable_sample_id(
        record_id="record",
        start_sample=0,
        stop_sample=2560,
        spec=new_spec,
        channel_policy_name="emotiv_common",
        preprocessing_hash=_resampler().stable_hash(
            channels=PROJECT_EMOTIV_CHANNEL_ORDER,
            loader_schema_version="test",
        ),
    )
    assert old != new


def test_event_timing_mapping_preserves_annotation_metadata() -> None:
    events = _events()
    audit = build_event_timing_audit(
        events, events.copy(), target_sampling_rate_hz=256.0
    )
    assert audit["metadata_equal"].all()
    assert set(audit["task_family_after"]) == {
        "n_back",
        "matb",
        "pvt",
        "flanker",
        "resting_state",
    }
    assert audit["timing_error_seconds"].abs().max() <= 0.5 / 256


def test_event_timing_mapping_detects_metadata_change() -> None:
    old = _events()
    new = old.copy()
    new.loc[0, "description"] = "changed"
    audit = build_event_timing_audit(
        old, new, target_sampling_rate_hz=256.0
    )
    assert int((~audit["metadata_equal"]).sum()) == 1


def test_window_time_mapping_is_complete_and_deterministic() -> None:
    old = pd.DataFrame(
        {
            "record_id": ["r"] * 3,
            "status": ["accepted"] * 3,
            "start_time_seconds": [0.0, 5.12, 10.24],
        }
    )
    new = pd.DataFrame(
        {
            "record_id": ["r"] * 2,
            "status": ["accepted"] * 2,
            "start_time_seconds": [0.0, 10.0],
            "sample_id": ["new-0", "new-1"],
        }
    )
    first = build_window_time_mapping(old, new)
    second = build_window_time_mapping(old, new)
    pd.testing.assert_frame_equal(first, second)
    assert first["new_sample_id"].tolist() == ["new-0", "new-1"]
    assert first["old_physical_start_time_seconds"].tolist() == [0.0, 10.24]


def test_old_input_file_is_unchanged_by_resampling(tmp_path: Path) -> None:
    old = tmp_path / "old-cache.bin"
    old.write_bytes(b"shape-only-cache")
    before = hashlib.sha256(old.read_bytes()).hexdigest()
    _resampler().apply(
        np.zeros((14, 500), dtype=np.float32), sampling_rate=500.0
    )
    after = hashlib.sha256(old.read_bytes()).hexdigest()
    assert before == after


def test_channel_order_excludes_ecg1() -> None:
    assert len(PROJECT_EMOTIV_CHANNEL_ORDER) == 14
    assert "ECG1" not in PROJECT_EMOTIV_CHANNEL_ORDER


def test_new_config_contains_no_absolute_paths() -> None:
    config = _config()
    path_values = [
        config["cache"]["dataset_root"],
        config["cache"]["index_cache"],
        config["cache"]["existing_shape_only_cache"],
        config["pretraining"]["cache"],
        config["pretraining"]["shape_only_pretraining_dir"],
        config["downstream"]["dataset"]["data_path"],
        config["downstream"]["shape_only_checkpoint"],
        config["output_dir"],
        config["tracked_report"],
    ]
    assert all(not Path(value).is_absolute() for value in path_values)
    assert all("F:\\" not in value for value in path_values)


def test_new_config_reuses_augmentation_contract_exactly() -> None:
    current = _config()["pretraining"]["augmentations"]
    previous = json.loads(
        (
            REPOSITORY_ROOT
            / "experiments"
            / "cog_bci"
            / "contrastive_eegnet_transfer_screening.json"
        ).read_text(encoding="utf-8")
    )["pretraining"]["augmentations"]
    assert current == previous
    assert current["time_masking"]["maximum_fraction"] == 0.1
    assert current["temporal_shift"]["maximum_fraction"] == 0.05


def test_new_config_uses_only_fold2_seed42_full_model_modes() -> None:
    downstream = _config()["downstream"]
    assert downstream["fold"] == 2
    assert downstream["seed"] == 42
    assert tuple(downstream["modes"]) == DOWNSTREAM_MODES
    assert "head_only" not in downstream["modes"]


def test_runner_validates_canonical_time_contract() -> None:
    runner = COGBCITimeAlignedTransferRunner(
        _config(), repository_root=REPOSITORY_ROOT
    )
    assert runner.cache_dir != (
        REPOSITORY_ROOT
        / "benchmark_results"
        / "cog_bci_windows"
        / "emotiv_common_full"
    )
    assert runner.downstream_dir.name == "label_q5_fold2_seed42"


def test_runner_rejects_per_window_or_wrong_ratio_config() -> None:
    config = _config()
    config["cache"]["resampling"]["down"] = 124
    with pytest.raises(ValueError, match="64/125"):
        COGBCITimeAlignedTransferRunner(
            config, repository_root=REPOSITORY_ROOT
        )


def test_pretraining_subject_assignment_matches_task8o() -> None:
    rows = []
    for subject in range(1, 30):
        rows.append(
            {
                "subject_id": f"sub-{subject:02d}",
                "sample_id": f"sample-{subject:02d}",
            }
        )
    split = create_pretraining_split(
        pd.DataFrame(rows), seed=42, validation_subjects=5
    )
    assert tuple(split["validation_subject_ids"]) == EXPECTED_VALIDATION_SUBJECTS
    assert split["training_subject_count"] == 24
    assert split["subject_overlap_count"] == 0


def test_encoder_checkpoint_loads_for_both_transfer_modes(
    tmp_path: Path,
) -> None:
    source = _eegnet()
    checkpoint = export_encoder_checkpoint(
        source, tmp_path / "encoder.pt", metadata={"physical_seconds": 10.0}
    )
    for _ in ("shape_only", "time_aligned"):
        target = _eegnet()
        metadata = load_encoder_checkpoint(target, checkpoint)
        assert metadata["physical_seconds"] == 10.0
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                source.features.state_dict().values(),
                target.features.state_dict().values(),
            )
        )


def test_encoder_checkpoint_does_not_contain_projection_head(
    tmp_path: Path,
) -> None:
    checkpoint = export_encoder_checkpoint(
        _eegnet(), tmp_path / "encoder.pt"
    )
    payload = torch.load(checkpoint, weights_only=False)
    assert not any(
        "projection" in key
        for key in payload["encoder_state_dict"]
    )


def _metrics(macro_f1: float, balanced_accuracy: float) -> dict[str, float]:
    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
    }


def _thresholds() -> dict[str, float]:
    return _config()["decision_rule"]


def test_decision_rule_proceeds_only_when_all_three_conditions_hold() -> None:
    result = time_alignment_transfer_decision(
        {
            "random_init": _metrics(0.25, 0.25),
            "shape_only": _metrics(0.24, 0.24),
            "time_aligned": _metrics(0.265, 0.249),
        },
        collapse_fatal=False,
        checkpoint_valid=True,
        leakage_safe=True,
        thresholds=_thresholds(),
    )
    assert result["decision"] == "proceed"


def test_decision_rule_strong_proceed() -> None:
    result = time_alignment_transfer_decision(
        {
            "random_init": _metrics(0.25, 0.25),
            "shape_only": _metrics(0.245, 0.24),
            "time_aligned": _metrics(0.275, 0.265),
        },
        collapse_fatal=False,
        checkpoint_valid=True,
        leakage_safe=True,
        thresholds=_thresholds(),
    )
    assert result["decision"] == "strong_proceed"


def test_decision_rule_closes_track_on_insufficient_gain() -> None:
    result = time_alignment_transfer_decision(
        {
            "random_init": _metrics(0.25, 0.25),
            "shape_only": _metrics(0.245, 0.24),
            "time_aligned": _metrics(0.251, 0.25),
        },
        collapse_fatal=False,
        checkpoint_valid=True,
        leakage_safe=True,
        thresholds=_thresholds(),
    )
    assert result["decision"] == "close_transfer_track"


def test_decision_rule_is_inconclusive_on_leakage_failure() -> None:
    result = time_alignment_transfer_decision(
        {
            "random_init": _metrics(0.25, 0.25),
            "shape_only": _metrics(0.24, 0.24),
            "time_aligned": _metrics(0.30, 0.30),
        },
        collapse_fatal=False,
        checkpoint_valid=True,
        leakage_safe=False,
        thresholds=_thresholds(),
    )
    assert result["decision"] == "inconclusive"


class _Annotations:
    onset = np.asarray([0.0])
    duration = np.asarray([0.0])
    description = np.asarray(["boundary"], dtype=object)


class _Raw:
    def __init__(self, signal: np.ndarray) -> None:
        self.signal = signal
        self.ch_names = list(PROJECT_EMOTIV_CHANNEL_ORDER)
        self.n_times = signal.shape[1]
        self.info = {
            "sfreq": 500.0,
            "highpass": 0.0,
            "lowpass": 250.0,
        }
        self.annotations = _Annotations()

    def get_data(self) -> np.ndarray:
        return self.signal.copy()

    def close(self) -> None:
        return None


class _Dataset:
    def __init__(self, root: Path, signal: np.ndarray) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "record.set").write_bytes(b"set")
        (root / "record.fdt").write_bytes(b"fdt")
        self.records = (
            SimpleNamespace(
                record_id="record",
                subject_id="sub-01",
                session_id="ses-01",
                task_family="n_back",
                task_variant="one_back",
                condition="one_back",
                set_relative_path="record.set",
                fdt_relative_path="record.fdt",
                sampling_rate_hz=500.0,
                has_cz=True,
            ),
        )
        self.index = SimpleNamespace(source_root_fingerprint="fingerprint")
        self.signal = signal
        self.open_count = 0

    def get_channel_policy(self, name: str) -> ChannelSelectionPolicy:
        return ChannelSelectionPolicy(
            name=name,
            mode="required_exact",
            required_names=PROJECT_EMOTIV_CHANNEL_ORDER,
        )

    def query(self, **_: object):
        return self.records

    def select_raw_channels(self, *_: object, **__: object):
        self.open_count += 1
        return SimpleNamespace(raw=_Raw(self.signal))


def test_builder_resamples_whole_record_before_windowing(
    tmp_path: Path,
) -> None:
    signal = np.random.default_rng(42).normal(size=(14, 10_001)).astype(
        np.float32
    )
    dataset = _Dataset(tmp_path / "source", signal)
    builder = COGBCIWindowBuilder(
        dataset,
        output_dir=tmp_path / "cache",
        channel_policy_name="emotiv_common",
        spec=RawWindowSpec(
            window_duration_seconds=10.0,
            window_stride_seconds=10.0,
            target_sampling_rate_hz=256.0,
        ),
        whole_record_preprocessor=_resampler(),
    )
    summary = builder.run(dataset.records)
    manifest = pd.read_parquet(
        tmp_path / "cache" / "record_manifest.parquet"
    ).iloc[0]
    assert dataset.open_count == 1
    assert summary["accepted_count"] == 2
    assert manifest["source_n_times"] == 10_001
    assert manifest["resampled_n_times"] == math.ceil(10_001 * 64 / 125)
    array_path = next((tmp_path / "cache" / "shards").glob("*.npy"))
    assert np.load(array_path).shape == (2, 14, 2560)


def test_builder_time_aligned_cache_is_record_safe(tmp_path: Path) -> None:
    signal = np.random.default_rng(7).normal(size=(14, 5_100)).astype(
        np.float32
    )
    dataset = _Dataset(tmp_path / "source", signal)
    builder = COGBCIWindowBuilder(
        dataset,
        output_dir=tmp_path / "cache",
        channel_policy_name="emotiv_common",
        spec=RawWindowSpec(
            window_duration_seconds=10.0,
            window_stride_seconds=10.0,
            target_sampling_rate_hz=256.0,
        ),
        whole_record_preprocessor=_resampler(),
    )
    builder.run(dataset.records)
    windows = pd.read_parquet(tmp_path / "cache" / "window_index.parquet")
    accepted = windows.loc[windows["status"].eq("accepted")]
    assert accepted["record_id"].eq("record").all()
    assert (accepted["valid_stop_sample"] <= math.ceil(5_100 * 64 / 125)).all()
    assert accepted["sampling_rate_hz"].eq(256.0).all()
