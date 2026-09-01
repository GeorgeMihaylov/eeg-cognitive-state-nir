from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bench.datasets.channel_contracts import (
    PROJECT_EMOTIV_CHANNEL_ORDER,
    ChannelSelectionPolicy,
)
from bench.datasets.cog_bci_window_cache import (
    COGBCIWindowBuilder,
    COGBCIWindowCacheError,
    RawWindowSpec,
    audit_window_index,
    build_preprocessing_spec,
    compute_window_qc,
    enumerate_record_windows,
    stable_sample_id,
)


class FakeAnnotations:
    def __init__(
        self,
        descriptions: list[str] | None = None,
        onsets: list[float] | None = None,
        durations: list[float] | None = None,
    ) -> None:
        self.description = np.asarray(descriptions or [], dtype=object)
        self.onset = np.asarray(onsets or [], dtype=float)
        self.duration = np.asarray(
            durations or [0.0] * len(self.description), dtype=float
        )


class FakeRaw:
    def __init__(
        self,
        signal: np.ndarray,
        channels: tuple[str, ...],
        *,
        annotations: FakeAnnotations | None = None,
        sfreq: float = 500.0,
    ) -> None:
        self._signal = signal
        self.ch_names = list(channels)
        self.n_times = signal.shape[1]
        self.info = {
            "sfreq": sfreq,
            "highpass": 0.0,
            "lowpass": sfreq / 2.0,
        }
        self.annotations = annotations or FakeAnnotations()
        self.closed = False

    def get_data(self) -> np.ndarray:
        return self._signal.copy()

    def close(self) -> None:
        self.closed = True


def _record(
    root: Path,
    *,
    subject: str = "sub-01",
    session: str = "ses-01",
    family: str = "n_back",
    variant: str = "one_back",
) -> SimpleNamespace:
    directory = root / subject / session / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    set_path = directory / f"{variant}.set"
    fdt_path = directory / f"{variant}.fdt"
    set_path.write_bytes(b"set")
    fdt_path.write_bytes(b"fdt")
    return SimpleNamespace(
        record_id=f"cog_bci::{subject}::{session}::{variant}::run-na",
        subject_id=subject,
        session_id=session,
        task_family=family,
        task_variant=variant,
        condition=None,
        set_relative_path=set_path.relative_to(root).as_posix(),
        fdt_relative_path=fdt_path.relative_to(root).as_posix(),
        sampling_rate_hz=500.0,
    )


class FakeDataset:
    def __init__(
        self,
        root: Path,
        records: list[SimpleNamespace],
        *,
        signal_by_record: dict[str, np.ndarray],
        annotations_by_record: dict[str, FakeAnnotations] | None = None,
    ) -> None:
        self.root = root
        self.records = tuple(records)
        self.index = SimpleNamespace(source_root_fingerprint="root-fingerprint")
        self.signal_by_record = signal_by_record
        self.annotations_by_record = annotations_by_record or {}
        self.opened: list[FakeRaw] = []
        self.selection_copy_flags: list[bool] = []

    def get_channel_policy(self, name: str) -> ChannelSelectionPolicy:
        if name == "emotiv_common":
            channels = PROJECT_EMOTIV_CHANNEL_ORDER
        elif name == "cog_bci_common":
            channels = tuple(f"C{index:02d}" for index in range(62))
        else:
            raise ValueError(name)
        return ChannelSelectionPolicy(
            name=name, mode="required_exact", required_names=channels
        )

    def query(
        self,
        *,
        subject_ids=None,
        session_ids=None,
        task_families=None,
        task_variants=None,
    ):
        return tuple(
            record
            for record in self.records
            if (subject_ids is None or record.subject_id in subject_ids)
            and (session_ids is None or record.session_id in session_ids)
            and (task_families is None or record.task_family in task_families)
            and (task_variants is None or record.task_variant in task_variants)
        )

    def select_raw_channels(
        self, record_id, policy, *, preload=False, copy=False
    ):
        del preload
        self.selection_copy_flags.append(bool(copy))
        raw = FakeRaw(
            self.signal_by_record[record_id],
            policy.required_names,
            annotations=self.annotations_by_record.get(record_id),
        )
        self.opened.append(raw)
        return SimpleNamespace(raw=raw)


def _builder(
    tmp_path: Path,
    *,
    policy: str = "emotiv_common",
    n_samples: int = 6000,
    signal: np.ndarray | None = None,
    spec: RawWindowSpec | None = None,
    annotations: FakeAnnotations | None = None,
):
    root = tmp_path / "source"
    record = _record(root)
    channels = 14 if policy == "emotiv_common" else 62
    if signal is None:
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(channels, n_samples)).astype(np.float32)
    dataset = FakeDataset(
        root,
        [record],
        signal_by_record={record.record_id: signal},
        annotations_by_record=(
            {} if annotations is None else {record.record_id: annotations}
        ),
    )
    builder = COGBCIWindowBuilder(
        dataset,
        output_dir=tmp_path / "cache",
        channel_policy_name=policy,
        spec=spec or RawWindowSpec(),
    )
    return builder, record, dataset


@pytest.mark.parametrize(
    ("n_samples", "duration", "stride", "expected"),
    [
        (1000, 1.0, 1.0, 2),
        (1000, 1.0, 0.5, 4),
        (1250, 1.0, 1.0, 3),
    ],
)
def test_window_count_is_deterministic(
    n_samples: int, duration: float, stride: float, expected: int
) -> None:
    spec = RawWindowSpec(
        window_duration_seconds=duration,
        window_stride_seconds=stride,
    )
    rows = enumerate_record_windows(n_samples, 500.0, spec)
    assert len(rows) == expected
    assert rows == enumerate_record_windows(n_samples, 500.0, spec)


def test_windows_never_cross_continuous_segments() -> None:
    spec = RawWindowSpec(
        window_duration_seconds=1.0,
        window_stride_seconds=0.5,
    )
    rows = enumerate_record_windows(
        2000, 500.0, spec, segments=[(0, 900), (900, 2000)]
    )
    assert all(
        row["valid_stop_sample"] <= 900 or row["start_sample"] >= 900
        for row in rows
    )
    assert {row["segment_index"] for row in rows} == {0, 1}


def test_incomplete_window_drop_and_keep_policy() -> None:
    dropped = enumerate_record_windows(
        1250,
        500.0,
        RawWindowSpec(
            window_duration_seconds=1.0,
            window_stride_seconds=1.0,
            drop_incomplete_window=True,
        ),
    )
    assert dropped[-1]["complete"] is False
    assert dropped[-1]["valid_stop_sample"] == 1250

    spec = RawWindowSpec(
        window_duration_seconds=1.0,
        window_stride_seconds=1.0,
        drop_incomplete_window=False,
        minimum_valid_fraction=0.5,
    )
    window = np.vstack(
        [
            np.linspace(0, 1, 500, dtype=np.float32),
            np.linspace(1, 2, 500, dtype=np.float32),
        ]
    )
    qc = compute_window_qc(
        window, valid_samples=250, expected_samples=500, spec=spec
    )
    assert qc["status"] == "accepted"
    assert qc["valid_sample_fraction"] == pytest.approx(0.5)


def test_stable_sample_id_depends_on_semantic_contract() -> None:
    base = RawWindowSpec()
    kwargs = {
        "record_id": "record-1",
        "start_sample": 0,
        "stop_sample": 2560,
        "spec": base,
        "channel_policy_name": "emotiv_common",
        "preprocessing_hash": "pre-a",
    }
    first = stable_sample_id(**kwargs)
    assert first == stable_sample_id(**kwargs)
    assert first != stable_sample_id(
        **{**kwargs, "channel_policy_name": "cog_bci_common"}
    )
    assert first != stable_sample_id(
        **{**kwargs, "preprocessing_hash": "pre-b"}
    )
    assert first != stable_sample_id(
        **{
            **kwargs,
            "spec": RawWindowSpec(window_stride_seconds=2.56),
        }
    )


def test_preprocessing_profiles_preserve_500_hz_and_disallow_car() -> None:
    for name in ("none", "bandpass", "notch", "bandpass_notch"):
        registered = build_preprocessing_spec(
            RawWindowSpec(
                preprocessing=name,
                allow_filtering_when_source_status_unknown=True,
            ),
            sampling_rate_hz=500.0,
        )
        assert registered.target_sampling_rate == 500.0
        assert registered.step("car").enabled is False
    with pytest.raises(NotImplementedError, match="does not resample"):
        build_preprocessing_spec(
            RawWindowSpec(target_sampling_rate_hz=256.0),
            sampling_rate_hz=500.0,
        )


def test_unconfirmed_segmentation_modes_fail_explicitly() -> None:
    with pytest.raises(NotImplementedError, match="not enabled"):
        RawWindowSpec(segmentation_mode="task_interval")
    with pytest.raises(NotImplementedError, match="not enabled"):
        RawWindowSpec(segmentation_mode="event_interval")


def test_qc_detects_nan_inf_and_constant_channels() -> None:
    spec = RawWindowSpec(
        window_duration_seconds=1.0,
        window_stride_seconds=1.0,
    )
    signal = np.vstack(
        [
            np.ones(500, dtype=np.float32),
            np.linspace(0, 1, 500, dtype=np.float32),
            np.linspace(1, 2, 500, dtype=np.float32),
        ]
    )
    signal[1, 10] = np.nan
    signal[2, 20] = np.inf
    qc = compute_window_qc(
        signal, valid_samples=500, expected_samples=500, spec=spec
    )
    assert qc["status"] == "rejected_nonfinite"
    assert qc["rejection_reason"] == "rejected_nonfinite"
    assert qc["has_nan"] is True
    assert qc["has_inf"] is True
    assert qc["constant_channel_count"] == 1
    assert np.isfinite(qc["absolute_max"])
    assert np.isfinite(qc["absolute_mean"])


@pytest.mark.parametrize(
    ("policy", "channels"), [("emotiv_common", 14), ("cog_bci_common", 62)]
)
def test_materialization_writes_float32_record_safe_cache(
    tmp_path: Path, policy: str, channels: int
) -> None:
    builder, record, dataset = _builder(
        tmp_path, policy=policy, n_samples=6000
    )
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            dataset.root / record.set_relative_path,
            dataset.root / record.fdt_relative_path,
        )
    }
    summary = builder.run([record])
    manifest = json.loads(
        (tmp_path / "cache" / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    record_frame = pd.read_parquet(
        tmp_path / "cache" / "record_manifest.parquet"
    )
    array = np.load(
        tmp_path / "cache" / record_frame.iloc[0]["output_relative_path"]
    )
    windows = pd.read_parquet(tmp_path / "cache" / "window_index.parquet")
    assert summary["accepted_count"] == 2
    assert array.shape == (2, channels, 2560)
    assert array.dtype == np.float32
    assert np.isfinite(array).all()
    assert windows["record_id"].nunique() == 1
    assert windows["subject_id"].nunique() == 1
    assert windows["session_id"].nunique() == 1
    assert windows.loc[windows["status"] == "accepted", "cache_offset"].tolist() == [
        0,
        1,
    ]
    assert set(windows["status"]) == {"accepted", "rejected_incomplete"}
    assert manifest["samples_per_window"] == 2560
    assert manifest["sampling_rate_hz"] == 500.0
    assert manifest["channel_count"] == channels
    assert "ECG1" not in manifest["channel_order"]
    assert dataset.selection_copy_flags == [True]
    record_manifest_path = (
        tmp_path / "cache" / record_frame.iloc[0]["manifest_relative_path"]
    )
    record_manifest = json.loads(
        record_manifest_path.read_text(encoding="utf-8")
    )
    assert record_manifest["dtype"] == "float32"
    assert record_manifest["channel_order"] == manifest["channel_order"]
    assert record_manifest["sampling_rate_hz"] == 500.0
    assert record_manifest["window_samples"] == 2560
    assert not any(
        Path(value).is_absolute()
        for value in record_frame["output_relative_path"]
    )
    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_hashes
    }
    assert all(raw.closed for raw in dataset.opened)


def test_internal_boundary_prevents_crossing_and_events_are_preserved(
    tmp_path: Path,
) -> None:
    annotations = FakeAnnotations(
        ["boundary", "610", "boundary", "611"],
        [0.0, 0.2, 5.5, 11.5],
        [12.0, 0.0, 0.0, 0.0],
    )
    builder, record, _ = _builder(
        tmp_path,
        n_samples=6000,
        spec=RawWindowSpec(
            window_duration_seconds=2.0,
            window_stride_seconds=2.0,
        ),
        annotations=annotations,
    )
    builder.run([record])
    windows = pd.read_parquet(tmp_path / "cache" / "window_index.parquet")
    split_sample = int(round(5.5 * 500))
    assert all(
        stop <= split_sample or start >= split_sample
        for start, stop in zip(
            windows["start_sample"], windows["valid_stop_sample"]
        )
    )
    assert windows["contains_task_start"].any()
    assert windows["contains_task_end"].any()
    events = pd.read_parquet(tmp_path / "cache" / "events.parquet")
    assert len(events) == 4
    assert events["is_boundary"].sum() == 2


def test_matb_end_marker_is_part_of_versioned_event_contract(
    tmp_path: Path,
) -> None:
    annotations = FakeAnnotations(["boundary", "MATBdiffend"], [0.0, 11.5])
    builder, record, _ = _builder(
        tmp_path, n_samples=6000, annotations=annotations
    )
    record.task_family = "matb"
    record.task_variant = "matb_difficult"
    builder.run([record])
    windows = pd.read_parquet(tmp_path / "cache" / "window_index.parquet")
    manifest = json.loads(
        (tmp_path / "cache" / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert windows["contains_task_end"].sum() == 1
    assert manifest["event_contract"]["task_end"]["matb"] == [
        "MATBdiffend",
        "MATBeasyend",
        "MATBmedend",
    ]


def test_none_skips_filter_and_filter_profile_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bench.datasets.cog_bci_window_cache as module

    calls: list[np.ndarray] = []

    def fake_apply(signals, *, sampling_rate, spec):
        del sampling_rate, spec
        calls.append(signals)
        return np.asarray(signals, dtype=np.float32)

    monkeypatch.setattr(module, "apply_preprocessing_spec", fake_apply)
    builder, record, _ = _builder(tmp_path / "none")
    builder.run([record])
    assert calls == []

    filtered = RawWindowSpec(
        preprocessing="bandpass_notch",
        allow_filtering_when_source_status_unknown=True,
    )
    builder, record, _ = _builder(tmp_path / "filtered", spec=filtered)
    builder.run([record])
    assert len(calls) == 1
    assert calls[0].shape[1] == 6000


def test_unknown_source_filter_status_requires_explicit_opt_in(
    tmp_path: Path,
) -> None:
    builder, record, _ = _builder(
        tmp_path,
        spec=RawWindowSpec(preprocessing="bandpass"),
    )
    with pytest.raises(COGBCIWindowCacheError, match="history is unknown"):
        builder.run([record])


def test_resume_verify_checksum_and_overwrite_contract(tmp_path: Path) -> None:
    builder, record, dataset = _builder(tmp_path)
    first = builder.run([record])
    second = builder.run([record], resume=True)
    verified = builder.run([record], verify_only=True)
    assert first["rebuilt_records"] == 1
    assert second["skipped_records"] == 1
    assert verified["verified_records"] == 1
    assert len(dataset.opened) == 1

    record_frame = pd.read_parquet(
        tmp_path / "cache" / "record_manifest.parquet"
    )
    array_path = (
        tmp_path / "cache" / record_frame.iloc[0]["output_relative_path"]
    )
    array_path.write_bytes(array_path.read_bytes() + b"corrupt")
    with pytest.raises(COGBCIWindowCacheError, match="Checksum mismatch"):
        builder.run([record], resume=True)
    overwritten = builder.run([record], overwrite=True)
    assert overwritten["rebuilt_records"] == 1
    assert builder.run([record], verify_only=True)["verified_records"] == 1


def test_config_hash_and_changed_source_are_rejected(tmp_path: Path) -> None:
    builder, record, dataset = _builder(tmp_path)
    builder.run([record])
    incompatible = COGBCIWindowBuilder(
        dataset,
        output_dir=tmp_path / "cache",
        channel_policy_name="emotiv_common",
        spec=RawWindowSpec(window_stride_seconds=2.56),
    )
    with pytest.raises(COGBCIWindowCacheError, match="config hash"):
        incompatible.run([record], resume=True)

    (dataset.root / record.fdt_relative_path).write_bytes(b"changed")
    with pytest.raises(COGBCIWindowCacheError, match="Source record changed"):
        builder.run([record], resume=True)
    assert incompatible.run([record], overwrite=True)["rebuilt_records"] == 1


def test_temporary_file_is_not_a_completed_shard(tmp_path: Path) -> None:
    builder, record, _ = _builder(tmp_path)
    shards = tmp_path / "cache" / "shards"
    shards.mkdir(parents=True)
    (shards / "orphan.npy.tmp").write_bytes(b"partial")
    summary = builder.run([record])
    assert summary["rebuilt_records"] == 1


def test_selection_is_sorted_and_can_choose_one_record_per_family(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    records = [
        _record(root, subject="sub-10", family="pvt", variant="pvt"),
        _record(root, subject="sub-01", family="n_back", variant="zero_back"),
        _record(root, subject="sub-01", family="n_back", variant="one_back"),
    ]
    signals = {
        record.record_id: np.random.default_rng(index).normal(
            size=(14, 3000)
        ).astype(np.float32)
        for index, record in enumerate(records)
    }
    dataset = FakeDataset(root, records, signal_by_record=signals)
    builder = COGBCIWindowBuilder(
        dataset,
        output_dir=tmp_path / "cache",
        channel_policy_name="emotiv_common",
        spec=RawWindowSpec(),
    )
    selected = builder.select_records(one_per_subject_family=True)
    assert len(selected) == 2
    assert [record.record_id for record in selected] == sorted(
        record.record_id for record in selected
    )


def test_window_index_audit_detects_duplicate_sample_id() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "same",
                "subject_id": "sub-01",
                "session_id": "ses-01",
                "record_id": "record-a",
                "record_group_id": "record-a",
                "start_sample": 0,
                "valid_stop_sample": 10,
                "status": "accepted",
            },
            {
                "sample_id": "same",
                "subject_id": "sub-02",
                "session_id": "ses-01",
                "record_id": "record-b",
                "record_group_id": "record-b",
                "start_sample": 0,
                "valid_stop_sample": 10,
                "status": "accepted",
            },
        ]
    )
    audit = audit_window_index(frame)
    assert audit["duplicate_sample_ids"] == 1
    assert audit["leakage_safe"] is False
