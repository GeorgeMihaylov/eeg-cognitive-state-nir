from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.core.abstract_dataset import BaseRecordDataset
from bench.bench_runner import BenchmarkRunner
from bench.datasets import cog_bci_dataset as cog
from bench.datasets.datasets_registry import DATASET_REGISTRY, get_dataset


def make_pair(
    root: Path,
    *,
    subject: str = "sub-01",
    session: str = "ses-S1",
    task: str = "zeroBACK",
) -> tuple[Path, Path]:
    directory = root / subject / session / "eeg"
    directory.mkdir(parents=True, exist_ok=True)
    set_path = directory / f"{task}.set"
    fdt_path = directory / f"{task}.fdt"
    set_path.write_bytes(b"set")
    fdt_path.write_bytes(b"fdt")
    return set_path, fdt_path


def fake_header(path: Path) -> dict[str, object]:
    has_cz = "sub-10" in path.as_posix()
    channels = ("Fp1", "Fz", *((("Cz",) if has_cz else ())), "ECG1")
    return {
        "sampling_rate_hz": 500.0,
        "n_samples": 5000,
        "duration_seconds": 10.0,
        "channel_names_total": channels,
        "eeg_channel_names": tuple(name for name in channels if name != "ECG1"),
        "auxiliary_channel_names": ("ECG1",),
        "mne_channel_types": tuple("eeg" for _ in channels),
        "event_count": 2,
        "event_types": ("10", "11"),
        "reference": "custom_ref_off",
        "montage_status": "auxiliary_missing_only",
        "channels_without_scalp_position": ("ECG1",),
        "eeg_channels_without_scalp_position": (),
        "data_units": (),
        "reader": "mock-mne(preload=False)",
    }


@pytest.fixture
def patched_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cog, "_read_record_header", fake_header)


def dataset_config(root: Path, **updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "data_path": root,
        "use_index_cache": False,
        "require_canonical_complete": False,
    }
    config.update(updates)
    return config


def test_discovers_complete_set_fdt_pair(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path)
    dataset = cog.COGBCIDataset(dataset_config(tmp_path))
    assert len(dataset.records) == 1
    assert dataset.records[0].set_relative_path.endswith("zeroBACK.set")
    assert dataset.records[0].fdt_relative_path.endswith("zeroBACK.fdt")


def test_missing_fdt_is_rejected(tmp_path: Path, patched_header: None) -> None:
    set_path, fdt_path = make_pair(tmp_path)
    fdt_path.unlink()
    with pytest.raises(cog.COGBCIIndexError, match="set=1, fdt=0"):
        cog.COGBCIRecordIndex.build(tmp_path)
    assert set_path.exists()


def test_missing_set_is_rejected(tmp_path: Path, patched_header: None) -> None:
    set_path, _ = make_pair(tmp_path)
    set_path.unlink()
    with pytest.raises(cog.COGBCIIndexError, match="set=0, fdt=1"):
        cog.COGBCIRecordIndex.build(tmp_path)


def test_duplicate_record_id_is_rejected(
    tmp_path: Path, patched_header: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_pair(tmp_path, task="zeroBACK")
    make_pair(tmp_path, task="oneBACK")
    monkeypatch.setattr(cog, "_record_id", lambda *args: "duplicate")
    with pytest.raises(cog.COGBCIIndexError, match="Duplicate record_id"):
        cog.COGBCIRecordIndex.build(tmp_path)


def test_unknown_subject_is_rejected(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path, subject="participant-01")
    with pytest.raises(cog.COGBCIIndexError, match="one subject"):
        cog.COGBCIRecordIndex.build(tmp_path)


def test_unknown_session_is_rejected(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path, session="session-A")
    with pytest.raises(cog.COGBCIIndexError, match="one session"):
        cog.COGBCIRecordIndex.build(tmp_path)


def test_unknown_task_is_rejected(tmp_path: Path, patched_header: None) -> None:
    make_pair(tmp_path, task="mystery")
    with pytest.raises(cog.COGBCIIndexError, match="Unknown COG-BCI task"):
        cog.COGBCIRecordIndex.build(tmp_path)


def test_record_id_is_stable_and_records_are_sorted(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path, subject="sub-10", session="ses-S2", task="PVT")
    make_pair(tmp_path, subject="sub-01", session="ses-S1", task="Flanker")
    first = cog.COGBCIRecordIndex.build(tmp_path)
    second = cog.COGBCIRecordIndex.build(tmp_path)
    assert [record.record_id for record in first.records] == [
        record.record_id for record in second.records
    ]
    assert [record.record_id for record in first.records] == sorted(
        record.record_id for record in first.records
    )
    assert first.to_dict() == second.to_dict()


def test_paths_and_serialization_are_relative(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path)
    document = cog.COGBCIRecordIndex.build(tmp_path).to_dict()
    serialized = json.dumps(document, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert ":\\" not in serialized
    record = document["records"][0]
    assert not Path(record["set_relative_path"]).is_absolute()
    assert not Path(record["fdt_relative_path"]).is_absolute()


def test_subject_and_session_filters_accept_string_ids(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path, subject="sub-01", session="ses-S1")
    make_pair(tmp_path, subject="sub-10", session="ses-S2")
    dataset = cog.COGBCIDataset(dataset_config(tmp_path))
    assert {record.subject_id for record in dataset.query(subject_ids=["1"])} == {
        "sub-01"
    }
    assert {record.session_id for record in dataset.query(session_ids=["ses-S2"])} == {
        "ses-02"
    }


def test_task_family_and_variant_filters_combine(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path, task="zeroBACK")
    make_pair(tmp_path, task="oneBACK")
    make_pair(tmp_path, task="MATBeasy")
    dataset = cog.COGBCIDataset(dataset_config(tmp_path))
    assert len(dataset.query(task_families=["n_back"])) == 2
    selected = dataset.query(
        task_families=["matb"], task_variants=["matb_easy"]
    )
    assert [record.task_label_raw for record in selected] == ["MATBeasy"]


def test_has_cz_and_layout_filters(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path, subject="sub-01")
    make_pair(tmp_path, subject="sub-10")
    dataset = cog.COGBCIDataset(dataset_config(tmp_path))
    without_cz = dataset.query(has_cz=False)
    with_cz = dataset.query(has_cz=True)
    assert [record.subject_id for record in without_cz] == ["sub-01"]
    assert [record.subject_id for record in with_cz] == ["sub-10"]
    layout = with_cz[0].channel_layout_id
    assert dataset.query(channel_layout_ids=[layout]) == with_cz


def test_unknown_filter_value_is_explicit(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path)
    dataset = cog.COGBCIDataset(dataset_config(tmp_path))
    with pytest.raises(ValueError, match="Unknown COG-BCI task_family"):
        dataset.query(task_families=["imagined_task"])


def test_ecg1_is_auxiliary_and_cz_is_preserved(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path, subject="sub-10")
    record = cog.COGBCIDataset(dataset_config(tmp_path)).records[0]
    assert record.auxiliary_channel_names == ("ECG1",)
    assert "ECG1" not in record.eeg_channel_names
    assert record.channel_count_auxiliary == 1
    assert record.channel_count_eeg == 3
    assert "Cz" in record.eeg_channel_names
    assert record.has_cz


def test_index_build_reads_headers_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_pair(tmp_path)
    calls: list[Path] = []

    def header(path: Path) -> dict[str, object]:
        calls.append(path)
        return fake_header(path)

    monkeypatch.setattr(cog, "_read_record_header", header)
    cog.COGBCIRecordIndex.build(tmp_path)
    assert [path.suffix for path in calls] == [".set"]


def test_open_raw_defaults_to_preload_false_and_eeg_only(
    tmp_path: Path, patched_header: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_pair(tmp_path)
    calls: list[dict[str, object]] = []

    class Raw:
        preload = False
        annotations = ("event",)

        def __init__(self) -> None:
            self.picked: list[str] | None = None

        def pick(self, names: list[str], ordered: bool) -> None:
            self.picked = names

    raw = Raw()

    def read_raw(path: str, **kwargs: object) -> Raw:
        calls.append({"path": path, **kwargs})
        raw.preload = bool(kwargs["preload"])
        return raw

    monkeypatch.setattr(
        cog,
        "_require_mne",
        lambda: SimpleNamespace(
            io=SimpleNamespace(read_raw_eeglab=read_raw)
        ),
    )
    dataset = cog.COGBCIDataset(dataset_config(tmp_path))
    opened = dataset.open_raw(dataset.records[0].record_id)
    assert calls[0]["preload"] is False
    assert opened.preload is False
    assert opened.picked == ["Fp1", "Fz"]
    assert opened.annotations == ("event",)


def test_open_raw_can_preserve_auxiliary_channels(
    tmp_path: Path, patched_header: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_pair(tmp_path)

    class Raw:
        def pick(self, names: list[str], ordered: bool) -> None:
            raise AssertionError("pick must not run when auxiliary channels are requested")

    monkeypatch.setattr(
        cog,
        "_require_mne",
        lambda: SimpleNamespace(
            io=SimpleNamespace(read_raw_eeglab=lambda *args, **kwargs: Raw())
        ),
    )
    dataset = cog.COGBCIDataset(dataset_config(tmp_path))
    dataset.open_raw(
        dataset.records[0].record_id,
        include_auxiliary_channels=True,
    )


def test_missing_mne_has_install_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cog.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError, match="python -m pip install"):
        cog._require_mne()


def test_cache_round_trip_does_not_reread_headers(
    tmp_path: Path, patched_header: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    cache = tmp_path / "cache" / "index.json"
    make_pair(root)
    first = cog.COGBCIDataset(
        dataset_config(root, use_index_cache=True, index_cache_path=cache)
    )
    first_record = first.records[0]
    monkeypatch.setattr(
        cog,
        "_read_record_header",
        lambda path: pytest.fail("cache load reread EEGLAB header"),
    )
    second = cog.COGBCIDataset(
        dataset_config(root, use_index_cache=True, index_cache_path=cache)
    )
    assert second.records[0] == first_record


def test_incompatible_cache_schema_is_rejected(
    tmp_path: Path, patched_header: None
) -> None:
    root = tmp_path / "data"
    cache = tmp_path / "index.json"
    make_pair(root)
    index = cog.COGBCIRecordIndex.build(root)
    document = index.to_dict()
    document["schema_version"] = 999
    cache.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(cog.COGBCICacheError, match="schema"):
        cog.COGBCIRecordIndex.load(cache, root)


def test_stale_source_fingerprint_is_rejected(
    tmp_path: Path, patched_header: None
) -> None:
    root = tmp_path / "data"
    cache = tmp_path / "index.json"
    _, fdt_path = make_pair(root)
    cog.COGBCIRecordIndex.build(root).save(cache)
    fdt_path.write_bytes(b"changed")
    with pytest.raises(cog.COGBCICacheError, match="fingerprint"):
        cog.COGBCIRecordIndex.load(cache, root)


def test_record_round_trip_accepts_windows_separators(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path)
    record = cog.COGBCIRecordIndex.build(tmp_path).records[0]
    document = record.to_dict()
    document["set_relative_path"] = document["set_relative_path"].replace("/", "\\")
    document["fdt_relative_path"] = document["fdt_relative_path"].replace("/", "\\")
    restored = cog.COGBCIRecord.from_dict(document)
    assert "\\" not in restored.set_relative_path
    assert restored.record_id == record.record_id


def test_no_targets_resampling_or_source_changes(
    tmp_path: Path, patched_header: None
) -> None:
    set_path, fdt_path = make_pair(tmp_path)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (set_path, fdt_path)
    }
    record = cog.COGBCIRecordIndex.build(tmp_path).records[0]
    fields = set(record.to_dict())
    assert not any("target" in field for field in fields)
    assert not any("label_q5" in field for field in fields)
    assert not any("resampl" in field for field in fields)
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (set_path, fdt_path)
    }
    assert before == after


def test_registry_uses_record_level_contract_without_materializing(
    tmp_path: Path,
) -> None:
    assert DATASET_REGISTRY["cog_bci"] is cog.COGBCIDataset
    dataset = get_dataset(
        "cog_bci",
        dataset_config(tmp_path),
    )
    assert isinstance(dataset, BaseRecordDataset)
    assert not hasattr(dataset, "load")


def test_expected_structure_mismatch_is_diagnostic(
    tmp_path: Path, patched_header: None
) -> None:
    make_pair(tmp_path)
    dataset = cog.COGBCIDataset(
        dataset_config(tmp_path, require_canonical_complete=True)
    )
    with pytest.raises(cog.COGBCIIndexError, match="observed=.*expected"):
        _ = dataset.records


def test_benchmark_runner_rejects_record_level_dataset_clearly(
    tmp_path: Path,
) -> None:
    config = {
        "output_dir": tmp_path / "output",
        "datasets": {
            "cog_bci": dataset_config(tmp_path),
        },
        "models": {},
    }
    runner = BenchmarkRunner(config)
    with pytest.raises(TypeError, match="record-level.*Window materialization"):
        runner.load_dataset("cog_bci")
