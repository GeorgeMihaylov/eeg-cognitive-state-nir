from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bench.datasets.channel_contracts import (
    PROJECT_EMOTIV_CHANNEL_ORDER,
    ChannelHarmonizationError,
    ChannelMappingEntry,
    ChannelSelectionPolicy,
    apply_channel_policy,
    build_cog_bci_channel_policy,
    channel_contract_json,
    compute_common_eeg_channel_order,
    load_cog_bci_emotiv_mapping,
)
from bench.datasets.cog_bci_dataset import COGBCIDataset
from bench.datasets.raw_eeg_window_dataset import CANONICAL_EEG_CHANNELS
from bench.data_quality.cog_bci_channel_audit import _project_contract_audit


def _layout(
    eeg: tuple[str, ...],
    *,
    auxiliary: tuple[str, ...] = ("ECG1",),
    layout_id: str = "layout-test",
):
    return SimpleNamespace(
        channel_layout_id=layout_id,
        channel_names_total=(*eeg, *auxiliary),
        eeg_channel_names=eeg,
        auxiliary_channel_names=auxiliary,
        has_cz="Cz" in eeg,
    )


def _raw(names: tuple[str, ...], *, auxiliary: tuple[str, ...] = ("ECG1",)):
    mne = pytest.importorskip("mne")
    types = ["ecg" if name in auxiliary else "eeg" for name in names]
    info = mne.create_info(list(names), sfreq=500.0, ch_types=types)
    return mne.io.RawArray(
        np.arange(len(names) * 64, dtype=np.float64).reshape(len(names), 64),
        info,
        verbose="ERROR",
    )


def test_project_emotiv_contract_is_raw_pipeline_source_of_truth():
    assert CANONICAL_EEG_CHANNELS is PROJECT_EMOTIV_CHANNEL_ORDER
    assert PROJECT_EMOTIV_CHANNEL_ORDER == (
        "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7",
        "EEG.P7", "EEG.O1", "EEG.O2", "EEG.P8", "EEG.T8",
        "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4",
    )


def test_common_order_preserves_first_layout_instead_of_sorting():
    records = (
        _layout(("F7", "AF3", "Cz", "F3"), layout_id="with-cz"),
        _layout(("F7", "AF3", "F3"), layout_id="without-cz"),
    )
    assert compute_common_eeg_channel_order(records) == ("F7", "AF3", "F3")


def test_common_order_rejects_empty_collection():
    with pytest.raises(ChannelHarmonizationError, match="empty"):
        compute_common_eeg_channel_order(())


def test_native_policy_preserves_eeg_order_and_excludes_ecg1():
    record = _layout(("F7", "AF3", "Cz"))
    policy = build_cog_bci_channel_policy("cog_bci_native", records=[record])
    validation = policy.validate(record)
    assert validation.valid
    assert validation.output_names == ("F7", "AF3", "Cz")
    assert validation.excluded_auxiliary == ("ECG1",)


def test_native_policy_preserves_cz_when_present():
    record = _layout(("AF3", "Cz"))
    policy = build_cog_bci_channel_policy("cog_bci_native", records=[record])
    assert policy.select_names(record) == ["AF3", "Cz"]


def test_common_policy_excludes_cz_missing_from_one_layout():
    records = (_layout(("AF3", "Cz")), _layout(("AF3",), layout_id="no-cz"))
    policy = build_cog_bci_channel_policy("cog_bci_common", records=records)
    assert policy.required_names == ("AF3",)
    assert policy.select_names(records[0]) == ["AF3"]
    assert policy.select_names(records[1]) == ["AF3"]


def test_emotiv_policy_uses_project_order_and_explicit_namespace_aliases():
    source_names = tuple(name.removeprefix("EEG.") for name in reversed(
        PROJECT_EMOTIV_CHANNEL_ORDER
    ))
    record = _layout(source_names)
    policy = build_cog_bci_channel_policy("emotiv_common", records=[record])
    validation = policy.validate(record)
    assert validation.valid
    assert validation.output_names == PROJECT_EMOTIV_CHANNEL_ORDER
    assert validation.source_names == tuple(
        name.removeprefix("EEG.") for name in PROJECT_EMOTIV_CHANNEL_ORDER
    )


def test_exact_match_has_priority_over_alias():
    policy = ChannelSelectionPolicy(
        name="test",
        mode="mapped",
        required_names=("EEG.AF3",),
        mappings=(
            ChannelMappingEntry(
                emotiv_channel="EEG.AF3",
                cog_bci_channel="AF3",
                match_type="explicit_alias_match",
                evidence="test",
            ),
        ),
    )
    validation = policy.validate(_layout(("AF3", "EEG.AF3")))
    assert validation.source_names == ("EEG.AF3",)


def test_multiple_present_aliases_are_ambiguous():
    policy = ChannelSelectionPolicy(
        name="test",
        mode="mapped",
        required_names=("EEG.AF3",),
        mappings=(
            ChannelMappingEntry("EEG.AF3", "AF3", "explicit_alias_match", "a"),
            ChannelMappingEntry("EEG.AF3", "AF3-alt", "explicit_alias_match", "b"),
        ),
    )
    validation = policy.validate(_layout(("AF3", "AF3-alt")))
    assert not validation.valid
    assert validation.ambiguous_required == ("EEG.AF3",)
    with pytest.raises(ChannelHarmonizationError, match="ambiguous"):
        policy.select_names(_layout(("AF3", "AF3-alt")))


def test_missing_required_channel_is_explicit():
    policy = ChannelSelectionPolicy(
        name="test",
        mode="required_exact",
        required_names=("AF3", "AF4"),
    )
    validation = policy.validate(_layout(("AF3",)))
    assert validation.missing_required == ("AF4",)
    with pytest.raises(ChannelHarmonizationError, match="AF4"):
        policy.select_names(_layout(("AF3",)))


def test_case_and_whitespace_are_not_silently_normalized():
    policy = ChannelSelectionPolicy(
        name="test",
        mode="required_exact",
        required_names=("AF3",),
    )
    for present in ("af3", " AF3"):
        assert not policy.validate(_layout((present,))).valid


def test_case_collisions_are_rejected_as_ambiguous():
    policy = ChannelSelectionPolicy(name="native", mode="native")
    validation = policy.validate(_layout(("AF3", "af3")))
    assert not validation.valid
    assert set(validation.ambiguous_required) == {"AF3", "af3"}


def test_auxiliary_channel_cannot_satisfy_required_eeg():
    policy = ChannelSelectionPolicy(
        name="test",
        mode="required_exact",
        required_names=("ECG1",),
    )
    assert not policy.validate(_layout(("AF3",))).valid


def test_apply_native_policy_returns_copy_and_preserves_original():
    raw = _raw(("AF3", "F7", "ECG1"))
    original_data = raw.get_data().copy()
    policy = ChannelSelectionPolicy(name="native", mode="native")
    result = apply_channel_policy(raw, policy)
    assert result.raw is not raw
    assert raw.ch_names == ["AF3", "F7", "ECG1"]
    assert result.raw.ch_names == ["AF3", "F7"]
    np.testing.assert_array_equal(raw.get_data(), original_data)


def test_apply_copy_false_is_explicitly_in_place():
    raw = _raw(("AF3", "ECG1"))
    policy = ChannelSelectionPolicy(name="native", mode="native")
    result = apply_channel_policy(raw, policy, copy=False)
    assert result.raw is raw
    assert raw.ch_names == ["AF3"]
    assert result.provenance["copy"] is False


def test_apply_policy_preserves_sampling_rate_samples_and_filter_state():
    raw = _raw(("AF3", "F7", "ECG1"))
    before = (
        raw.info["sfreq"],
        raw.n_times,
        raw.info["highpass"],
        raw.info["lowpass"],
        raw.info["custom_ref_applied"],
    )
    result = apply_channel_policy(
        raw, ChannelSelectionPolicy(name="native", mode="native")
    )
    after = (
        result.raw.info["sfreq"],
        result.raw.n_times,
        result.raw.info["highpass"],
        result.raw.info["lowpass"],
        result.raw.info["custom_ref_applied"],
    )
    assert after == before
    assert result.provenance["resampling"] is False
    assert result.provenance["filtering"] is False
    assert result.provenance["rereferencing"] is False


def test_apply_mapped_policy_renames_output_to_project_contract():
    raw = _raw(("AF3", "F7", "ECG1"))
    expected_f7 = raw.get_data(picks=["F7"]).copy()
    policy = ChannelSelectionPolicy(
        name="emotiv_common",
        mode="mapped",
        required_names=("EEG.F7", "EEG.AF3"),
        mappings=(
            ChannelMappingEntry("EEG.F7", "F7", "explicit_alias_match", "test"),
            ChannelMappingEntry("EEG.AF3", "AF3", "explicit_alias_match", "test"),
        ),
    )
    result = apply_channel_policy(raw, policy)
    assert result.raw.ch_names == ["EEG.F7", "EEG.AF3"]
    assert result.selected_names == ("EEG.F7", "EEG.AF3")
    assert result.provenance["selected_source_names"] == ["F7", "AF3"]
    np.testing.assert_array_equal(
        result.raw.get_data(picks=["EEG.F7"]), expected_f7
    )


def test_apply_policy_reports_original_layout_and_selection():
    raw = _raw(("AF3", "ECG1"))
    record = _layout(("AF3",), layout_id="layout-source")
    result = apply_channel_policy(
        raw,
        ChannelSelectionPolicy(name="native", mode="native"),
        record_metadata=record,
    )
    assert result.provenance["policy_name"] == "native"
    assert result.provenance["source_layout_id"] == "layout-source"
    assert result.provenance["source_channel_names"] == ["AF3", "ECG1"]
    assert result.provenance["selected_channel_names"] == ["AF3"]


def test_apply_policy_fails_when_raw_disagrees_with_record_metadata():
    raw = _raw(("F7", "ECG1"))
    record = _layout(("AF3",))
    with pytest.raises(ChannelHarmonizationError, match="Raw object is missing"):
        apply_channel_policy(
            raw,
            ChannelSelectionPolicy(name="native", mode="native"),
            record_metadata=record,
        )


def test_mapping_artifact_matches_production_order_and_is_complete():
    mappings = load_cog_bci_emotiv_mapping()
    assert tuple(item.emotiv_channel for item in mappings) == (
        PROJECT_EMOTIV_CHANNEL_ORDER
    )
    assert len({item.cog_bci_channel for item in mappings}) == 14
    assert {item.match_type for item in mappings} == {"explicit_alias_match"}


def test_mapping_serialization_is_deterministic():
    record = _layout(tuple(
        name.removeprefix("EEG.") for name in PROJECT_EMOTIV_CHANNEL_ORDER
    ))
    first = build_cog_bci_channel_policy("emotiv_common", records=[record])
    second = build_cog_bci_channel_policy("emotiv_common", records=[record])
    assert channel_contract_json(first) == channel_contract_json(second)
    assert json.loads(channel_contract_json(first))["name"] == "emotiv_common"


def test_unknown_policy_is_explicit():
    with pytest.raises(ChannelHarmonizationError, match="Unknown"):
        build_cog_bci_channel_policy("not-a-policy", records=[_layout(("AF3",))])


def test_unknown_policy_mode_is_explicit():
    with pytest.raises(ChannelHarmonizationError, match="mode"):
        ChannelSelectionPolicy(name="bad", mode="other").validate(
            _layout(("AF3",))
        )


def test_dataset_get_channel_policy_uses_complete_record_index():
    records = (_layout(("AF3", "Cz")), _layout(("AF3",), layout_id="no-cz"))
    dataset = object.__new__(COGBCIDataset)
    dataset._index = SimpleNamespace(records=records)
    policy = dataset.get_channel_policy("cog_bci_common")
    assert policy.required_names == ("AF3",)


def test_dataset_select_raw_channels_preserves_old_open_contract(monkeypatch):
    record = _layout(("AF3",), layout_id="layout-one")
    record.record_id = "record-one"
    dataset = object.__new__(COGBCIDataset)
    dataset._index = SimpleNamespace(records=(record,))
    raw = _raw(("AF3", "ECG1"))
    calls = []

    def fake_open(record_id, *, preload=False, include_auxiliary_channels=None):
        calls.append((record_id, preload, include_auxiliary_channels))
        return raw

    monkeypatch.setattr(dataset, "open_raw", fake_open)
    result = dataset.select_raw_channels(
        "record-one", "cog_bci_native", preload=False
    )
    assert calls == [("record-one", False, True)]
    assert result.raw.ch_names == ["AF3"]
    assert raw.ch_names == ["AF3", "ECG1"]


def test_cz_interpolation_policy_is_not_implemented():
    with pytest.raises(ChannelHarmonizationError, match="Unknown"):
        build_cog_bci_channel_policy(
            "cog_bci_cz_interpolated", records=[_layout(("AF3",))]
        )


def test_project_contract_audit_checks_both_sources_and_shard_manifest(tmp_path):
    complete_layout = [
        "EEG.Counter", *PROJECT_EMOTIV_CHANNEL_ORDER, "EEG.RawCq"
    ]
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame({
        "source": ["gpn_data", "Old_EEG"],
        "eeg_columns": [repr(complete_layout), repr(complete_layout)],
    }).to_csv(catalog_path, index=False)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for name in ("gpn", "old"):
        (cache_dir / f"{name}.json").write_text(
            json.dumps({"channels": list(PROJECT_EMOTIV_CHANNEL_ORDER)}),
            encoding="utf-8",
        )

    audit = _project_contract_audit(catalog_path, cache_dir)

    assert audit["catalog_sources_same_signal_order"] is True
    assert audit["raw_cache_matches_contract"] is True
    assert audit["raw_cache_shards_checked"] == 2
    assert set(audit["catalog_sources"]) == {"gpn_data", "Old_EEG"}
