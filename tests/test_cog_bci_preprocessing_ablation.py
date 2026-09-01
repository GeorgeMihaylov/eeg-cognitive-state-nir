from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import bench.preprocessing.cog_bci_preprocessing as preprocessing_module
from bench.analysis.cog_bci_preprocessing_ablation import (
    _build_sample_mapping,
    select_preprocessing,
    spectral_qc_features,
)
from bench.datasets.cog_bci_window_cache import RawWindowSpec, stable_sample_id
from bench.preprocessing.cog_bci_preprocessing import (
    VARIANT_ORDER,
    build_preprocessing_variants,
)


def _variants():
    return {
        variant.variant_id: variant for variant in build_preprocessing_variants()
    }


def test_variant_matrix_is_exact_and_stable() -> None:
    variants = build_preprocessing_variants()
    assert tuple(variant.variant_id for variant in variants) == VARIANT_ORDER
    assert [variant.to_dict()["operation_order"] for variant in variants] == [
        ["identity"],
        ["demean"],
        ["notch"],
        ["bandpass"],
        ["demean", "notch"],
        ["demean", "bandpass"],
        ["bandpass", "notch"],
        ["demean", "bandpass", "notch"],
    ]
    assert len({variant.stable_hash() for variant in variants}) == 8


def test_raw_variant_preserves_float32_values_exactly() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(14, 4096)).astype(np.float32)
    before = values.copy()
    result = _variants()["A_raw"].apply(values, sampling_rate=500.0)
    assert np.array_equal(result, before)
    assert np.array_equal(values, before)


def test_whole_record_demean_removes_each_channel_mean() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(14, 4096)).astype(np.float32)
    values += np.arange(14, dtype=np.float32)[:, None]
    result = _variants()["B_record_demean"].apply(
        values, sampling_rate=500.0
    )
    assert np.allclose(result.mean(axis=1), 0.0, atol=2e-6)
    first_window = result[:, :2048]
    assert not np.allclose(first_window.mean(axis=1), 0.0, atol=1e-8)


def test_combination_calls_filters_once_and_in_documented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float]] = []

    def fake_sosfiltfilt(sos, values, **kwargs):
        calls.append(("bandpass", float(values.mean())))
        return values + 1.0

    def fake_filtfilt(numerator, denominator, values, **kwargs):
        calls.append(("notch", float(values.mean())))
        return values + 1.0

    monkeypatch.setattr(
        preprocessing_module, "sosfiltfilt", fake_sosfiltfilt
    )
    monkeypatch.setattr(preprocessing_module, "filtfilt", fake_filtfilt)
    values = np.full((2, 128), 7.0, dtype=np.float32)
    result = _variants()["H_demean_bandpass_notch"].apply(
        values, sampling_rate=500.0
    )
    assert calls[0][0] == "bandpass"
    assert calls[0][1] == pytest.approx(0.0)
    assert calls[1] == ("notch", pytest.approx(1.0))
    assert np.allclose(result, 2.0)


def test_notch_reduces_synthetic_50_hz_component() -> None:
    sampling_rate = 500.0
    time = np.arange(5000) / sampling_rate
    signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 50 * time)
    values = np.tile(signal, (2, 1)).astype(np.float32)
    result = _variants()["C_notch"].apply(values, sampling_rate=sampling_rate)
    frequencies = np.fft.rfftfreq(values.shape[1], 1 / sampling_rate)
    index = int(np.argmin(np.abs(frequencies - 50)))
    before = np.abs(np.fft.rfft(values[0]))[index]
    after = np.abs(np.fft.rfft(result[0]))[index]
    assert after < before * 0.05


def test_bandpass_reduces_dc_and_preserves_shape_on_short_record() -> None:
    sampling_rate = 500.0
    time = np.arange(200) / sampling_rate
    values = np.tile(
        4.0 + np.sin(2 * np.pi * 10 * time), (14, 1)
    ).astype(np.float32)
    result = _variants()["D_bandpass"].apply(
        values, sampling_rate=sampling_rate
    )
    assert result.shape == values.shape
    assert np.isfinite(result).all()
    assert abs(float(result.mean())) < 0.1


def test_filter_spec_resolves_library_phase_padding_and_width() -> None:
    spec = _variants()["G_bandpass_notch"].to_dict()
    assert spec["filter_library"] == "scipy.signal"
    assert spec["library_version"]
    assert spec["bandpass"]["phase"] == "zero_phase_forward_backward"
    assert spec["bandpass"]["padding"]["padlen_samples"] > 0
    assert spec["notch"]["padding"]["padlen_samples"] == 9
    assert spec["notch"]["width_hz"] == pytest.approx(50 / 30)


def test_spectral_qc_is_finite_and_contains_required_features() -> None:
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(3, 14, 2560)).astype(np.float32)
    result = spectral_qc_features(windows, sampling_rate=500.0)
    assert np.isfinite(result.to_numpy()).all()
    assert {
        "dc_magnitude",
        "within_record_channel_std",
        "dc_std_ratio",
        "power_0_1",
        "power_1_45",
        "power_49_51",
        "line_to_1_45_ratio",
        "theta_power",
        "alpha_power",
        "beta_power",
        "theta_alpha",
        "theta_beta",
    } == set(result)


def test_sample_id_changes_with_preprocessing_hash() -> None:
    spec = RawWindowSpec()
    raw, notch = _variants()["A_raw"], _variants()["C_notch"]
    common = {
        "record_id": "record-1",
        "start_sample": 0,
        "stop_sample": 2560,
        "spec": spec,
        "channel_policy_name": "emotiv_common",
    }
    raw_id = stable_sample_id(
        **common, preprocessing_hash=raw.stable_hash(channels=["F3"])
    )
    notch_id = stable_sample_id(
        **common, preprocessing_hash=notch.stable_hash(channels=["F3"])
    )
    assert raw_id != notch_id
    assert raw_id == stable_sample_id(
        **common, preprocessing_hash=raw.stable_hash(channels=["F3"])
    )


def test_mapping_preserves_boundaries_and_class_contract() -> None:
    raw = pd.DataFrame(
        {
            "sample_id": ["raw-1", "raw-2"],
            "record_id": ["r", "r"],
            "window_index": [0, 1],
            "start_sample": [0, 2560],
            "stop_sample": [2560, 5120],
            "start_time_seconds": [0.0, 5.12],
            "stop_time_seconds": [5.12, 10.24],
            "subject_id": ["s", "s"],
            "session_id": ["1", "1"],
            "target": [2, 2],
            "class_name": ["two_back", "two_back"],
        }
    )
    variant = raw.rename(columns={"sample_id": "old"}).copy()
    variant["sample_id"] = ["new-1", "new-2"]
    variant["status"] = "accepted"
    mapping = _build_sample_mapping(raw, variant)
    assert mapping["raw_sample_id"].tolist() == ["raw-1", "raw-2"]
    assert mapping["variant_sample_id"].tolist() == ["new-1", "new-2"]
    assert mapping["target"].tolist() == [2, 2]


def _selection_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for variant, score in (
        ("A_raw", 0.40),
        ("C_notch", 0.42),
        ("D_bandpass", 0.41),
        ("G_bandpass_notch", 0.39),
        ("B_record_demean", 0.38),
        ("E_demean_notch", 0.37),
        ("F_demean_bandpass", 0.36),
        ("H_demean_bandpass_notch", 0.35),
    ):
        for fold in range(1, 6):
            for model in ("a", "b"):
                fold_rows.append(
                    {
                        "variant_id": variant,
                        "preprocessing_name": variant,
                        "fold": fold,
                        "model": model,
                        "validation_macro_f1": score,
                        "validation_ordinal_mae": 0.6,
                        "test_macro_f1": 0.1 if variant == "C_notch" else 0.9,
                    }
                )
    qc = pd.DataFrame(
        {
            "variant_id": list(VARIANT_ORDER),
            "median_line_to_1_45_ratio": [
                1.0,
                1.0,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ],
            "nonfinite_values": 0,
        }
    )
    return pd.DataFrame(fold_rows), qc


def test_selection_uses_inner_validation_and_not_outer_test() -> None:
    folds, qc = _selection_frames()
    _, selected = select_preprocessing(
        folds,
        qc,
        tie_tolerance=1e-6,
        max_ordinal_mae_increase=0.1,
    )
    assert selected["selected_variant_id"] == "C_notch"
    assert selected["outer_test_used_for_selection"] is False


def test_selection_tie_prefers_raw_then_simple_non_demean() -> None:
    folds, qc = _selection_frames()
    folds["validation_macro_f1"] = 0.4
    table, selected = select_preprocessing(
        folds,
        qc,
        tie_tolerance=1e-6,
        max_ordinal_mae_increase=0.1,
    )
    assert selected["selected_variant_id"] == "A_raw"
    assert table.loc[
        table["variant_id"].eq("B_record_demean"), "simplicity_rank"
    ].iloc[0] > table.loc[
        table["variant_id"].eq("G_bandpass_notch"), "simplicity_rank"
    ].iloc[0]


def test_tracked_config_has_no_absolute_paths() -> None:
    config = json.loads(
        (
            __import__("pathlib").Path(__file__).parents[1]
            / "experiments/cog_bci/nback_preprocessing_ablation.json"
        ).read_text(encoding="utf-8")
    )
    encoded = json.dumps(config)
    assert "F:\\\\" not in encoded
    assert config["deep_check"]["fold"] == 1
    assert config["deep_check"]["max_epochs"] == 15
