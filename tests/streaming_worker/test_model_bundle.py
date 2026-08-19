import json

import numpy as np
import pytest

from apps.streaming_worker.model_bundle import load_model_bundle
from cogstate.model_zoo.factory import build_model


def raw_manifest(**overrides):
    payload = {
        "version": "shallow-test-v1",
        "model_type": "torch_shallow_convnet",
        "input_mode": "raw_eeg",
        "input_layout": "batch,1,channels,time",
        "sample_rate": 128,
        "channels": ["C1", "C2", "C3", "C4"],
        "window_seconds": 2,
        "n_times": 256,
        "class_names": ["low", "medium", "high"],
        "preprocessing": {
            "bandpass_low_hz": 1,
            "bandpass_high_hz": 45,
            "notch_hz": 50,
            "faster": False,
            "filter_mode": "causal",
        },
        "bootstrap": True,
        "diagnostic_only": True,
        "model_file": "model.pt",
    }
    payload.update(overrides)
    return payload


def raw_preprocessing():
    return {
        "bandpass_low_hz": 1,
        "bandpass_high_hz": 45,
        "notch_hz": 50,
        "faster": False,
        "filter_mode": "causal",
    }


def test_raw_eeg_bootstrap_accepts_shallow_convnet_window(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(raw_manifest()), encoding="utf-8"
    )
    bundle = load_model_bundle(
        tmp_path,
        sample_rate=128,
        channels=("C1", "C2", "C3", "C4"),
        window_seconds=2,
        preprocessing=raw_preprocessing(),
        allow_bootstrap=True,
        device="cpu",
    )

    probabilities = bundle.predict_proba(np.zeros((1, 4, 256), dtype=np.float32))

    assert bundle.manifest.input_mode == "raw_eeg"
    assert set(probabilities) == {"low", "medium", "high"}
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_raw_eeg_bundle_rejects_preprocessing_mismatch(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(raw_manifest()), encoding="utf-8"
    )
    incompatible = raw_preprocessing()
    incompatible["faster"] = True

    with pytest.raises(ValueError, match="preprocessing.faster"):
        load_model_bundle(
            tmp_path,
            sample_rate=128,
            channels=("C1", "C2", "C3", "C4"),
            window_seconds=2,
            preprocessing=incompatible,
            allow_bootstrap=True,
            device="cpu",
        )


def test_saved_shallow_convnet_weights_load_as_raw_bundle(tmp_path):
    estimator = build_model(
        "torch_shallow_convnet",
        "classification",
        (1, 4, 256),
        3,
        {
            "sampling_rate": 128,
            "channel_names": ["C1", "C2", "C3", "C4"],
            "standardize": False,
            "device": "cpu",
        },
    )
    estimator.is_fitted_ = True
    estimator.model.eval()
    estimator.save(tmp_path / "model.pt")
    manifest = raw_manifest(bootstrap=False, diagnostic_only=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    bundle = load_model_bundle(
        tmp_path,
        sample_rate=128,
        channels=("C1", "C2", "C3", "C4"),
        window_seconds=2,
        preprocessing=raw_preprocessing(),
        allow_bootstrap=False,
        device="cpu",
    )
    probabilities = bundle.predict_proba(np.zeros((1, 4, 256), dtype=np.float32))

    assert bundle.estimator.model_metadata["sampling_rate"] == 128
    assert bundle.estimator.model_metadata["channel_names"] == [
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_bundle_contract_rejects_wrong_feature_profile(tmp_path):
    manifest = {
        "version": "trained-v1",
        "model_type": "logistic_regression",
        "n_features": 10,
        "sample_rate": 128,
        "channels": ["C1", "C2"],
        "window_seconds": 2,
        "feature_profile": "full",
        "diagnostic_only": False,
        "model_file": "model.joblib",
        "scaler_file": None,
        "selector_file": None,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_profile"):
        load_model_bundle(
            tmp_path,
            n_features=10,
            sample_rate=128,
            channels=("C1", "C2"),
            window_seconds=2,
            preprocessing={},
            feature_profile="lightweight",
            allow_bootstrap=False,
        )


def test_bundle_contract_rejects_legacy_feature_semantics(tmp_path):
    manifest = {
        "version": "trained-v1",
        "model_type": "logistic_regression",
        "n_features": 10,
        "sample_rate": 128,
        "channels": ["C1", "C2"],
        "window_seconds": 2,
        "feature_profile": "lightweight",
        "diagnostic_only": False,
        "model_file": "model.joblib",
        "scaler_file": None,
        "selector_file": None,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_schema_version"):
        load_model_bundle(
            tmp_path,
            n_features=10,
            sample_rate=128,
            channels=("C1", "C2"),
            window_seconds=2,
            preprocessing={},
            feature_profile="lightweight",
            allow_bootstrap=False,
        )
