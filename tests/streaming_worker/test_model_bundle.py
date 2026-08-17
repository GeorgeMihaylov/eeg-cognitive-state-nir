import json

import pytest

from apps.streaming_worker.model_bundle import load_model_bundle


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
            feature_profile="lightweight",
            allow_bootstrap=False,
        )
