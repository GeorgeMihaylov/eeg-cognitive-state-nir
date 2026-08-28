from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from apps.streaming_worker.config import WorkerConfig
from apps.streaming_worker.contracts import (
    feature_schema_hash,
    preprocessing_contract,
    preprocessing_hash,
)
from apps.streaming_worker.model_bundle import load_model_bundle
from apps.streaming_worker.runtime import StreamingRuntime
from apps.streaming_worker.scientific import plan_experiment, select_training_rows
from apps.streaming_worker.api.app import create_app
from cogstate.features.streaming import build_lightweight_pipeline
from cogstate.model_zoo.ML.multitask import PMMultiTaskClassifier
from cogstate.preprocessing.filtering import FilterConfig, StreamingFilter
from cogstate.protocol import EEG_CHANNELS, PM_METRICS


def _real_bundle(path: Path, *, n_features: int, schema_hash: str, pre_hash: str) -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(90, n_features))
    base = np.repeat(np.arange(3), 30)
    targets = np.column_stack([np.roll(base, index) for index in range(7)])
    estimator = PMMultiTaskClassifier(
        "logistic_regression", params={"max_iter": 200, "random_state": 42}
    ).fit(features, targets)
    path.mkdir(parents=True)
    joblib.dump(estimator, path / "model.joblib")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "version": "unit-real-v1",
                "model_type": "logistic_regression_multitask_q3",
                "diagnostic_only": False,
                "model_file": "model.joblib",
                "scaler_file": None,
                "selector_file": None,
                "imputer_file": None,
                "n_features": n_features,
                "sample_rate": 256,
                "channels": list(EEG_CHANNELS),
                "window_seconds": 10,
                "feature_profile": "lightweight",
                "feature_schema_hash": schema_hash,
                "preprocessing_hash": pre_hash,
                "target_metrics": list(PM_METRICS),
            }
        ),
        encoding="utf-8",
    )


def _contract() -> tuple[int, str, str]:
    names = build_lightweight_pipeline(256).feature_names(14)
    pre = preprocessing_contract(
        sample_rate=256,
        bandpass_enabled=False,
        bandpass_low_hz=1,
        bandpass_high_hz=45,
        notch_enabled=False,
        notch_hz=50,
        faster=False,
    )
    return len(names), feature_schema_hash(names), preprocessing_hash(pre)


def test_lightweight_feature_contract_is_locked_and_deterministic() -> None:
    first = build_lightweight_pipeline(256).feature_names(14)
    second = build_lightweight_pipeline(256).feature_names(14)
    assert first == second
    assert len(first) == 336
    assert feature_schema_hash(first) == (
        "62736110455c3423c27edff6f42769579b0ee57212099cdf979be92331d77f72"
    )
    assert sum("entropy" in name.lower() for name in first) == 0
    assert sum("connect" in name.lower() for name in first) == 0


def test_streaming_filter_has_real_identity_bypass() -> None:
    signal = np.arange(42, dtype=float).reshape(3, 14)
    filt = StreamingFilter(
        FilterConfig(
            sample_rate=256,
            bandpass_enabled=False,
            notch_enabled=False,
        ),
        n_channels=14,
    )
    np.testing.assert_array_equal(filt.process(signal), signal)
    filt.reset()
    np.testing.assert_array_equal(filt.process(signal), signal)


def test_real_bundle_requires_matching_scientific_hashes(tmp_path: Path) -> None:
    n_features, schema_hash, pre_hash = _contract()
    artifact = tmp_path / "bundle"
    _real_bundle(artifact, n_features=n_features, schema_hash=schema_hash, pre_hash=pre_hash)
    bundle = load_model_bundle(
        artifact,
        n_features=n_features,
        sample_rate=256,
        channels=EEG_CHANNELS,
        window_seconds=10,
        feature_profile="lightweight",
        feature_schema_hash_value=schema_hash,
        preprocessing_hash_value=pre_hash,
        allow_bootstrap=False,
    )
    probabilities = bundle.predict_pm_proba(np.zeros(n_features))
    assert bundle.manifest.diagnostic_only is False
    assert tuple(probabilities) == PM_METRICS
    for values in probabilities.values():
        assert set(values) == {"low", "medium", "high"}
        assert np.isfinite(list(values.values())).all()
        assert sum(values.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="preprocessing hash differs"):
        load_model_bundle(
            artifact,
            n_features=n_features,
            sample_rate=256,
            channels=EEG_CHANNELS,
            window_seconds=10,
            feature_profile="lightweight",
            feature_schema_hash_value=schema_hash,
            preprocessing_hash_value="wrong",
            allow_bootstrap=False,
        )
    with pytest.raises(ValueError, match="feature_profile"):
        load_model_bundle(
            artifact,
            n_features=n_features,
            sample_rate=256,
            channels=EEG_CHANNELS,
            window_seconds=10,
            feature_profile="full",
            feature_schema_hash_value=schema_hash,
            preprocessing_hash_value=pre_hash,
            allow_bootstrap=False,
        )


def test_scientific_config_forbids_bootstrap(tmp_path: Path) -> None:
    payload = {
        "source": {"type": "replay", "path": str(tmp_path / "missing.npy")},
        "model": {"artifact_dir": str(tmp_path / "missing"), "allow_bootstrap": False},
    }
    assert WorkerConfig.from_dict(payload).model.allow_bootstrap is False


def test_training_selection_is_deterministic_and_record_bounded() -> None:
    rows = []
    for record in ("a", "b"):
        for index in range(10):
            rows.append(
                {
                    "sample_id": f"{record}-{index:02d}",
                    "record_id": record,
                    "t_start": index * 10.0,
                    **{f"target_{metric}": index / 9 for metric in PM_METRICS},
                }
            )
    frame = pd.DataFrame(rows)
    per_record = np.repeat(np.arange(3), [3, 4, 3])
    labels = np.column_stack([np.tile(per_record, 2) for _ in PM_METRICS])
    first = select_training_rows(frame, labels, max_windows_per_record=4)
    second = select_training_rows(frame, labels, max_windows_per_record=4)
    assert first["sample_id"].tolist() == second["sample_id"].tolist()
    assert first.groupby("record_id").size().max() == 4


def test_runtime_writes_raw_postprocessed_and_total_latency(tmp_path: Path) -> None:
    n_features, schema_hash, pre_hash = _contract()
    artifact = tmp_path / "bundle"
    _real_bundle(artifact, n_features=n_features, schema_hash=schema_hash, pre_hash=pre_hash)
    replay = np.random.default_rng(7).normal(size=(2560, 14)).astype(np.float32)
    replay_path = tmp_path / "replay.npy"
    np.save(replay_path, replay)
    output = tmp_path / "predictions.jsonl"
    config = WorkerConfig.from_dict(
        {
            "source": {"type": "replay", "path": str(replay_path), "realtime": False},
            "preprocessing": {
                "bandpass_enabled": False,
                "notch_enabled": False,
                "faster": False,
            },
            "features": {"profile": "lightweight"},
            "model": {"artifact_dir": str(artifact), "allow_bootstrap": False},
            "output": {"console": False, "jsonl_path": str(output)},
        }
    )
    runtime = StreamingRuntime(config)
    runtime.run()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert runtime.processed_windows == 1
    assert rows[0]["diagnostic_model"] is False
    assert rows[0]["raw_prediction"] is not None
    assert rows[0]["postprocessed_prediction"] is not None
    assert "total_processing" in rows[0]["stage_latencies_ms"]
    assert set(rows[0]["raw_prediction"]["target_probabilities"]) == set(PM_METRICS)

    second_output = tmp_path / "predictions-second.jsonl"
    second_payload = {
        "source": {"type": "replay", "path": str(replay_path), "realtime": False},
        "preprocessing": {"bandpass_enabled": False, "notch_enabled": False, "faster": False},
        "features": {"profile": "lightweight"},
        "model": {"artifact_dir": str(artifact), "allow_bootstrap": False},
        "output": {"console": False, "jsonl_path": str(second_output)},
    }
    StreamingRuntime(WorkerConfig.from_dict(second_payload)).run()
    second = json.loads(second_output.read_text(encoding="utf-8").splitlines()[0])
    assert second["raw_prediction"]["target_probabilities"] == rows[0]["raw_prediction"]["target_probabilities"]


def test_api_exposes_real_model_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    n_features, schema_hash, pre_hash = _contract()
    artifact = tmp_path / "bundle"
    _real_bundle(artifact, n_features=n_features, schema_hash=schema_hash, pre_hash=pre_hash)
    replay_path = tmp_path / "replay.npy"
    np.save(replay_path, np.random.default_rng(8).normal(size=(2560, 14)).astype(np.float32))
    config = WorkerConfig.from_dict(
        {
            "source": {"type": "replay", "path": str(replay_path), "realtime": False},
            "preprocessing": {"bandpass_enabled": False, "notch_enabled": False, "faster": False},
            "features": {"profile": "lightweight"},
            "model": {"artifact_dir": str(artifact), "allow_bootstrap": False},
            "output": {"console": False, "jsonl_path": None},
            "api": {"autostart_worker": True},
        }
    )
    api_jsonl = tmp_path / "api-predictions.jsonl"
    monkeypatch.setenv("COGSTATE_STREAMING_JSONL_PATH", str(api_jsonl))
    app = create_app(config=config)
    assert app.state.streaming_service.config.output.jsonl_path == str(api_jsonl)
    with TestClient(app) as client:
        for _ in range(100):
            status = client.get("/v1/status").json()
            if status["processed_windows"]:
                break
            time.sleep(0.01)
        assert status["processed_windows"] == 1
        assert client.get("/health").status_code == 200
        assert status["model_version"] == "unit-real-v1"
        assert status["diagnostic_model"] is False
        latest = client.get("/v1/predictions/latest").json()["prediction"]
        assert latest["diagnostic_model"] is False
        assert set(latest["raw_prediction"]["target_probabilities"]) == set(PM_METRICS)


@pytest.mark.skipif(
    not Path("data/interim/raw_eeg_window_index_w10_raw_v3.parquet").exists(),
    reason="Canonical raw cache is not available",
)
def test_real_scientific_plan_is_participant_disjoint_and_fold_locked() -> None:
    plan = plan_experiment("configs/streaming_scientific_v1.yaml")
    assert plan["outer_fold"] == 1
    assert plan["participant_overlap"] == []
    assert not set(plan["train_participant_ids"]) & set(plan["test_participant_ids"])
    assert plan["feature_count"] == 399


@pytest.mark.skipif(
    not Path("data/interim/raw_eeg_window_index_w10_raw_v3.parquet").exists(),
    reason="Canonical raw cache is not available",
)
def test_lightweight_plan_reuses_full_scientific_cohort_and_signal_contract() -> None:
    full = plan_experiment("configs/streaming_scientific_v1.yaml")
    light = plan_experiment("configs/streaming_scientific_lightweight_v1.yaml")
    assert light["outer_fold"] == full["outer_fold"] == 1
    assert light["train_participant_ids"] == full["train_participant_ids"]
    assert light["test_participant_ids"] == full["test_participant_ids"]
    assert light["training_sample_ids_hash"] == full["training_sample_ids_hash"]
    assert light["q3_thresholds_hash"] == full["q3_thresholds_hash"]
    assert light["signal_preprocessing_hash"] == full["preprocessing_hash"]
    assert light["feature_count"] == 336
    assert light["feature_schema_hash"] != full["feature_schema_hash"]
    assert light["participant_overlap"] == []
