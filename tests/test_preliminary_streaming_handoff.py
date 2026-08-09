from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bench.experiments.preliminary_streaming_handoff import (
    HANDOFF_SCHEMA_VERSION,
    _read_csv_records,
    _run_manifest_row,
    _target_ids,
    _target_slug,
    measure_single_window_latency,
)
from model_zoo import build_model


def test_preliminary_config_is_fold_one_raw_and_has_fourteen_targets() -> None:
    path = Path("experiments/targets/preliminary_streaming_handoff_shallow_fold1.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["schema_version"] == HANDOFF_SCHEMA_VERSION
    assert config["evaluation"]["folds"] == [1]
    assert config["evaluation"]["precomputed_fold_column"] == "outer_fold"
    assert config["validation"]["group_column"] == "record_group_id"
    assert config["raw_preprocessing"]["bandpass"]["enabled"] is False
    assert config["raw_preprocessing"]["notch"]["enabled"] is False
    assert len(_target_ids(config["targets"])) == 14


def test_windows_safe_target_slugs_are_unique() -> None:
    target_ids = _target_ids(
        ["attention", "engagement", "excitement", "stress", "relaxation", "interest", "focus"]
    )
    slugs = [_target_slug(target_id) for target_id in target_ids]

    assert len(slugs) == len(set(slugs)) == 14
    assert max(map(len, slugs)) < max(map(len, target_ids))


def test_empty_partial_csv_is_resume_safe(tmp_path: Path) -> None:
    empty = tmp_path / "latency.csv"
    empty.write_text("\n", encoding="utf-8")
    assert _read_csv_records(empty) == []

    pd.DataFrame([{"target_id": "pm_attention_regression"}]).to_csv(
        empty, index=False
    )
    assert _read_csv_records(empty) == [
        {"target_id": "pm_attention_regression"}
    ]


def test_run_manifest_omits_inapplicable_nan_fields() -> None:
    row = _run_manifest_row({
        "target_id": "pm_attention_regression",
        "status": "completed",
        "target_transform": np.nan,
    })
    assert row == {
        "target_id": "pm_attention_regression",
        "status": "completed",
    }


def test_single_window_latency_contract_on_cpu() -> None:
    adapter = build_model(
        "torch_shallow_convnet",
        "regression",
        input_shape=(1, 2, 64),
        num_outputs=1,
        params={
            "n_filters": 2,
            "temporal_kernel_samples": 5,
            "pool_size": 8,
            "pool_stride": 4,
            "dropout": 0.0,
            "device": "cpu",
            "standardize": True,
            "random_state": 42,
        },
    )
    adapter.feature_mean_ = np.zeros(2, dtype=np.float32)
    adapter.feature_scale_ = np.ones(2, dtype=np.float32)
    adapter.is_fitted_ = True
    rows = measure_single_window_latency(
        adapter,
        np.zeros((1, 2, 64), dtype=np.float32),
        warmup=1,
        repetitions=3,
    )

    assert [row["latency_mode"] for row in rows] == [
        "model_only",
        "channel_normalization_plus_model",
    ]
    assert all(row["iterations"] == 3 for row in rows)
    assert all(row["batch_size"] == 1 for row in rows)
    assert all(np.isfinite(row["p95_ms"]) for row in rows)
