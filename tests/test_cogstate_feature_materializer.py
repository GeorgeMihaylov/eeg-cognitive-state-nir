from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.features.cogstate_feature_cache import (
    FEATURE_MATRIX_NAME,
    build_canonical_feature_index,
    load_feature_cache,
    materialize_cogstate_features,
    plan_cogstate_feature_cache,
)
from cogstate.features import FeaturePipeline, FeaturePipelineConfig


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(4, 2, 128)).astype(np.float32)
    raw_path = tmp_path / "raw.npy"
    np.save(raw_path, raw)
    manifest = pd.DataFrame(
        {
            "sample_id": [10, 11, 12, 13],
            "record_id": ["r1", "r1", "r2", "r2"],
            "record_group_id": ["g1", "g1", "g2", "g2"],
            "subject_id": ["s1", "s1", "s2", "s2"],
            "status": ["ok"] * 4,
            "cache_file": [str(raw_path)] * 4,
            "cache_offset": list(range(4)),
            "n_channels": [2] * 4,
            "n_samples_expected": [128] * 4,
            "preprocessing_hash": ["raw-hash"] * 4,
            "label_q5": [0, 1, 2, 0],
            "target_focus": [0.1, 0.2, 0.3, 0.4],
            "outer_fold": [1, 1, 2, 2],
            "source": ["a"] * 4,
            "t_start": [0.0, 0.5, 0.0, 0.5],
            "t_end": [0.5, 1.0, 0.5, 1.0],
        }
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    logical = pd.DataFrame(
        {"record_group_id": ["g1", "g2"], "selected_record_id": ["r1", "r2"]}
    )
    logical_path = tmp_path / "logical.parquet"
    logical.to_parquet(logical_path, index=False)
    profile = {
        "schema_version": "cogstate-features-v1",
        "selection": "none",
        "feature_pipeline": {
            "sample_rate": 128,
            "channel_names": ["C1", "C2"],
            "groups": {
                "spectral": False,
                "statistical": True,
                "entropy": False,
                "connectivity": False,
            },
            "statistical": {},
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return manifest_path, logical_path, profile_path, raw_path


def test_tiny_materialization_reload_and_direct_transform(tmp_path: Path) -> None:
    manifest, logical, profile, raw_path = _fixture(tmp_path)
    output = tmp_path / "features"
    summary = materialize_cogstate_features(
        manifest_path=manifest,
        logical_recording_map_path=logical,
        cache_path_root=tmp_path,
        feature_profile_path=profile,
        output_dir=output,
        chunk_size=2,
    )
    matrix, index, names, cache_manifest = load_feature_cache(output)

    assert summary["rows"] == 4
    assert summary["n_features"] == len(names) == 20
    assert matrix.shape == (4, 20)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    assert index["sample_id"].tolist() == [10, 11, 12, 13]
    assert "label_q5" not in index.columns
    assert "target_focus" not in index.columns
    assert cache_manifest["identity"]["target_columns_present"] is False
    assert summary["target_columns_present"] is False
    assert cache_manifest["status"] == "complete"
    pipeline = FeaturePipeline(
        FeaturePipelineConfig.from_mapping(json.loads(profile.read_text()))
    )
    raw = np.load(raw_path)
    expected = pipeline.transform_window(raw[2].T).astype(np.float32)
    np.testing.assert_allclose(matrix[2], expected, rtol=1e-6, atol=1e-6)


def test_completed_cache_is_resume_safe(tmp_path: Path) -> None:
    manifest, logical, profile, _ = _fixture(tmp_path)
    output = tmp_path / "features"
    kwargs = dict(
        manifest_path=manifest,
        logical_recording_map_path=logical,
        cache_path_root=tmp_path,
        feature_profile_path=profile,
        output_dir=output,
        chunk_size=2,
    )
    first = materialize_cogstate_features(**kwargs)
    before = (output / FEATURE_MATRIX_NAME).read_bytes()
    second = materialize_cogstate_features(**kwargs, resume=True)
    assert first["cache_identity_hash"] == second["cache_identity_hash"]
    assert (output / FEATURE_MATRIX_NAME).read_bytes() == before


def test_duplicate_sample_id_is_rejected(tmp_path: Path) -> None:
    manifest, logical, _, _ = _fixture(tmp_path)
    frame = pd.read_parquet(manifest)
    frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    frame.to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_canonical_feature_index(manifest, logical)


def test_incompatible_resume_identity_is_rejected(tmp_path: Path) -> None:
    manifest, logical, profile, _ = _fixture(tmp_path)
    output = tmp_path / "features"
    materialize_cogstate_features(
        manifest_path=manifest,
        logical_recording_map_path=logical,
        cache_path_root=tmp_path,
        feature_profile_path=profile,
        output_dir=output,
    )
    payload = json.loads(profile.read_text())
    payload["feature_pipeline"]["groups"]["spectral"] = True
    profile.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible feature cache identity"):
        materialize_cogstate_features(
            manifest_path=manifest,
            logical_recording_map_path=logical,
            cache_path_root=tmp_path,
            feature_profile_path=profile,
            output_dir=output,
            resume=True,
        )


def test_plan_only_is_target_free_deterministic_and_writes_nothing(
    tmp_path: Path,
) -> None:
    manifest, logical, profile, _ = _fixture(tmp_path)
    output = tmp_path / "planned-cache"
    kwargs = dict(
        manifest_path=manifest,
        logical_recording_map_path=logical,
        cache_path_root=tmp_path,
        feature_profile_path=profile,
        output_dir=output,
    )
    first = plan_cogstate_feature_cache(**kwargs)
    second = plan_cogstate_feature_cache(**kwargs)
    assert first == second
    assert first["status"] == "plan_only"
    assert first["expected_matrix_shape"] == [4, 20]
    assert first["source_target_columns_excluded"] == [
        "label_q5",
        "target_focus",
    ]
    assert first["target_columns_present"] is False
    assert first["identity"]["target_columns_present"] is False
    assert not output.exists()


def test_loader_rejects_legacy_index_with_target_columns(tmp_path: Path) -> None:
    manifest, logical, profile, _ = _fixture(tmp_path)
    output = tmp_path / "features"
    materialize_cogstate_features(
        manifest_path=manifest,
        logical_recording_map_path=logical,
        cache_path_root=tmp_path,
        feature_profile_path=profile,
        output_dir=output,
    )
    index_path = output / "feature_index.parquet"
    index = pd.read_parquet(index_path)
    index["label_q5"] = 0
    index.to_parquet(index_path, index=False)
    with pytest.raises(ValueError, match="contains target columns"):
        load_feature_cache(output)
