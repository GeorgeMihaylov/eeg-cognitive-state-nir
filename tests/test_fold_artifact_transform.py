from __future__ import annotations

import numpy as np
import pandas as pd

from bench.preprocessing.fold_artifact_transform import (
    ArtifactTransformedRawView,
    FoldArtifactConfig,
    FoldArtifactTransform,
    select_calibration_indices,
)
from cogstate.preprocessing.artifact_removal import IcaConfig


class TinyRawView:
    is_lazy_raw_eeg = True

    def __init__(self, values: np.ndarray, manifest: pd.DataFrame) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.manifest = manifest.reset_index(drop=True)
        self.shape = self.values.shape
        self.ndim = 4
        self.dtype = np.dtype(np.float32)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        if np.isscalar(index):
            return self.values[int(index)]
        positions = np.arange(len(self))[index]
        return TinyRawView(self.values[positions], self.manifest.iloc[positions])


def _view(seed: int = 42) -> TinyRawView:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(8, 1, 4, 300)).astype(np.float32)
    manifest = pd.DataFrame({
        "sample_id": np.arange(8),
        "subject_id": np.repeat(["s1", "s2", "s3", "s4"], 2),
        "record_id": [f"r{i // 2}" for i in range(8)],
        "record_group_id": [f"g{i // 2}" for i in range(8)],
        "t_start": np.tile([0.0, 10.0], 4),
        "status": "ok",
    })
    return TinyRawView(values, manifest)


def test_calibration_selection_is_deterministic_and_train_bounded() -> None:
    view = _view()
    first = select_calibration_indices(view.manifest, max_windows=4)
    second = select_calibration_indices(view.manifest, max_windows=4)
    np.testing.assert_array_equal(first, second)
    assert set(view.manifest.iloc[first]["subject_id"]) == {"s1", "s2", "s3", "s4"}


def test_raw_variant_is_identity_and_does_not_fit_ica_or_faster() -> None:
    view = _view()
    transform = FoldArtifactTransform(FoldArtifactConfig("raw", sample_rate=100)).fit(
        view, fold=1
    )
    wrapped = ArtifactTransformedRawView(view, transform)
    np.testing.assert_array_equal(wrapped[0], view[0])
    assert transform.ica_ is None
    assert transform.manifest_["calibration_windows"] == 0
    assert transform.manifest_["faster_semantics"] == "not_applied"


def test_fold_ica_fits_once_on_train_and_reuses_state_for_test() -> None:
    train = _view(1)
    test = _view(2)
    test.manifest["subject_id"] = "outer-test"
    transform = FoldArtifactTransform(
        FoldArtifactConfig(
            "ica",
            sample_rate=100,
            calibration_max_windows=4,
            ica=IcaConfig(n_components=4, max_iter=300, random_state=42),
        )
    ).fit(train, fold=1)
    state_hash = transform.manifest_["ica_state_hash"]
    wrapped = ArtifactTransformedRawView(test, transform)
    first = wrapped[0]
    second = wrapped[0]
    np.testing.assert_allclose(first, second)
    assert np.isfinite(first).all()
    assert transform.fit_count_ == 1
    assert transform.transform_calls_ == 2
    assert transform.transform_seconds_ > 0
    assert transform.runtime_diagnostics()["transform_calls"] == 2
    assert transform.manifest_["ica_fit_scope"] == "outer_train_only"
    assert "outer-test" not in transform.manifest_["outer_train_participants"]
    assert transform.manifest_["ica_state_hash"] == state_hash


def test_faster_ica_order_and_lazy_channel_normalization() -> None:
    view = _view(3)
    transform = FoldArtifactTransform(
        FoldArtifactConfig(
            "faster_ica",
            sample_rate=100,
            calibration_max_windows=4,
            ica=IcaConfig(n_components=4, max_iter=300, random_state=42),
        )
    ).fit(view, fold=2)
    wrapped = ArtifactTransformedRawView(view, transform)
    mean, scale = wrapped.compute_channel_statistics()
    normalized = wrapped.with_channel_normalization(mean, scale)
    assert normalized[0].shape == (1, 4, 300)
    assert np.isfinite(normalized[0]).all()
    assert transform.manifest_["operation_order"] == [
        "canonical_raw_cache",
        "apply_faster_per_window",
        "fold_fitted_ica_transform",
        "train_only_channel_normalization_in_torch_adapter",
    ]


def test_transformed_window_cache_is_shared_with_slices_and_normalized_views() -> None:
    view = _view(4)
    transform = FoldArtifactTransform(
        FoldArtifactConfig("faster", sample_rate=100)
    ).fit(view, fold=1)
    wrapped = ArtifactTransformedRawView(
        view, transform, cache_transformed_windows=True
    )
    first = wrapped[0]
    sliced = wrapped[[0, 1]]
    np.testing.assert_allclose(sliced[0], first)
    normalized = sliced.with_channel_normalization(
        np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    )
    np.testing.assert_allclose(normalized[0], first)
    assert transform.transform_calls_ == 1
    assert wrapped.cache_diagnostics() == {
        "enabled": True,
        "entries": 1,
        "hits": 2,
        "misses": 1,
        "estimated_bytes": 4 * 300 * 4,
    }
